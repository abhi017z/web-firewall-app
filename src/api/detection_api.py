from __future__ import annotations

import os
import time
import uuid
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Security, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from fastapi.security.api_key import APIKeyHeader
import torch

from ..models.inference import InferenceEngine
from ..preprocessing.tokenizer import HTTPRequestTokenizer
from ..storage.detection_store import detection_store
from ..utils.config import load_config, CONFIG_PATH
from ..utils.logger import logger, setup_logging
import json as _json
from pathlib import Path as _Path


class DetectionRequest(BaseModel):
    method: str
    path: str
    query_params: Dict[str, str]
    headers: Dict[str, str]
    body: str = ""


class DetectionResponse(BaseModel):
    is_anomaly: bool
    confidence: float
    anomaly_score: float
    timestamp: str
    request_id: str


class DetectionHistoryResponse(BaseModel):
    id: int
    request_id: str
    timestamp: float
    method: str
    path: str
    anomaly_score: float
    is_anomaly: bool
    model_version: int
    threshold: float
    client_ip_hash: Optional[str]
    notes: Optional[str]
    created_at: str


class DetectionStatsResponse(BaseModel):
    total_requests: int
    total_anomalies: int
    avg_score: float
    detection_rate: float
    first_detection: Optional[float]
    last_detection: Optional[float]


app = FastAPI(title="WAF Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the detection history UI
static_path = _Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


CONFIG = load_config(CONFIG_PATH)
ENGINE: InferenceEngine | None = None
TOKENIZER: HTTPRequestTokenizer | None = None
API_KEY = os.environ.get("WAF_API_KEY", "dev-key")
REQUIRE_API_KEY = os.environ.get("WAF_REQUIRE_API_KEY", "false").strip().lower() in {"1", "true", "yes", "on"}
METRICS = {"requests": 0, "anomalies": 0}
_RATE_WINDOW_S = 60
_RATE_LIMIT = 1000
_RATE_BUCKET: dict[str, list[float]] = {}

# Swagger-friendly API key (enables Authorize button)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> bool:
    if not REQUIRE_API_KEY:
        return True
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


def _auth(request: Request) -> None:
    if not REQUIRE_API_KEY:
        return
    key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _rate_limit(request: Request) -> None:
    now = time.time()
    ip = request.client.host if request.client else "unknown"
    bucket = _RATE_BUCKET.setdefault(ip, [])
    # Drop old
    threshold = now - _RATE_WINDOW_S
    while bucket and bucket[0] < threshold:
        bucket.pop(0)
    if len(bucket) >= _RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    bucket.append(now)


def _append_detection_log(record: Dict[str, object]) -> None:
    try:
        logs_dir = _Path(os.path.dirname(os.path.dirname(__file__))) / ".." / "logs" / "detection_logs"
        logs_dir = logs_dir.resolve()
        os.makedirs(str(logs_dir), exist_ok=True)
        with open(str(logs_dir / "detections.jsonl"), "a", encoding="utf-8") as f:
            f.write(_json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


@app.on_event("startup")
async def load_model() -> None:  # noqa: D401
    setup_logging(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "logs", "detection_logs"))
    global ENGINE, TOKENIZER
    threshold = float(CONFIG.get("detection", {}).get("threshold", 0.75))
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models", "checkpoints", "best.pt")
    vocab_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models", "checkpoints", "vocab.json")
    ENGINE = InferenceEngine(model_path=model_path, threshold=threshold)
    try:
        ENGINE.load_model()
        ckpt = torch.load(model_path, map_location="cpu")
        meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
        saved_threshold = meta.get("threshold")
        if saved_threshold is not None:
            ENGINE.threshold = float(saved_threshold)
        elif meta.get("threshold_percentile") is not None:
            logger.info("Checkpoint has no calibrated threshold; using config fallback.")
    except FileNotFoundError:
        logger.warning("Model checkpoint not found, API will return dummy responses until trained.")
        ENGINE.model = None  # type: ignore[assignment]
    TOKENIZER = HTTPRequestTokenizer(vocab_size=int(CONFIG.get("model", {}).get("vocab_size", 10000)))
    if os.path.exists(vocab_path):
        try:
            TOKENIZER.load_vocab(vocab_path)
        except Exception as exc:
            logger.warning("Failed to load tokenizer vocab from %s: %s", vocab_path, exc)
    else:
        logger.warning("Tokenizer vocab not found at %s; inference may be inaccurate until retrained.", vocab_path)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def home() -> FileResponse:
    static_file = _Path(__file__).parent / "static" / "control_center.html"
    return FileResponse(static_file)


@app.get("/metrics")
async def metrics() -> Dict[str, int]:
    return dict(METRICS)


@app.post("/threshold")
async def update_threshold(value: float, _: bool = Security(verify_api_key)) -> Dict[str, float]:
    global ENGINE
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    ENGINE.update_threshold(value)
    return {"threshold": value}


def _encode_request(payload: DetectionRequest, max_len: int) -> Dict[str, List[int]]:
    assert TOKENIZER is not None
    composed = f"{payload.method} {payload.path}?" + "&".join(f"{k}={v}" for k, v in sorted(payload.query_params.items()))
    if payload.body:
        composed += " BODY:" + payload.body
    return TOKENIZER.encode(composed, max_length=max_len)


@app.post("/detect", response_model=DetectionResponse)
async def detect_anomaly(payload: DetectionRequest, request: Request, _: bool = Security(verify_api_key)) -> DetectionResponse:
    _rate_limit(request)
    global ENGINE
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    METRICS["requests"] += 1
    if ENGINE.model is None:
        rid = str(uuid.uuid4())
        resp = DetectionResponse(is_anomaly=False, confidence=0.0, anomaly_score=0.0, timestamp=str(time.time()), request_id=rid)
        _append_detection_log({
            "request_id": rid,
            "method": payload.method,
            "path": payload.path,
            "query_params": payload.query_params,
            "headers": payload.headers,
            "is_anomaly": resp.is_anomaly,
            "anomaly_score": resp.anomaly_score,
            "confidence": resp.confidence,
            "ts": resp.timestamp,
        })
        return resp
    max_len_cfg = int(CONFIG.get("model", {}).get("max_seq_length", 512))
    if ENGINE.model is not None:
        max_len_model = int(getattr(getattr(ENGINE.model, "positional", None), "num_embeddings", max_len_cfg))
        max_len = min(max_len_cfg, max_len_model)
    else:
        max_len = max_len_cfg
    encoded = _encode_request(payload, max_len)
    result = ENGINE.predict_single(encoded["input_ids"], encoded["attention_mask"])
    rid = str(uuid.uuid4())
    if result["is_anomaly"]:  # type: ignore[index]
        METRICS["anomalies"] += 1
    resp = DetectionResponse(
        is_anomaly=bool(result["is_anomaly"]),  # type: ignore[index]
        confidence=float(result["confidence"]),  # type: ignore[index]
        anomaly_score=float(result["anomaly_score"]),  # type: ignore[index]
        timestamp=str(time.time()),
        request_id=rid,
    )
    
    # Store in structured detection store (non-blocking)
    detection_store.store_detection(
        request_id=rid,
        method=payload.method,
        path=payload.path,
        query_params=payload.query_params,
        headers=payload.headers,
        body=payload.body,
        anomaly_score=resp.anomaly_score,
        is_anomaly=resp.is_anomaly,
        model_version=getattr(ENGINE, 'model_version', 1),
        threshold=getattr(ENGINE, 'threshold', 8.5),
        client_ip=request.client.host if request.client else None
    )
    
    # Keep legacy logging for backward compatibility
    _append_detection_log({
        "request_id": rid,
        "method": payload.method,
        "path": payload.path,
        "query_params": payload.query_params,
        "headers": payload.headers,
        "is_anomaly": resp.is_anomaly,
        "anomaly_score": resp.anomaly_score,
        "confidence": resp.confidence,
        "ts": resp.timestamp,
    })
    return resp


@app.post("/detect/batch")
async def detect_batch(requests: List[DetectionRequest], request: Request, _: bool = Security(verify_api_key)):
    _rate_limit(request)
    global ENGINE
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    METRICS["requests"] += len(requests)
    if ENGINE.model is None:
        out = []
        for _ in requests:
            rec = {
                "is_anomaly": False,
                "confidence": 0.0,
                "anomaly_score": 0.0,
                "timestamp": str(time.time()),
                "request_id": str(uuid.uuid4()),
            }
            out.append(rec)
            _append_detection_log({"request_id": rec["request_id"], "batch": True, **rec})
        return out
    max_len_cfg = int(CONFIG.get("model", {}).get("max_seq_length", 512))
    if ENGINE.model is not None:
        max_len_model = int(getattr(getattr(ENGINE.model, "positional", None), "num_embeddings", max_len_cfg))
        max_len = min(max_len_cfg, max_len_model)
    else:
        max_len = max_len_cfg
    encoded_inputs = []
    encoded_masks = []
    for p in requests:
        enc = _encode_request(p, max_len)
        encoded_inputs.append(enc["input_ids"])  # type: ignore[index]
        encoded_masks.append(enc["attention_mask"])  # type: ignore[index]
    results = ENGINE.predict_batch(encoded_inputs, encoded_masks)
    out = []
    for i, r in enumerate(results):
        if r["is_anomaly"]:  # type: ignore[index]
            METRICS["anomalies"] += 1
        rec = {
            "is_anomaly": bool(r["is_anomaly"]),  # type: ignore[index]
            "confidence": float(r["confidence"]),  # type: ignore[index]
            "anomaly_score": float(r["anomaly_score"]),  # type: ignore[index]
            "timestamp": str(time.time()),
            "request_id": str(uuid.uuid4()),
        }
        out.append(rec)
        _append_detection_log({"request_id": rec["request_id"], "batch": True, **rec})
        
        # Store each batch item in structured store
        detection_store.store_detection(
            request_id=rec["request_id"],
            method=requests[i].method,
            path=requests[i].path,
            query_params=requests[i].query_params,
            headers=requests[i].headers,
            body=requests[i].body,
            anomaly_score=rec["anomaly_score"],
            is_anomaly=rec["is_anomaly"],
            model_version=getattr(ENGINE, 'model_version', 1),
            threshold=getattr(ENGINE, 'threshold', 8.5),
            client_ip=request.client.host if request.client else None,
            notes="batch_request"
        )
    return out


@app.get("/logs")
async def detection_logs_dashboard() -> FileResponse:
    static_file = _Path(__file__).parent / "static" / "logs_dashboard.html"
    return FileResponse(static_file)


@app.get("/logs/json", response_model=List[DetectionHistoryResponse])
async def get_detection_logs_json(
    from_time: Optional[str] = Query(None, description="Start time (ISO format or timestamp)"),
    to_time: Optional[str] = Query(None, description="End time (ISO format or timestamp)"),
    is_anomaly: Optional[bool] = Query(None, description="Filter by anomaly status"),
    min_score: Optional[float] = Query(None, description="Minimum anomaly score"),
    max_score: Optional[float] = Query(None, description="Maximum anomaly score"),
    path_pattern: Optional[str] = Query(None, description="Path pattern to match"),
    limit: int = Query(100, description="Maximum results"),
    offset: int = Query(0, description="Results offset"),
    _: bool = Security(verify_api_key)
) -> List[DetectionHistoryResponse]:
    """
    Get detection history with filtering options.
    
    Supports filtering by time range, anomaly status, score range, and path patterns.
    Results are ordered by most recent first.
    """
    
    # Parse timestamps
    from_timestamp = None
    to_timestamp = None
    
    if from_time:
        try:
            # Try parsing as ISO format first
            from_timestamp = datetime.fromisoformat(from_time.replace('Z', '+00:00')).timestamp()
        except ValueError:
            # Try parsing as raw timestamp
            try:
                from_timestamp = float(from_time)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid from_time format")
    
    if to_time:
        try:
            to_timestamp = datetime.fromisoformat(to_time.replace('Z', '+00:00')).timestamp()
        except ValueError:
            try:
                to_timestamp = float(to_time)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid to_time format")
    
    # Query detection store
    detections = detection_store.get_detections(
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
        is_anomaly=is_anomaly,
        min_score=min_score,
        max_score=max_score,
        path_pattern=path_pattern,
        limit=limit,
        offset=offset
    )
    
    # Convert to response format
    return [
        DetectionHistoryResponse(
            id=d["id"],
            request_id=d["request_id"],
            timestamp=d["timestamp"],
            method=d["method"],
            path=d["path"],
            anomaly_score=d["anomaly_score"],
            is_anomaly=bool(d["is_anomaly"]),
            model_version=d["model_version"],
            threshold=d["threshold"],
            client_ip_hash=d["client_ip_hash"],
            notes=d["notes"],
            created_at=d["created_at"]
        )
        for d in detections
    ]


@app.post("/replay/{request_id}", response_model=DetectionResponse)
async def replay_detection(
    request_id: str,
    _: bool = Security(verify_api_key)
) -> DetectionResponse:
    """
    Replay a stored request through the current model.
    
    Useful for debugging and comparing how the current model would score
    a previously seen request.
    """
    
    # Get original detection record
    original = detection_store.get_detection_by_id(request_id)
    if not original:
        raise HTTPException(status_code=404, detail="Request ID not found")
    
    # Parse the normalized request back to components
    try:
        request_data = _json.loads(original["normalized_request"])
    except _json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid stored request data")
    
    # Create detection request
    payload = DetectionRequest(
        method=request_data["method"],
        path=request_data["path"],
        query_params=request_data.get("query_params", {}),
        headers=request_data.get("headers", {}),
        body=request_data.get("body", "")
    )
    
    # Run through current model
    global ENGINE
    if ENGINE is None or ENGINE.model is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    
    max_len_cfg = int(CONFIG.get("model", {}).get("max_seq_length", 512))
    if ENGINE.model is not None:
        max_len_model = int(getattr(getattr(ENGINE.model, "positional", None), "num_embeddings", max_len_cfg))
        max_len = min(max_len_cfg, max_len_model)
    else:
        max_len = max_len_cfg
    
    encoded = _encode_request(payload, max_len)
    result = ENGINE.predict_single(encoded["input_ids"], encoded["attention_mask"])
    
    # Generate new request ID for replay
    replay_id = f"replay-{request_id}-{str(uuid.uuid4())[:8]}"
    
    resp = DetectionResponse(
        is_anomaly=bool(result["is_anomaly"]),  # type: ignore[index]
        confidence=float(result["confidence"]),  # type: ignore[index]
        anomaly_score=float(result["anomaly_score"]),  # type: ignore[index]
        timestamp=str(time.time()),
        request_id=replay_id,
    )
    
    # Store replay result with notes
    detection_store.store_detection(
        request_id=replay_id,
        method=payload.method,
        path=payload.path,
        query_params=payload.query_params,
        headers=payload.headers,
        body=payload.body,
        anomaly_score=resp.anomaly_score,
        is_anomaly=resp.is_anomaly,
        model_version=getattr(ENGINE, 'model_version', 1),
        threshold=getattr(ENGINE, 'threshold', 8.5),
        notes=f"replay_of_{request_id}"
    )
    
    return resp


@app.get("/stats")
async def detection_stats_dashboard() -> FileResponse:
    static_file = _Path(__file__).parent / "static" / "stats_dashboard.html"
    return FileResponse(static_file)


@app.get("/stats/json", response_model=DetectionStatsResponse)
async def get_detection_stats(_: bool = Security(verify_api_key)) -> DetectionStatsResponse:
    """
    Get detection statistics and performance metrics.
    
    Provides overview of total requests, anomalies, scores, and detection rates.
    """
    stats = detection_store.get_stats()
    
    return DetectionStatsResponse(
        total_requests=stats["total_requests"],
        total_anomalies=stats["total_anomalies"],
        avg_score=stats["avg_score"] or 0.0,
        detection_rate=stats["detection_rate"],
        first_detection=stats["first_detection"],
        last_detection=stats["last_detection"]
    )


@app.post("/cleanup")
async def cleanup_old_records(
    retention_days: int = Query(30, description="Days to retain records"),
    _: bool = Security(verify_api_key)
) -> Dict[str, int]:
    """
    Clean up old detection records beyond retention period.
    
    Default retention is 30 days for privacy compliance.
    """
    deleted_count = detection_store.cleanup_old_records(retention_days)
    return {"deleted_records": deleted_count}


@app.get("/history")
async def detection_history_ui():
    """
    Serve the detection history web UI.
    
    Provides a user-friendly interface to view, filter, and replay detections.
    """
    static_file = _Path(__file__).parent / "static" / "detection_history.html"
    return FileResponse(static_file)


