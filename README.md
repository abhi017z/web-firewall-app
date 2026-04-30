# Transformer-based Web Application Firewall (WAF)

This repository contains a working prototype of a Web Application Firewall that uses a Transformer model to flag anomalous HTTP requests. It was built for the Smart India Hackathon (SIH) and comes with simple integration points for Nginx/Apache and demo apps to showcase end‑to‑end behavior.

The focus is practicality: small, readable code, a repeatable demo, and clear setup instructions.

## Quick links

- Detection History (local): [http://localhost:8000/history](http://localhost:8000/history)
- Detection API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Update API docs: [http://localhost:8001/docs](http://localhost:8001/docs)

## What this does

- Detects suspicious requests in real time using a trained Transformer (no rule lists)
- Stores detection events for audit, filtering, replay, and incremental retraining
- Provides a small web UI for browsing detection history
- Ships with scripts to generate benign data, train quickly, and run a live demo

## Prerequisites

- Python 3.10+
- macOS/Linux (tested on macOS)
- Optional: Docker and Docker Compose

## Quick start (local)

```bash
cd transformer-waf
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start APIs (two terminals or run in background)
PYTHONPATH=. uvicorn src.api.detection_api:app --host 0.0.0.0 --port 8000
PYTHONPATH=. uvicorn src.api.update_api:app --host 0.0.0.0 --port 8001
```

Generate a small dataset and train a lightweight model (enough for the demo):

```bash
PYTHONPATH=. python scripts/generate_benign.py
PYTHONPATH=. python scripts/train_quick.py
```

Run the guided demo (recommended for judges):

```bash
./scripts/demo_presentation.sh
```

You can also start everything with one helper script:

```bash
./scripts/start_demo_services.sh
```

## How to use it during evaluation

1. Open Detection API docs: `http://localhost:8000/docs`
2. Send a benign request to `/detect` (see examples below)
3. Send an attack‑like request (SQLi/XSS sample) and note the anomaly score
4. Open the history UI: `http://localhost:8000/history` and filter recent events
5. (Optional) Trigger retraining via Update API: `http://localhost:8001/docs`

Example requests:

```bash
# Benign
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" -H "X-API-Key: dev-key" \
  -d '{"method":"GET","path":"/products","query_params":{"q":"laptop"},"headers":{},"body":""}'

# Attack‑like (XSS)
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" -H "X-API-Key: dev-key" \
  -d '{"method":"GET","path":"/search","query_params":{"q":"<script>alert(1)</script>"},"headers":{},"body":""}'
```

History UI: `http://localhost:8000/history`  
Swagger: `http://localhost:8000/docs` and `http://localhost:8001/docs`

## Docker (optional)

```bash
docker-compose up -d
# WAF API  : http://localhost:8000
# Update API: http://localhost:8001
# Nginx     : http://localhost:80
# Sample apps: http://localhost:8081-8083
```

## Project layout (short version)

```
transformer-waf/
├── src/
│   ├── api/            # FastAPI apps (detection/update)
│   ├── ingestion/      # Batch/stream log ingestion
│   ├── preprocessing/  # Parser, normalizer, tokenizer
│   ├── models/         # Transformer model + training
│   ├── storage/        # SQLite store for detections
│   └── utils/          # Config, logging
├── integration/        # Nginx/Apache helpers
├── scripts/            # Train/demo/test scripts
├── sample_apps/        # Small demo services
└── tests/              # Unit tests and attack payloads
```

## Tech stack (simple)

- FastAPI + Uvicorn: lightweight web servers for the Detection API (port 8000) and Update API (port 8001).
- Python + PyTorch: trains and runs the Transformer model that assigns an anomaly score to requests.
- SQLite: stores structured detection events so you can filter, review, and replay later.
- Nginx/Apache (optional): integration examples to mirror/forward traffic to the detector for evaluation.
- Docker Compose (optional): spins up APIs, demo apps, and Nginx quickly for a live demo.

Why a Transformer? It learns patterns from normal requests (paths, params, headers) and flags out‑of‑pattern inputs, so you don’t need to hand‑craft rules.

## How detection works (in short)

1. The request is parsed and normalized (remove noise, standardize fields).
2. It is tokenized into a sequence the model understands.
3. The Transformer produces an anomaly score.
4. If the score crosses the threshold, we mark it as suspicious and log it.
5. All events (benign and suspicious) can be browsed in the History UI and used for retraining.

## API endpoints (simple)

Base URL for Detection API: `http://localhost:8000`

- POST `/detect`: Analyze one HTTP request. Returns `is_anomaly` and `score`.

  Example body:

  ```json
  {
    "method": "GET",
    "path": "/search",
    "query_params": {"q": "<script>alert(1)</script>"},
    "headers": {},
    "body": ""
  }
  ```

- POST `/detect/batch`: Analyze many requests at once. Same schema, but send a list.

- GET `/logs`: List detection events with optional filters like `is_anomaly`, `limit`, `path`.

  Example: `GET /logs?is_anomaly=true&limit=10`

- POST `/replay/{request_id}`: Re-run a past request with the current model (useful after retraining).

- GET `/stats`: Summary counts (how many anomalies, totals, etc.).

- GET `/metrics`: Performance metrics (latencies, throughput) for quick checks.

- POST `/threshold`: Update the anomaly cutoff. Lower = more sensitive.

- POST `/cleanup`: Remove old records based on retention settings.

- GET `/history`: Simple web page to browse and filter events.

Base URL for Update API: `http://localhost:8001`

- POST `/retrain`: Start an incremental retraining job on new benign data.
- GET `/retrain/status`: Check the current training job’s progress.
- POST `/add_benign_data`: Add fresh benign samples to the training set.

## Configuration

Key settings live in `config.yaml` (threshold, retention, ports). Reasonable defaults are provided for the demo.

## Model artifacts

To keep the repo lean, large checkpoint files are not stored in Git. The quick‑train script creates a small checkpoint locally under `models/checkpoints/`. If you need a heavier model for experiments, store it outside Git (e.g., release assets or object storage) and point the loader to its path.

## Troubleshooting

- If ports are occupied, stop old `uvicorn` processes or change ports in `config.yaml`.
- If the history page is empty, send a few requests first, then refresh.
- If `pip install` fails, ensure Python 3.10+ and try `pip install --upgrade pip`.

## License

Built for Smart India Hackathon 2025.

— Team SecuraFormer
