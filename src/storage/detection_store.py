from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class DetectionStore:
	def __init__(self, db_path: Optional[Path] = None) -> None:
		root = Path(__file__).resolve().parents[2]
		default_db = root / "logs" / "detection_logs" / "detections.db"
		self.db_path = db_path or default_db
		self.db_path.parent.mkdir(parents=True, exist_ok=True)
		self._lock = threading.Lock()
		self._initialize()

	def _get_conn(self) -> sqlite3.Connection:
		conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
		conn.row_factory = sqlite3.Row
		return conn

	def _initialize(self) -> None:
		with self._get_conn() as conn:
			conn.execute(
				"""
				CREATE TABLE IF NOT EXISTS detections (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					request_id TEXT NOT NULL UNIQUE,
					timestamp REAL NOT NULL,
					method TEXT NOT NULL,
					path TEXT NOT NULL,
					anomaly_score REAL NOT NULL,
					is_anomaly INTEGER NOT NULL,
					model_version INTEGER NOT NULL,
					threshold REAL NOT NULL,
					client_ip_hash TEXT,
					normalized_request TEXT NOT NULL,
					notes TEXT,
					created_at TEXT NOT NULL
				)
				"""
			)
			conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp)")
			conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_is_anomaly ON detections(is_anomaly)")
			conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_path ON detections(path)")
			conn.commit()

	def store_detection(
		self,
		request_id: str,
		method: str,
		path: str,
		query_params: Dict[str, Any],
		headers: Dict[str, Any],
		body: str,
		anomaly_score: float,
		is_anomaly: bool,
		model_version: int,
		threshold: float,
		client_ip: Optional[str] = None,
		notes: Optional[str] = None,
	) -> None:
		now = time.time()
		payload = {
			"method": method,
			"path": path,
			"query_params": query_params or {},
			"headers": headers or {},
			"body": body or "",
		}
		normalized_request = json.dumps(payload, ensure_ascii=False)
		client_ip_hash = hashlib.sha256(client_ip.encode("utf-8")).hexdigest() if client_ip else None
		created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))

		with self._lock:
			with self._get_conn() as conn:
				conn.execute(
					"""
					INSERT OR REPLACE INTO detections (
						request_id, timestamp, method, path, anomaly_score,
						is_anomaly, model_version, threshold, client_ip_hash,
						normalized_request, notes, created_at
					) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
					""",
					(
						request_id,
						now,
						method,
						path,
						float(anomaly_score),
						1 if is_anomaly else 0,
						int(model_version),
						float(threshold),
						client_ip_hash,
						normalized_request,
						notes,
						created_at,
					),
				)
				conn.commit()

	def get_detections(
		self,
		from_timestamp: Optional[float] = None,
		to_timestamp: Optional[float] = None,
		is_anomaly: Optional[bool] = None,
		min_score: Optional[float] = None,
		max_score: Optional[float] = None,
		path_pattern: Optional[str] = None,
		limit: int = 100,
		offset: int = 0,
	) -> List[Dict[str, Any]]:
		where_parts: List[str] = []
		params: List[Any] = []

		if from_timestamp is not None:
			where_parts.append("timestamp >= ?")
			params.append(float(from_timestamp))
		if to_timestamp is not None:
			where_parts.append("timestamp <= ?")
			params.append(float(to_timestamp))
		if is_anomaly is not None:
			where_parts.append("is_anomaly = ?")
			params.append(1 if is_anomaly else 0)
		if min_score is not None:
			where_parts.append("anomaly_score >= ?")
			params.append(float(min_score))
		if max_score is not None:
			where_parts.append("anomaly_score <= ?")
			params.append(float(max_score))
		if path_pattern:
			where_parts.append("path LIKE ?")
			params.append(f"%{path_pattern}%")

		where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
		params.extend([max(1, int(limit)), max(0, int(offset))])

		query = (
			"SELECT id, request_id, timestamp, method, path, anomaly_score, is_anomaly, "
			"model_version, threshold, client_ip_hash, notes, created_at "
			"FROM detections "
			f"{where_clause} "
			"ORDER BY timestamp DESC "
			"LIMIT ? OFFSET ?"
		)

		with self._get_conn() as conn:
			rows = conn.execute(query, params).fetchall()

		return [dict(row) for row in rows]

	def get_detection_by_id(self, request_id: str) -> Optional[Dict[str, Any]]:
		with self._get_conn() as conn:
			row = conn.execute(
				"""
				SELECT id, request_id, timestamp, method, path, anomaly_score, is_anomaly,
					   model_version, threshold, client_ip_hash, normalized_request,
					   notes, created_at
				FROM detections
				WHERE request_id = ?
				""",
				(request_id,),
			).fetchone()
		return dict(row) if row else None

	def get_stats(self) -> Dict[str, Any]:
		with self._get_conn() as conn:
			row = conn.execute(
				"""
				SELECT
					COUNT(*) AS total_requests,
					COALESCE(SUM(is_anomaly), 0) AS total_anomalies,
					AVG(anomaly_score) AS avg_score,
					MIN(timestamp) AS first_detection,
					MAX(timestamp) AS last_detection
				FROM detections
				"""
			).fetchone()

		total_requests = int(row["total_requests"] or 0)
		total_anomalies = int(row["total_anomalies"] or 0)
		# Return detection rate as percentage for direct display in UI.
		detection_rate = ((total_anomalies / total_requests) * 100.0) if total_requests else 0.0

		return {
			"total_requests": total_requests,
			"total_anomalies": total_anomalies,
			"avg_score": float(row["avg_score"]) if row["avg_score"] is not None else None,
			"detection_rate": detection_rate,
			"first_detection": float(row["first_detection"]) if row["first_detection"] is not None else None,
			"last_detection": float(row["last_detection"]) if row["last_detection"] is not None else None,
		}

	def cleanup_old_records(self, retention_days: int = 30) -> int:
		cutoff = time.time() - (max(1, int(retention_days)) * 86400)
		with self._lock:
			with self._get_conn() as conn:
				cur = conn.execute("DELETE FROM detections WHERE timestamp < ?", (cutoff,))
				conn.commit()
				return int(cur.rowcount or 0)


detection_store = DetectionStore()

