#!/usr/bin/env bash
set -euo pipefail

# Simple load test using hey if available, else curl loop
URL=${1:-http://localhost:8000/detect}
API_KEY=${API_KEY:-dev-key}

if command -v hey >/dev/null 2>&1; then
  echo "Running hey: 1000 requests, 50 concurrent"
  hey -z 10s -c 50 -H "X-API-Key: $API_KEY" -m POST -T application/json -d '{"method":"GET","path":"/search","query_params":{"q":"hello"},"headers":{},"body":""}' "$URL"
else
  echo "hey not found; running curl loop (not accurate for throughput). Press Ctrl+C to stop."
  for i in $(seq 1 200); do
    curl -s -X POST "$URL" -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" -d '{"method":"GET","path":"/search","query_params":{"q":"hello"},"headers":{},"body":""}' >/dev/null &
  done
  wait
  echo "Done curl load burst"
fi


