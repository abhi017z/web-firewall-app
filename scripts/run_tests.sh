#!/usr/bin/env bash
set -euo pipefail

# Adjust if your server/port or API key is different
URL="http://localhost:8000/detect"
API_KEY="dev-key"

TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

send_and_parse() {
  local payload_file="$1"
  local response
  response=$(curl -s -X POST "$URL" \
    -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
    -d @"$payload_file")
  python3 - <<PY - "$response"
import sys, json
resp = json.loads(sys.argv[1])
print(json.dumps({
  "is_anomaly": resp.get("is_anomaly"),
  "anomaly_score": resp.get("anomaly_score")
}))
PY
}

declare -a TESTS
i=0

# 1 Benign - simple search
cat > "$TMPDIR/payload_$i.json" <<'JSON'
{"method":"GET","path":"/search","query_params":{"q":"hello world"},"headers":{"User-Agent":"curl"},"body":""}
JSON
TESTS+=("Benign - simple search|false|$TMPDIR/payload_$i.json")
i=$((i+1))

# 2 Benign - login POST
cat > "$TMPDIR/payload_$i.json" <<'JSON'
{"method":"POST","path":"/login","query_params":{},"headers":{"Content-Type":"application/x-www-form-urlencoded"},"body":"username=alice&password=secret"}
JSON
TESTS+=("Benign - login POST|false|$TMPDIR/payload_$i.json")
i=$((i+1))

# 3 SQLi (classic)
cat > "$TMPDIR/payload_$i.json" <<'JSON'
{"method":"GET","path":"/login","query_params":{"id":"' OR '1'='1"},"headers":{},"body":""}
JSON
TESTS+=("SQLi - classic|true|$TMPDIR/payload_$i.json")
i=$((i+1))

# 4 SQLi (stacked queries)
cat > "$TMPDIR/payload_$i.json" <<'JSON'
{"method":"GET","path":"/user","query_params":{"id":"1; DROP TABLE users; --"},"headers":{},"body":""}
JSON
TESTS+=("SQLi - stacked/DDL|true|$TMPDIR/payload_$i.json")
i=$((i+1))

# 5 XSS (reflected)
cat > "$TMPDIR/payload_$i.json" <<'JSON'
{"method":"GET","path":"/search","query_params":{"q":"<script>alert('xss')</script>"},"headers":{},"body":""}
JSON
TESTS+=("XSS - reflected|true|$TMPDIR/payload_$i.json")
i=$((i+1))

# 6 Path Traversal / LFI
cat > "$TMPDIR/payload_$i.json" <<'JSON'
{"method":"GET","path":"/download","query_params":{"file":"../../../../etc/passwd"},"headers":{},"body":""}
JSON
TESTS+=("Path Traversal|true|$TMPDIR/payload_$i.json")
i=$((i+1))

# 7 Command Injection
cat > "$TMPDIR/payload_$i.json" <<'JSON'
{"method":"GET","path":"/ping","query_params":{"ip":"127.0.0.1; rm -rf /"},"headers":{},"body":""}
JSON
TESTS+=("Command Injection|true|$TMPDIR/payload_$i.json")
i=$((i+1))

# 8 SSRF / metadata access
cat > "$TMPDIR/payload_$i.json" <<'JSON'
{"method":"GET","path":"/fetch","query_params":{"url":"http://169.254.169.254/latest/meta-data"},"headers":{},"body":""}
JSON
TESTS+=("SSRF - metadata|true|$TMPDIR/payload_$i.json")
i=$((i+1))

# 9 Encoded SQLi
cat > "$TMPDIR/payload_$i.json" <<'JSON'
{"method":"GET","path":"/product","query_params":{"id":"%27%20OR%20%271%27%3D%271"},"headers":{},"body":""}
JSON
TESTS+=("SQLi - encoded|true|$TMPDIR/payload_$i.json")
i=$((i+1))

# 10 Large body (flood)
BIGSTR=$(python3 - <<'PY'
print("A"*5000)
PY
)
cat > "$TMPDIR/payload_$i.json" <<JSON
{"method":"POST","path":"/upload","query_params":{},"headers":{"Content-Type":"application/json"},"body":"$BIGSTR"}
JSON
TESTS+=("Large body - flood|true|$TMPDIR/payload_$i.json")
i=$((i+1))

# 11 Malicious header
cat > "$TMPDIR/payload_$i.json" <<'JSON'
{"method":"GET","path":"/","query_params":{},"headers":{"X-Forwarded-For":"127.0.0.1, 192.168.1.5, 10.0.0.1; rm -rf /"},"body":""}
JSON
TESTS+=("Malicious header|true|$TMPDIR/payload_$i.json")
i=$((i+1))

# 12 JSON injection (malformed-ish body)
cat > "$TMPDIR/payload_$i.json" <<'JSON'
{"method":"POST","path":"/api","query_params":{},"headers":{"Content-Type":"application/json"},"body":"{\"user\": \"admin\", \"isAdmin\": true} --"}
JSON
TESTS+=("JSON injection|true|$TMPDIR/payload_$i.json")
i=$((i+1))

# 13 Strange path
cat > "$TMPDIR/payload_$i.json" <<'JSON'
{"method":"GET","path":"/....//....//admin///login","query_params":{},"headers":{},"body":""}
JSON
TESTS+=("Strange path|true|$TMPDIR/payload_$i.json")
i=$((i+1))

# 14 Unicode benign
cat > "$TMPDIR/payload_$i.json" <<'JSON'
{"method":"GET","path":"/search","query_params":{"q":"हेलो दुनिया こんにちは"},"headers":{},"body":""}
JSON
TESTS+=("Unicode benign|false|$TMPDIR/payload_$i.json")
i=$((i+1))

total=${#TESTS[@]}
passed=0
failed=0
echo "Running $total tests against $URL"
echo "----------------------------------------"

for entry in "${TESTS[@]}"; do
  IFS='|' read -r name expect file <<< "$entry"
  printf "%-30s " "$name"
  out=$(send_and_parse "$file") || { echo "ERROR contacting server"; failed=$((failed+1)); continue; }
  is_anomaly=$(echo "$out" | python3 -c 'import sys, json; d=json.load(sys.stdin); print(d.get("is_anomaly"))')
  score=$(echo "$out" | python3 -c 'import sys, json; d=json.load(sys.stdin); print(d.get("anomaly_score"))')
  if [ "$is_anomaly" = "True" ] || [ "$is_anomaly" = "true" ]; then
    is_anom_bool=true
  else
    is_anom_bool=false
  fi
  if [ "$is_anom_bool" = "$expect" ]; then
    echo "PASS (is_anomaly=$is_anomaly, score=$score)"
    passed=$((passed+1))
  else
    echo "FAIL (is_anomaly=$is_anomaly, score=$score, expected=$expect)"
    failed=$((failed+1))
  fi
done

echo "----------------------------------------"
echo "Total: $total | Passed: $passed | Failed: $failed"
if [ "$failed" -gt 0 ]; then
  echo "Some tests failed. Consider: raise threshold, add benign data, retrain, recalibrate."
fi


