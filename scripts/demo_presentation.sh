#!/bin/bash

# SIH WAF Demonstration Script
# This script automates the complete demonstration for judges/interviewers

set -e

echo "🎯Transformer-based WAF Demonstration"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

BASE_URL="http://localhost:8000"
UPDATE_URL="http://localhost:8001"
API_KEY="dev-key"

echo -e "${BLUE}Phase 1: System Status Check${NC}"
echo "----------------------------"

# Check if APIs are running
echo "Checking WAF Detection API..."
if curl -s -f "$BASE_URL/health" > /dev/null; then
    echo -e "${GREEN}✓ Detection API is running${NC}"
else
    echo -e "${RED}✗ Detection API not running. Please start it first.${NC}"
    echo "Run: cd transformer-waf && source venv/bin/activate && PYTHONPATH=. python -m uvicorn src.api.detection_api:app --host 0.0.0.0 --port 8000"
    exit 1
fi

echo "Checking WAF Update API..."
if curl -s -f "$UPDATE_URL/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Update API is running${NC}"
else
    echo -e "${YELLOW}⚠ Update API not running (optional for basic demo)${NC}"
fi

echo
echo -e "${BLUE}Phase 2: Current System Metrics${NC}"
echo "-------------------------------"
curl -s -H "X-API-Key: $API_KEY" "$BASE_URL/metrics" | python -m json.tool

echo
echo -e "${BLUE}Phase 3: Benign Traffic Testing${NC}"
echo "-------------------------------"

BENIGN_REQUESTS=(
    '{"method":"GET","path":"/search","query_params":{"q":"hello"},"headers":{},"body":""}'
    '{"method":"GET","path":"/products","query_params":{"category":"electronics"},"headers":{},"body":""}'
    '{"method":"GET","path":"/about","query_params":{},"headers":{},"body":""}'
    '{"method":"POST","path":"/contact","query_params":{},"headers":{"Content-Type":"application/json"},"body":"{\"name\":\"John\",\"email\":\"john@example.com\"}"}'
)

echo "Testing benign requests (should NOT be flagged as anomalies):"
for i in "${!BENIGN_REQUESTS[@]}"; do
    echo -e "${YELLOW}Request $((i+1)):${NC} ${BENIGN_REQUESTS[i]}"
    RESPONSE=$(curl -s -X POST "$BASE_URL/detect" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $API_KEY" \
        -d "${BENIGN_REQUESTS[i]}")
    
    IS_ANOMALY=$(echo "$RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['is_anomaly'])")
    SCORE=$(echo "$RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['anomaly_score'])")
    
    if [ "$IS_ANOMALY" = "False" ] || [ "$IS_ANOMALY" = "false" ]; then
        echo -e "${GREEN}✓ PASS: Not flagged as anomaly (score: $SCORE)${NC}"
    else
        echo -e "${RED}✗ FAIL: Incorrectly flagged as anomaly (score: $SCORE)${NC}"
    fi
    echo
done

echo -e "${BLUE}Phase 4: Attack Detection Testing${NC}"
echo "--------------------------------"

ATTACK_REQUESTS=(
    '{"method":"GET","path":"/login","query_params":{"id":"'\'' OR '\''1'\''='\''1"},"headers":{},"body":""}' # SQL Injection
    '{"method":"GET","path":"/search","query_params":{"q":"<script>alert(1)</script>"},"headers":{},"body":""}' # XSS
    '{"method":"GET","path":"/files","query_params":{"file":"../../../etc/passwd"},"headers":{},"body":""}' # Path Traversal
    '{"method":"POST","path":"/upload","query_params":{},"headers":{},"body":"<?php system($_GET['\''cmd'\'']); ?>"}' # Code Injection
)

ATTACK_NAMES=("SQL Injection" "XSS Attack" "Path Traversal" "Code Injection")

echo "Testing attack requests (SHOULD be flagged as anomalies):"
for i in "${!ATTACK_REQUESTS[@]}"; do
    echo -e "${YELLOW}${ATTACK_NAMES[i]}:${NC} ${ATTACK_REQUESTS[i]}"
    RESPONSE=$(curl -s -X POST "$BASE_URL/detect" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $API_KEY" \
        -d "${ATTACK_REQUESTS[i]}")
    
    IS_ANOMALY=$(echo "$RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['is_anomaly'])")
    SCORE=$(echo "$RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['anomaly_score'])")
    
    if [ "$IS_ANOMALY" = "True" ] || [ "$IS_ANOMALY" = "true" ]; then
        echo -e "${GREEN}✓ PASS: Correctly detected as anomaly (score: $SCORE)${NC}"
    else
        echo -e "${RED}✗ FAIL: Missed attack (score: $SCORE)${NC}"
    fi
    echo
done

echo -e "${BLUE}Phase 5: Batch Detection Demo${NC}"
echo "----------------------------"

echo "Testing batch detection with mixed traffic:"
BATCH_REQUEST='[
    {"method":"GET","path":"/search","query_params":{"q":"hello"},"headers":{},"body":""},
    {"method":"GET","path":"/login","query_params":{"id":"'\'' OR '\''1'\''='\''1"},"headers":{},"body":""},
    {"method":"GET","path":"/products","query_params":{"category":"electronics"},"headers":{},"body":""},
    {"method":"GET","path":"/search","query_params":{"q":"<script>alert(1)</script>"},"headers":{},"body":""}
]'

curl -s -X POST "$BASE_URL/detect/batch" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d "$BATCH_REQUEST" | python -m json.tool

echo
echo -e "${BLUE}Phase 6: Performance Metrics${NC}"
echo "---------------------------"

echo "Current detection threshold:"
curl -s -H "X-API-Key: $API_KEY" "$BASE_URL/threshold"
echo

echo
echo "Final system metrics:"
curl -s -H "X-API-Key: $API_KEY" "$BASE_URL/metrics" | python -m json.tool

echo
echo -e "${BLUE}Phase 7: Continuous Learning Demo${NC}"
echo "--------------------------------"

if curl -s -f "$UPDATE_URL/health" > /dev/null 2>&1; then
    echo "Adding new benign data for continuous learning:"
    NEW_BENIGN='[
        {"method":"GET","path":"/dashboard","query_params":{},"headers":{},"body":""},
        {"method":"GET","path":"/profile","query_params":{"user_id":"123"},"headers":{},"body":""},
        {"method":"POST","path":"/settings","query_params":{},"headers":{"Content-Type":"application/json"},"body":"{\"theme\":\"dark\"}"}
    ]'
    
    curl -s -X POST "$UPDATE_URL/add_benign_data" \
        -H "Content-Type: application/json" \
        -d "$NEW_BENIGN" | python -m json.tool
    
    echo
    echo "Checking retraining status:"
    curl -s "$UPDATE_URL/retrain/status" | python -m json.tool
else
    echo -e "${YELLOW}Update API not available - skipping continuous learning demo${NC}"
fi

echo
echo -e "${GREEN}🎉 Demonstration Complete!${NC}"
echo "=========================="
echo
echo -e "${BLUE}Key Points Demonstrated:${NC}"
echo "• ✓ Real-time anomaly detection"
echo "• ✓ Batch processing capability"
echo "• ✓ Attack pattern recognition (SQL Injection, XSS, Path Traversal, Code Injection)"
echo "• ✓ Benign traffic handling"
echo "• ✓ API-based architecture"
echo "• ✓ Continuous learning capability"
echo "• ✓ Performance monitoring"
echo
echo -e "${YELLOW}Next Steps for Live Demo:${NC}"
echo "1. Show Swagger UI: http://localhost:8000/docs"
echo "2. Show detection logs: tail -f logs/detection_logs/detections.jsonl"
echo "3. Demonstrate nginx integration"
echo "4. Run load testing: ./scripts/load_test.sh"
echo "5. Show training data and model architecture"
