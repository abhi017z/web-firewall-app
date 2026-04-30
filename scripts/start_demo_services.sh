#!/bin/bash

# Start all services needed for SIH demonstration
# Run this before the demo presentation

echo "🚀 Starting SIH WAF Demo Services"
echo "================================="

PROJECT_ROOT="/Users/kuldeepraj/SIH Part 2/transformer-waf"
cd "$PROJECT_ROOT"

# Kill any existing processes on the ports
echo "Cleaning up existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8001 | xargs kill -9 2>/dev/null || true
lsof -ti:8081 | xargs kill -9 2>/dev/null || true
lsof -ti:8082 | xargs kill -9 2>/dev/null || true
lsof -ti:8083 | xargs kill -9 2>/dev/null || true

sleep 2

echo "Starting WAF Detection API (port 8000)..."
source venv/bin/activate
PYTHONPATH=. nohup python -m uvicorn src.api.detection_api:app --host 0.0.0.0 --port 8000 --workers 1 > logs/detection_api.log 2>&1 &
DETECTION_PID=$!

echo "Starting WAF Update API (port 8001)..."
PYTHONPATH=. nohup python -m uvicorn src.api.update_api:app --host 0.0.0.0 --port 8001 --workers 1 > logs/update_api.log 2>&1 &
UPDATE_PID=$!

echo "Starting Sample App 1 (port 8081)..."
cd sample_apps/app1
nohup uvicorn main:app --host 0.0.0.0 --port 8081 > ../../logs/app1.log 2>&1 &
APP1_PID=$!

echo "Starting Sample App 2 (port 8082)..."
cd ../app2
nohup uvicorn main:app --host 0.0.0.0 --port 8082 > ../../logs/app2.log 2>&1 &
APP2_PID=$!

echo "Starting Sample App 3 (port 8083)..."
cd ../app3
nohup uvicorn main:app --host 0.0.0.0 --port 8083 > ../../logs/app3.log 2>&1 &
APP3_PID=$!

cd "$PROJECT_ROOT"

echo "Waiting for services to start..."
sleep 5

echo
echo "🔍 Checking service status..."

# Check Detection API
if curl -s -f http://localhost:8000/health > /dev/null; then
    echo "✅ Detection API (8000) - RUNNING"
else
    echo "❌ Detection API (8000) - FAILED"
fi

# Check Update API
if curl -s -f http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ Update API (8001) - RUNNING"
else
    echo "❌ Update API (8001) - FAILED"
fi

# Check Sample Apps
if curl -s -f http://localhost:8081/ > /dev/null; then
    echo "✅ Sample App 1 (8081) - RUNNING"
else
    echo "❌ Sample App 1 (8081) - FAILED"
fi

if curl -s -f http://localhost:8082/ > /dev/null; then
    echo "✅ Sample App 2 (8082) - RUNNING"
else
    echo "❌ Sample App 2 (8082) - FAILED"
fi

if curl -s -f http://localhost:8083/ > /dev/null; then
    echo "✅ Sample App 3 (8083) - RUNNING"
else
    echo "❌ Sample App 3 (8083) - FAILED"
fi

echo
echo "📊 Service URLs:"
echo "• Detection API: http://localhost:8000/docs"
echo "• Update API: http://localhost:8001/docs"
echo "• Sample App 1: http://localhost:8081/"
echo "• Sample App 2: http://localhost:8082/"
echo "• Sample App 3: http://localhost:8083/"

echo
echo "🎯 Ready for demonstration!"
echo "Run: ./scripts/demo_presentation.sh"

echo
echo "📝 Process IDs (for cleanup):"
echo "Detection API: $DETECTION_PID"
echo "Update API: $UPDATE_PID"
echo "Sample App 1: $APP1_PID"
echo "Sample App 2: $APP2_PID"
echo "Sample App 3: $APP3_PID"

# Save PIDs for cleanup
echo "$DETECTION_PID $UPDATE_PID $APP1_PID $APP2_PID $APP3_PID" > .demo_pids
