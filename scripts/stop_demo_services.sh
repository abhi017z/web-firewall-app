#!/bin/bash

# Stop all demo services

echo "🛑 Stopping SIH WAF Demo Services"
echo "================================="

PROJECT_ROOT="/Users/kuldeepraj/SIH Part 2/transformer-waf"
cd "$PROJECT_ROOT"

if [ -f .demo_pids ]; then
    echo "Stopping services using saved PIDs..."
    PIDS=$(cat .demo_pids)
    for pid in $PIDS; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping process $pid"
            kill "$pid"
        fi
    done
    rm .demo_pids
fi

echo "Cleaning up any remaining processes on demo ports..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8001 | xargs kill -9 2>/dev/null || true
lsof -ti:8081 | xargs kill -9 2>/dev/null || true
lsof -ti:8082 | xargs kill -9 2>/dev/null || true
lsof -ti:8083 | xargs kill -9 2>/dev/null || true

echo "✅ All demo services stopped"
