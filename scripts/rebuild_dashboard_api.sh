#!/bin/bash
# Rebuild Dashboard API with Fixed Table Names
# This script rebuilds the API service after fixing QuestDB table references

set -e

cd /srv/ai-trading-system

echo "=== Rebuilding API with fixed table names ==="
docker-compose build api 2>&1 | tail -5

echo ""
echo "=== Stopping and removing old API container ==="
docker-compose stop api
docker-compose rm -f api

echo ""
echo "=== Starting new API container ==="
docker-compose up -d api

echo ""
echo "=== Waiting for API to be healthy (30 seconds) ==="
sleep 30

echo ""
echo "=== Testing Dashboard Endpoints ==="
echo ""
echo "1. Watchlist (should show 939 symbols):"
curl -s http://localhost:8000/api/dashboard/watchlist/all | jq '{total, sample: .symbols[0:5]}'

echo ""
echo "2. Services Health (should show 7 healthy):"
curl -s http://localhost:8000/api/dashboard/services/health | jq '{total, healthy, all_healthy}'

echo ""
echo "3. Market Summary (should show 17M+ bars):"
curl -s http://localhost:8000/api/dashboard/market/summary | jq '{total_bars, unique_symbols, latest_timestamp}'

echo ""
echo "4. Processing Stats:"
curl -s http://localhost:8000/api/dashboard/processing/stats | jq '{total_symbols, mode, bars_per_hour}'

echo ""
echo "=== Dashboard API Rebuild Complete ===" 
