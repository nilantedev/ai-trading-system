#!/bin/bash
# Fix Docker image corruption and rebuild API

cd /srv/ai-trading-system

echo "Removing corrupted containers and images..."
docker rm -f $(docker ps -aq --filter "name=trading-api") 2>/dev/null || true
docker rmi ai-trading-system_api 2>/dev/null || true

echo "Rebuilding API image..."
docker-compose build --no-cache api

echo "Starting API..."
docker-compose up -d api

echo "Waiting for API to start..."
sleep 5

echo "Checking API status..."
docker ps | grep trading-api

echo "Testing API endpoint..."
curl -s http://localhost:8000/health || echo "API not responding yet"

echo "Done!"
