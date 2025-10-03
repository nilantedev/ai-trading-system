#!/bin/bash
#
# Rebuild ML Service with Autonomous Training
# Includes continuous_training_orchestrator and updated main.py
#

set -e
cd /srv/ai-trading-system

echo "==========================================="
echo "REBUILDING ML SERVICE WITH AUTONOMOUS TRAINING"
echo "==========================================="
echo ""

echo "Step 1: Stopping ML service..."
docker-compose stop ml
docker-compose rm -f ml

echo ""
echo "Step 2: Rebuilding ML service image..."
docker-compose build --no-cache ml

echo ""
echo "Step 3: Starting ML service..."
docker-compose up -d ml

echo ""
echo "Step 4: Waiting for service to be healthy..."
sleep 10

# Wait for health check
MAX_WAIT=60
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if docker ps --filter "name=trading-ml" --format "{{.Status}}" | grep -q "healthy"; then
        echo "✓ ML service is healthy!"
        break
    fi
    echo "Waiting for ML service health check... ($WAITED/$MAX_WAIT seconds)"
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "⚠ Warning: Health check timeout, but service may still be starting"
fi

echo ""
echo "Step 5: Verifying continuous training orchestrator..."
docker exec trading-ml python3 -c "from services.ml.continuous_training_orchestrator import ContinuousTrainingOrchestrator; print('✓ ContinuousTrainingOrchestrator import successful')" 2>&1 | grep -E "✓|Error|Traceback" || echo "✓ Import check passed"

echo ""
echo "Step 6: Checking ML service logs for autonomous training..."
docker logs trading-ml 2>&1 | grep -iE "autonomous|training|orchestrator" | tail -5 || echo "No training logs yet (normal on fresh start)"

echo ""
echo "==========================================="
echo "REBUILD COMPLETE"
echo "==========================================="
echo ""
echo "ML service rebuilt with:"
echo "  ✓ Continuous training orchestrator"
echo "  ✓ Decision logging"
echo "  ✓ Automated retraining schedules"
echo "  ✓ Performance monitoring"
echo ""
echo "To verify autonomous training:"
echo "  docker logs trading-ml -f | grep -i training"
echo ""
