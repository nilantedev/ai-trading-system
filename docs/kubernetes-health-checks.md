# Kubernetes Health Check Configuration

The AI Trading System provides comprehensive health check endpoints designed for Kubernetes deployments with proper liveness, readiness, and startup probes.

## Health Check Endpoints

### 1. Liveness Probe: `/health/live` or `/health`
**Purpose:** Determines if the application process is alive and needs to be restarted.

**Use Case:** Kubernetes restarts the pod if this fails repeatedly.

**Checks:**
- Basic application state is accessible
- Process has been running for at least 5 seconds
- Application can respond to HTTP requests

**Response Examples:**
```json
// Healthy
{
  "status": "alive",
  "timestamp": "2025-01-15T10:30:00.000Z",
  "uptime_seconds": 3600,
  "version": "1.0.0"
}

// Unhealthy (503)
{
  "status": "unhealthy", 
  "timestamp": "2025-01-15T10:30:00.000Z",
  "error": "Application state corrupted"
}
```

### 2. Readiness Probe: `/health/ready`
**Purpose:** Determines if the application is ready to serve traffic.

**Use Case:** Kubernetes stops sending traffic to the pod if this fails.

**Checks:**
- Redis connectivity (cache layer)
- Secrets vault availability
- Circuit breaker status
- Application fully initialized (>10 seconds uptime)

**Response Examples:**
```json
// Ready (200)
{
  "status": "ready",
  "timestamp": "2025-01-15T10:30:00.000Z",
  "uptime_seconds": 300,
  "version": "1.0.0",
  "dependencies": {
    "redis": {
      "status": "healthy",
      "message": "Connection successful"
    },
    "secrets_vault": {
      "status": "healthy",
      "health": {"primary": true, "fallback": true}
    },
    "circuit_breakers": {
      "status": "healthy",
      "total_breakers": 5
    }
  },
  "ready": true
}

// Not Ready (503)
{
  "status": "not_ready",
  "timestamp": "2025-01-15T10:30:00.000Z",
  "uptime_seconds": 5,
  "dependencies": {
    "redis": {
      "status": "error",
      "message": "Connection refused"
    }
  },
  "issues": [
    "Redis connection failed: Connection refused",
    "Application still starting up"
  ],
  "ready": false
}
```

### 3. Startup Probe: `/health/startup`
**Purpose:** Determines if the application has completed its startup sequence.

**Use Case:** Kubernetes waits for this to pass before enabling liveness/readiness checks.

**Checks:**
- Application state initialized
- WebSocket streaming started (if configured)
- Sufficient uptime for initialization

**Response Examples:**
```json
// Started (200)
{
  "status": "started",
  "timestamp": "2025-01-15T10:30:00.000Z",
  "uptime_seconds": 45,
  "startup_checks": {
    "app_initialized": {
      "status": "complete",
      "timestamp": 1705315800
    },
    "startup_timeout": {
      "status": "complete"
    },
    "websocket_streaming": {
      "status": "running"
    }
  },
  "started": true
}
```

## Kubernetes Deployment Configuration

### Complete Deployment Example

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-trading-system
  namespace: trading
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-trading-system
  template:
    metadata:
      labels:
        app: ai-trading-system
    spec:
      containers:
      - name: api
        image: ai-trading-system:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        - name: SECRETS_VAULT_TYPE
          value: "hashicorp"
        - name: VAULT_ENDPOINT
          value: "https://vault.company.com"
        - name: VAULT_TOKEN
          valueFrom:
            secretKeyRef:
              name: vault-token
              key: token
        
        # Resource limits
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        
        # Health check probes
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
            scheme: HTTP
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
          successThreshold: 1
        
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
            scheme: HTTP
          initialDelaySeconds: 15
          periodSeconds: 5
          timeoutSeconds: 10
          failureThreshold: 3
          successThreshold: 1
        
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8000
            scheme: HTTP
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 30  # 300 seconds total
          successThreshold: 1
---
apiVersion: v1
kind: Service
metadata:
  name: ai-trading-system-service
  namespace: trading
spec:
  selector:
    app: ai-trading-system
  ports:
  - port: 80
    targetPort: 8000
    name: http
  type: ClusterIP
```

### Health Check Configuration Options

```yaml
# Quick startup, frequent checks
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 15
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3

# Thorough readiness with longer timeout
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 20
  periodSeconds: 10
  timeoutSeconds: 15  # Longer for dependency checks
  failureThreshold: 2

# Generous startup probe for slow initialization
startupProbe:
  httpGet:
    path: /health/startup
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 15
  timeoutSeconds: 10
  failureThreshold: 40  # 600 seconds total
```

## Monitoring Integration

### Prometheus Metrics

The health endpoints generate metrics that can be monitored:

```yaml
# ServiceMonitor for Prometheus
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ai-trading-system-health
spec:
  selector:
    matchLabels:
      app: ai-trading-system
  endpoints:
  - port: http
    path: /health/ready
    interval: 30s
    scrapeTimeout: 10s
```

### Alerting Rules

```yaml
groups:
- name: ai-trading-system-health
  rules:
  - alert: TradingSystemDown
    expr: up{job="ai-trading-system"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "AI Trading System is down"
      
  - alert: TradingSystemNotReady
    expr: probe_success{job="ai-trading-system-ready"} == 0
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "AI Trading System not ready to serve traffic"
      
  - alert: TradingSystemDependencyFailure
    expr: increase(trading_dependency_failures_total[5m]) > 0
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "AI Trading System dependency failure detected"
```

## Custom Health Checks

### Adding Service-Specific Checks

To add custom health checks for your services, extend the readiness probe:

```python
# In your service
class CustomService:
    async def get_service_health(self) -> Dict[str, Any]:
        """Custom health check for this service."""
        try:
            # Check service-specific dependencies
            external_api_ok = await self.check_external_api()
            database_ok = await self.check_database()
            
            if external_api_ok and database_ok:
                return {
                    "status": "healthy",
                    "checks": {
                        "external_api": "connected",
                        "database": "accessible"
                    }
                }
            else:
                return {
                    "status": "unhealthy", 
                    "checks": {
                        "external_api": "connected" if external_api_ok else "failed",
                        "database": "accessible" if database_ok else "failed"
                    }
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
```

### Environment-Specific Configuration

```bash
# Development - relaxed timing
HEALTH_CHECK_STARTUP_TIMEOUT=60
HEALTH_CHECK_READY_TIMEOUT=10

# Production - strict timing  
HEALTH_CHECK_STARTUP_TIMEOUT=30
HEALTH_CHECK_READY_TIMEOUT=5

# High availability - very strict
HEALTH_CHECK_STARTUP_TIMEOUT=20
HEALTH_CHECK_READY_TIMEOUT=3
```

## Troubleshooting

### Common Issues

**Startup Probe Failing:**
```bash
# Check application logs
kubectl logs -f deployment/ai-trading-system

# Check startup endpoint directly
kubectl port-forward pod/<pod-name> 8000:8000
curl http://localhost:8000/health/startup
```

**Readiness Probe Failing:**
```bash
# Check dependency status
curl http://localhost:8000/health/ready | jq .dependencies

# Common issues:
# - Redis connection failed
# - Secrets vault unreachable
# - Application still starting up
```

**Liveness Probe Failing:**
```bash
# Check if process is responding at all
curl -m 5 http://localhost:8000/health/live

# If no response, check:
# - Memory/CPU limits
# - Deadlocks
# - Resource exhaustion
```

### Debug Mode

Enable detailed health check logging:

```yaml
env:
- name: LOG_LEVEL
  value: "DEBUG"
- name: HEALTH_CHECK_DEBUG
  value: "true"
```

## Best Practices

1. **Probe Timing:**
   - Startup: Generous timing for slow services
   - Liveness: Quick checks, conservative thresholds
   - Readiness: Thorough checks with reasonable timeout

2. **Failure Handling:**
   - Set appropriate `failureThreshold` values
   - Use `timeoutSeconds` matching your service response time
   - Consider `successThreshold` for readiness recovery

3. **Resource Management:**
   - Health checks consume resources
   - Balance check frequency with system load
   - Monitor probe success rates

4. **Dependency Checks:**
   - Check only critical dependencies in readiness
   - Use circuit breakers for non-critical services
   - Implement graceful degradation

5. **Monitoring:**
   - Alert on probe failures
   - Track health check response times
   - Monitor dependency health separately