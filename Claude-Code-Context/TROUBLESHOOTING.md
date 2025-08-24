# Troubleshooting Guide & Common Issues

**Last Updated**: August 21, 2025  
**Purpose**: Comprehensive troubleshooting guide for development and deployment  
**Usage**: Reference when encountering issues during any development phase  

---

## 🚨 Emergency Procedures

### CRITICAL: System Down - Immediate Recovery Steps
```bash
# 1. Stop everything safely
make emergency-stop

# 2. Check system resources
df -h  # Check disk space
free -h  # Check memory
top  # Check CPU and processes

# 3. Check Docker status
docker system df  # Check Docker disk usage
docker system prune -f  # Clean up if needed

# 4. Restart infrastructure in dependency order
make start-infrastructure
make health-check-infrastructure
make start-services
make health-check-all
```

### CRITICAL: Data Corruption Detected
```bash
# 1. Immediate isolation
make stop-all-services
make backup-current-state

# 2. Assess damage
make validate-data-integrity
make check-backup-availability

# 3. Recovery procedures
make restore-from-backup <backup-timestamp>
make validate-restored-data
make restart-services
```

---

## 🐛 Common Development Issues

### Phase 1: Foundation Infrastructure Issues

#### Issue: Docker Services Won't Start
**Symptoms**: Services fail to start, port binding errors, resource exhaustion
```bash
# Diagnosis
docker-compose ps  # Check service status
docker-compose logs <service-name>  # Check specific logs
ss -tlnp | grep <port>  # Check port conflicts

# Solutions
# Port conflict:
docker-compose down && docker-compose up -d

# Resource exhaustion:
docker system prune -f
docker volume prune -f

# Permission issues:
sudo chown -R $USER:$USER .
sudo chmod -R 755 infrastructure/
```

#### Issue: Shared Library Import Errors
**Symptoms**: "ModuleNotFoundError", "cannot find crate", import failures
```bash
# Diagnosis
cd shared/python-common && pip list | grep trading-common
cd shared/rust-common && cargo check

# Solutions
# Python library not installed:
cd shared/python-common && pip install -e .

# Rust library build issues:
cd shared/rust-common && cargo clean && cargo build

# Path issues:
export PYTHONPATH="${PYTHONPATH}:$(pwd)/shared/python-common"
```

#### Issue: Monitoring Stack Not Working
**Symptoms**: Grafana dashboards empty, Prometheus targets down
```bash
# Diagnosis
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[].health'
curl -s http://localhost:3001/api/health

# Solutions
# Prometheus config issue:
docker-compose restart prometheus
docker-compose logs prometheus

# Grafana data source issue:
# Check Grafana data source configuration points to prometheus:9090
```

### Phase 2: Database Issues

#### Issue: QuestDB Connection Failures
**Symptoms**: Connection timeout, SQL execution errors, slow queries
```bash
# Diagnosis
curl -s "http://localhost:9000/exec?query=SELECT%201"
docker-compose logs questdb

# Solutions
# Connection timeout:
# Check QuestDB memory allocation in docker-compose.yml
# Increase QuestDB_CAIRO_MAX_UNCOMMITTED_ROWS

# Slow queries:
# Add proper indexes, check PARTITION BY clauses
# Monitor query execution with QuestDB web console
```

#### Issue: Redis Memory Issues
**Symptoms**: Redis OOM errors, eviction warnings, connection refused
```bash
# Diagnosis
redis-cli info memory
redis-cli config get maxmemory*

# Solutions
# Memory limit exceeded:
redis-cli config set maxmemory 64gb
redis-cli config set maxmemory-policy allkeys-lru

# Too many connections:
redis-cli config set maxclients 10000
```

#### Issue: Weaviate Vector Search Problems
**Symptoms**: Slow vector queries, embedding failures, schema errors
```bash
# Diagnosis
curl -s http://localhost:8080/v1/schema | jq .
curl -s http://localhost:8080/v1/.well-known/ready

# Solutions
# Schema recreation needed:
# Delete and recreate Weaviate collections
# Update vectorizer configuration

# Performance issues:
# Check HNSW index parameters
# Adjust ef and efConstruction values
```

### Phase 3: Message Infrastructure Issues

#### Issue: Pulsar Broker Failures
**Symptoms**: Message delivery failures, consumer lag, broker disconnections
```bash
# Diagnosis
curl -s http://localhost:8082/admin/v2/brokers/health
docker-compose logs pulsar

# Solutions
# Broker startup issues:
# Increase Pulsar memory allocation
# Check BookKeeper disk space

# Message routing problems:
# Verify topic partitioning
# Check consumer subscription types
```

#### Issue: Message Serialization Errors
**Symptoms**: Avro schema validation failures, deserialization errors
```bash
# Diagnosis
# Check Avro schema registry
# Validate message format compatibility

# Solutions
# Schema evolution issues:
# Update schema with backward compatibility
# Implement schema migration logic

# Serialization format errors:
# Validate Pydantic models match Avro schemas
# Check data type compatibility
```

### Phase 4: AI Model Issues

#### Issue: Local LLM Performance Problems
**Symptoms**: Slow inference, memory errors, model loading failures
```bash
# Diagnosis
curl -s http://localhost:8003/health
nvidia-smi  # If using GPU
htop  # Check CPU/memory usage

# Solutions
# Model loading failures:
# Check available disk space for models
# Verify model file integrity

# Slow inference:
# Adjust batch size and workers
# Consider model quantization
# Monitor CPU/memory allocation
```

#### Issue: API Rate Limiting
**Symptoms**: 429 errors from OpenAI/Anthropic, quota exceeded messages
```bash
# Diagnosis
# Check API usage tracking logs
# Review current rate limits and quotas

# Solutions
# Immediate:
# Switch to local model fallback
# Implement exponential backoff

# Long-term:
# Implement intelligent request routing
# Cache frequent requests
# Use smaller models for simple tasks
```

### Phase 5-6: Service Integration Issues

#### Issue: Inter-Service Communication Failures
**Symptoms**: 503 errors, message delivery failures, timeout errors
```bash
# Diagnosis
make health-check-all
docker-compose logs | grep ERROR
ss -tlnp  # Check listening ports

# Solutions
# Service discovery issues:
# Verify service names in docker-compose
# Check network connectivity between containers

# Timeout issues:
# Increase timeout configurations
# Check service resource allocation
```

#### Issue: Performance Degradation
**Symptoms**: High latency, CPU spikes, memory leaks
```bash
# Diagnosis
htop  # System resources
docker stats  # Container resources
curl -s http://localhost:9090/api/v1/query?query=rate(http_requests_total[5m])

# Solutions
# CPU bottlenecks:
# Profile application code
# Optimize database queries
# Consider horizontal scaling

# Memory leaks:
# Monitor garbage collection
# Check for unclosed connections
# Review caching strategies
```

### Phase 7: Agent System Issues

#### Issue: Agent Consensus Failures
**Symptoms**: No trading decisions, conflicting agent outputs, consensus timeouts
```bash
# Diagnosis
# Check agent health endpoints
# Review consensus algorithm logs
# Monitor agent performance metrics

# Solutions
# Consensus timeout:
# Adjust timeout parameters
# Check agent response times
# Review voting thresholds

# Conflicting decisions:
# Review agent confidence scores
# Check training data quality
# Implement tie-breaking mechanisms
```

#### Issue: Agent Performance Degradation
**Symptoms**: Low accuracy, high error rates, slow responses
```bash
# Diagnosis
# Review agent accuracy metrics
# Check model drift indicators
# Monitor resource usage per agent

# Solutions
# Model drift:
# Retrain models with recent data
# Implement online learning
# Update feature engineering

# Resource contention:
# Scale agent instances
# Optimize model serving
# Implement load balancing
```

### Phase 8: Dashboard Issues

#### Issue: Real-time Updates Not Working
**Symptoms**: Stale data in dashboard, WebSocket disconnections
```bash
# Diagnosis
# Check WebSocket connection status
# Review browser console for errors
# Monitor server-side WebSocket logs

# Solutions
# Connection issues:
# Implement reconnection logic
# Check proxy configurations
# Review CORS settings

# Data staleness:
# Verify pub/sub message flow
# Check data transformation pipeline
# Review caching strategies
```

#### Issue: Frontend Performance Problems
**Symptoms**: Slow page loads, UI freezing, memory leaks in browser
```bash
# Diagnosis
# Use browser dev tools performance tab
# Check bundle size and loading times
# Monitor memory usage in browser

# Solutions
# Large bundle size:
# Implement code splitting
# Optimize dependencies
# Use dynamic imports

# Memory leaks:
# Review React component lifecycle
# Check for event listener cleanup
# Optimize state management
```

---

## 🔧 Diagnostic Tools & Commands

### System Health Checks
```bash
# Quick health check for all services
make health-check-all

# Detailed system diagnostics
make system-diagnostics

# Performance monitoring
make performance-monitor

# Resource usage check
make resource-usage
```

### Service-Specific Diagnostics
```bash
# Database health
make db-health-check

# Message system health
make messaging-health-check

# AI model health
make model-health-check

# Application service health
make service-health-check
```

### Log Analysis
```bash
# View recent logs from all services
docker-compose logs --tail=100 --follow

# Search for errors across all logs
make search-logs "ERROR"

# Service-specific logs
docker-compose logs <service-name>

# Export logs for analysis
make export-logs <date-range>
```

### Performance Analysis
```bash
# CPU and memory profiling
make profile-system

# Database query analysis
make analyze-db-performance

# Network latency testing
make test-network-latency

# Load testing
make load-test <service>
```

---

## 📊 Monitoring and Alerting

### Key Metrics to Monitor
- **System Resources**: CPU, memory, disk, network
- **Service Health**: Response times, error rates, availability
- **Database Performance**: Query times, connection pools, cache hit rates
- **Message System**: Throughput, latency, queue depths
- **Application Metrics**: Trading decisions, risk calculations, P&L

### Alert Thresholds
```yaml
critical_alerts:
  cpu_usage: >90%
  memory_usage: >95%
  disk_usage: >85%
  error_rate: >5%
  response_time: >1000ms

warning_alerts:
  cpu_usage: >70%
  memory_usage: >80%
  disk_usage: >70%
  error_rate: >1%
  response_time: >500ms
```

### Dashboard URLs for Monitoring
- **Grafana**: http://localhost:3001
- **Prometheus**: http://localhost:9090
- **QuestDB Console**: http://localhost:9000
- **Traefik Dashboard**: http://localhost:8080
- **Pulsar Manager**: http://localhost:8082

---

## 🚨 Known Issues & Workarounds

### Issue: Port 8080 Conflict with Other Applications
**Workaround**: Modify Traefik ports in docker-compose.yml to use 8070/8470

### Issue: QuestDB Startup Slow on First Run
**Workaround**: Increase startup timeout and pre-allocate disk space

### Issue: Weaviate Memory Usage High
**Workaround**: Adjust HNSW parameters and implement vector compression

### Issue: Pulsar High Disk Usage
**Workaround**: Configure retention policies and cleanup scripts

### Issue: Model Server GPU Memory Issues
**Workaround**: Implement model unloading and CPU fallback

---

## 📞 Escalation Procedures

### Level 1: Development Issues
1. Check this troubleshooting guide
2. Review relevant documentation
3. Search project issues and logs
4. Try common solutions

### Level 2: Infrastructure Issues
1. Check server resources and capacity
2. Review network and security settings
3. Validate configuration files
4. Restart services in dependency order

### Level 3: Critical System Failure
1. Implement emergency stop procedures
2. Activate backup and recovery plans
3. Document all steps taken
4. Post-incident analysis and documentation

---

## 🔄 Recovery Procedures

### Service Recovery
```bash
# Single service recovery
docker-compose restart <service-name>
make health-check <service-name>

# Full system recovery
make stop-all
make start-infrastructure
make start-services
make validate-system
```

### Data Recovery
```bash
# Database recovery
make restore-database <backup-date>
make validate-data-integrity

# Configuration recovery
make restore-config <backup-date>
make validate-config

# Model recovery
make restore-models <backup-date>
make validate-models
```

### State Recovery
```bash
# Application state recovery
make restore-app-state <checkpoint>
make validate-app-state

# Trading state recovery
make restore-trading-state <timestamp>
make validate-positions
make reconcile-with-broker
```

---

**🔄 This guide covers the most common issues encountered during development.**  
**Update when**: New issues are discovered, solutions are found, or procedures change