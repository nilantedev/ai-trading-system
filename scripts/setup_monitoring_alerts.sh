#!/bin/bash
#
# Monitoring Alerts Setup Script for AI Trading System
# Configures Prometheus alerting and notification channels
#
# Usage: ./setup_monitoring_alerts.sh [options]
#
# Options:
#   --slack-webhook <url>    Slack webhook URL for notifications
#   --email <address>        Email address for alerts
#   --pagerduty-key <key>    PagerDuty integration key
#   --telegram-token <token> Telegram bot token
#   --test                   Send test alert to verify configuration
#   --validate               Validate alert rules syntax
#   -h, --help              Show this help message
#

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROMETHEUS_CONFIG="/srv/trading/config/prometheus"
ALERTMANAGER_CONFIG="/srv/trading/config/alertmanager"
GRAFANA_CONFIG="/srv/trading/config/grafana"
SLACK_WEBHOOK=""
EMAIL=""
PAGERDUTY_KEY=""
TELEGRAM_TOKEN=""
TEST_ALERT=false
VALIDATE_ONLY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --slack-webhook)
            SLACK_WEBHOOK="$2"
            shift 2
            ;;
        --email)
            EMAIL="$2"
            shift 2
            ;;
        --pagerduty-key)
            PAGERDUTY_KEY="$2"
            shift 2
            ;;
        --telegram-token)
            TELEGRAM_TOKEN="$2"
            shift 2
            ;;
        --test)
            TEST_ALERT=true
            shift
            ;;
        --validate)
            VALIDATE_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Monitoring Alerts Setup Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --slack-webhook <url>    Slack webhook URL"
            echo "  --email <address>        Email for alerts"
            echo "  --pagerduty-key <key>    PagerDuty integration key"
            echo "  --telegram-token <token> Telegram bot token"
            echo "  --test                   Send test alert"
            echo "  --validate               Validate alert rules"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Create required directories
setup_directories() {
    log_info "Setting up configuration directories..."
    
    sudo mkdir -p "$PROMETHEUS_CONFIG/rules"
    sudo mkdir -p "$ALERTMANAGER_CONFIG"
    sudo mkdir -p "$GRAFANA_CONFIG/provisioning/notifiers"
    
    sudo chown -R $(whoami):$(whoami) "$PROMETHEUS_CONFIG"
    sudo chown -R $(whoami):$(whoami) "$ALERTMANAGER_CONFIG"
    sudo chown -R $(whoami):$(whoami) "$GRAFANA_CONFIG"
    
    log_info "✓ Directories created"
}

# Create production alert rules
create_alert_rules() {
    log_info "Creating production alert rules..."
    
    cat > "$PROMETHEUS_CONFIG/rules/production_alerts.yml" <<'EOF'
# Production Alert Rules for AI Trading System
groups:
  - name: critical_alerts
    interval: 30s
    rules:
      # System availability
      - alert: APIDown
        expr: up{job="trading-api"} == 0
        for: 1m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "CRITICAL: Trading API is down"
          description: "Trading API has been down for {{ $value }} minutes"
          runbook_url: "https://wiki.internal/runbooks/api-down"
      
      # Data pipeline alerts
      - alert: DataPipelineStalled
        expr: time() - data_pipeline_last_success_timestamp > 300
        for: 5m
        labels:
          severity: critical
          team: data
        annotations:
          summary: "Data pipeline stalled"
          description: "No new data processed for {{ $value | humanizeDuration }}"
      
      # Trading risk alerts
      - alert: PortfolioDrawdownCritical
        expr: trading_portfolio_drawdown_percent > 15
        for: 1m
        labels:
          severity: critical
          team: trading
          auto_close_positions: true
        annotations:
          summary: "CRITICAL: Portfolio drawdown {{ $value }}%"
          description: "Portfolio drawdown exceeded critical threshold"
          action: "Auto-closing all positions"
      
      - alert: UnauthorizedTrading
        expr: trading_unauthorized_order_attempts > 0
        for: 10s
        labels:
          severity: critical
          team: security
        annotations:
          summary: "SECURITY: Unauthorized trading attempt"
          description: "{{ $value }} unauthorized trading attempts detected"
      
      # Model performance
      - alert: ModelPerformanceDegraded
        expr: ml_model_sharpe_ratio < 0.5
        for: 30m
        labels:
          severity: warning
          team: ml
        annotations:
          summary: "Model performance degraded"
          description: "Model {{ $labels.model }} Sharpe ratio: {{ $value }}"
      
      - alert: PredictionAnomalyDetected
        expr: ml_prediction_anomaly_score > 3
        for: 5m
        labels:
          severity: warning
          team: ml
        annotations:
          summary: "Anomalous predictions detected"
          description: "Model {{ $labels.model }} anomaly score: {{ $value }}"

  - name: performance_alerts
    interval: 1m
    rules:
      # Latency alerts
      - alert: HighAPILatency
        expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "High API latency detected"
          description: "99th percentile latency: {{ $value }}s"
      
      # Database performance
      - alert: DatabaseSlowQueries
        expr: rate(postgresql_slow_queries_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "High rate of slow database queries"
          description: "{{ $value }} slow queries per second"
      
      # Cache performance
      - alert: CacheMissRateHigh
        expr: rate(redis_cache_misses_total[5m]) / rate(redis_cache_requests_total[5m]) > 0.3
        for: 10m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "High cache miss rate"
          description: "Cache miss rate: {{ $value | humanizePercentage }}"

  - name: resource_alerts
    interval: 1m
    rules:
      # Memory alerts
      - alert: HighMemoryUsage
        expr: (container_memory_usage_bytes / container_spec_memory_limit_bytes) > 0.9
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "High memory usage"
          description: "Container {{ $labels.container }} memory usage: {{ $value | humanizePercentage }}"
      
      # Disk alerts
      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) < 0.1
        for: 5m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "Critical: Low disk space"
          description: "Only {{ $value | humanizePercentage }} disk space remaining"
      
      # GPU monitoring (for ML workloads)
      - alert: GPUMemoryExhausted
        expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.95
        for: 5m
        labels:
          severity: warning
          team: ml
        annotations:
          summary: "GPU memory nearly exhausted"
          description: "GPU {{ $labels.gpu }} memory usage: {{ $value | humanizePercentage }}"

  - name: security_alerts
    interval: 30s
    rules:
      # Authentication alerts
      - alert: BruteForceAttempt
        expr: sum(rate(auth_failed_attempts_total[1m])) by (ip) > 10
        for: 1m
        labels:
          severity: critical
          team: security
        annotations:
          summary: "Potential brute force attack"
          description: "{{ $value }} failed login attempts from IP {{ $labels.ip }}"
      
      # Rate limiting
      - alert: RateLimitBypass
        expr: rate_limit_bypass_attempts > 0
        for: 10s
        labels:
          severity: critical
          team: security
        annotations:
          summary: "Rate limit bypass attempt detected"
          description: "{{ $value }} bypass attempts detected"
      
      # API abuse
      - alert: APIAbuseDetected
        expr: sum(rate(http_requests_total[1m])) by (client_id) > 1000
        for: 2m
        labels:
          severity: warning
          team: security
        annotations:
          summary: "Potential API abuse"
          description: "Client {{ $labels.client_id }} making {{ $value }} requests/min"

  - name: business_metrics
    interval: 5m
    rules:
      # Trading performance
      - alert: DailyPnLNegative
        expr: trading_daily_pnl < -1000
        for: 10m
        labels:
          severity: warning
          team: trading
        annotations:
          summary: "Negative daily P&L"
          description: "Daily P&L: ${{ $value }}"
      
      # Order execution
      - alert: OrderExecutionFailureRate
        expr: rate(trading_order_failures_total[5m]) / rate(trading_order_attempts_total[5m]) > 0.05
        for: 10m
        labels:
          severity: warning
          team: trading
        annotations:
          summary: "High order failure rate"
          description: "Order failure rate: {{ $value | humanizePercentage }}"
      
      # Market data quality
      - alert: StaleMarketData
        expr: time() - market_data_last_update_timestamp > 60
        for: 2m
        labels:
          severity: warning
          team: data
        annotations:
          summary: "Market data is stale"
          description: "No market data updates for {{ $value | humanizeDuration }}"
EOF
    
    log_info "✓ Alert rules created"
}

# Configure Alertmanager
configure_alertmanager() {
    log_info "Configuring Alertmanager..."
    
    cat > "$ALERTMANAGER_CONFIG/alertmanager.yml" <<EOF
# Alertmanager Configuration
global:
  resolve_timeout: 5m
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@trading-system.com'
  smtp_auth_username: '${EMAIL}'
  smtp_auth_password: '\${SMTP_PASSWORD}'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'
  
  routes:
  - match:
      severity: critical
    receiver: critical
    continue: true
    
  - match:
      team: security
    receiver: security
    continue: true
    
  - match:
      team: trading
    receiver: trading
    continue: true
    
  - match:
      team: ml
    receiver: ml
    continue: true

receivers:
- name: 'default'
  webhook_configs:
  - url: 'http://localhost:8000/api/v1/webhooks/alerts'
    send_resolved: true
EOF
    
    # Add Slack configuration if webhook provided
    if [ -n "$SLACK_WEBHOOK" ]; then
        cat >> "$ALERTMANAGER_CONFIG/alertmanager.yml" <<EOF

- name: 'critical'
  slack_configs:
  - api_url: '$SLACK_WEBHOOK'
    channel: '#critical-alerts'
    title: 'Critical Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}{{ end }}'
    send_resolved: true
    color: '{{ if eq .Status "firing" }}danger{{ else }}good{{ end }}'
EOF
    fi
    
    # Add email configuration if provided
    if [ -n "$EMAIL" ]; then
        cat >> "$ALERTMANAGER_CONFIG/alertmanager.yml" <<EOF

- name: 'security'
  email_configs:
  - to: '$EMAIL'
    headers:
      Subject: 'Security Alert: {{ .GroupLabels.alertname }}'
    html: |
      <h2>Security Alert</h2>
      {{ range .Alerts }}
      <p><b>{{ .Annotations.summary }}</b></p>
      <p>{{ .Annotations.description }}</p>
      {{ end }}
EOF
    fi
    
    # Add PagerDuty configuration if key provided
    if [ -n "$PAGERDUTY_KEY" ]; then
        cat >> "$ALERTMANAGER_CONFIG/alertmanager.yml" <<EOF

- name: 'trading'
  pagerduty_configs:
  - service_key: '$PAGERDUTY_KEY'
    description: '{{ .GroupLabels.alertname }}'
    details:
      firing: '{{ range .Alerts.Firing }}{{ .Annotations.description }}{{ end }}'
EOF
    fi
    
    # Add Telegram configuration if token provided
    if [ -n "$TELEGRAM_TOKEN" ]; then
        cat >> "$ALERTMANAGER_CONFIG/alertmanager.yml" <<EOF

- name: 'ml'
  webhook_configs:
  - url: 'https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage'
    http_config:
      bearer_token: '$TELEGRAM_TOKEN'
    send_resolved: true
EOF
    fi
    
    log_info "✓ Alertmanager configured"
}

# Configure Grafana alerting
configure_grafana_alerts() {
    log_info "Configuring Grafana alerts..."
    
    cat > "$GRAFANA_CONFIG/provisioning/notifiers/notifiers.yml" <<EOF
# Grafana Alert Notification Channels
notifiers:
  - name: slack-critical
    type: slack
    uid: slack-critical
    org_id: 1
    is_default: false
    send_reminder: true
    frequency: 1h
    disable_resolve_message: false
    settings:
      url: "$SLACK_WEBHOOK"
      recipient: "#critical-alerts"
      username: "Grafana"
      icon_emoji: ":grafana:"
      
  - name: email-alerts
    type: email
    uid: email-alerts
    org_id: 1
    is_default: true
    settings:
      addresses: "$EMAIL"
      
  - name: webhook-trading
    type: webhook
    uid: webhook-trading
    org_id: 1
    settings:
      url: "http://localhost:8000/api/v1/webhooks/grafana"
      httpMethod: "POST"
EOF
    
    log_info "✓ Grafana alerting configured"
}

# Create monitoring dashboards
create_monitoring_dashboards() {
    log_info "Creating monitoring dashboards..."
    
    cat > "$GRAFANA_CONFIG/dashboards/alerts-overview.json" <<'EOF'
{
  "dashboard": {
    "title": "Alert Overview",
    "panels": [
      {
        "title": "Active Alerts",
        "type": "graph",
        "targets": [
          {
            "expr": "ALERTS{alertstate='firing'}",
            "legendFormat": "{{ alertname }}"
          }
        ]
      },
      {
        "title": "Alert History",
        "type": "table",
        "targets": [
          {
            "expr": "ALERTS",
            "format": "table"
          }
        ]
      },
      {
        "title": "Alert Rate by Severity",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (severity) (ALERTS)",
            "legendFormat": "{{ severity }}"
          }
        ]
      }
    ]
  }
}
EOF
    
    log_info "✓ Monitoring dashboards created"
}

# Validate alert rules
validate_alert_rules() {
    log_info "Validating alert rules..."
    
    if command -v promtool &> /dev/null; then
        promtool check rules "$PROMETHEUS_CONFIG/rules/*.yml"
        
        if [ $? -eq 0 ]; then
            log_info "✓ Alert rules are valid"
        else
            log_error "Alert rules validation failed"
            exit 1
        fi
    else
        log_warning "promtool not found - skipping validation"
        log_info "Install prometheus-tools to validate rules"
    fi
}

# Test alert configuration
test_alerts() {
    log_info "Sending test alert..."
    
    # Create test alert
    cat > /tmp/test_alert.json <<EOF
{
  "alerts": [
    {
      "status": "firing",
      "labels": {
        "alertname": "TestAlert",
        "severity": "info",
        "team": "platform"
      },
      "annotations": {
        "summary": "Test Alert",
        "description": "This is a test alert to verify notification channels"
      },
      "generatorURL": "http://localhost:9090/",
      "startsAt": "$(date -Iseconds)"
    }
  ]
}
EOF
    
    # Send to Alertmanager
    if curl -s -X POST \
        -H "Content-Type: application/json" \
        -d @/tmp/test_alert.json \
        "http://localhost:9093/api/v1/alerts"; then
        log_info "✓ Test alert sent successfully"
        log_info "Check your notification channels for the test alert"
    else
        log_error "Failed to send test alert"
    fi
    
    rm -f /tmp/test_alert.json
}

# Setup alert aggregation
setup_alert_aggregation() {
    log_info "Setting up alert aggregation..."
    
    cat > "$PROMETHEUS_CONFIG/alert_aggregation.yml" <<EOF
# Alert Aggregation Rules
aggregation_rules:
  - name: noise_reduction
    conditions:
      - alert_count > 10
      - time_window: 5m
    action: group_and_throttle
    
  - name: cascade_prevention
    conditions:
      - related_alerts > 5
    action: suppress_children
    
  - name: business_hours
    conditions:
      - time: "Mon-Fri 09:00-18:00"
      - severity: "warning"
    action: batch_notifications
EOF
    
    log_info "✓ Alert aggregation configured"
}

# Create runbooks
create_runbooks() {
    log_info "Creating alert runbooks..."
    
    mkdir -p "$PROMETHEUS_CONFIG/runbooks"
    
    cat > "$PROMETHEUS_CONFIG/runbooks/api-down.md" <<EOF
# API Down Runbook

## Alert: APIDown

### Description
The Trading API service is not responding to health checks.

### Impact
- No new trades can be placed
- Existing positions cannot be managed
- Market data collection may be interrupted

### Investigation Steps
1. Check service status:
   \`\`\`bash
   docker-compose ps trading-api
   docker-compose logs --tail=100 trading-api
   \`\`\`

2. Check system resources:
   \`\`\`bash
   df -h
   free -h
   docker stats
   \`\`\`

3. Check database connectivity:
   \`\`\`bash
   docker-compose exec postgres pg_isready
   \`\`\`

### Resolution Steps
1. Restart the service:
   \`\`\`bash
   docker-compose restart trading-api
   \`\`\`

2. If restart fails, check logs for errors:
   \`\`\`bash
   docker-compose logs trading-api | grep ERROR
   \`\`\`

3. Escalate to on-call if service doesn't recover within 5 minutes

### Prevention
- Implement health check improvements
- Add redundancy with multiple API instances
- Regular load testing
EOF
    
    log_info "✓ Runbooks created"
}

# Generate summary report
generate_summary() {
    log_info ""
    log_info "========================================="
    log_info "MONITORING ALERTS CONFIGURATION SUMMARY"
    log_info "========================================="
    log_info ""
    log_info "Alert Rules:"
    log_info "  ✓ Critical alerts configured"
    log_info "  ✓ Performance monitoring enabled"
    log_info "  ✓ Security alerts active"
    log_info "  ✓ Business metrics tracked"
    log_info ""
    log_info "Notification Channels:"
    
    if [ -n "$SLACK_WEBHOOK" ]; then
        log_info "  ✓ Slack: Configured"
    else
        log_info "  ⚠ Slack: Not configured"
    fi
    
    if [ -n "$EMAIL" ]; then
        log_info "  ✓ Email: $EMAIL"
    else
        log_info "  ⚠ Email: Not configured"
    fi
    
    if [ -n "$PAGERDUTY_KEY" ]; then
        log_info "  ✓ PagerDuty: Configured"
    else
        log_info "  ⚠ PagerDuty: Not configured"
    fi
    
    if [ -n "$TELEGRAM_TOKEN" ]; then
        log_info "  ✓ Telegram: Configured"
    else
        log_info "  ⚠ Telegram: Not configured"
    fi
    
    log_info ""
    log_info "Files Created:"
    log_info "  - $PROMETHEUS_CONFIG/rules/production_alerts.yml"
    log_info "  - $ALERTMANAGER_CONFIG/alertmanager.yml"
    log_info "  - $GRAFANA_CONFIG/provisioning/notifiers/notifiers.yml"
    log_info "  - $PROMETHEUS_CONFIG/runbooks/"
}

# Main execution
main() {
    log_info "Starting monitoring alerts setup..."
    
    # Setup directories
    setup_directories
    
    # Validate only mode
    if [ "$VALIDATE_ONLY" = true ]; then
        validate_alert_rules
        exit 0
    fi
    
    # Create configurations
    create_alert_rules
    configure_alertmanager
    configure_grafana_alerts
    create_monitoring_dashboards
    setup_alert_aggregation
    create_runbooks
    
    # Validate configuration
    validate_alert_rules
    
    # Test alerts if requested
    if [ "$TEST_ALERT" = true ]; then
        test_alerts
    fi
    
    # Generate summary
    generate_summary
    
    log_info ""
    log_info "========================================="
    log_info "MONITORING ALERTS SETUP COMPLETED"
    log_info "========================================="
    log_info ""
    log_info "Next steps:"
    log_info "1. Configure notification channels:"
    log_info "   $0 --slack-webhook <url> --email <address>"
    log_info "2. Reload Prometheus configuration:"
    log_info "   docker-compose exec prometheus kill -HUP 1"
    log_info "3. Start Alertmanager:"
    log_info "   docker-compose up -d alertmanager"
    log_info "4. Test alerts:"
    log_info "   $0 --test"
}

# Run main function
main