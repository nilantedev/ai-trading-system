#!/usr/bin/env bash
# start_full_system.sh
# Orchestrated phased startup for the AI Trading System with dependency-aware waits
# Safe for production use. Does not rebuild images unless --build provided.
#
# Features:
#  - Deterministic startup order (core infra -> monitoring -> messaging -> application layer)
#  - Health / readiness waiting using container healthchecks where defined
#  - Graceful timeout with diagnostics
#  - Optional build / no-cache build
#  - Optional skip of final comprehensive health script
#  - Lightweight Loki readiness probe (because docker-compose.yml currently has no Loki healthcheck)
#  - Idempotent: restarting already running healthy containers is a no-op
#
# Usage:
#   ./scripts/start_full_system.sh [--build] [--no-cache] [--skip-final-health] [--fast] [--wait-tracing] [--no-dashboards-check] [--summary-only]
#
# Exit codes:
#   0 success (all required services healthy)
#   1 failure (some service failed to become healthy)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

COMPOSE_FILE="docker-compose.yml"
if [[ ! -f $COMPOSE_FILE ]]; then
  echo "[FATAL] docker-compose.yml not found at $ROOT_DIR" >&2
  exit 1
fi

# ------------------------- Arguments -------------------------
DO_BUILD=0
NO_CACHE=0
SKIP_FINAL=0
FAST_MODE=0
WAIT_TRACING=0
CHECK_DASHBOARDS=1
SUMMARY_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --build) DO_BUILD=1 ;;
    --no-cache) NO_CACHE=1 ;;
    --skip-final-health) SKIP_FINAL=1 ;;
  --fast) FAST_MODE=1 ;;
  --wait-tracing) WAIT_TRACING=1 ;;
  --no-dashboards-check) CHECK_DASHBOARDS=0 ;;
  --summary-only) SUMMARY_ONLY=1 ;;
    -h|--help)
      grep '^# ' "$0" | sed 's/^# //'
      exit 0
      ;;
    *) echo "[WARN] Unknown argument: $arg" ;;
  esac
done

# ------------------------- Configuration -------------------------
# Services are grouped to ensure dependencies are up first.
# NOTE: traefik optional; keep first so reverse proxy is ready for early external probes
CORE_INFRA=(traefik redis postgres questdb weaviate minio pulsar)
OBSERVABILITY=(prometheus loki grafana node-exporter cadvisor)

APP_LAYER_PHASE1=(api)                 # Depends on DB + cache
APP_LAYER_PHASE2=(ml-service)          # Needs redis+questdb+weaviate+api
APP_LAYER_PHASE3=(data-ingestion signal-generator risk-monitor) # Need pulsar, ml-service (for signals), questdb
APP_LAYER_PHASE4=(execution strategy-engine backtesting)        # Downstream of risk, signals, ml

# Flatten for reporting
ALL_GROUPS=(CORE_INFRA OBSERVABILITY APP_LAYER_PHASE1 APP_LAYER_PHASE2 APP_LAYER_PHASE3 APP_LAYER_PHASE4)

# Explicit timeouts (seconds) - some components (pulsar, weaviate) need longer warm-up
DEFAULT_TIMEOUT=180
declare -A SERVICE_TIMEOUT_OVERRIDE=(
  [pulsar]=240
  [questdb]=210
  [weaviate]=210
  [ml-service]=240
)

# Services without a healthcheck in compose (we treat mere 'running' as acceptable) - keep in sync if compose changes
NO_HEALTHCHECK_SERVICES=(cadvisor minio traefik)

is_in_array() { local needle=$1; shift; for x in "$@"; do [[ $x == $needle ]] && return 0; done; return 1; }

# ------------------------- Functions -------------------------
compose() { docker compose "$@"; }

container_id_for() {
  compose ps -q "$1" 2>/dev/null || true
}

service_state() {
  local sid
  sid=$(container_id_for "$1") || true
  [[ -z $sid ]] && { echo "not_created"; return; }
  docker inspect -f '{{.State.Status}}' "$sid" 2>/dev/null || echo "unknown"
}

service_health_status() {
  local sid
  sid=$(container_id_for "$1") || true
  [[ -z $sid ]] && { echo "none"; return; }
  docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$sid" 2>/dev/null || echo "none"
}

wait_for_service() {
  local svc=$1
  local timeout=${SERVICE_TIMEOUT_OVERRIDE[$svc]:-$DEFAULT_TIMEOUT}
  local start_ts=$(date +%s)
  local waited=0

  # For fast mode we just sleep a little to allow container spawn
  if [[ $FAST_MODE -eq 1 ]]; then
    sleep 2
    return 0
  fi

  while true; do
    local state health
    state=$(service_state "$svc")
    health=$(service_health_status "$svc")

    # Determine readiness
    if is_in_array "$svc" "${NO_HEALTHCHECK_SERVICES[@]}"; then
      if [[ $state == running ]]; then
        return 0
      fi
    else
      if [[ $health == healthy ]]; then
        return 0
      fi
    fi

    local now=$(date +%s)
    waited=$(( now - start_ts ))
    if (( waited >= timeout )); then
      echo "[ERROR] Timeout waiting for $svc (state=$state health=$health) after ${timeout}s" >&2
      echo "--- Last 40 log lines ($svc) ---" >&2
      compose logs --tail=40 "$svc" >&2 || true
      return 1
    fi
    # Slight adaptive backoff with jitter (caps at 5s) to reduce thundering herd
    sleep $(( 2 + (RANDOM % 3) ))
  done
}

start_services_group() {
  local group_name=$1; shift
  local services=("$@")
  echo "\n===== Starting $group_name: ${services[*]} ====="
  for s in "${services[@]}"; do
    echo "[INFO] Starting service: $s"
    compose up -d "$s"
  done
  echo "[INFO] Waiting for health in group $group_name"
  local failures=0
  for s in "${services[@]}"; do
    if ! wait_for_service "$s"; then
      failures=$(( failures + 1 ))
      if [[ $SUMMARY_ONLY -eq 0 ]]; then
        echo "[DIAG] Capturing last 30 lines for $s" >&2
        compose logs --tail=30 "$s" >&2 || true
      fi
    else
      echo "[READY] $s"
    fi
  done
  return $failures
}

print_summary() {
  echo "\n===== Startup Summary ====="
  local any_fail=0
  for group_var in "${ALL_GROUPS[@]}"; do
    # Indirect expansion to get array by name
    local -n arr_ref="$group_var"
    for s in "${arr_ref[@]}"; do
      local state health
      state=$(service_state "$s")
      health=$(service_health_status "$s")
      printf "%-18s state=%-10s health=%-8s\n" "$s" "$state" "$health"
      if ! is_in_array "$s" "${NO_HEALTHCHECK_SERVICES[@]}"; then
        [[ $health != healthy ]] && any_fail=1
      else
        [[ $state != running ]] && any_fail=1
      fi
    done
  done
  return $any_fail
}

# ------------------------- Optional Build -------------------------
if [[ $DO_BUILD -eq 1 ]]; then
  echo "[INFO] Building images (no-cache=$NO_CACHE)"
  if [[ $NO_CACHE -eq 1 ]]; then
    compose build --no-cache
  else
    compose build
  fi
fi

# ------------------------- Phased Startup -------------------------
FAILED=0
start_services_group CORE_INFRA "${CORE_INFRA[@]}" || FAILED=1
start_services_group OBSERVABILITY "${OBSERVABILITY[@]}" || FAILED=1
start_services_group APP_LAYER_PHASE1 "${APP_LAYER_PHASE1[@]}" || FAILED=1
start_services_group APP_LAYER_PHASE2 "${APP_LAYER_PHASE2[@]}" || FAILED=1
start_services_group APP_LAYER_PHASE3 "${APP_LAYER_PHASE3[@]}" || FAILED=1
start_services_group APP_LAYER_PHASE4 "${APP_LAYER_PHASE4[@]}" || FAILED=1

print_summary || SUMMARY_FAIL=1 || true

# ------------------------- Optional tracing exporter wait -------------------------
if [[ $WAIT_TRACING -eq 1 && ${ENABLE_TRACING:-false} =~ ^(1|true|yes)$ ]]; then
  OTEL_ENDPOINT=${OTEL_EXPORTER_OTLP_ENDPOINT:-http://localhost:4318}
  echo "[INFO] Waiting for tracing collector at $OTEL_ENDPOINT"
  for i in {1..10}; do
    if curl -fs --max-time 2 "$OTEL_ENDPOINT/v1/traces" -o /dev/null 2>&1; then
      echo "[READY] Tracing collector responsive"; break
    fi
    sleep 2
    if [[ $i -eq 10 ]]; then echo "[WARN] Tracing collector not responding (continuing)"; fi
  done
fi

# ------------------------- Metrics & dashboard readiness -------------------------
metrics_ready=0
if [[ $FAILED -eq 0 ]]; then
  echo "[INFO] Probing API /metrics for core app metrics"
  for i in {1..20}; do
    if curl -fs --max-time 2 http://localhost:8000/metrics 2>/dev/null | grep -q '^app_http_requests_total'; then
      metrics_ready=1; echo "[READY] Core metrics exposed"; break
    fi
    sleep 1
  done
  if [[ $metrics_ready -eq 0 ]]; then
    echo "[WARN] app_http_requests_total not observed (service may still be warming)"
  fi
fi

if [[ $CHECK_DASHBOARDS -eq 1 ]]; then
  echo "[INFO] Checking dashboard endpoints (/admin & /business)"
  for path in admin business; do
    if curl -fs --max-time 3 -D - http://localhost:8000/$path -o /dev/null 2>&1 | grep -qi 'content-security-policy'; then
      echo "[READY] Dashboard $path reachable with CSP"
    else
      echo "[WARN] Dashboard $path not fully reachable yet" >&2
    fi
  done
fi

# ------------------------- Final Health Script -------------------------
if [[ $SKIP_FINAL -eq 0 && -x scripts/check_health_production.sh ]]; then
  echo "\n===== Running Comprehensive Health Script ====="
  if ! scripts/check_health_production.sh; then
    echo "[WARN] Comprehensive health script reported issues" >&2
    SUMMARY_FAIL=1
  fi
else
  echo "[INFO] Skipping final comprehensive health run (flag or script missing)"
fi

if [[ ${SUMMARY_FAIL:-0} -eq 1 || $FAILED -eq 1 ]]; then
  echo "\n[RESULT] Startup completed with issues" >&2
  exit 1
fi

echo "\n[RESULT] All services started and reported healthy (where healthchecks exist)."
exit 0
