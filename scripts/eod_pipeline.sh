#!/usr/bin/env bash
# EOD Operational Pipeline
# Purpose: Run nightly coverage verification, targeted backfills (options/news/social),
#          trigger ML retraining, and update readiness artifacts.
# Safety: Idempotent via marker files; avoids re-running expensive slices already completed.
# Usage (cron example UTC 01:15 Mon-Fri):
# 15 1 * * 1-5 /srv/ai-trading-system/scripts/eod_pipeline.sh >> /var/log/eod_pipeline.log 2>&1

set -euo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${ROOT_DIR}/scripts"
ARTIFACTS_DIR="${ROOT_DIR}"
MARKERS_DIR="${ROOT_DIR}/data_markers"
LOG_PREFIX="[EOD]"

mkdir -p "${MARKERS_DIR}" || true

PYTHON_BIN="${PYTHON_BIN:-python3}"
INGEST_BASE_URL="${INGEST_BASE_URL:-http://127.0.0.1:8002}"  # Data ingestion API

# Configurable environment knobs
EOD_OPTIONS_UNDERLYINGS="${EOD_OPTIONS_UNDERLYINGS:-AAPL,MSFT,SPY,QQQ}"
EOD_OPTIONS_WINDOW_DAYS="${EOD_OPTIONS_WINDOW_DAYS:-30}"
EOD_OPTIONS_MAX_CONTRACTS="${EOD_OPTIONS_MAX_CONTRACTS:-250}"  # tune per provider limits
EOD_OPTIONS_PACE_SECONDS="${EOD_OPTIONS_PACE_SECONDS:-0.25}"

EOD_NEWS_LOOKBACK_YEARS="${EOD_NEWS_LOOKBACK_YEARS:-3}"
EOD_NEWS_BATCH_DAYS="${EOD_NEWS_BATCH_DAYS:-14}"
EOD_NEWS_MAX_ARTICLES="${EOD_NEWS_MAX_ARTICLES:-80}"

EOD_SOCIAL_LOOKBACK_MONTHS="${EOD_SOCIAL_LOOKBACK_MONTHS:-6}"
EOD_SOCIAL_BATCH_HOURS="${EOD_SOCIAL_BATCH_HOURS:-6}"

RETRAIN_ENABLED="${RETRAIN_ENABLED:-true}"  # set false to skip ML retrain stage

# Time helpers
TODAY_UTC=$(date -u +%Y-%m-%d)
NOW_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

log(){ echo "${LOG_PREFIX} $*"; }
warn(){ echo "${LOG_PREFIX} [WARN] $*" >&2; }
err(){ echo "${LOG_PREFIX} [ERROR] $*" >&2; }

require_cmd(){ command -v "$1" >/dev/null 2>&1 || { err "Required command '$1' not found"; exit 12; }; }

require_cmd "${PYTHON_BIN}"

if ! ${PYTHON_BIN} -c "import httpx" >/dev/null 2>&1; then
  warn "Python 'httpx' not installed; attempting pip install (one-time)";
  ${PYTHON_BIN} -m pip install --quiet httpx || warn "Failed to install httpx; backfill_driver may fail";
fi

BACKFILL_DRIVER="${SCRIPTS_DIR}/backfill_driver.py"
COVERAGE_SCRIPT="${SCRIPTS_DIR}/verify_historical_coverage.py"
TRAIN_TRIGGER="${SCRIPTS_DIR}/trigger_retraining.py"

if [[ ! -f "${BACKFILL_DRIVER}" ]]; then
  err "Missing backfill_driver.py at ${BACKFILL_DRIVER}"; exit 13;
fi

# Step 1: Export latest coverage (equities/options)
log "Exporting coverage artifacts..."
set +e
${PYTHON_BIN} "${BACKFILL_DRIVER}" --base-url "${INGEST_BASE_URL}" coverage || warn "Coverage export had non-zero exit"
set -e

# Helper: run options backfill for last N days per underlying if marker absent
run_options_backfill(){
  local underlyings_csv=$1
  local days_back=$2
  local max_contracts=$3
  local pace=$4
  local end_date=$(date -u +%Y-%m-%d)
  local start_date=$(date -u -d "${end_date} -${days_back} days" +%Y-%m-%d)
  IFS=',' read -r -a UNDER_ARR <<< "${underlyings_csv}"
  for u in "${UNDER_ARR[@]}"; do
    local marker="${MARKERS_DIR}/options_${u}_${start_date}_${end_date}.marker"
    if [[ -f "${marker}" ]]; then
      log "Options slice already processed for ${u} ${start_date}-${end_date}, skipping"
      continue
    fi
    log "Backfilling options chain ${u} window=${start_date}:${end_date} max_contracts=${max_contracts} pace=${pace}";
    set +e
    ${PYTHON_BIN} "${BACKFILL_DRIVER}" --base-url "${INGEST_BASE_URL}" options --symbols "${u}" \
      --start "${start_date}" --end "${end_date}" --max-contracts "${max_contracts}" --pace "${pace}" 2>&1 | sed 's/^/[options]/'
    local rc=$?
    set -e
    if [[ ${rc} -eq 0 ]]; then
      touch "${marker}";
    else
      warn "Options backfill failed for ${u} rc=${rc}";
    fi
  done
}

# Helper: run news backfill (year-sliced) with batching, using markers per year
run_news_backfill(){
  local years=$1
  local batch_days=$2
  local max_articles=$3
  local syms_csv=$4
  local current_year=$(date -u +%Y)
  local start_year=$(( current_year - years ))
  IFS=',' read -r -a SYM_ARR <<< "${syms_csv}"
  local syms_joined=$(printf "%s," "${SYM_ARR[@]}")
  syms_joined=${syms_joined%,}
  for (( y=start_year; y<current_year; y++)); do
    local start="${y}-01-01"
    local end="${y}-12-31"
    local marker="${MARKERS_DIR}/news_${y}.marker"
    if [[ -f "${marker}" ]]; then
      log "News year ${y} already processed"; continue; fi
    log "Backfilling news year ${y} symbols=${syms_joined} batch_days=${batch_days} max_articles=${max_articles}";
    set +e
    ${PYTHON_BIN} "${BACKFILL_DRIVER}" --base-url "${INGEST_BASE_URL}" news --symbols "${syms_joined}" \
      --start "${start}" --end "${end}" --batch-days "${batch_days}" --max-articles "${max_articles}" 2>&1 | sed 's/^/[news]/'
    local rc=$?
    set -e
    if [[ ${rc} -eq 0 ]]; then
      touch "${marker}"; else warn "News backfill failed for year ${y} rc=${rc}"; fi
  done
}

# Helper: run social backfill last N months in month slices
run_social_backfill(){
  local months=$1
  local batch_hours=$2
  local syms_csv=$3
  local end_month=$(date -u +%Y-%m-01)
  local start_ts=$(date -u -d "${months} months ago" +%Y-%m-01)
  IFS=',' read -r -a SYM_ARR <<< "${syms_csv}"
  local syms_joined=$(printf "%s," "${SYM_ARR[@]}")
  syms_joined=${syms_joined%,}
  local cursor=${start_ts}
  while [[ "${cursor}" < "${end_month}" ]]; do
    local slice_start=${cursor}
    local slice_end=$(date -u -d "${slice_start} +1 month -1 day" +%Y-%m-%d)
    local marker="${MARKERS_DIR}/social_${slice_start}.marker"
    if [[ -f "${marker}" ]]; then
      log "Social month ${slice_start} already processed"; cursor=$(date -u -d "${slice_start} +1 month" +%Y-%m-01); continue; fi
    log "Backfilling social slice ${slice_start}..${slice_end} symbols=${syms_joined} batch_hours=${batch_hours}";
    set +e
    ${PYTHON_BIN} "${BACKFILL_DRIVER}" --base-url "${INGEST_BASE_URL}" social --symbols "${syms_joined}" \
      --start "${slice_start}" --end "${slice_end}" --batch-hours "${batch_hours}" 2>&1 | sed 's/^/[social]/'
    local rc=$?
    set -e
    if [[ ${rc} -eq 0 ]]; then
      touch "${marker}"; else warn "Social backfill failed for ${slice_start} rc=${rc}"; fi
    cursor=$(date -u -d "${slice_start} +1 month" +%Y-%m-01)
  done
}

# Step 2: Targeted backfills (options/news/social)
log "Evaluating and running targeted backfills (idempotent)..."
run_options_backfill "${EOD_OPTIONS_UNDERLYINGS}" "${EOD_OPTIONS_WINDOW_DAYS}" "${EOD_OPTIONS_MAX_CONTRACTS}" "${EOD_OPTIONS_PACE_SECONDS}"
run_news_backfill "${EOD_NEWS_LOOKBACK_YEARS}" "${EOD_NEWS_BATCH_DAYS}" "${EOD_NEWS_MAX_ARTICLES}" "${EOD_OPTIONS_UNDERLYINGS}"
run_social_backfill "${EOD_SOCIAL_LOOKBACK_MONTHS}" "${EOD_SOCIAL_BATCH_HOURS}" "${EOD_OPTIONS_UNDERLYINGS}"

# Step 3: Re-run historical coverage verification
if [[ -f "${COVERAGE_SCRIPT}" ]]; then
  log "Re-running historical coverage verification..."
  set +e
  ${PYTHON_BIN} "${COVERAGE_SCRIPT}" 2>&1 | sed 's/^/[coverage]/'
  set -e
else
  warn "Coverage script not found at ${COVERAGE_SCRIPT}"
fi

# Step 4: ML retraining trigger (optional)
if [[ "${RETRAIN_ENABLED}" == "true" ]]; then
  if [[ -f "${TRAIN_TRIGGER}" ]]; then
    log "Triggering ML retraining sequence..."
    set +e
    ${PYTHON_BIN} "${TRAIN_TRIGGER}" --symbols "${EOD_OPTIONS_UNDERLYINGS}" 2>&1 | sed 's/^/[retrain]/'
    set -e
  else
    warn "Retraining trigger script missing: ${TRAIN_TRIGGER} (skip)"
  fi
else
  log "RETRAIN_ENABLED=false -> skipping ML retraining stage"
fi

# Step 5: Write success marker & summary
SUCCESS_MARKER="${MARKERS_DIR}/last_eod_success"
{
  echo "timestamp=${NOW_TS}"
  echo "options_underlyings=${EOD_OPTIONS_UNDERLYINGS}"
  echo "options_days=${EOD_OPTIONS_WINDOW_DAYS}"
  echo "news_years=${EOD_NEWS_LOOKBACK_YEARS}"
  echo "social_months=${EOD_SOCIAL_LOOKBACK_MONTHS}"
} > "${SUCCESS_MARKER}" || warn "Failed to write success marker"

log "EOD pipeline completed at ${NOW_TS}"