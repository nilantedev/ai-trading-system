#!/bin/bash
# logs_tail.sh - Advanced unified log tail & filter utility for AI Trading System
# Usage examples:
#   ./scripts/logs_tail.sh                                 # tail all service logs (follow)
#   ./scripts/logs_tail.sh api ml                          # tail only api & ml-service
#   ./scripts/logs_tail.sh --since 10m error               # show last 10m logs and filter 'error'
#   ./scripts/logs_tail.sh --no-follow api                 # one-shot recent logs for api
#   ./scripts/logs_tail.sh --grep 'exception|fail' --since 2h
#   ./scripts/logs_tail.sh --json --json-field message     # parse JSON log lines & extract field(s)
#   ./scripts/logs_tail.sh --highlight 'ERROR|CRITICAL'    # highlight patterns
#   ./scripts/logs_tail.sh --squelch 'HealthProbe'         # suppress noisy patterns
#   ./scripts/logs_tail.sh --fail-on-error --since 5m      # exit 2 if any ERROR lines observed
#   ./scripts/logs_tail.sh --rate 50                       # cap output to 50 lines/sec per container
# Environment variables:
#   LOG_DATETIME=yes            -> prefix each line with container name & timestamp
#   LOG_SHOW_CONTAINER=no       -> hide container name prefix
#   LOG_COLOR=no                -> disable color (overrides tty detection)

set -euo pipefail

FOLLOW=true
SINCE="30m"
GREP_EXPR=""
CONTAINERS=()
COLOR=true
JSON_MODE=false
JSON_FIELDS=""
HIGHLIGHT_EXPR=""
SQUELCH_EXPR=""
FAIL_ON_ERROR=false
MAX_LPS=0

if [[ ! -t 1 ]]; then COLOR=false; fi
if [[ "${LOG_COLOR:-yes}" == no ]]; then COLOR=false; fi

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
if [[ "$COLOR" == false ]]; then RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; NC=''; fi

while [[ $# -gt 0 ]]; do
  case $1 in
    --since) SINCE="$2"; shift 2 ;;
    --grep|-g) GREP_EXPR="$2"; shift 2 ;;
  --no-follow) FOLLOW=false; shift ;;
  --json) JSON_MODE=true; shift ;;
  --json-field) JSON_FIELDS="$2"; shift 2 ;;
  --highlight) HIGHLIGHT_EXPR="$2"; shift 2 ;;
  --squelch) SQUELCH_EXPR="$2"; shift 2 ;;
  --fail-on-error) FAIL_ON_ERROR=true; shift ;;
  --rate) MAX_LPS="$2"; shift 2 ;;
    --help|-h)
      sed -n '1,40p' "$0"; exit 0 ;;
    --) shift; break ;;
    -*) echo "Unknown option $1"; exit 1 ;;
    *) CONTAINERS+=("$1"); shift ;;
  esac
done

if [[ ${#CONTAINERS[@]} -eq 0 ]]; then
  # default set (exclude infra noise if desired)
  CONTAINERS=(api ml data-ingestion signal-generator execution risk-monitor strategy-engine backtesting)
fi

RESOLVED=()
for name in "${CONTAINERS[@]}"; do
  case "$name" in
    api) c=trading-api ;;
    ml|ml-service) c=trading-ml ;;
    data-ingestion|ingestion) c=trading-data-ingestion ;;
    signal-generator|signals) c=trading-signal-generator ;;
    execution|order-execution) c=trading-execution ;;
    risk-monitor|risk) c=trading-risk-monitor ;;
    strategy-engine|strategy) c=trading-strategy-engine ;;
    backtesting|backtest) c=trading-backtesting ;;
    redis) c=trading-redis ;;
    postgres|db) c=trading-postgres ;;
    pulsar) c=trading-pulsar ;;
    weaviate) c=trading-weaviate ;;
    *) c="$name" ;;
  esac
  RESOLVED+=("$c")
done

_colorize_severity() {
  local line="$1"
  if [[ $COLOR == false ]]; then echo "$line"; return; fi
  if echo "$line" | grep -qiE '\\bCRITICAL|FATAL\\b'; then
    echo -e "${RED}${BOLD}$line${NC}"; return
  elif echo "$line" | grep -qiE '\\bERROR\\b'; then
    echo -e "${RED}$line${NC}"; return
  elif echo "$line" | grep -qiE '\\bWARN|WARNING\\b'; then
    echo -e "${YELLOW}$line${NC}"; return
  elif echo "$line" | grep -qiE '\\bINFO\\b'; then
    echo -e "${GREEN}$line${NC}"; return
  elif echo "$line" | grep -qiE '\\bDEBUG|TRACE\\b'; then
    echo -e "${CYAN}$line${NC}"; return
  fi
  echo -e "$line"
}

_apply_highlight() {
  local line="$1"; if [[ -z "$HIGHLIGHT_EXPR" ]]; then echo "$line"; return; fi
  # Bold yellow for matches (sed portable) - avoid color collision by re-applying severity after
  if [[ $COLOR == true ]]; then
    echo "$line" | sed -E "s/($HIGHLIGHT_EXPR)/$(printf '\\033[1;33m')\1$(printf '\\033[0m')/g"
  else
    echo "$line" | sed -E "s/($HIGHLIGHT_EXPR)/**\1**/g"
  fi
}

_json_extract() {
  local line="$1"; if [[ $JSON_MODE == false ]]; then echo "$line"; return; fi
  if ! echo "$line" | grep -q '^{' ; then echo "$line"; return; fi
  if ! command -v jq >/dev/null 2>&1; then echo "$line"; return; fi
  local jq_expr='.'
  if [[ -n "$JSON_FIELDS" ]]; then
    IFS=',' read -r -a farr <<< "$JSON_FIELDS"
    local parts=()
    for f in "${farr[@]}"; do parts+=("\"$f\": .${f}"); done
    jq_expr="{${parts[*]}}"
  fi
  echo "$line" | jq -c "$jq_expr" 2>/dev/null || echo "$line"
}

FORMAT() {
  local container="$1"
  local last_ts=0 tokens=0
  while IFS= read -r line; do
    # Squelch noisy lines early
    if [[ -n "$SQUELCH_EXPR" ]] && echo "$line" | grep -qiE "$SQUELCH_EXPR"; then
      continue
    fi
    # Rate limiting
    if [[ $MAX_LPS -gt 0 ]]; then
      local now=$(date +%s)
      if [[ $now -ne $last_ts ]]; then
        last_ts=$now; tokens=$MAX_LPS
      fi
      if [[ $tokens -le 0 ]]; then
        continue
      fi
      tokens=$((tokens-1))
    fi
    # JSON extraction
    line=$(_json_extract "$line")
    # Apply severity color first
    line=$(_colorize_severity "$line")
    # Highlight custom expressions (post severity coloring may nest—acceptable)
    line=$(_apply_highlight "$line")
    local prefix=""
    if [[ "${LOG_SHOW_CONTAINER:-yes}" != no ]]; then
      if [[ "${LOG_DATETIME:-no}" == yes ]]; then
        prefix="[$(date +%H:%M:%S)] ${CYAN}${container}${NC} | "
      else
        prefix="${CYAN}${container}${NC} | "
      fi
    fi
    echo -e "${prefix}${line}"
    if [[ $FAIL_ON_ERROR == true ]] && echo "$line" | grep -qiE '\\b(ERROR|CRITICAL|FATAL)\\b'; then
      export _LOGS_ERROR_DETECTED=1
    fi
  done
}

STREAM() {
  local c="$1"
  if ! docker ps --format '{{.Names}}' | grep -q "^${c}$"; then
    echo -e "${YELLOW}Skipping ${c} (not running)${NC}" >&2
    return
  fi
  local base_cmd=(docker logs "${c}" --since "$SINCE")
  if [[ "$FOLLOW" == true ]]; then base_cmd+=(--follow); fi
  "${base_cmd[@]}" 2>&1 | {
    if [[ -n "$GREP_EXPR" ]]; then grep -Ei --line-buffered "$GREP_EXPR" || true; else cat; fi
  } | FORMAT "$c"
}

# Parallel streaming (if follow) or sequential (no-follow)
if [[ "$FOLLOW" == true ]]; then
  for c in "${RESOLVED[@]}"; do STREAM "$c" & done
  wait
else
  for c in "${RESOLVED[@]}"; do STREAM "$c"; done
fi

if [[ $FAIL_ON_ERROR == true && ${_LOGS_ERROR_DETECTED:-0} -eq 1 ]]; then
  exit 2
fi
