#!/bin/bash
# docker_hygiene.sh - Safe container image & volume hygiene utility
# Purpose: Provide auditable cleanup of dangling images, unused build cache, and orphaned volumes
#          with safeguards (dry-run by default, retention window).
#
# Usage:
#   ./scripts/docker_hygiene.sh                # Dry run report only
#   ./scripts/docker_hygiene.sh --prune        # Prune dangling images & build cache (respect retention)
#   ./scripts/docker_hygiene.sh --prune-all    # ALSO prune unused volumes (requires --force)
#   ./scripts/docker_hygiene.sh --prune-networks # Prune unused docker networks (requires --force)
#   ./scripts/docker_hygiene.sh --clean-logs   # Rotate/truncate logs (retain last N MiB)
#   ./scripts/docker_hygiene.sh --clean-ollama-tmp # Remove Ollama temp cache safely
#   ./scripts/docker_hygiene.sh --force --prune-all --older-than 14
#   Env Vars:
#       HYGIENE_RETENTION_DAYS (default 7) - Do not remove images created within this many days
#       HYGIENE_LABEL_FILTER (optional) - Additional docker image filter, e.g. "label=maintainer=ai-trading"
#
# Safety Principles:
#   - No destructive action without explicit flag (--prune / --prune-all)
#   - Requires --force when removing volumes (stateful risk)
#   - Retention window excludes recent images even if dangling
#   - Summarizes reclaimed space estimates prior to action
#
set -euo pipefail

RETENTION_DAYS=${HYGIENE_RETENTION_DAYS:-7}
LABEL_FILTER=${HYGIENE_LABEL_FILTER:-}
PRUNE_IMAGES=false
PRUNE_VOLUMES=false
PRUNE_NETWORKS=false
CLEAN_LOGS=false
CLEAN_OLLAMA_TMP=false
FORCE=false
OLDER_THAN_OVERRIDE=""
LOG_RETENTION_MIB=${LOG_RETENTION_MIB:-64}
OLLAMA_TMP_DIRS=("/mnt/fastdrive/trading/models/ollama/tmp" "/mnt/fastdrive/trading/models/ollama/.tmp" "/var/lib/ollama/tmp")

usage() {
  sed -n '1,40p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prune) PRUNE_IMAGES=true; shift ;;
    --prune-all) PRUNE_IMAGES=true; PRUNE_VOLUMES=true; shift ;;
    --prune-networks) PRUNE_NETWORKS=true; shift ;;
    --clean-logs) CLEAN_LOGS=true; shift ;;
    --clean-ollama-tmp) CLEAN_OLLAMA_TMP=true; shift ;;
    --force) FORCE=true; shift ;;
    --older-than) OLDER_THAN_OVERRIDE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -n "$OLDER_THAN_OVERRIDE" ]]; then
  if [[ ! "$OLDER_THAN_OVERRIDE" =~ ^[0-9]+$ ]]; then
    echo "--older-than requires integer days" >&2; exit 2
  fi
  RETENTION_DAYS=$OLDER_THAN_OVERRIDE
fi

# Detect docker
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker not installed or not in PATH" >&2; exit 3
fi

NOW_EPOCH=$(date +%s)
CUTOFF_EPOCH=$(( NOW_EPOCH - RETENTION_DAYS*86400 ))

bold() { tput bold 2>/dev/null || true; }
reset() { tput sgr0 2>/dev/null || true; }
blue() { echo -e "\033[0;34m$1\033[0m"; }
green() { echo -e "\033[0;32m$1\033[0m"; }
yellow() { echo -e "\033[1;33m$1\033[0m"; }
red() { echo -e "\033[0;31m$1\033[0m"; }

header() { echo; echo "$(bold)=== $1 ===$(reset)"; }

header "Docker Hygiene Report"
echo "Timestamp: $(date -Iseconds)"
echo "Retention (days): $RETENTION_DAYS"
[[ -n "$LABEL_FILTER" ]] && echo "Label filter: $LABEL_FILTER"

# 1. Dangling images (candidate list)
header "Dangling Images"
# Get image ID, created date, size
DANGLING_JSON=$(docker images -f dangling=true --format '{{.ID}} {{.CreatedAt}} {{.Repository}}:{{.Tag}} {{.Size}}' || true)
if [[ -z "$DANGLING_JSON" ]]; then
  echo "None"
else
  printf '%-20s %-20s %-45s %-8s %-8s\n' IMAGE_ID CREATED_AT REF SIZE ELIGIBLE
  while read -r line; do
    [[ -z "$line" ]] && continue
    ID=$(echo "$line" | awk '{print $1}')
    CREATED_AT_RAW=$(echo "$line" | awk '{print $2" "$3" "$4}')
    REF=$(echo "$line" | awk '{print $5}')
    SIZE=$(echo "$line" | awk '{print $6}')
    # Convert created date to epoch (best effort; locale independent format assumed)
    CREATED_EPOCH=$(date -d "$CREATED_AT_RAW" +%s 2>/dev/null || echo 0)
    if [[ $CREATED_EPOCH -eq 0 ]]; then
      ELIGIBLE="?"
    else
      if [[ $CREATED_EPOCH -lt $CUTOFF_EPOCH ]]; then ELIGIBLE="yes"; else ELIGIBLE="no"; fi
    fi
    printf '%-20s %-20s %-45s %-8s %-8s\n' "$ID" "$CREATED_AT_RAW" "$REF" "$SIZE" "$ELIGIBLE"
  done <<< "$DANGLING_JSON"
fi

# 2. Build cache size
header "Builder Cache"
CACHE_INFO=$(docker system df --format '{{json .}}' 2>/dev/null | head -n1 || true)
if [[ -n "$CACHE_INFO" ]]; then
  docker system df | sed 's/^/  /'
else
  echo "Unavailable"
fi

# 3. Unused volumes
header "Unused Volumes"
UNUSED_VOLS=$(docker volume ls -qf dangling=true || true)
if [[ -z "$UNUSED_VOLS" ]]; then
  echo "None"
else
  COUNT=$(echo "$UNUSED_VOLS" | wc -l | tr -d ' ')
  echo "Count: $COUNT"
  echo "$UNUSED_VOLS" | sed 's/^/  - /'
fi

# 4. Planned actions (dry-run if no prune flags)
header "Planned Actions"
if [[ $PRUNE_IMAGES == false && $PRUNE_VOLUMES == false ]]; then
  echo "Dry run only (no --prune flags supplied)."
else
  echo "Prune images: $PRUNE_IMAGES"
  echo "Prune volumes: $PRUNE_VOLUMES (force required)"
  echo "Force: $FORCE"
fi
echo "Prune networks: $PRUNE_NETWORKS (force required)"
echo "Clean logs: $CLEAN_LOGS (retention ${LOG_RETENTION_MIB}MiB)"
echo "Clean Ollama tmp: $CLEAN_OLLAMA_TMP"

# Abort here if nothing to prune
if [[ $PRUNE_IMAGES == false && $PRUNE_VOLUMES == false && $PRUNE_NETWORKS == false && $CLEAN_LOGS == false && $CLEAN_OLLAMA_TMP == false ]]; then
  exit 0
fi

# Confirm destructive actions
if [[ $PRUNE_VOLUMES == true && $FORCE == false ]]; then
  red "Refusing to prune volumes without --force"
  exit 4
fi

echo
read -r -p "Proceed with pruning (y/N)? " RESP
if [[ ! "$RESP" =~ ^[Yy]$ ]]; then
  echo "Aborted by user"; exit 0
fi

echo
header "Executing Prune"

if [[ $PRUNE_IMAGES == true ]]; then
  # Filter to dangling images older than retention
  PRUNE_IDS=$(docker images -f dangling=true -q | sort -u || true)
  PRUNE_LIST=()
  for id in $PRUNE_IDS; do
    CREATED=$(docker image inspect -f '{{.Created}}' "$id" 2>/dev/null || true)
    CREATED_EPOCH=$(date -d "$CREATED" +%s 2>/dev/null || echo 0)
    if [[ $CREATED_EPOCH -ne 0 && $CREATED_EPOCH -lt $CUTOFF_EPOCH ]]; then
      PRUNE_LIST+=("$id")
    fi
  done
  if [[ ${#PRUNE_LIST[@]} -eq 0 ]]; then
    echo "No dangling images exceed retention window ($RETENTION_DAYS d)."
  else
    echo "Removing ${#PRUNE_LIST[@]} dangling images older than $RETENTION_DAYS days"
    printf '%s\n' "${PRUNE_LIST[@]}" | xargs -r docker rmi -f
  fi
  # Optionally prune build cache (safe)
  echo "Pruning build cache (unused layers)"
  docker builder prune -f >/dev/null 2>&1 || true
fi

if [[ $PRUNE_VOLUMES == true ]]; then
  if [[ $FORCE == true ]]; then
    echo "Pruning unused volumes"
    echo "$UNUSED_VOLS" | xargs -r docker volume rm
  fi
fi

if [[ $PRUNE_NETWORKS == true ]]; then
  if [[ $FORCE == false ]]; then
    red "Refusing to prune networks without --force"; exit 5
  fi
  echo "Pruning unused docker networks"
  docker network prune -f || true
fi

if [[ $CLEAN_LOGS == true ]]; then
  echo "Cleaning container json-file logs (retain last ${LOG_RETENTION_MIB}MiB per file)"
  # Find json logs under docker's containers dir and truncate in-place
  DOCKER_CONT_DIR="/var/lib/docker/containers"
  if [[ -d "$DOCKER_CONT_DIR" ]]; then
    find "$DOCKER_CONT_DIR" -type f -name "*json.log" 2>/dev/null | while read -r LOGF; do
      if command -v truncate >/dev/null 2>&1; then
        sudo -n truncate -s ${LOG_RETENTION_MIB}M "$LOGF" 2>/dev/null || true
      else
        # Fallback: copy tail to temp then move back
        T=$(mktemp) || true
        tail -c $((LOG_RETENTION_MIB*1024*1024)) "$LOGF" > "$T" 2>/dev/null || true
        cat "$T" > "$LOGF" 2>/dev/null || true
        rm -f "$T" 2>/dev/null || true
      fi
    done
  fi
fi

if [[ $CLEAN_OLLAMA_TMP == true ]]; then
  echo "Cleaning Ollama temporary cache directories"
  for d in "${OLLAMA_TMP_DIRS[@]}"; do
    if [[ -d "$d" ]]; then
      echo " - Removing stale files in $d"
      find "$d" -type f -mtime +2 -print -delete 2>/dev/null || true
      find "$d" -type d -empty -delete 2>/dev/null || true
    fi
  done
fi

echo
header "Post-Prune Summary"
docker system df | sed 's/^/  /'

echo "Done."
