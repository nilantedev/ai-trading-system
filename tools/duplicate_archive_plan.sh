#!/usr/bin/env bash
# Dry-run planner for archiving duplicate JSON artifacts from /srv top-level into /srv/archive/<timestamp>
# Safe policy: do not touch code/vendor; only plan moves for known artifact patterns.
# Patterns: coverage_snapshot*.json, coverage_snapshot_full*.json, coverage_run_latest.json,
#           coverage_summary_consolidated.json, coverage_verify_new.json,
#           *_seed_report*.json, *_seed_checkpoint*.json, backfill_progress.json,
#           storage_projection_*.json, equities_seed_report.json, options_seed_report.json,
#           news_seed_report.json, social_seed_report.json
# Output: tools/duplicate_archive_plan.json (list of {source, dest})

set -euo pipefail
TS=$(date +%Y%m%d_%H%M%S)
ARCHIVE_DIR=/srv/archive/$TS
mkdir -p /srv/ai-trading-system/tools
PLAN_JSON=/srv/ai-trading-system/tools/duplicate_archive_plan.json
> "$PLAN_JSON"

echo '[' >> "$PLAN_JSON"
first=1
declare -A seen
# Candidate patterns at /srv top level only
cd /srv
patterns=(
  'coverage_snapshot*.json'
  'coverage_snapshot_full*.json'
  'coverage_run_latest.json'
  'coverage_summary_consolidated.json'
  'coverage_verify_new.json'
  '*_seed_report*.json'
  '*_seed_checkpoint*.json'
  'backfill_progress.json'
  'storage_projection_*.json'
  'equities_seed_report.json'
  'options_seed_report.json'
  'news_seed_report.json'
  'social_seed_report.json'
)

for pat in "${patterns[@]}"; do
  for f in $(ls -1d $pat 2>/dev/null || true); do
    # Skip directories and ensure file exists
    [ -f "$f" ] || continue
    # Skip if the canonical copy under repo also exists and is identical by name (we keep repo copy)
    base=$(basename "$f")
    # Search for same-named file under repo (not exact path); prefer keeping repo copy
    repo_matches=$(find /srv/ai-trading-system -maxdepth 2 -type f -name "$base" | wc -l)
    if [ "$repo_matches" -ge 1 ]; then
      dest="$ARCHIVE_DIR/$base"
      if [ -z "${seen[$base]:-}" ]; then
        if [ $first -eq 0 ]; then echo ',' >> "$PLAN_JSON"; fi
        first=0
        seen[$base]=1
        echo "  {\"source\": \"/srv/$base\", \"dest\": \"$dest\"}" >> "$PLAN_JSON"
      fi
    fi
  done
done

echo ']' >> "$PLAN_JSON"

echo "Plan written to $PLAN_JSON"
echo "To execute, review the JSON then run:"
echo "  mkdir -p $ARCHIVE_DIR && jq -r '.[] | \"mv \\(.source) \\(.dest)\"' $PLAN_JSON | bash"
