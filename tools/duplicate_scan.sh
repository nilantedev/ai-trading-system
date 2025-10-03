#!/usr/bin/env bash
# Produce a dry-run duplicate/overlap report across repo docs/scripts to aid safe cleanup.
# Output JSON at tools/duplicate_report.json and a human summary.
set -euo pipefail
ROOT=${ROOT:-"/srv/ai-trading-system"}
OUT_JSON="$ROOT/tools/duplicate_report.json"
TMP=$(mktemp)

emit() { echo "$1" >> "$TMP"; }

emit '{"groups": ['
FIRST=true
# Look for repeated filenames in docs/scripts with same basename across directories
mapfile -t FILES < <(find "$ROOT" -type f \( -name "*.md" -o -name "*.sh" -o -name "*.py" -o -name "*.json" \) \
  ! -path "*/node_modules/*" ! -path "*/venv/*" 2>/dev/null)

# Group by basename
declare -A GROUP
for f in "${FILES[@]}"; do
  base=$(basename "$f")
  GROUP["$base"]+="$f\n"
done

for base in "${!GROUP[@]}"; do
  paths=$(echo -e "${GROUP[$base]}" | sed '/^$/d')
  count=$(echo "$paths" | wc -l | tr -d ' ')
  if [ "$count" -gt 1 ]; then
    # Emit a group entry
    $FIRST || emit ','
    FIRST=false
    emit '{'
    emit "  \"basename\": \"$base\","
    emit "  \"count\": $count,"
    emit "  \"paths\": ["
    i=0
    while IFS= read -r p; do
      i=$((i+1))
      sep=','; [ $i -eq $count ] && sep=''
      emit "    \"$p\"$sep"
    done <<< "$paths"
    emit '  ]'
    emit '}'
  fi
done
emit ']}'

mv "$TMP" "$OUT_JSON"
echo "Wrote duplicate groups report to $OUT_JSON"
# Print a short human summary (top 10 groups)
jq -r '.groups | sort_by(-.count) | .[0:10][] | "- \(.basename): \(.count) copies"' "$OUT_JSON" 2>/dev/null || true
