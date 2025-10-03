#!/bin/bash
#
# Production File Cleanup
# Removes duplicate scripts and non-production files
# Keeps only essential production files
#

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  PRODUCTION FILE CLEANUP${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Archive directory
ARCHIVE_DIR="archive/cleanup-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCHIVE_DIR"

echo -e "${BLUE}→${NC} Archiving duplicate and obsolete files..."

# Duplicate watchlist scripts (keep only production_watchlist_manager.sh and update_backfill_queue.sh)
DUPLICATES=(
    "scripts/populate_watchlist.sh"
    "scripts/sync_watchlist_from_options.sh"
    "scripts/sync_optionable_watchlist.sh"
    "scripts/daily_watchlist_sync.sh"
    "scripts/discover_optionable_symbols.py"
    "scripts/sync_optionable_symbols.py"
    "scripts/install_daily_sync_cron.sh"
)

REMOVED=0
for file in "${DUPLICATES[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" "$ARCHIVE_DIR/" && echo -e "${GREEN}✓${NC} Archived: $file" && REMOVED=$((REMOVED + 1))
    fi
done

# Duplicate documentation files
DOC_DUPLICATES=(
    "FINAL_SYSTEM_STATUS.md"
    "CLEANUP_RECOMMENDATIONS.md"
    "PRODUCTION_HARDENING_REPORT_APPENDIX.md"
    "OVERLAP_EVALUATION_START.txt"
    "README_DASHBOARD_ENHANCEMENTS.md"
)

for file in "${DOC_DUPLICATES[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" "$ARCHIVE_DIR/" && echo -e "${GREEN}✓${NC} Archived: $file" && REMOVED=$((REMOVED + 1))
    fi
done

# Old checkpoint/report files (keep only latest)
OLD_CHECKPOINTS=(
    "equities_seed_checkpoint.json"
    "equities_seed_report.json"
    "options_seed_checkpoint.json"
    "options_seed_report.json"
    "news_seed_checkpoint.json"
    "news_seed_report.json"
    "social_seed_checkpoint.json"
    "social_seed_report.json"
    "cleanup_report.json"
    "value_score_histogram.json"
    "backfill_progress.json"
)

for file in "${OLD_CHECKPOINTS[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" "$ARCHIVE_DIR/" && echo -e "${GREEN}✓${NC} Archived: $file" && REMOVED=$((REMOVED + 1))
    fi
done

# Coverage and verification snapshots (keep latest only)
COVERAGE_FILES=(
    "coverage_run_latest.json"
    "coverage_snapshot_full.json"
    "coverage_snapshot.json"
    "coverage_summary_consolidated.json"
    "coverage_verify_new.json"
    "logs_coverage.txt"
)

for file in "${COVERAGE_FILES[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" "$ARCHIVE_DIR/" && echo -e "${GREEN}✓${NC} Archived: $file" && REMOVED=$((REMOVED + 1))
    fi
done

# Retention reports (keep in archive, not root)
if [ -f "retention_applied_questdb.json" ]; then
    mv "retention_applied_questdb.json" "$ARCHIVE_DIR/" && REMOVED=$((REMOVED + 1))
fi
if [ -f "retention_dryrun_questdb.json" ]; then
    mv "retention_dryrun_questdb.json" "$ARCHIVE_DIR/" && REMOVED=$((REMOVED + 1))
fi

# Production readiness reports (consolidate)
if [ -f "production_readiness_report.json" ]; then
    mv "production_readiness_report.json" "$ARCHIVE_DIR/" && REMOVED=$((REMOVED + 1))
fi

echo ""
echo -e "${BLUE}→${NC} Creating production file index..."

# Create index of archived files
cat > "$ARCHIVE_DIR/README.md" << 'EOF'
# Archived Files - Production Cleanup

This directory contains files archived during production cleanup.

## Duplicate Scripts Removed
- Multiple watchlist sync scripts consolidated into `production_watchlist_manager.sh`
- Obsolete Python scripts replaced with pure bash solutions

## Old Checkpoint/Report Files
- Historical seed checkpoints and reports
- Coverage snapshots moved to archive
- Retention reports archived

## Documentation Cleanup
- Duplicate or superseded documentation files
- Old status reports replaced by current `SYSTEM_STATUS_FINAL.md`

All functionality preserved in production scripts.
EOF

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  CLEANUP COMPLETE${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Files Archived:  ${GREEN}$REMOVED${NC}"
echo -e "  Archive Location: ${BLUE}$ARCHIVE_DIR${NC}"
echo ""
echo -e "${GREEN}✓ Production file structure optimized${NC}"
echo ""

# List remaining production files
echo -e "${BLUE}→${NC} Essential production files retained:"
echo ""
echo "CONFIGURATION FILES:"
ls -1 .env docker-compose.yml alembic.ini requirements*.txt 2>/dev/null | sed 's/^/  /'
echo ""
echo "DOCUMENTATION:"
ls -1 *.md 2>/dev/null | grep -v "^archive" | sed 's/^/  /'
echo ""
echo "PRODUCTION SCRIPTS:"
ls -1 scripts/*.sh 2>/dev/null | wc -l | xargs -I {} echo "  {} shell scripts"
ls -1 scripts/*.py 2>/dev/null | wc -l | xargs -I {} echo "  {} Python scripts"
echo ""
