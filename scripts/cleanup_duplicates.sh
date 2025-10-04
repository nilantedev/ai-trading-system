#!/bin/bash

# Cleanup Duplicate Documentation and Scripts
# Consolidates similar files and removes redundant documentation

echo "═══════════════════════════════════════════════════════════════"
echo "  DUPLICATE FILES CLEANUP"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Create backup directory
BACKUP_DIR="/srv/archive/cleanup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR/scripts"
mkdir -p "$BACKUP_DIR/docs"

echo "Backup directory: $BACKUP_DIR"
echo ""

# Track what we're keeping vs removing
echo "📋 ANALYSIS OF FILES:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Documentation files to review
echo "Documentation Files:"
echo ""
echo "KEEP - Main documentation:"
echo "  ✓ CONFIGURATION.md (system configuration)"
echo "  ✓ PRODUCTION_STATUS.md (current production status)"
echo "  ✓ COMPREHENSIVE_DATA_REPORT.md (latest data report)"
echo ""
echo "REMOVE - Redundant/duplicate documentation:"
echo "  ✗ FINAL_VERIFICATION_REPORT.md (superseded by COMPREHENSIVE_DATA_REPORT.md)"
echo ""

# Check if we should remove
if [ -f "/srv/ai-trading-system/FINAL_VERIFICATION_REPORT.md" ]; then
    echo "Moving FINAL_VERIFICATION_REPORT.md to archive..."
    mv "/srv/ai-trading-system/FINAL_VERIFICATION_REPORT.md" "$BACKUP_DIR/docs/"
fi

echo ""
echo "Scripts - Verification/Status Scripts:"
echo ""
echo "KEEP - Essential production scripts:"
echo "  ✓ production_readiness_check.sh (main production check)"
echo "  ✓ holistic_system_health.sh (comprehensive health)"
echo "  ✓ check_ml_trading_status.sh (ML & trading specific)"
echo "  ✓ check_backfill_status_all.sh (backfill tracking)"
echo "  ✓ comprehensive_data_collection_status.sh (data collection)"
echo ""
echo "CONSOLIDATE/REMOVE - Redundant scripts:"
echo "  ✗ complete_system_verification.sh (redundant with production_readiness_check.sh)"
echo "  ✗ final_verification.sh (redundant with holistic_system_health.sh)"
echo "  ✗ verify_data_collection.sh (redundant with comprehensive_data_collection_status.sh)"
echo "  ✗ check_questdb_data.sh (functionality in comprehensive scripts)"
echo "  ✗ check_postgres_data.sh (functionality in comprehensive scripts)"
echo "  ✗ check_backfill_status.sh (superseded by check_backfill_status_all.sh)"
echo ""

# Remove redundant verification scripts
cd /srv/ai-trading-system/scripts

for script in complete_system_verification.sh final_verification.sh verify_data_collection.sh check_questdb_data.sh check_postgres_data.sh; do
    if [ -f "$script" ]; then
        echo "Archiving: $script"
        mv "$script" "$BACKUP_DIR/scripts/"
    fi
done

# Check backfill_status might be useful, let's compare
if [ -f "check_backfill_status.sh" ]; then
    echo "Archiving: check_backfill_status.sh (superseded by check_backfill_status_all.sh)"
    mv "check_backfill_status.sh" "$BACKUP_DIR/scripts/"
fi

echo ""
echo "KEEP - Operational scripts:"
echo "  ✓ automated_watchlist_update.sh (automation)"
echo "  ✓ trigger_watchlist_backfill.sh (backfill management)"
echo "  ✓ docker_hygiene.sh (cleanup)"
echo "  ✓ production_cleanup.sh (maintenance)"
echo "  ✓ quick_sync_and_backfill.sh (operational)"
echo "  ✓ eod_pipeline.sh (end-of-day processing)"
echo "  ✓ start_full_system.sh (startup)"
echo ""

# Check for other potential duplicates
echo "Checking for other duplicates..."
echo ""

# investor_readiness_report.sh might be redundant if we have production reports
if [ -f "investor_readiness_report.sh" ]; then
    echo "REVIEW: investor_readiness_report.sh"
    echo "  This generates investor reports. Keep if used, archive if redundant."
    # Keep it for now as it serves a different purpose
fi

# Tools directory
echo ""
echo "Tools Directory:"
if [ -d "/srv/ai-trading-system/tools" ]; then
    echo "  ✓ /srv/ai-trading-system/tools/ (utilities, keep)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "CLEANUP SUMMARY:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

archived_count=$(find "$BACKUP_DIR" -type f | wc -l)
echo "Files archived: $archived_count"
echo "Backup location: $BACKUP_DIR"
echo ""

# List remaining production scripts
echo "PRODUCTION SCRIPTS RETAINED:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd /srv/ai-trading-system/scripts
ls -1 *.sh | sort | while read script; do
    size=$(ls -lh "$script" | awk '{print $5}')
    printf "  %-50s %8s\n" "$script" "$size"
done

echo ""
echo "PRODUCTION DOCUMENTATION RETAINED:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd /srv/ai-trading-system
ls -1 *.md 2>/dev/null | sort | while read doc; do
    size=$(ls -lh "$doc" | awk '{print $5}')
    printf "  %-50s %8s\n" "$doc" "$size"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Cleanup Complete!"
echo ""
echo "To restore archived files:"
echo "  cp -r $BACKUP_DIR/* /srv/ai-trading-system/"
echo "═══════════════════════════════════════════════════════════════"
