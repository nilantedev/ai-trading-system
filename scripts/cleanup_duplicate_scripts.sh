#!/bin/bash
###############################################################################
# Script Cleanup - Remove Duplicates and Obsolete Files
# Keep only production-ready, consolidated scripts
###############################################################################

cd /srv/ai-trading-system/scripts

echo "═══════════════════════════════════════════════════════════════"
echo "  SCRIPT CLEANUP - REMOVING DUPLICATES"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Backup before deleting
mkdir -p /srv/archive/scripts_backup_$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/srv/archive/scripts_backup_$(date +%Y%m%d_%H%M%S)"

echo "Creating backup in: $BACKUP_DIR"
cp -r /srv/ai-trading-system/scripts/*.sh /srv/ai-trading-system/scripts/*.py "$BACKUP_DIR/" 2>/dev/null

echo ""
echo "═══ REMOVING DUPLICATE HEALTH CHECK SCRIPTS ═══"
echo ""
echo "Keeping: production_readiness_check.sh (most comprehensive)"
echo "Keeping: holistic_system_health.sh (detailed checks)"
echo "Keeping: check_backfill_status.sh (specific purpose)"
echo ""
echo "Removing duplicates:"

# Remove duplicate health/status checks (keep the best ones)
for file in \
    comprehensive_health_check.sh \
    final_production_status.sh \
    final_system_status.sh \
    show_system_status.sh \
    system_ready_check.sh \
    verify_production_system.sh \
    verify_trading_ready_comprehensive.sh \
    check_data_alignment.sh \
    check_data_coverage.sh
do
    if [ -f "$file" ]; then
        echo "  - $file"
        rm -f "$file"
    fi
done

echo ""
echo "═══ REMOVING DUPLICATE MEMORY/CLEANUP SCRIPTS ═══"
echo ""
echo "Keeping: docker_hygiene.sh (cleanup utility)"
echo "Keeping: production_cleanup.sh (production-specific)"
echo ""
echo "Removing duplicates:"

for file in \
    api_memory_analysis.sh \
    memory_analysis_and_optimization.sh \
    memory_status_summary.sh \
    system_cleanup_and_verification.sh
do
    if [ -f "$file" ]; then
        echo "  - $file"
        rm -f "$file"
    fi
done

echo ""
echo "═══ REMOVING DUPLICATE WATCHLIST SCRIPTS ═══"
echo ""
echo "Keeping: automated_watchlist_update.sh (complete automation)"
echo "Keeping: trigger_watchlist_backfill.sh (backfill trigger)"
echo ""
echo "Removing duplicates:"

for file in \
    manual_watchlist_sync.py \
    production_watchlist_manager.sh \
    setup_watchlist_automation.sh \
    sync_watchlist_from_discovery.sh \
    sync_watchlist_now.sh \
    update_backfill_queue.sh
do
    if [ -f "$file" ]; then
        echo "  - $file"
        rm -f "$file"
    fi
done

echo ""
echo "═══ REMOVING DUPLICATE BACKFILL/ALIGNMENT SCRIPTS ═══"
echo ""
echo "Keeping: backfill_driver.py (primary backfill tool)"
echo "Keeping: quick_sync_and_backfill.sh (quick operation)"
echo ""
echo "Removing duplicates:"

for file in \
    sync_and_backfill.py \
    align_data_to_watchlist.py \
    execute_questdb_alignment.sh \
    generate_alignment_report.sh \
    questdb_alignment_strategy.sh \
    cleanup_non_watchlist_data.sh
do
    if [ -f "$file" ]; then
        echo "  - $file"
        rm -f "$file"
    fi
done

echo ""
echo "═══ REMOVING OTHER OBSOLETE SCRIPTS ═══"
echo ""

for file in \
    verify_autonomous_training.sh \
    verify_live_trading_system.sh \
    verify_model_routing.sh \
    activate_production_system.sh \
    comprehensive_fix.sh \
    fix_production_issues.sh \
    optimize_system_readiness.sh \
    rebuild_api_clean.sh \
    rebuild_ml_with_training.sh \
    review_git_changes.sh
do
    if [ -f "$file" ]; then
        echo "  - $file"
        rm -f "$file"
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  CLEANUP COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Backup location: $BACKUP_DIR"
echo ""
echo "Remaining production scripts:"
ls -lh *.sh *.py 2>/dev/null | wc -l | xargs -I {} echo "  {} scripts"
echo ""
echo "Key scripts preserved:"
echo "  ✓ production_readiness_check.sh - System health and readiness"
echo "  ✓ holistic_system_health.sh - Detailed health check"
echo "  ✓ check_backfill_status.sh - Backfill monitoring"
echo "  ✓ automated_watchlist_update.sh - Watchlist automation"
echo "  ✓ trigger_watchlist_backfill.sh - Backfill trigger"
echo "  ✓ backfill_driver.py - Primary backfill tool"
echo "  ✓ docker_hygiene.sh - Container cleanup"
echo "  ✓ enforce_retention*.py - Data retention"
echo ""
