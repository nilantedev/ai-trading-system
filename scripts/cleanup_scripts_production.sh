#!/bin/bash
#
# Production Scripts Cleanup
# Removes non-production, duplicate, and testing scripts
# Keeps only essential production scripts
#

set +e  # Continue on errors
cd /srv/ai-trading-system/scripts

echo "==========================================="
echo "PRODUCTION SCRIPTS CLEANUP"
echo "==========================================="
echo ""

# Create archive directory for removed scripts
ARCHIVE_DIR="/srv/archive/scripts_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCHIVE_DIR"

echo "Archive directory: $ARCHIVE_DIR"
echo ""

# Scripts to DELETE (move to archive)
REMOVE_SCRIPTS=(
    "audit_duplicates.sh"
    "ci_smoke.sh"
    "cleanup_dry_run.sh"
    "cleanup_workspace.sh"
    "comprehensive_trading_readiness.sh"
    "dependency_diff.py"
    "diagnose_news_social_persistence.py"
    "diagnose_options_persistence.py"
    "diagnose_service.sh"
    "check_intelligence_status.sh"
    "equities_reconciliation.py"
    "generate_backfill_report.py"
    "generate_gap_targets.py"
    "generate_sbom.sh"
    "hmac_signer.py"
    "production_cleanup.sh"
    "production_readiness_check.py"
    "production_system_audit.sh"
    "publish_indicator_analysis.py"
    "pulsar_indicator_smoke.py"
    "quality_check.sh"
    "reindex_equity_vectors.py"
    "reset_environment.sh"
    "run_backfills.py"
    "run_backfills_container.py"
    "run_historical_backfill.py"
    "security_scan.sh"
    "seed_base.py"
    "seed_equities.py"
    "seed_news.py"
    "seed_options.py"
    "seed_social.py"
    "smoke_dashboards.py"
    "smoke_hosts.sh"
    "stream_load_test.py"
    "test_auth_password_flows.py"
    "test_dashboard_enhancements.sh"
    "test_form_login.py"
    "test_intelligence_system.sh"
    "test_stream_dlq.py"
    "trigger_backfills_and_vectors.sh"
    "trigger_retraining.py"
    "validate_compose_mounts.sh"
    "validate_prometheus_rules.py"
    "verify_dashboard_enhancement.sh"
    "verify_env.sh"
    "verify_historical_coverage.py"
    "verify_trading_ready.sh"
    "weaviate_probe.py"
    "bulk_load_market_data.py"
    "collect_news_once.py"
    "collect_social_sentiment_once.py"
)

echo "Removing non-production scripts..."
REMOVED=0
for script in "${REMOVE_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        mv "$script" "$ARCHIVE_DIR/"
        echo "  ✓ Removed: $script"
        ((REMOVED++))
    fi
done

# Remove test directories
if [ -d "tests" ]; then
    mv tests "$ARCHIVE_DIR/"
    echo "  ✓ Removed: tests/"
    ((REMOVED++))
fi

# Remove __pycache__
if [ -d "__pycache__" ]; then
    rm -rf __pycache__
    echo "  ✓ Removed: __pycache__/"
    ((REMOVED++))
fi

echo ""
echo "==========================================="
echo "CLEANUP SUMMARY"
echo "==========================================="
echo "Removed: $REMOVED items"
echo "Archive: $ARCHIVE_DIR"
echo ""
echo "Remaining production scripts:"
ls -1 *.sh *.py 2>/dev/null | wc -l
echo ""
echo "✓ Cleanup complete!"
