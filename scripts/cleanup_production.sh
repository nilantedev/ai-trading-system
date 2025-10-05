#!/bin/bash
#
# Production Cleanup - Remove Duplicate Documentation and Scripts
# Keep only essential production files
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ARCHIVE_DIR="/srv/archive/cleanup_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$ARCHIVE_DIR/docs"
mkdir -p "$ARCHIVE_DIR/scripts"

echo "========================================"
echo "PRODUCTION FILE CLEANUP"
echo "========================================"
echo ""
echo "Archive: $ARCHIVE_DIR"
echo ""

# Duplicate Documentation Files
echo "1. Archiving Duplicate Documentation"
echo "   ================================="

DUPLICATE_DOCS=(
    "AI_WEAVIATE_INTEGRATION_STATUS.md"
    "PRODUCTION_READY_STATUS.md"
    "SYSTEM_IMPLEMENTATION_COMPLETE.md"
    "SYSTEM_UPDATE_SUMMARY_20251005.md"
)

KEEP_DOCS=(
    "README.md"
    "CONFIGURATION.md"
    "WATCHLIST_MANAGEMENT.md"
    "FINAL_STATUS_REPORT.md"
    "PRODUCTION_DEPLOYMENT_COMPLETE.md"
)

DOC_COUNT=0
for doc in "${DUPLICATE_DOCS[@]}"; do
    if [ -f "$PROJECT_DIR/$doc" ]; then
        echo "   📦 $doc"
        mv "$PROJECT_DIR/$doc" "$ARCHIVE_DIR/docs/"
        DOC_COUNT=$((DOC_COUNT + 1))
    fi
done
echo "   Archived: $DOC_COUNT docs"
echo ""

# Duplicate Scripts
echo "2. Archiving Duplicate Scripts"
echo "   ============================"

DUPLICATE_SCRIPTS=(
    "final_system_check.sh"
    "final_system_verification.sh"
    "quick_status.sh"
    "run_symbol_discovery.sh"
    "cleanup_duplicate_scripts.sh"
)

KEEP_SCRIPTS=(
    "complete_system_check.sh"
    "verify_trading_system.sh"
    "verify_updates.sh"
    "rebuild_dashboard_api.sh"
    "update_dashboards_production.sh"
    "activate_continuous_processing.sh"
    "docker_hygiene.sh"
    "logs_tail.sh"
    "eod_pipeline.sh"
    "optimize_databases.sh"
    "quick_reference.sh"
    "access_dashboards.sh"
)

SCRIPT_COUNT=0
for script in "${DUPLICATE_SCRIPTS[@]}"; do
    if [ -f "$SCRIPT_DIR/$script" ]; then
        echo "   📦 $script"
        mv "$SCRIPT_DIR/$script" "$ARCHIVE_DIR/scripts/"
        SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
    fi
done
echo "   Archived: $SCRIPT_COUNT scripts"
echo ""

# List Production Files
echo "3. Production Files Kept"
echo "   ====================="
echo ""
echo "   Documentation (${#KEEP_DOCS[@]}):"
for doc in "${KEEP_DOCS[@]}"; do
    if [ -f "$PROJECT_DIR/$doc" ]; then
        echo "      ✓ $doc"
    fi
done
echo ""
echo "   Scripts (${#KEEP_SCRIPTS[@]}):"
for script in "${KEEP_SCRIPTS[@]}"; do
    if [ -f "$SCRIPT_DIR/$script" ]; then
        echo "      ✓ $script"
    fi
done
echo ""

# Summary
echo "========================================"
echo "CLEANUP COMPLETE"
echo "========================================"
echo ""
echo "Archived: $DOC_COUNT documentation files"
echo "Archived: $SCRIPT_COUNT script files"
echo "Location: $ARCHIVE_DIR"
echo ""
echo "Production files retained:"
echo "  - ${#KEEP_DOCS[@]} documentation files"
echo "  - ${#KEEP_SCRIPTS[@]} operational scripts"
echo ""
