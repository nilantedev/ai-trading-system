#!/bin/bash
#
# Root Directory Cleanup
# Removes duplicate documentation, temporary files, and non-production assets
#

set +e  # Continue on errors
cd /srv/ai-trading-system

echo "==========================================="
echo "ROOT DIRECTORY CLEANUP"
echo "==========================================="
echo ""

# Create archive directory for removed files
ARCHIVE_DIR="/srv/archive/root_cleanup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCHIVE_DIR"

echo "Archive directory: $ARCHIVE_DIR"
echo ""

REMOVED=0

# Remove duplicate/old documentation
echo "Removing duplicate/old documentation..."
if [ -f "DASHBOARD_ENHANCEMENT_SUMMARY.md" ]; then
    mv "DASHBOARD_ENHANCEMENT_SUMMARY.md" "$ARCHIVE_DIR/"
    echo "  ✓ Removed: DASHBOARD_ENHANCEMENT_SUMMARY.md (superseded by DASHBOARD_IMPLEMENTATION_COMPLETE.md)"
    ((REMOVED++))
fi

# Remove temporary scripts
echo "Removing temporary scripts..."
if [ -f "check_password_length.py" ]; then
    mv "check_password_length.py" "$ARCHIVE_DIR/"
    echo "  ✓ Removed: check_password_length.py"
    ((REMOVED++))
fi

if [ -f "system_audit_comprehensive.sh" ]; then
    mv "system_audit_comprehensive.sh" "$ARCHIVE_DIR/"
    echo "  ✓ Removed: system_audit_comprehensive.sh (superseded by verify_trading_ready_comprehensive.sh)"
    ((REMOVED++))
fi

# Remove HTTP file (appears to be a test artifact)
if [ -f "HTTP" ]; then
    mv "HTTP" "$ARCHIVE_DIR/"
    echo "  ✓ Removed: HTTP (test artifact)"
    ((REMOVED++))
fi

# Remove starting file (appears to be a temp file)
if [ -f "starting" ]; then
    mv "starting" "$ARCHIVE_DIR/"
    echo "  ✓ Removed: starting (temp file)"
    ((REMOVED++))
fi

# Clean up logs directory old files
echo "Cleaning up old logs..."
if [ -d "logs" ]; then
    # Move logs older than 7 days to archive
    find logs -type f -mtime +7 -name "*.log" -o -name "*.txt" 2>/dev/null | while read logfile; do
        mkdir -p "$ARCHIVE_DIR/logs"
        mv "$logfile" "$ARCHIVE_DIR/logs/"
        echo "  ✓ Archived old log: $(basename $logfile)"
        ((REMOVED++))
    done
fi

# Clean up artifacts directory
echo "Cleaning up old artifacts..."
if [ -d "artifacts" ]; then
    # Move artifacts older than 30 days to archive
    find artifacts -type f -mtime +30 2>/dev/null | while read artifact; do
        mkdir -p "$ARCHIVE_DIR/artifacts"
        mv "$artifact" "$ARCHIVE_DIR/artifacts/"
        echo "  ✓ Archived old artifact: $(basename $artifact)"
        ((REMOVED++))
    done
fi

# Clean up backtest-results
echo "Cleaning up old backtest results..."
if [ -d "backtest-results" ]; then
    # Move backtest results older than 30 days
    find backtest-results -type f -mtime +30 2>/dev/null | while read result; do
        mkdir -p "$ARCHIVE_DIR/backtest-results"
        mv "$result" "$ARCHIVE_DIR/backtest-results/"
        echo "  ✓ Archived old backtest: $(basename $result)"
        ((REMOVED++))
    done
fi

# Remove any .pyc files
echo "Removing Python cache files..."
find . -type f -name "*.pyc" -delete 2>/dev/null && echo "  ✓ Removed .pyc files"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null && echo "  ✓ Removed __pycache__ directories"

echo ""
echo "==========================================="
echo "CLEANUP SUMMARY"
echo "==========================================="
echo "Removed/Archived: $REMOVED+ items"
echo "Archive: $ARCHIVE_DIR"
echo ""

# Show remaining root files
echo "Production files in root:"
ls -1 *.{md,txt,yml,ini,sql,py} 2>/dev/null | grep -v "__pycache__"

echo ""
echo "✓ Root cleanup complete!"
