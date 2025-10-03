#!/bin/bash
# Production Repository Cleanup Script
# Removes non-production files and prepares repo for clean GitHub push

set -e

echo "=== PRODUCTION REPOSITORY CLEANUP ==="
echo "Current directory: $(pwd)"
echo ""

# Define non-production files to remove
NON_PROD_FILES=(
    # Root level non-production docs
    "INVESTOR_EXECUTIVE_SUMMARY.md"
    
    # Archived directories (already in archive/)
    # Keep archive/ itself but it's gitignored
    
    # Root level checkpoint/report files
    "/srv/equities_seed_checkpoint.json"
    "/srv/equities_seed_report.json"
    "/srv/options_seed_checkpoint.json"
    "/srv/value_score_histogram.json"
    "/srv/cleanup_report.json"
)

# Check for files to remove
echo "Checking for non-production files to remove..."
REMOVED_COUNT=0

for file in "${NON_PROD_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  Removing: $file"
        rm -f "$file"
        ((REMOVED_COUNT++))
    fi
done

# Remove git-staged files that were deleted from previous operations
echo ""
echo "Checking git status for deleted files..."
DELETED_FILES=$(git status --short | grep "^AD" | awk '{print $2}' || true)

if [ -n "$DELETED_FILES" ]; then
    echo "Found deleted files to unstage:"
    echo "$DELETED_FILES"
    echo ""
    echo "$DELETED_FILES" | while read -r file; do
        if [ -n "$file" ]; then
            echo "  Removing from git: $file"
            git rm --cached "$file" 2>/dev/null || true
        fi
    done
fi

# Check for large files that shouldn't be in repo
echo ""
echo "Checking for large files (>1MB)..."
LARGE_FILES=$(find . -type f -size +1M ! -path "./.git/*" ! -path "./archive/*" ! -path "./logs/*" ! -path "./.venv/*" ! -path "./data_markers/*" ! -path "./artifacts/*" | head -20)

if [ -n "$LARGE_FILES" ]; then
    echo "⚠ WARNING: Found large files:"
    echo "$LARGE_FILES" | while read -r file; do
        SIZE=$(du -h "$file" | cut -f1)
        echo "  $SIZE - $file"
    done
    echo ""
    echo "Review these files to ensure they're properly gitignored"
else
    echo "✓ No large files found"
fi

# Check for log files
echo ""
echo "Checking for log files..."
LOG_FILES=$(find . -maxdepth 2 -name "*.log" ! -path "./.git/*" ! -path "./logs/*" ! -path "./archive/*" || true)
if [ -n "$LOG_FILES" ]; then
    echo "⚠ WARNING: Found log files in root:"
    echo "$LOG_FILES"
else
    echo "✓ No log files in root directories"
fi

# Summary
echo ""
echo "=== CLEANUP SUMMARY ==="
echo "Files removed: $REMOVED_COUNT"
echo ""

# Check git status
echo "Current git status:"
git status --short | head -30

echo ""
echo "✓ Cleanup complete"
echo ""
echo "Next steps:"
echo "1. Review the git status above"
echo "2. Stage production files: git add <files>"
echo "3. Commit changes: git commit -m 'Production cleanup'"
echo "4. Push to GitHub: git push origin main"
