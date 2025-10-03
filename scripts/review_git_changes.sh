#!/bin/bash
# Review Git Changes Before Push
# Comprehensive review of all changes to ensure safe push to GitHub

set -e

cd /srv/ai-trading-system

echo "=== GIT CHANGES REVIEW ==="
echo ""

# Summary
echo "1. CHANGE SUMMARY"
echo "=================="
ADDED=$(git status --short | grep "^A" | wc -l)
MODIFIED=$(git status --short | grep "^M" | wc -l)
DELETED=$(git status --short | grep "^D" | wc -l)
RENAMED=$(git status --short | grep "^R" | wc -l)
UNSTAGED=$(git status --short | grep "^.M" | wc -l)

echo "Files added:     $ADDED"
echo "Files modified:  $MODIFIED"
echo "Files deleted:   $DELETED"
echo "Files renamed:   $RENAMED"
echo "Unstaged:        $UNSTAGED"
echo ""

# Check for sensitive files
echo "2. SENSITIVE FILE CHECK"
echo "======================="
SENSITIVE=$(git status --short | grep -iE "secret|password|key|credential|token|\.env$|\.pem$|\.crt$" || true)
if [ -n "$SENSITIVE" ]; then
    echo "⚠ WARNING: Potential sensitive files detected:"
    echo "$SENSITIVE"
    echo ""
else
    echo "✓ No sensitive files detected in staged changes"
    echo ""
fi

# Check for large files
echo "3. LARGE FILE CHECK"
echo "==================="
LARGE_FILES=$(git diff --cached --name-only | while read -r file; do
    if [ -f "$file" ] && [ $(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null) -gt 1048576 ]; then
        SIZE=$(du -h "$file" | cut -f1)
        echo "$SIZE - $file"
    fi
done)

if [ -n "$LARGE_FILES" ]; then
    echo "⚠ WARNING: Large files detected (>1MB):"
    echo "$LARGE_FILES"
    echo ""
else
    echo "✓ No large files (>1MB) in staged changes"
    echo ""
fi

# List new files being added
echo "4. NEW FILES BEING ADDED"
echo "========================"
git status --short | grep "^A" | head -20
echo ""

# List deleted files
echo "5. FILES BEING DELETED"
echo "======================"
git status --short | grep "^D" | head -20
echo ""

# Check unstaged changes
echo "6. UNSTAGED CHANGES"
echo "==================="
UNSTAGED_FILES=$(git status --short | grep "^.M" || true)
if [ -n "$UNSTAGED_FILES" ]; then
    echo "⚠ You have unstaged changes:"
    echo "$UNSTAGED_FILES"
    echo ""
else
    echo "✓ No unstaged changes"
    echo ""
fi

# Modified files summary
echo "7. MODIFIED FILES (first 20)"
echo "============================"
git status --short | grep "^M" | head -20
echo ""

# Summary recommendation
echo "=== RECOMMENDATIONS ==="
echo ""
if [ -n "$SENSITIVE" ]; then
    echo "❌ DO NOT PUSH - Sensitive files detected!"
    echo "   Review and remove from staging first"
    exit 1
fi

if [ -n "$LARGE_FILES" ]; then
    echo "⚠ CAUTION - Large files detected"
    echo "  Consider using Git LFS or .gitignore"
fi

echo "✓ Repository appears safe for push"
echo ""
echo "Next steps:"
echo "  git add .gitignore"
echo "  git commit -m 'Production cleanup and repository preparation'"
echo "  git push origin main --force-with-lease"
echo ""
echo "Use --force-with-lease instead of --force for safety"
