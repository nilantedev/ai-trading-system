#!/bin/bash
# Professional Dashboard Update - Remove placeholders, add real data
# This script updates both business and admin dashboards to be investor-ready

set -e

cd /srv/ai-trading-system

echo "=== UPDATING DASHBOARDS TO PRODUCTION STANDARD ==="
echo ""
echo "Objective: Create professional, investor-ready dashboards"
echo "- Remove all placeholder content"
echo "- Remove academic/PhD references" 
echo "- Add real-time data visualizations"
echo "- Professional minimal design"
echo ""

# Backup current templates
echo "Creating backups..."
cp api/templates/business/dashboard_v2.html api/templates/business/dashboard_v2.html.backup
cp api/templates/admin/dashboard.html api/templates/admin/dashboard.html.backup

echo "Backups created."
echo ""

# Delete unused duplicate files
echo "Removing unused dashboard.html (dashboard_v2.html is active)..."
rm -f api/templates/business/dashboard.html

echo ""
echo "=== Dashboard files ready for update ==="
echo "Business: api/templates/business/dashboard_v2.html"
echo "Admin: api/templates/admin/dashboard.html"
echo ""
echo "Next: Update templates with professional design and real data endpoints"
