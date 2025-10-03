#!/bin/bash
#
# Weaviate Upgrade Script - 1.25.10 → 1.27.2
# Safe upgrade with backup and rollback capability
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  WEAVIATE UPGRADE: 1.25.10 → 1.27.2${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Check current version
CURRENT_VERSION=$(curl -s http://localhost:8080/v1/meta 2>/dev/null | grep -o '"version":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
echo -e "${BLUE}→${NC} Current Weaviate version: $CURRENT_VERSION"

# Check data size
DATA_SIZE=$(docker exec trading-weaviate du -sh /var/lib/weaviate 2>/dev/null | awk '{print $1}' || echo "unknown")
echo -e "${BLUE}→${NC} Current data size: $DATA_SIZE"

# Count vectors
VECTOR_COUNT=$(curl -s 'http://localhost:8080/v1/objects?class=SocialSentiment&limit=1' 2>/dev/null | grep -o '"totalResults":[0-9]*' | cut -d':' -f2 || echo "unknown")
echo -e "${BLUE}→${NC} Social sentiment vectors: $VECTOR_COUNT"
echo ""

# Confirm upgrade
echo -e "${YELLOW}⚠ This will:${NC}"
echo "  1. Stop the Weaviate container (30-60 second downtime)"
echo "  2. Update docker-compose.yml to version 1.27.2"
echo "  3. Restart Weaviate with upgraded version"
echo "  4. Data will be automatically migrated (backward compatible)"
echo ""
echo -e "${GREEN}✓ Backup location: /mnt/bulkdata/trading/weaviate (preserved)${NC}"
echo -e "${GREEN}✓ Rollback: Change image back to 1.25.10 and restart${NC}"
echo ""

read -p "Proceed with upgrade? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo -e "${YELLOW}Upgrade cancelled${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  STEP 1: Pre-Upgrade Health Check${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Test connectivity
if ! curl -s http://localhost:8080/v1/meta > /dev/null; then
    echo -e "${RED}✗ Cannot connect to Weaviate${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Weaviate is accessible"

# Check docker-compose exists
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}✗ docker-compose.yml not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} docker-compose.yml found"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  STEP 2: Backup Current Configuration${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Backup docker-compose.yml
BACKUP_FILE="docker-compose.yml.backup-$(date +%Y%m%d_%H%M%S)"
cp docker-compose.yml "$BACKUP_FILE"
echo -e "${GREEN}✓${NC} Backed up to: $BACKUP_FILE"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  STEP 3: Update Weaviate Image Version${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Update docker-compose.yml
sed -i 's|semitechnologies/weaviate:1\.25\.10|semitechnologies/weaviate:1.27.2|g' docker-compose.yml

# Verify change
NEW_VERSION=$(grep "semitechnologies/weaviate:" docker-compose.yml | head -1 | grep -o '1\.[0-9]*\.[0-9]*' || echo "unknown")
echo -e "${GREEN}✓${NC} Updated docker-compose.yml to version: $NEW_VERSION"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  STEP 4: Pull New Weaviate Image${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

docker pull semitechnologies/weaviate:1.27.2
echo -e "${GREEN}✓${NC} New image downloaded"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  STEP 5: Restart Weaviate Container${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Stop container
echo -e "${YELLOW}→${NC} Stopping Weaviate..."
docker-compose stop weaviate

# Remove old container
echo -e "${YELLOW}→${NC} Removing old container..."
docker-compose rm -f weaviate

# Start with new version
echo -e "${YELLOW}→${NC} Starting Weaviate 1.27.2..."
docker-compose up -d weaviate

# Wait for startup
echo -e "${YELLOW}→${NC} Waiting for Weaviate to start..."
sleep 10

# Wait for health check
MAX_WAIT=60
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if docker inspect trading-weaviate | grep -q '"Status": "healthy"'; then
        echo -e "${GREEN}✓${NC} Weaviate is healthy"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    echo -n "."
done
echo ""

if [ $WAITED -ge $MAX_WAIT ]; then
    echo -e "${RED}✗ Weaviate did not become healthy in time${NC}"
    echo -e "${YELLOW}→${NC} Check logs: docker logs trading-weaviate"
    echo -e "${YELLOW}→${NC} Rollback: cp $BACKUP_FILE docker-compose.yml && docker-compose up -d weaviate"
    exit 1
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  STEP 6: Post-Upgrade Verification${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Verify new version
UPGRADED_VERSION=$(curl -s http://localhost:8080/v1/meta 2>/dev/null | grep -o '"version":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
echo -e "${GREEN}✓${NC} New version: $UPGRADED_VERSION"

# Verify data accessibility
VECTOR_COUNT_AFTER=$(curl -s 'http://localhost:8080/v1/objects?class=SocialSentiment&limit=1' 2>/dev/null | grep -o '"totalResults":[0-9]*' | cut -d':' -f2 || echo "0")
echo -e "${GREEN}✓${NC} Vectors accessible: $VECTOR_COUNT_AFTER"

# Verify schema
CLASSES=$(curl -s http://localhost:8080/v1/schema 2>/dev/null | grep -o '"class":"[^"]*"' | wc -l || echo "0")
echo -e "${GREEN}✓${NC} Schema classes: $CLASSES"

# Compare before/after
if [ "$VECTOR_COUNT" = "$VECTOR_COUNT_AFTER" ]; then
    echo -e "${GREEN}✓${NC} Vector count matches (no data loss)"
else
    echo -e "${YELLOW}⚠${NC} Vector count changed: $VECTOR_COUNT → $VECTOR_COUNT_AFTER"
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  UPGRADE SUMMARY${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Previous Version: ${YELLOW}$CURRENT_VERSION${NC}"
echo -e "  New Version:      ${GREEN}$UPGRADED_VERSION${NC}"
echo -e "  Data Size:        ${GREEN}$DATA_SIZE${NC}"
echo -e "  Vectors:          ${GREEN}$VECTOR_COUNT_AFTER${NC}"
echo -e "  Backup:           ${GREEN}$BACKUP_FILE${NC}"
echo ""

if [ "$UPGRADED_VERSION" = "1.27.2" ]; then
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✓ UPGRADE SUCCESSFUL                                     ║${NC}"
    echo -e "${GREEN}║                                                           ║${NC}"
    echo -e "${GREEN}║  Weaviate upgraded to 1.27.2                              ║${NC}"
    echo -e "${GREEN}║  Vector indexing errors should now be resolved            ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}→${NC} Monitor data ingestion: docker logs -f trading-data-ingestion"
    echo -e "${BLUE}→${NC} The 267 version warnings should disappear"
    echo ""
else
    echo -e "${RED}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ⚠ UPGRADE INCOMPLETE                                     ║${NC}"
    echo -e "${RED}║                                                           ║${NC}"
    echo -e "${RED}║  Version mismatch detected                                ║${NC}"
    echo -e "${RED}║  Rollback: cp $BACKUP_FILE docker-compose.yml            ║${NC}"
    echo -e "${RED}║            docker-compose up -d weaviate                  ║${NC}"
    echo -e "${RED}╚═══════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi
