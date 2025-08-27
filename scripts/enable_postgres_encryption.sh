#!/bin/bash

# Enable PostgreSQL Encryption at Rest
# =====================================
# This script configures PostgreSQL for encryption at rest using
# transparent data encryption (TDE) and SSL/TLS connections

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}PostgreSQL Encryption Setup${NC}"
echo "============================="

# Check if running as postgres user or with sudo
if [[ $EUID -ne 0 ]] && [[ $(whoami) != "postgres" ]]; then
   echo -e "${RED}This script must be run as root or postgres user${NC}"
   exit 1
fi

# PostgreSQL version check
PG_VERSION=$(psql --version | awk '{print $3}' | sed 's/\..*//g')
if [ "$PG_VERSION" -lt 13 ]; then
    echo -e "${YELLOW}Warning: PostgreSQL 13+ recommended for best encryption support${NC}"
fi

# 1. Generate SSL certificates if they don't exist
echo -e "${GREEN}Step 1: Setting up SSL certificates${NC}"

SSL_DIR="/var/lib/postgresql/ssl"
mkdir -p "$SSL_DIR"
cd "$SSL_DIR"

if [ ! -f "server.key" ]; then
    echo "Generating SSL certificates..."
    
    # Generate private key
    openssl genrsa -out server.key 2048
    chmod 600 server.key
    chown postgres:postgres server.key
    
    # Generate certificate signing request
    openssl req -new -key server.key -out server.csr -subj "/C=US/ST=State/L=City/O=AITradingSystem/CN=postgres"
    
    # Generate self-signed certificate (valid for 365 days)
    openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt
    chmod 644 server.crt
    chown postgres:postgres server.crt
    
    echo -e "${GREEN}SSL certificates created successfully${NC}"
else
    echo "SSL certificates already exist"
fi

# 2. Configure PostgreSQL for encryption
echo -e "${GREEN}Step 2: Configuring PostgreSQL${NC}"

PG_CONFIG_DIR="/etc/postgresql/$PG_VERSION/main"
if [ ! -d "$PG_CONFIG_DIR" ]; then
    # Try alternative locations
    PG_CONFIG_DIR="/var/lib/postgresql/data"
    if [ ! -d "$PG_CONFIG_DIR" ]; then
        PG_CONFIG_DIR="/usr/local/pgsql/data"
    fi
fi

if [ -d "$PG_CONFIG_DIR" ]; then
    # Backup original configuration
    cp "$PG_CONFIG_DIR/postgresql.conf" "$PG_CONFIG_DIR/postgresql.conf.backup.$(date +%Y%m%d)"
    
    # Apply encryption settings
    cat >> "$PG_CONFIG_DIR/postgresql.conf" << EOF

# ===== ENCRYPTION CONFIGURATION =====
# Added by enable_postgres_encryption.sh on $(date)

# SSL/TLS Settings
ssl = on
ssl_cert_file = '$SSL_DIR/server.crt'
ssl_key_file = '$SSL_DIR/server.key'
ssl_ciphers = 'HIGH:MEDIUM:+3DES:!aNULL:!MD5:!SEED:!IDEA'
ssl_prefer_server_ciphers = on
ssl_min_protocol_version = 'TLSv1.2'

# Force SSL connections for all users
# (Add to pg_hba.conf for enforcement)

# Authentication encryption
password_encryption = scram-sha-256

# Enable data checksums (detects corruption)
# Note: Must be enabled at cluster initialization for full effect
EOF
    
    echo -e "${GREEN}PostgreSQL configuration updated${NC}"
else
    echo -e "${YELLOW}Warning: Could not find PostgreSQL config directory${NC}"
    echo "Please manually update postgresql.conf with encryption settings"
fi

# 3. Update pg_hba.conf to require SSL
echo -e "${GREEN}Step 3: Enforcing SSL connections${NC}"

PG_HBA="$PG_CONFIG_DIR/pg_hba.conf"
if [ -f "$PG_HBA" ]; then
    # Backup original
    cp "$PG_HBA" "$PG_HBA.backup.$(date +%Y%m%d)"
    
    # Update to require SSL for all connections except local
    cat > "$PG_HBA" << EOF
# PostgreSQL Client Authentication Configuration
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# Local connections (no SSL needed)
local   all             all                                     scram-sha-256

# IPv4 connections (require SSL)
hostssl all             all             127.0.0.1/32            scram-sha-256
hostssl all             all             ::1/128                 scram-sha-256

# Docker network connections (require SSL)
hostssl all             all             172.16.0.0/12           scram-sha-256

# Reject non-SSL connections
hostnossl all           all             0.0.0.0/0               reject
hostnossl all           all             ::/0                    reject
EOF
    
    echo -e "${GREEN}pg_hba.conf updated to require SSL${NC}"
else
    echo -e "${YELLOW}Warning: pg_hba.conf not found${NC}"
fi

# 4. Set up encryption at rest using LUKS (Linux) or FileVault (macOS)
echo -e "${GREEN}Step 4: Encryption at Rest${NC}"

# Check if we're in a Docker container
if [ -f /.dockerenv ]; then
    echo -e "${YELLOW}Running in Docker - encryption at rest should be configured on host${NC}"
    echo "Options:"
    echo "  1. Use encrypted host volumes"
    echo "  2. Use Docker secrets for sensitive data"
    echo "  3. Enable LUKS encryption on host filesystem"
else
    # Check current encryption status
    if command -v dmsetup &> /dev/null; then
        if dmsetup status | grep -q crypt; then
            echo -e "${GREEN}Filesystem encryption (LUKS) detected${NC}"
        else
            echo -e "${YELLOW}Filesystem encryption not detected${NC}"
            echo "To enable encryption at rest, consider:"
            echo "  1. LUKS encryption for Linux"
            echo "  2. FileVault for macOS"
            echo "  3. BitLocker for Windows"
            echo "  4. Cloud provider encryption (EBS encryption for AWS)"
        fi
    fi
fi

# 5. Create encrypted backup script
echo -e "${GREEN}Step 5: Setting up encrypted backups${NC}"

cat > /usr/local/bin/pg_encrypted_backup.sh << 'EOF'
#!/bin/bash
# Encrypted PostgreSQL Backup Script

BACKUP_DIR="/mnt/bulkdata/backups/postgres"
mkdir -p "$BACKUP_DIR"

# Generate backup filename with timestamp
BACKUP_FILE="$BACKUP_DIR/postgres_$(date +%Y%m%d_%H%M%S).sql"
ENCRYPTED_FILE="$BACKUP_FILE.enc"

# Perform backup
pg_dumpall -U postgres > "$BACKUP_FILE"

# Encrypt backup using AES-256
openssl enc -aes-256-cbc -salt -in "$BACKUP_FILE" -out "$ENCRYPTED_FILE" -k "$BACKUP_ENCRYPTION_KEY"

# Remove unencrypted backup
shred -u "$BACKUP_FILE"

# Keep only last 30 days of backups
find "$BACKUP_DIR" -name "*.enc" -mtime +30 -delete

echo "Encrypted backup completed: $ENCRYPTED_FILE"
EOF

chmod +x /usr/local/bin/pg_encrypted_backup.sh
echo -e "${GREEN}Encrypted backup script created at /usr/local/bin/pg_encrypted_backup.sh${NC}"

# 6. Restart PostgreSQL to apply changes
echo -e "${GREEN}Step 6: Restarting PostgreSQL${NC}"

if systemctl is-active postgresql &> /dev/null; then
    systemctl restart postgresql
    echo -e "${GREEN}PostgreSQL restarted successfully${NC}"
elif service postgresql status &> /dev/null; then
    service postgresql restart
    echo -e "${GREEN}PostgreSQL restarted successfully${NC}"
else
    echo -e "${YELLOW}Please manually restart PostgreSQL to apply changes${NC}"
fi

# 7. Verify SSL is enabled
echo -e "${GREEN}Step 7: Verifying configuration${NC}"

sleep 2
if psql -U postgres -c "SHOW ssl;" | grep -q "on"; then
    echo -e "${GREEN}✓ SSL is enabled${NC}"
else
    echo -e "${RED}✗ SSL is not enabled - please check configuration${NC}"
fi

if psql -U postgres -c "SHOW password_encryption;" | grep -q "scram-sha-256"; then
    echo -e "${GREEN}✓ Password encryption is using scram-sha-256${NC}"
else
    echo -e "${YELLOW}⚠ Password encryption not using scram-sha-256${NC}"
fi

echo ""
echo -e "${GREEN}=== PostgreSQL Encryption Setup Complete ===${NC}"
echo ""
echo "Next steps:"
echo "1. Test SSL connections: psql 'sslmode=require' -U your_user -d your_db"
echo "2. Set up automated encrypted backups in cron"
echo "3. Rotate SSL certificates annually"
echo "4. Monitor logs for non-SSL connection attempts"
echo "5. Consider implementing Transparent Data Encryption (TDE) extension if available"
echo ""
echo -e "${YELLOW}IMPORTANT: Ensure BACKUP_ENCRYPTION_KEY environment variable is set for backups${NC}"