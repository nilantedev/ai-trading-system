#!/bin/bash
#
# Production Secrets Rotation Script for AI Trading System
# Rotates sensitive credentials and API keys safely
#
# Usage: ./rotate_secrets.sh [options]
#
# Options:
#   --type <type>         Type of secret to rotate (db|redis|jwt|api|all)
#   --env <environment>   Environment (production|staging) (default: production)
#   --backup              Create backup before rotation
#   --force               Force rotation without confirmation
#   --dry-run             Show what would be rotated without making changes
#   -h, --help           Show this help message
#

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENV_FILE=".env.production"
BACKUP_DIR="./backups/secrets"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SECRET_TYPE="all"
ENVIRONMENT="production"
CREATE_BACKUP=true
FORCE=false
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            SECRET_TYPE="$2"
            shift 2
            ;;
        --env)
            ENVIRONMENT="$2"
            ENV_FILE=".env.$2"
            shift 2
            ;;
        --backup)
            CREATE_BACKUP=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Secrets Rotation Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --type <type>         Type of secret to rotate (db|redis|jwt|api|all)"
            echo "  --env <environment>   Environment (production|staging)"
            echo "  --backup              Create backup before rotation"
            echo "  --force               Force rotation without confirmation"
            echo "  --dry-run             Show what would be rotated"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Generate secure random password
generate_password() {
    local length=${1:-32}
    openssl rand -base64 $length | tr -d "=+/" | cut -c1-$length
}

# Generate secure API key
generate_api_key() {
    local prefix=${1:-"sk"}
    local key=$(openssl rand -hex 32)
    echo "${prefix}_${key}"
}

# Generate JWT secret
generate_jwt_secret() {
    openssl rand -base64 64 | tr -d "\n"
}

# Backup current secrets
backup_secrets() {
    if [ "$CREATE_BACKUP" = true ] && [ "$DRY_RUN" = false ]; then
        log_info "Creating backup of current secrets..."
        
        mkdir -p "$BACKUP_DIR"
        
        # Encrypt the backup
        if command -v gpg &> /dev/null; then
            gpg --symmetric --cipher-algo AES256 --output "$BACKUP_DIR/secrets_${TIMESTAMP}.env.gpg" "$ENV_FILE"
            log_info "✓ Encrypted backup created: $BACKUP_DIR/secrets_${TIMESTAMP}.env.gpg"
        else
            cp "$ENV_FILE" "$BACKUP_DIR/secrets_${TIMESTAMP}.env"
            chmod 600 "$BACKUP_DIR/secrets_${TIMESTAMP}.env"
            log_warning "GPG not available - created unencrypted backup (restricted permissions)"
        fi
    fi
}

# Update secret in environment file
update_secret() {
    local key=$1
    local value=$2
    
    if [ "$DRY_RUN" = true ]; then
        log_debug "[DRY-RUN] Would update: $key=<redacted>"
    else
        # Use sed to update the value
        sed -i.bak "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
        log_info "✓ Updated: $key"
    fi
}

# Rotate database passwords
rotate_db_secrets() {
    log_info "Rotating database secrets..."
    
    local db_password=$(generate_password 32)
    local db_root_password=$(generate_password 32)
    
    update_secret "DB_PASSWORD" "$db_password"
    update_secret "DB_ROOT_PASSWORD" "$db_root_password"
    
    if [ "$DRY_RUN" = false ]; then
        # Update database passwords if service is running
        if docker ps | grep -q trading-postgres; then
            log_info "Updating PostgreSQL passwords..."
            
            # Create SQL script to update passwords
            cat > /tmp/rotate_db.sql <<EOF
ALTER USER trading_user WITH PASSWORD '${db_password}';
ALTER USER postgres WITH PASSWORD '${db_root_password}';
EOF
            
            # Execute password rotation
            docker exec -i trading-postgres psql -U postgres < /tmp/rotate_db.sql
            rm -f /tmp/rotate_db.sql
            
            log_info "✓ Database passwords updated"
        else
            log_warning "PostgreSQL not running - manual password update required"
        fi
    fi
}

# Rotate Redis password
rotate_redis_secret() {
    log_info "Rotating Redis password..."
    
    local redis_password=$(generate_password 32)
    update_secret "REDIS_PASSWORD" "$redis_password"
    
    if [ "$DRY_RUN" = false ]; then
        # Update Redis password if service is running
        if docker ps | grep -q trading-redis; then
            log_info "Updating Redis password..."
            
            # Update Redis configuration
            docker exec trading-redis redis-cli CONFIG SET requirepass "$redis_password"
            
            log_info "✓ Redis password updated"
        else
            log_warning "Redis not running - manual password update required"
        fi
    fi
}

# Rotate JWT secrets
rotate_jwt_secrets() {
    log_info "Rotating JWT secrets..."
    
    local jwt_secret=$(generate_jwt_secret)
    local jwt_refresh_secret=$(generate_jwt_secret)
    
    update_secret "JWT_SECRET" "$jwt_secret"
    update_secret "JWT_REFRESH_SECRET" "$jwt_refresh_secret"
    
    if [ "$DRY_RUN" = false ]; then
        log_warning "JWT secrets rotated - all existing tokens will be invalidated"
        log_warning "Users will need to re-authenticate"
    fi
}

# Rotate API keys
rotate_api_keys() {
    log_info "Rotating API keys..."
    
    # Internal API keys
    local api_key=$(generate_api_key "api")
    local webhook_secret=$(generate_api_key "whsec")
    
    update_secret "API_KEY" "$api_key"
    update_secret "WEBHOOK_SECRET" "$webhook_secret"
    
    # Monitoring secrets
    local grafana_password=$(generate_password 24)
    update_secret "GRAFANA_PASSWORD" "$grafana_password"
    
    if [ "$DRY_RUN" = false ]; then
        log_info "✓ API keys rotated"
        log_warning "Update any external systems using these API keys"
    fi
}

# Rotate MinIO secrets
rotate_minio_secrets() {
    log_info "Rotating MinIO secrets..."
    
    local minio_root_user=$(generate_password 16)
    local minio_root_password=$(generate_password 32)
    
    update_secret "MINIO_ROOT_USER" "$minio_root_user"
    update_secret "MINIO_ROOT_PASSWORD" "$minio_root_password"
    
    if [ "$DRY_RUN" = false ]; then
        if docker ps | grep -q trading-minio; then
            log_warning "MinIO credentials rotated - restart required"
        fi
    fi
}

# Rotate Weaviate secrets  
rotate_weaviate_secrets() {
    log_info "Rotating Weaviate secrets..."
    
    local weaviate_key=$(generate_api_key "wv")
    update_secret "WEAVIATE_AUTHENTICATION_APIKEY_ALLOWED_KEYS" "$weaviate_key"
    
    if [ "$DRY_RUN" = false ]; then
        log_info "✓ Weaviate API key rotated"
    fi
}

# Rotate encryption keys
rotate_encryption_keys() {
    log_info "Rotating encryption keys..."
    
    local secret_key=$(generate_password 64)
    local backup_key=$(openssl rand -base64 32 | tr -d "\n")
    
    update_secret "SECRET_KEY" "$secret_key"
    update_secret "BACKUP_ENCRYPTION_KEY" "$backup_key"
    
    if [ "$DRY_RUN" = false ]; then
        log_warning "Encryption keys rotated - existing encrypted data may need re-encryption"
    fi
}

# Verify rotated secrets
verify_rotation() {
    if [ "$DRY_RUN" = true ]; then
        return 0
    fi
    
    log_info "Verifying rotated secrets..."
    
    # Check environment file syntax
    if bash -n "$ENV_FILE" 2>/dev/null; then
        log_info "✓ Environment file syntax valid"
    else
        log_error "Environment file has syntax errors"
        return 1
    fi
    
    # Check for empty values
    local empty_count=$(grep -c "=$" "$ENV_FILE" || true)
    if [ "$empty_count" -gt 0 ]; then
        log_warning "Found $empty_count empty secret values"
    fi
    
    # Test service connectivity with new credentials
    if docker ps | grep -q trading-system; then
        log_info "Testing service connectivity..."
        
        # Test Redis connection
        if docker ps | grep -q trading-redis; then
            REDIS_PASS=$(grep "^REDIS_PASSWORD=" "$ENV_FILE" | cut -d= -f2)
            if docker exec trading-redis redis-cli -a "$REDIS_PASS" ping &>/dev/null; then
                log_info "✓ Redis connection successful"
            else
                log_warning "Redis connection failed with new password"
            fi
        fi
    fi
}

# Apply rotated secrets to running services
apply_secrets() {
    if [ "$DRY_RUN" = true ]; then
        log_debug "[DRY-RUN] Would restart services to apply new secrets"
        return 0
    fi
    
    log_info "Applying rotated secrets..."
    
    # Check if services are running
    if docker-compose ps | grep -q "Up"; then
        read -p "Restart services to apply new secrets? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Restarting services..."
            docker-compose down
            docker-compose up -d
            
            # Wait for services to be healthy
            sleep 10
            
            # Check service health
            if docker-compose ps | grep -q "healthy"; then
                log_info "✓ Services restarted successfully"
            else
                log_warning "Some services may not be healthy - check logs"
            fi
        else
            log_warning "Services not restarted - new secrets not applied"
            log_warning "Run 'docker-compose restart' to apply changes"
        fi
    else
        log_info "Services not running - secrets will be used on next start"
    fi
}

# Generate rotation report
generate_report() {
    local report_file="./logs/secret_rotation_${TIMESTAMP}.log"
    mkdir -p ./logs
    
    {
        echo "Secret Rotation Report"
        echo "======================"
        echo "Timestamp: $(date)"
        echo "Environment: $ENVIRONMENT"
        echo "Secret Types: $SECRET_TYPE"
        echo "Dry Run: $DRY_RUN"
        echo ""
        echo "Rotated Secrets:"
        echo "----------------"
        
        case "$SECRET_TYPE" in
            db)
                echo "- Database passwords"
                ;;
            redis)
                echo "- Redis password"
                ;;
            jwt)
                echo "- JWT secrets"
                ;;
            api)
                echo "- API keys"
                ;;
            all)
                echo "- All secrets"
                ;;
        esac
        
        echo ""
        echo "Backup Location: $BACKUP_DIR/secrets_${TIMESTAMP}.env.gpg"
        echo ""
        echo "Actions Required:"
        echo "----------------"
        echo "1. Restart services to apply new secrets"
        echo "2. Update external systems with new API keys"
        echo "3. Notify team members of rotation"
        echo "4. Test all integrations"
    } > "$report_file"
    
    log_info "Rotation report saved: $report_file"
}

# Main execution
main() {
    log_info "Starting secret rotation..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Secret Type: $SECRET_TYPE"
    
    # Check if environment file exists
    if [ ! -f "$ENV_FILE" ]; then
        log_error "Environment file not found: $ENV_FILE"
        exit 1
    fi
    
    # Confirm rotation
    if [ "$FORCE" = false ] && [ "$DRY_RUN" = false ]; then
        log_warning "This will rotate secrets and may disrupt services"
        read -p "Continue with secret rotation? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Rotation cancelled"
            exit 0
        fi
    fi
    
    # Create backup
    backup_secrets
    
    # Rotate secrets based on type
    case "$SECRET_TYPE" in
        db)
            rotate_db_secrets
            ;;
        redis)
            rotate_redis_secret
            ;;
        jwt)
            rotate_jwt_secrets
            ;;
        api)
            rotate_api_keys
            ;;
        all)
            rotate_db_secrets
            rotate_redis_secret
            rotate_jwt_secrets
            rotate_api_keys
            rotate_minio_secrets
            rotate_weaviate_secrets
            rotate_encryption_keys
            ;;
        *)
            log_error "Unknown secret type: $SECRET_TYPE"
            exit 1
            ;;
    esac
    
    # Verify rotation
    verify_rotation
    
    # Apply secrets if not dry run
    if [ "$DRY_RUN" = false ]; then
        apply_secrets
    fi
    
    # Generate report
    generate_report
    
    log_info "========================================="
    log_info "SECRET ROTATION COMPLETED"
    log_info "========================================="
    
    if [ "$DRY_RUN" = true ]; then
        log_info "Dry run completed - no changes made"
    else
        log_warning "IMPORTANT: Secrets have been rotated"
        log_warning "1. Restart services to apply changes"
        log_warning "2. Update external systems with new credentials"
        log_warning "3. Test all integrations"
    fi
}

# Run main function
main