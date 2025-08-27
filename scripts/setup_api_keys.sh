#!/bin/bash
#
# API Key Setup and Validation Script for AI Trading System
# Configures and validates external API keys for market data providers
#
# Usage: ./setup_api_keys.sh [options]
#
# Options:
#   --provider <name>     Specific provider to setup (alpaca|polygon|finnhub|alphavantage|all)
#   --validate            Validate existing API keys
#   --test                Test API connectivity
#   --encrypt             Encrypt API keys for secure storage
#   --import <file>       Import API keys from encrypted file
#   --export <file>       Export API keys to encrypted file
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
SECURE_STORAGE="/srv/trading/config/api_keys"
PROVIDER="all"
VALIDATE=false
TEST_MODE=false
ENCRYPT=false
IMPORT_FILE=""
EXPORT_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --validate)
            VALIDATE=true
            shift
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        --encrypt)
            ENCRYPT=true
            shift
            ;;
        --import)
            IMPORT_FILE="$2"
            shift 2
            ;;
        --export)
            EXPORT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "API Key Setup Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --provider <name>     Specific provider (alpaca|polygon|finnhub|alphavantage|all)"
            echo "  --validate            Validate existing API keys"
            echo "  --test                Test API connectivity"
            echo "  --encrypt             Encrypt API keys"
            echo "  --import <file>       Import from encrypted file"
            echo "  --export <file>       Export to encrypted file"
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

# Setup secure storage directory
setup_secure_storage() {
    if [ ! -d "$SECURE_STORAGE" ]; then
        log_info "Creating secure storage directory..."
        sudo mkdir -p "$SECURE_STORAGE"
        sudo chmod 700 "$SECURE_STORAGE"
        sudo chown $(whoami):$(whoami) "$SECURE_STORAGE"
    fi
}

# Validate API key format
validate_key_format() {
    local key_name=$1
    local key_value=$2
    
    case "$key_name" in
        ALPACA_API_KEY)
            if [[ ! "$key_value" =~ ^PK[A-Z0-9]{18,}$ ]]; then
                log_warning "Alpaca API key format appears invalid"
                return 1
            fi
            ;;
        POLYGON_API_KEY)
            if [[ ${#key_value} -lt 32 ]]; then
                log_warning "Polygon API key appears too short"
                return 1
            fi
            ;;
        FINNHUB_API_KEY)
            if [[ ! "$key_value" =~ ^[a-zA-Z0-9]{20,}$ ]]; then
                log_warning "Finnhub API key format appears invalid"
                return 1
            fi
            ;;
        ALPHA_VANTAGE_API_KEY)
            if [[ ${#key_value} -lt 16 ]]; then
                log_warning "Alpha Vantage API key appears too short"
                return 1
            fi
            ;;
    esac
    
    return 0
}

# Setup Alpaca API keys
setup_alpaca() {
    log_info "Setting up Alpaca API keys..."
    
    if [ "$VALIDATE" = true ]; then
        # Check existing keys
        if grep -q "^ALPACA_API_KEY=" "$ENV_FILE" 2>/dev/null; then
            EXISTING_KEY=$(grep "^ALPACA_API_KEY=" "$ENV_FILE" | cut -d= -f2)
            if [ -n "$EXISTING_KEY" ]; then
                log_info "✓ Alpaca API key already configured"
                
                if [ "$TEST_MODE" = true ]; then
                    test_alpaca_connection "$EXISTING_KEY"
                fi
                return 0
            fi
        fi
    fi
    
    echo "Enter Alpaca API Key (or press Enter to skip):"
    read -s ALPACA_KEY
    
    if [ -n "$ALPACA_KEY" ]; then
        echo "Enter Alpaca Secret Key:"
        read -s ALPACA_SECRET
        
        if validate_key_format "ALPACA_API_KEY" "$ALPACA_KEY"; then
            # Update environment file
            update_env_var "ALPACA_API_KEY" "$ALPACA_KEY"
            update_env_var "ALPACA_SECRET_KEY" "$ALPACA_SECRET"
            update_env_var "ALPACA_BASE_URL" "https://paper-api.alpaca.markets"
            
            log_info "✓ Alpaca API keys configured"
            
            if [ "$TEST_MODE" = true ]; then
                test_alpaca_connection "$ALPACA_KEY"
            fi
        fi
    else
        log_info "Skipping Alpaca setup"
    fi
}

# Test Alpaca connection
test_alpaca_connection() {
    local api_key=$1
    
    log_info "Testing Alpaca API connection..."
    
    RESPONSE=$(curl -s -H "APCA-API-KEY-ID: $api_key" \
        "https://paper-api.alpaca.markets/v2/account" \
        -w "\n%{http_code}" | tail -n1)
    
    if [ "$RESPONSE" = "200" ]; then
        log_info "✓ Alpaca API connection successful"
    else
        log_error "Alpaca API connection failed (HTTP $RESPONSE)"
    fi
}

# Setup Polygon API key
setup_polygon() {
    log_info "Setting up Polygon.io API key..."
    
    if [ "$VALIDATE" = true ]; then
        if grep -q "^POLYGON_API_KEY=" "$ENV_FILE" 2>/dev/null; then
            EXISTING_KEY=$(grep "^POLYGON_API_KEY=" "$ENV_FILE" | cut -d= -f2)
            if [ -n "$EXISTING_KEY" ]; then
                log_info "✓ Polygon API key already configured"
                
                if [ "$TEST_MODE" = true ]; then
                    test_polygon_connection "$EXISTING_KEY"
                fi
                return 0
            fi
        fi
    fi
    
    echo "Enter Polygon.io API Key (or press Enter to skip):"
    read -s POLYGON_KEY
    
    if [ -n "$POLYGON_KEY" ]; then
        if validate_key_format "POLYGON_API_KEY" "$POLYGON_KEY"; then
            update_env_var "POLYGON_API_KEY" "$POLYGON_KEY"
            log_info "✓ Polygon API key configured"
            
            if [ "$TEST_MODE" = true ]; then
                test_polygon_connection "$POLYGON_KEY"
            fi
        fi
    else
        log_info "Skipping Polygon setup"
    fi
}

# Test Polygon connection
test_polygon_connection() {
    local api_key=$1
    
    log_info "Testing Polygon API connection..."
    
    RESPONSE=$(curl -s "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-01?apiKey=$api_key" \
        -w "\n%{http_code}" | tail -n1)
    
    if [ "$RESPONSE" = "200" ]; then
        log_info "✓ Polygon API connection successful"
    else
        log_error "Polygon API connection failed (HTTP $RESPONSE)"
    fi
}

# Setup Finnhub API key
setup_finnhub() {
    log_info "Setting up Finnhub API key..."
    
    if [ "$VALIDATE" = true ]; then
        if grep -q "^FINNHUB_API_KEY=" "$ENV_FILE" 2>/dev/null; then
            EXISTING_KEY=$(grep "^FINNHUB_API_KEY=" "$ENV_FILE" | cut -d= -f2)
            if [ -n "$EXISTING_KEY" ]; then
                log_info "✓ Finnhub API key already configured"
                
                if [ "$TEST_MODE" = true ]; then
                    test_finnhub_connection "$EXISTING_KEY"
                fi
                return 0
            fi
        fi
    fi
    
    echo "Enter Finnhub API Key (or press Enter to skip):"
    read -s FINNHUB_KEY
    
    if [ -n "$FINNHUB_KEY" ]; then
        if validate_key_format "FINNHUB_API_KEY" "$FINNHUB_KEY"; then
            update_env_var "FINNHUB_API_KEY" "$FINNHUB_KEY"
            log_info "✓ Finnhub API key configured"
            
            if [ "$TEST_MODE" = true ]; then
                test_finnhub_connection "$FINNHUB_KEY"
            fi
        fi
    else
        log_info "Skipping Finnhub setup"
    fi
}

# Test Finnhub connection
test_finnhub_connection() {
    local api_key=$1
    
    log_info "Testing Finnhub API connection..."
    
    RESPONSE=$(curl -s "https://finnhub.io/api/v1/quote?symbol=AAPL&token=$api_key" \
        -w "\n%{http_code}" | tail -n1)
    
    if [ "$RESPONSE" = "200" ]; then
        log_info "✓ Finnhub API connection successful"
    else
        log_error "Finnhub API connection failed (HTTP $RESPONSE)"
    fi
}

# Setup Alpha Vantage API key
setup_alphavantage() {
    log_info "Setting up Alpha Vantage API key..."
    
    if [ "$VALIDATE" = true ]; then
        if grep -q "^ALPHA_VANTAGE_API_KEY=" "$ENV_FILE" 2>/dev/null; then
            EXISTING_KEY=$(grep "^ALPHA_VANTAGE_API_KEY=" "$ENV_FILE" | cut -d= -f2)
            if [ -n "$EXISTING_KEY" ]; then
                log_info "✓ Alpha Vantage API key already configured"
                
                if [ "$TEST_MODE" = true ]; then
                    test_alphavantage_connection "$EXISTING_KEY"
                fi
                return 0
            fi
        fi
    fi
    
    echo "Enter Alpha Vantage API Key (or press Enter to skip):"
    read -s ALPHA_KEY
    
    if [ -n "$ALPHA_KEY" ]; then
        if validate_key_format "ALPHA_VANTAGE_API_KEY" "$ALPHA_KEY"; then
            update_env_var "ALPHA_VANTAGE_API_KEY" "$ALPHA_KEY"
            log_info "✓ Alpha Vantage API key configured"
            
            if [ "$TEST_MODE" = true ]; then
                test_alphavantage_connection "$ALPHA_KEY"
            fi
        fi
    else
        log_info "Skipping Alpha Vantage setup"
    fi
}

# Test Alpha Vantage connection
test_alphavantage_connection() {
    local api_key=$1
    
    log_info "Testing Alpha Vantage API connection..."
    
    RESPONSE=$(curl -s "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey=$api_key" \
        -w "\n%{http_code}" | tail -n1)
    
    if [ "$RESPONSE" = "200" ]; then
        log_info "✓ Alpha Vantage API connection successful"
    else
        log_error "Alpha Vantage API connection failed (HTTP $RESPONSE)"
    fi
}

# Update environment variable
update_env_var() {
    local key=$1
    local value=$2
    
    # Check if key exists
    if grep -q "^${key}=" "$ENV_FILE"; then
        # Update existing
        sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
    else
        # Add new
        echo "${key}=${value}" >> "$ENV_FILE"
    fi
}

# Encrypt API keys
encrypt_api_keys() {
    log_info "Encrypting API keys..."
    
    setup_secure_storage
    
    # Extract API keys from environment file
    grep -E "^(ALPACA|POLYGON|FINNHUB|ALPHA_VANTAGE|NEWS).*API" "$ENV_FILE" > "$SECURE_STORAGE/api_keys.tmp" || true
    
    if [ -s "$SECURE_STORAGE/api_keys.tmp" ]; then
        # Encrypt with GPG
        if command -v gpg &> /dev/null; then
            gpg --symmetric --cipher-algo AES256 --output "$SECURE_STORAGE/api_keys.enc" "$SECURE_STORAGE/api_keys.tmp"
            rm -f "$SECURE_STORAGE/api_keys.tmp"
            chmod 600 "$SECURE_STORAGE/api_keys.enc"
            log_info "✓ API keys encrypted and stored securely"
        else
            log_warning "GPG not available - storing unencrypted (restricted permissions)"
            mv "$SECURE_STORAGE/api_keys.tmp" "$SECURE_STORAGE/api_keys.txt"
            chmod 600 "$SECURE_STORAGE/api_keys.txt"
        fi
    else
        log_warning "No API keys found to encrypt"
    fi
}

# Import API keys from encrypted file
import_api_keys() {
    local import_file=$1
    
    if [ ! -f "$import_file" ]; then
        log_error "Import file not found: $import_file"
        exit 1
    fi
    
    log_info "Importing API keys from: $import_file"
    
    # Decrypt file
    if [[ "$import_file" == *.enc || "$import_file" == *.gpg ]]; then
        if command -v gpg &> /dev/null; then
            gpg --decrypt "$import_file" > /tmp/api_keys_import.tmp
        else
            log_error "GPG required to decrypt file"
            exit 1
        fi
    else
        cp "$import_file" /tmp/api_keys_import.tmp
    fi
    
    # Import keys
    while IFS='=' read -r key value; do
        if [ -n "$key" ] && [ -n "$value" ]; then
            update_env_var "$key" "$value"
            log_info "✓ Imported: $key"
        fi
    done < /tmp/api_keys_import.tmp
    
    # Cleanup
    rm -f /tmp/api_keys_import.tmp
    
    log_info "✓ API keys imported successfully"
}

# Export API keys to encrypted file
export_api_keys() {
    local export_file=$1
    
    log_info "Exporting API keys to: $export_file"
    
    # Extract API keys
    grep -E "^(ALPACA|POLYGON|FINNHUB|ALPHA_VANTAGE|NEWS).*API" "$ENV_FILE" > /tmp/api_keys_export.tmp || true
    
    if [ ! -s /tmp/api_keys_export.tmp ]; then
        log_warning "No API keys found to export"
        rm -f /tmp/api_keys_export.tmp
        return
    fi
    
    # Encrypt if GPG available
    if command -v gpg &> /dev/null; then
        gpg --symmetric --cipher-algo AES256 --output "$export_file" /tmp/api_keys_export.tmp
        log_info "✓ API keys exported (encrypted)"
    else
        cp /tmp/api_keys_export.tmp "$export_file"
        chmod 600 "$export_file"
        log_warning "API keys exported (unencrypted - GPG not available)"
    fi
    
    # Cleanup
    rm -f /tmp/api_keys_export.tmp
}

# Validate all configured API keys
validate_all_keys() {
    log_info "Validating all configured API keys..."
    
    local all_valid=true
    
    # Check each provider
    for provider in ALPACA POLYGON FINNHUB ALPHA_VANTAGE; do
        KEY_VAR="${provider}_API_KEY"
        if grep -q "^${KEY_VAR}=" "$ENV_FILE" 2>/dev/null; then
            KEY_VALUE=$(grep "^${KEY_VAR}=" "$ENV_FILE" | cut -d= -f2)
            if [ -n "$KEY_VALUE" ]; then
                log_info "Found ${provider} API key"
                
                if [ "$TEST_MODE" = true ]; then
                    case "$provider" in
                        ALPACA)
                            test_alpaca_connection "$KEY_VALUE"
                            ;;
                        POLYGON)
                            test_polygon_connection "$KEY_VALUE"
                            ;;
                        FINNHUB)
                            test_finnhub_connection "$KEY_VALUE"
                            ;;
                        ALPHA_VANTAGE)
                            test_alphavantage_connection "$KEY_VALUE"
                            ;;
                    esac
                fi
            else
                log_warning "${provider} API key is empty"
                all_valid=false
            fi
        fi
    done
    
    if [ "$all_valid" = true ]; then
        log_info "✓ All configured API keys are valid"
    else
        log_warning "Some API keys need attention"
    fi
}

# Generate API key summary
generate_summary() {
    log_info "API Key Configuration Summary:"
    log_info "=============================="
    
    for provider in ALPACA POLYGON FINNHUB ALPHA_VANTAGE NEWS; do
        KEY_VAR="${provider}_API_KEY"
        if grep -q "^${KEY_VAR}=" "$ENV_FILE" 2>/dev/null; then
            KEY_VALUE=$(grep "^${KEY_VAR}=" "$ENV_FILE" | cut -d= -f2)
            if [ -n "$KEY_VALUE" ]; then
                # Mask the key for security
                MASKED_KEY="${KEY_VALUE:0:4}...${KEY_VALUE: -4}"
                echo -e "${GREEN}✓${NC} ${provider}: Configured (${MASKED_KEY})"
            else
                echo -e "${YELLOW}⚠${NC} ${provider}: Empty"
            fi
        else
            echo -e "${RED}✗${NC} ${provider}: Not configured"
        fi
    done
}

# Main execution
main() {
    log_info "Starting API key setup..."
    
    # Check if environment file exists
    if [ ! -f "$ENV_FILE" ]; then
        log_error "Environment file not found: $ENV_FILE"
        log_info "Creating from template..."
        cp .env.example "$ENV_FILE"
    fi
    
    # Handle import/export operations
    if [ -n "$IMPORT_FILE" ]; then
        import_api_keys "$IMPORT_FILE"
        exit 0
    fi
    
    if [ -n "$EXPORT_FILE" ]; then
        export_api_keys "$EXPORT_FILE"
        exit 0
    fi
    
    # Validate existing keys
    if [ "$VALIDATE" = true ]; then
        validate_all_keys
        generate_summary
        exit 0
    fi
    
    # Setup API keys based on provider
    case "$PROVIDER" in
        alpaca)
            setup_alpaca
            ;;
        polygon)
            setup_polygon
            ;;
        finnhub)
            setup_finnhub
            ;;
        alphavantage)
            setup_alphavantage
            ;;
        all)
            setup_alpaca
            setup_polygon
            setup_finnhub
            setup_alphavantage
            ;;
        *)
            log_error "Unknown provider: $PROVIDER"
            exit 1
            ;;
    esac
    
    # Encrypt if requested
    if [ "$ENCRYPT" = true ]; then
        encrypt_api_keys
    fi
    
    # Generate summary
    generate_summary
    
    log_info "========================================="
    log_info "API KEY SETUP COMPLETED"
    log_info "========================================="
    log_info ""
    log_info "Next steps:"
    log_info "1. Test API connections: $0 --validate --test"
    log_info "2. Encrypt keys for security: $0 --encrypt"
    log_info "3. Restart services to use new keys"
}

# Run main function
main