#!/bin/bash
#
# SSL/TLS Certificate Setup Script for AI Trading System
# Configures SSL certificates for production deployment
#
# Usage: ./setup_ssl_certificates.sh [options]
#
# Options:
#   --domain <domain>     Domain name for certificates (default: trading.main-nilante.com)
#   --email <email>       Email for Let's Encrypt registration (default: admin@main-nilante.com)
#   --staging             Use Let's Encrypt staging environment for testing
#   --self-signed         Generate self-signed certificates for local testing
#   --wildcard            Request wildcard certificate (requires DNS challenge)
#   --check-only          Only check certificate status without renewal
#   -h, --help           Show this help message
#

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration defaults
DOMAIN="${DOMAIN:-trading.main-nilante.com}"
EMAIL="${EMAIL:-admin@main-nilante.com}"
CERT_PATH="/srv/trading/config/letsencrypt"
TRAEFIK_CONFIG="/srv/trading/config/traefik"
USE_STAGING=false
SELF_SIGNED=false
WILDCARD=false
CHECK_ONLY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --email)
            EMAIL="$2"
            shift 2
            ;;
        --staging)
            USE_STAGING=true
            shift
            ;;
        --self-signed)
            SELF_SIGNED=true
            shift
            ;;
        --wildcard)
            WILDCARD=true
            shift
            ;;
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        -h|--help)
            echo "SSL/TLS Certificate Setup Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --domain <domain>     Domain name for certificates (default: trading.main-nilante.com)"
            echo "  --email <email>       Email for Let's Encrypt registration"
            echo "  --staging             Use Let's Encrypt staging environment"
            echo "  --self-signed         Generate self-signed certificates"
            echo "  --wildcard            Request wildcard certificate"
            echo "  --check-only          Only check certificate status"
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

# Check if running as root or with sudo
check_permissions() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root or with sudo"
        exit 1
    fi
}

# Create required directories
setup_directories() {
    log_info "Setting up certificate directories..."
    
    mkdir -p "$CERT_PATH"
    mkdir -p "$TRAEFIK_CONFIG"
    mkdir -p "$TRAEFIK_CONFIG/dynamic"
    
    # Set proper permissions
    chmod 700 "$CERT_PATH"
    chmod 755 "$TRAEFIK_CONFIG"
    
    log_info "✓ Directories created"
}

# Check certificate status
check_certificate_status() {
    log_info "Checking existing certificate status..."
    
    if [ -f "$CERT_PATH/acme.json" ]; then
        # Check certificate expiry using openssl
        if command -v openssl &> /dev/null; then
            # Extract certificate from acme.json if possible
            log_info "Let's Encrypt certificates found in acme.json"
            
            # Check if certificate is valid
            CERT_INFO=$(cat "$CERT_PATH/acme.json" 2>/dev/null | grep -c "certificate" || true)
            if [ "$CERT_INFO" -gt 0 ]; then
                log_info "✓ Certificates exist in acme.json"
            fi
        fi
    elif [ -f "$CERT_PATH/cert.pem" ]; then
        if command -v openssl &> /dev/null; then
            EXPIRY=$(openssl x509 -enddate -noout -in "$CERT_PATH/cert.pem" | cut -d= -f2)
            EXPIRY_EPOCH=$(date -d "$EXPIRY" +%s)
            CURRENT_EPOCH=$(date +%s)
            DAYS_LEFT=$(( ($EXPIRY_EPOCH - $CURRENT_EPOCH) / 86400 ))
            
            if [ $DAYS_LEFT -lt 30 ]; then
                log_warning "Certificate expires in $DAYS_LEFT days (Expiry: $EXPIRY)"
                return 1
            else
                log_info "✓ Certificate valid for $DAYS_LEFT days (Expiry: $EXPIRY)"
                return 0
            fi
        fi
    else
        log_warning "No existing certificates found"
        return 1
    fi
}

# Generate self-signed certificates
generate_self_signed() {
    log_info "Generating self-signed certificates..."
    
    # Generate private key
    openssl genrsa -out "$CERT_PATH/privkey.pem" 4096
    
    # Generate certificate signing request
    openssl req -new \
        -key "$CERT_PATH/privkey.pem" \
        -out "$CERT_PATH/cert.csr" \
        -subj "/C=US/ST=State/L=City/O=AI Trading System/CN=$DOMAIN"
    
    # Generate self-signed certificate (valid for 365 days)
    openssl x509 -req \
        -days 365 \
        -in "$CERT_PATH/cert.csr" \
        -signkey "$CERT_PATH/privkey.pem" \
        -out "$CERT_PATH/cert.pem"
    
    # Create full chain (same as cert for self-signed)
    cp "$CERT_PATH/cert.pem" "$CERT_PATH/fullchain.pem"
    
    # Set proper permissions
    chmod 600 "$CERT_PATH/privkey.pem"
    chmod 644 "$CERT_PATH/cert.pem"
    chmod 644 "$CERT_PATH/fullchain.pem"
    
    log_info "✓ Self-signed certificates generated"
}

# Configure Traefik for SSL
configure_traefik() {
    log_info "Configuring Traefik for SSL/TLS..."
    
    # Create Traefik static configuration
    cat > "$TRAEFIK_CONFIG/traefik.yml" <<EOF
# Traefik Static Configuration
api:
  dashboard: true
  insecure: false

entryPoints:
  web:
    address: ":80"
    http:
      redirections:
        entrypoint:
          to: websecure
          scheme: https
          permanent: true
  websecure:
    address: ":443"
    http:
      tls:
        certResolver: letsencrypt
        domains:
          - main: "$DOMAIN"
$(if [ "$WILDCARD" = true ]; then echo "          - sans: \"*.$DOMAIN\""; fi)

providers:
  docker:
    endpoint: "unix:///var/run/docker.sock"
    exposedByDefault: false
    network: trading-frontend
  file:
    directory: /etc/traefik/dynamic
    watch: true

certificatesResolvers:
  letsencrypt:
    acme:
      email: "$EMAIL"
      storage: /letsencrypt/acme.json
      keyType: EC256
$(if [ "$USE_STAGING" = true ]; then echo "      caServer: https://acme-staging-v02.api.letsencrypt.org/directory"; fi)
$(if [ "$WILDCARD" = true ]; then
    echo "      dnsChallenge:"
    echo "        provider: cloudflare"
    echo "        delayBeforeCheck: 10"
else
    echo "      httpChallenge:"
    echo "        entryPoint: web"
fi)

metrics:
  prometheus:
    buckets:
      - 0.1
      - 0.3
      - 1.2
      - 5.0
    addEntryPointsLabels: true
    addServicesLabels: true

log:
  level: INFO
  format: json

accessLog:
  format: json
  filters:
    statusCodes:
      - "200-599"

# Security headers middleware
http:
  middlewares:
    security-headers:
      headers:
        customFrameOptions: "SAMEORIGIN"
        contentTypeNosniff: true
        browserXssFilter: true
        referrerPolicy: "strict-origin-when-cross-origin"
        permissionsPolicy: "camera=(), microphone=(), geolocation=()"
        customResponseHeaders:
          X-Robots-Tag: "none,noarchive,nosnippet,notranslate,noimageindex"
          Server: ""
          X-Powered-By: ""
        sslRedirect: true
        sslProxyHeaders:
          X-Forwarded-Proto: https
        stsIncludeSubdomains: true
        stsPreload: true
        stsSeconds: 63072000
        forceSTSHeader: true
EOF
    
    # Create dynamic configuration for self-signed certificates if needed
    if [ "$SELF_SIGNED" = true ]; then
        cat > "$TRAEFIK_CONFIG/dynamic/self-signed.yml" <<EOF
tls:
  certificates:
    - certFile: /letsencrypt/cert.pem
      keyFile: /letsencrypt/privkey.pem
EOF
    fi
    
    # Set proper permissions on acme.json
    touch "$CERT_PATH/acme.json"
    chmod 600 "$CERT_PATH/acme.json"
    
    log_info "✓ Traefik configuration created"
}

# Setup automatic certificate renewal
setup_auto_renewal() {
    log_info "Setting up automatic certificate renewal..."
    
    # Create renewal script
    cat > /usr/local/bin/renew-certificates.sh <<'EOF'
#!/bin/bash
# Certificate renewal script

CERT_PATH="/srv/trading/config/letsencrypt"
LOG_FILE="/var/log/cert-renewal.log"

echo "[$(date)] Starting certificate renewal check..." >> "$LOG_FILE"

# Traefik handles Let's Encrypt renewal automatically
# This script is for monitoring and alerting

# Check certificate expiry
if [ -f "$CERT_PATH/acme.json" ]; then
    # Check if Traefik is running
    if docker ps | grep -q trading-traefik; then
        echo "[$(date)] Traefik is running - certificates will auto-renew" >> "$LOG_FILE"
    else
        echo "[$(date)] WARNING: Traefik is not running!" >> "$LOG_FILE"
    fi
fi

# Send notification if certificates expire soon (optional)
# You can add email/slack notification here
EOF
    
    chmod +x /usr/local/bin/renew-certificates.sh
    
    # Create systemd timer for renewal checks
    cat > /etc/systemd/system/cert-renewal.service <<EOF
[Unit]
Description=Certificate Renewal Check
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/renew-certificates.sh
User=root
StandardOutput=journal
StandardError=journal
EOF
    
    cat > /etc/systemd/system/cert-renewal.timer <<EOF
[Unit]
Description=Run Certificate Renewal Check Daily
Requires=cert-renewal.service

[Timer]
OnCalendar=daily
RandomizedDelaySec=1h
Persistent=true

[Install]
WantedBy=timers.target
EOF
    
    # Enable and start the timer
    systemctl daemon-reload
    systemctl enable cert-renewal.timer
    systemctl start cert-renewal.timer
    
    log_info "✓ Automatic renewal configured"
}

# Verify SSL configuration
verify_ssl() {
    log_info "Verifying SSL configuration..."
    
    # Check if Traefik is running
    if docker ps | grep -q trading-traefik; then
        log_info "✓ Traefik is running"
        
        # Test HTTPS endpoint
        if command -v curl &> /dev/null; then
            RESPONSE=$(curl -sI "https://$DOMAIN" 2>&1 || true)
            if echo "$RESPONSE" | grep -q "HTTP/2 200\|HTTP/1.1 200"; then
                log_info "✓ HTTPS endpoint is responding"
            else
                log_warning "HTTPS endpoint not responding as expected"
            fi
        fi
    else
        log_warning "Traefik is not running - start it to enable SSL"
    fi
    
    # Check certificate files
    if [ -f "$CERT_PATH/acme.json" ] || [ -f "$CERT_PATH/cert.pem" ]; then
        log_info "✓ Certificate files present"
    else
        log_warning "Certificate files not found"
    fi
}

# Main execution
main() {
    log_info "Starting SSL/TLS certificate setup..."
    log_info "Domain: $DOMAIN"
    log_info "Email: $EMAIL"
    
    if [ "$CHECK_ONLY" = true ]; then
        check_certificate_status
        exit 0
    fi
    
    # Check permissions (skip for check-only mode)
    check_permissions
    
    # Setup directories
    setup_directories
    
    # Check existing certificates
    if check_certificate_status; then
        log_info "Valid certificates already exist"
        read -p "Do you want to reconfigure? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Keeping existing configuration"
            exit 0
        fi
    fi
    
    # Generate certificates based on mode
    if [ "$SELF_SIGNED" = true ]; then
        generate_self_signed
    else
        log_info "Let's Encrypt certificates will be obtained automatically by Traefik"
        log_info "Make sure ports 80 and 443 are accessible from the internet"
    fi
    
    # Configure Traefik
    configure_traefik
    
    # Setup auto-renewal
    setup_auto_renewal
    
    # Verify configuration
    verify_ssl
    
    log_info "========================================="
    log_info "SSL/TLS SETUP COMPLETED"
    log_info "========================================="
    log_info ""
    log_info "Next steps:"
    log_info "1. Ensure DNS A record points to server IP"
    log_info "2. Open ports 80 and 443 in firewall"
    log_info "3. Restart Traefik to apply configuration:"
    log_info "   docker-compose restart traefik"
    log_info ""
    if [ "$SELF_SIGNED" = true ]; then
        log_warning "Using self-signed certificates - browsers will show security warning"
    else
        log_info "Let's Encrypt certificates will be obtained on first HTTPS request"
    fi
}

# Run main function
main