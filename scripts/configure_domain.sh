#!/bin/bash
#
# Domain Configuration Script for AI Trading System
# Configures domain settings for production deployment
#
# Usage: ./configure_domain.sh [options]
#
# Options:
#   --domain <domain>     Primary domain name (default: trading.main-nilante.com)
#   --subdomain <sub>     Subdomain prefix (default: trading)
#   --ip <address>        Server IP address (default: auto-detect)
#   --check-dns           Check DNS configuration
#   --setup-nginx         Configure nginx reverse proxy
#   --setup-traefik       Configure Traefik labels
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
DOMAIN="main-nilante.com"
SUBDOMAIN="trading"
FULL_DOMAIN="${SUBDOMAIN}.${DOMAIN}"
SERVER_IP=""
CHECK_DNS=false
SETUP_NGINX=false
SETUP_TRAEFIK=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --domain)
            DOMAIN="$2"
            FULL_DOMAIN="${SUBDOMAIN}.${DOMAIN}"
            shift 2
            ;;
        --subdomain)
            SUBDOMAIN="$2"
            FULL_DOMAIN="${SUBDOMAIN}.${DOMAIN}"
            shift 2
            ;;
        --ip)
            SERVER_IP="$2"
            shift 2
            ;;
        --check-dns)
            CHECK_DNS=true
            shift
            ;;
        --setup-nginx)
            SETUP_NGINX=true
            shift
            ;;
        --setup-traefik)
            SETUP_TRAEFIK=true
            shift
            ;;
        -h|--help)
            echo "Domain Configuration Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --domain <domain>     Primary domain name"
            echo "  --subdomain <sub>     Subdomain prefix"
            echo "  --ip <address>        Server IP address"
            echo "  --check-dns           Check DNS configuration"
            echo "  --setup-nginx         Configure nginx reverse proxy"
            echo "  --setup-traefik       Configure Traefik labels"
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

# Auto-detect server IP if not provided
detect_server_ip() {
    if [ -z "$SERVER_IP" ]; then
        log_info "Auto-detecting server IP address..."
        
        # Try to get public IP
        SERVER_IP=$(curl -s ifconfig.me || curl -s icanhazip.com || echo "")
        
        if [ -z "$SERVER_IP" ]; then
            # Fallback to local IP
            SERVER_IP=$(hostname -I | awk '{print $1}')
        fi
        
        if [ -n "$SERVER_IP" ]; then
            log_info "Detected IP: $SERVER_IP"
        else
            log_error "Could not detect server IP address"
            exit 1
        fi
    fi
}

# Check DNS configuration
check_dns_configuration() {
    log_info "Checking DNS configuration for $FULL_DOMAIN..."
    
    # Check A record
    A_RECORD=$(dig +short A "$FULL_DOMAIN" 2>/dev/null || echo "")
    
    if [ -n "$A_RECORD" ]; then
        log_info "A record found: $FULL_DOMAIN -> $A_RECORD"
        
        if [ "$A_RECORD" = "$SERVER_IP" ]; then
            log_info "✓ DNS A record correctly points to server"
        else
            log_warning "DNS A record ($A_RECORD) doesn't match server IP ($SERVER_IP)"
        fi
    else
        log_warning "No A record found for $FULL_DOMAIN"
        log_info "Please configure DNS A record: $FULL_DOMAIN -> $SERVER_IP"
    fi
    
    # Check wildcard subdomain
    WILDCARD_RECORD=$(dig +short A "*.${DOMAIN}" 2>/dev/null || echo "")
    
    if [ -n "$WILDCARD_RECORD" ]; then
        log_info "Wildcard record found: *.${DOMAIN} -> $WILDCARD_RECORD"
    fi
    
    # Check specific service subdomains
    for service in api grafana prometheus questdb; do
        SERVICE_DOMAIN="${service}.${FULL_DOMAIN}"
        SERVICE_RECORD=$(dig +short A "$SERVICE_DOMAIN" 2>/dev/null || echo "")
        
        if [ -n "$SERVICE_RECORD" ]; then
            log_info "✓ ${service} subdomain configured: $SERVICE_DOMAIN -> $SERVICE_RECORD"
        else
            log_debug "No specific record for $SERVICE_DOMAIN (will use main domain)"
        fi
    done
    
    # Check reverse DNS
    PTR_RECORD=$(dig +short -x "$SERVER_IP" 2>/dev/null || echo "")
    if [ -n "$PTR_RECORD" ]; then
        log_info "Reverse DNS: $SERVER_IP -> $PTR_RECORD"
    fi
}

# Configure nginx reverse proxy
configure_nginx() {
    log_info "Configuring nginx reverse proxy for $FULL_DOMAIN..."
    
    # Create nginx configuration
    cat > /tmp/trading-system.conf <<EOF
# AI Trading System - Nginx Configuration
# Domain: $FULL_DOMAIN

upstream trading_api {
    server 127.0.0.1:8000;
    keepalive 32;
}

upstream grafana {
    server 127.0.0.1:3000;
}

upstream prometheus {
    server 127.0.0.1:9090;
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name $FULL_DOMAIN *.${FULL_DOMAIN};
    
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }
    
    location / {
        return 301 https://\$server_name\$request_uri;
    }
}

# Main HTTPS server
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name $FULL_DOMAIN;
    
    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/$FULL_DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$FULL_DOMAIN/privkey.pem;
    
    # SSL security settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384';
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_stapling on;
    ssl_stapling_verify on;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    # Main API
    location / {
        proxy_pass http://trading_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffering
        proxy_buffering off;
        proxy_request_buffering off;
    }
    
    # WebSocket support
    location /ws {
        proxy_pass http://trading_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket timeouts
        proxy_connect_timeout 7d;
        proxy_send_timeout 7d;
        proxy_read_timeout 7d;
    }
    
    # Grafana
    location /grafana/ {
        proxy_pass http://grafana/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # Prometheus
    location /prometheus/ {
        proxy_pass http://prometheus/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Basic auth for Prometheus (optional)
        # auth_basic "Prometheus";
        # auth_basic_user_file /etc/nginx/.htpasswd;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\\n";
        add_header Content-Type text/plain;
    }
    
    # Metrics endpoint (internal only)
    location /nginx_status {
        stub_status;
        allow 127.0.0.1;
        deny all;
    }
    
    # Deny access to hidden files
    location ~ /\\. {
        deny all;
        access_log off;
        log_not_found off;
    }
    
    # Logging
    access_log /var/log/nginx/${SUBDOMAIN}_access.log combined;
    error_log /var/log/nginx/${SUBDOMAIN}_error.log warn;
}
EOF
    
    log_info "✓ Nginx configuration created"
    
    # Check nginx syntax
    if command -v nginx &> /dev/null; then
        if nginx -t -c /tmp/trading-system.conf &>/dev/null; then
            log_info "✓ Nginx configuration syntax valid"
            
            # Install configuration
            sudo cp /tmp/trading-system.conf /etc/nginx/sites-available/trading-system.conf
            sudo ln -sf /etc/nginx/sites-available/trading-system.conf /etc/nginx/sites-enabled/
            
            log_info "✓ Nginx configuration installed"
            log_warning "Run 'sudo systemctl reload nginx' to apply changes"
        else
            log_error "Nginx configuration has syntax errors"
        fi
    else
        log_warning "Nginx not installed - configuration saved to /tmp/trading-system.conf"
    fi
}

# Configure Traefik labels
configure_traefik() {
    log_info "Configuring Traefik labels for Docker services..."
    
    # Create Traefik docker-compose override
    cat > docker-compose.traefik.yml <<EOF
# Traefik Configuration Override
# Domain: $FULL_DOMAIN

version: '3.8'

services:
  traefik:
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.api.rule=Host(\`$FULL_DOMAIN\`)"
      - "traefik.http.routers.api.entrypoints=websecure"
      - "traefik.http.routers.api.tls.certresolver=letsencrypt"
      - "traefik.http.routers.api.tls.domains[0].main=$FULL_DOMAIN"
      - "traefik.http.routers.api.tls.domains[0].sans=*.${FULL_DOMAIN}"
  
  api:
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.trading-api.rule=Host(\`$FULL_DOMAIN\`) || Host(\`api.${FULL_DOMAIN}\`)"
      - "traefik.http.routers.trading-api.entrypoints=websecure"
      - "traefik.http.routers.trading-api.tls.certresolver=letsencrypt"
      - "traefik.http.services.trading-api.loadbalancer.server.port=8000"
      - "traefik.http.routers.trading-api.middlewares=security-headers@file"
  
  grafana:
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.grafana.rule=Host(\`grafana.${FULL_DOMAIN}\`) || (Host(\`$FULL_DOMAIN\`) && PathPrefix(\`/grafana\`))"
      - "traefik.http.routers.grafana.entrypoints=websecure"
      - "traefik.http.routers.grafana.tls.certresolver=letsencrypt"
      - "traefik.http.services.grafana.loadbalancer.server.port=3000"
      - "traefik.http.routers.grafana.middlewares=security-headers@file"
  
  prometheus:
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.prometheus.rule=Host(\`prometheus.${FULL_DOMAIN}\`) || (Host(\`$FULL_DOMAIN\`) && PathPrefix(\`/prometheus\`))"
      - "traefik.http.routers.prometheus.entrypoints=websecure"
      - "traefik.http.routers.prometheus.tls.certresolver=letsencrypt"
      - "traefik.http.services.prometheus.loadbalancer.server.port=9090"
      - "traefik.http.routers.prometheus.middlewares=security-headers@file,auth@file"
  
  questdb:
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.questdb.rule=Host(\`questdb.${FULL_DOMAIN}\`) || (Host(\`$FULL_DOMAIN\`) && PathPrefix(\`/questdb\`))"
      - "traefik.http.routers.questdb.entrypoints=websecure"
      - "traefik.http.routers.questdb.tls.certresolver=letsencrypt"
      - "traefik.http.services.questdb.loadbalancer.server.port=9000"
      - "traefik.http.routers.questdb.middlewares=security-headers@file,auth@file"
EOF
    
    log_info "✓ Traefik configuration created: docker-compose.traefik.yml"
    
    # Update environment file with domain
    if [ -f ".env.production" ]; then
        if grep -q "^DOMAIN_NAME=" ".env.production"; then
            sed -i "s|^DOMAIN_NAME=.*|DOMAIN_NAME=$FULL_DOMAIN|" ".env.production"
        else
            echo "DOMAIN_NAME=$FULL_DOMAIN" >> ".env.production"
        fi
        
        if grep -q "^LETSENCRYPT_EMAIL=" ".env.production"; then
            log_info "Let's Encrypt email already configured"
        else
            read -p "Enter email for Let's Encrypt notifications: " LE_EMAIL
            echo "LETSENCRYPT_EMAIL=$LE_EMAIL" >> ".env.production"
        fi
        
        log_info "✓ Environment file updated with domain settings"
    fi
}

# Update /etc/hosts for local testing
update_hosts_file() {
    log_info "Updating /etc/hosts for local testing..."
    
    if grep -q "$FULL_DOMAIN" /etc/hosts; then
        log_info "Domain already in /etc/hosts"
    else
        echo "# AI Trading System" | sudo tee -a /etc/hosts
        echo "127.0.0.1 $FULL_DOMAIN" | sudo tee -a /etc/hosts
        echo "127.0.0.1 api.${FULL_DOMAIN}" | sudo tee -a /etc/hosts
        echo "127.0.0.1 grafana.${FULL_DOMAIN}" | sudo tee -a /etc/hosts
        echo "127.0.0.1 prometheus.${FULL_DOMAIN}" | sudo tee -a /etc/hosts
        echo "127.0.0.1 questdb.${FULL_DOMAIN}" | sudo tee -a /etc/hosts
        
        log_info "✓ /etc/hosts updated for local testing"
    fi
}

# Generate DNS instructions
generate_dns_instructions() {
    log_info ""
    log_info "========================================="
    log_info "DNS CONFIGURATION INSTRUCTIONS"
    log_info "========================================="
    log_info ""
    log_info "Add the following DNS records to your domain:"
    log_info ""
    log_info "A Records:"
    log_info "  $FULL_DOMAIN -> $SERVER_IP"
    log_info "  api.${FULL_DOMAIN} -> $SERVER_IP"
    log_info "  grafana.${FULL_DOMAIN} -> $SERVER_IP"
    log_info "  prometheus.${FULL_DOMAIN} -> $SERVER_IP"
    log_info "  questdb.${FULL_DOMAIN} -> $SERVER_IP"
    log_info ""
    log_info "Or use a wildcard:"
    log_info "  *.${FULL_DOMAIN} -> $SERVER_IP"
    log_info ""
    log_info "For Cloudflare:"
    log_info "1. Log into Cloudflare Dashboard"
    log_info "2. Select your domain: $DOMAIN"
    log_info "3. Go to DNS settings"
    log_info "4. Add A record:"
    log_info "   - Type: A"
    log_info "   - Name: $SUBDOMAIN"
    log_info "   - Content: $SERVER_IP"
    log_info "   - Proxy status: DNS only (for initial setup)"
    log_info ""
    log_info "After DNS propagation (5-30 minutes):"
    log_info "1. Test: dig $FULL_DOMAIN"
    log_info "2. Setup SSL: ./setup_ssl_certificates.sh --domain $FULL_DOMAIN"
    log_info "3. Enable Cloudflare proxy if desired"
}

# Main execution
main() {
    log_info "Starting domain configuration..."
    log_info "Domain: $FULL_DOMAIN"
    
    # Detect server IP
    detect_server_ip
    
    # Check DNS if requested
    if [ "$CHECK_DNS" = true ]; then
        check_dns_configuration
    fi
    
    # Setup nginx if requested
    if [ "$SETUP_NGINX" = true ]; then
        configure_nginx
    fi
    
    # Setup Traefik if requested
    if [ "$SETUP_TRAEFIK" = true ]; then
        configure_traefik
    fi
    
    # Update hosts file for local testing
    if [[ "$DOMAIN" == "localhost" || "$DOMAIN" == "127.0.0.1" ]]; then
        update_hosts_file
    fi
    
    # Generate DNS instructions
    generate_dns_instructions
    
    log_info ""
    log_info "========================================="
    log_info "DOMAIN CONFIGURATION COMPLETED"
    log_info "========================================="
    log_info ""
    log_info "Configuration Summary:"
    log_info "- Primary Domain: $FULL_DOMAIN"
    log_info "- Server IP: $SERVER_IP"
    log_info ""
    log_info "Next steps:"
    log_info "1. Configure DNS records as shown above"
    log_info "2. Wait for DNS propagation (5-30 minutes)"
    log_info "3. Run: ./setup_ssl_certificates.sh --domain $FULL_DOMAIN"
    log_info "4. Deploy application: ./deploy_production.sh"
}

# Run main function
main