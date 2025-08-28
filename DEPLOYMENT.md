# AI Trading System - Complete Deployment Guide

## Server Requirements
- **OS**: Ubuntu 24.04 LTS
- **RAM**: Minimum 32GB (256GB+ recommended for AI models)
- **Storage**: 100GB+ SSD
- **Python**: 3.11+ (3.13 compatible)
- **Ports**: 8000 (API), 5432 (PostgreSQL), 6379 (Redis)

## Step-by-Step Deployment Instructions

### Step 1: Connect to Your Server
```bash
# From your local machine
ssh nilante@168.119.145.135
```

### Step 2: Install System Dependencies
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+ and essential tools
sudo apt install -y python3.11 python3.11-venv python3-pip
sudo apt install -y git docker.io docker-compose
sudo apt install -y postgresql-client redis-tools
sudo apt install -y build-essential libpq-dev

# Add your user to docker group
sudo usermod -aG docker $USER
# LOGOUT AND LOGIN AGAIN for group changes to take effect
```

### Step 3: Clone the Repository
```bash
# Create deployment directory
cd /srv
sudo mkdir trading
sudo chown $USER:$USER trading
cd trading

# Clone repository (replace with your actual repo URL)
git clone https://github.com/YOUR_USERNAME/ai-trading-system.git
cd ai-trading-system
```

### Step 4: Set Up Python Environment
```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Step 5: Configure Environment Variables
```bash
# Copy environment template
cp .env.example .env

# Generate secure passwords
python3 -c "import secrets; print('JWT_SECRET=' + secrets.token_urlsafe(32))" >> .env
python3 -c "import secrets; print('DB_PASSWORD=' + secrets.token_urlsafe(32))" >> .env
python3 -c "import secrets; print('REDIS_PASSWORD=' + secrets.token_urlsafe(32))" >> .env

# Edit configuration
nano .env

# Add your API keys (get from providers):
# ALPACA_API_KEY=your_key_here
# ALPACA_SECRET_KEY=your_secret_here
# POLYGON_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here (optional)
```

### Step 6: Start Docker Services
```bash
# Start infrastructure services
docker-compose up -d postgres redis prometheus grafana

# Wait for services to be healthy (about 30 seconds)
sleep 30

# Check service status
docker-compose ps
```

### Step 7: Initialize Database
```bash
# Set Python path
export PYTHONPATH=/srv/trading/ai-trading-system:/srv/trading/ai-trading-system/shared/python-common:/srv/trading/ai-trading-system/services

# Run database migrations
alembic upgrade head

# Create admin user
python scripts/create_user_tables.py
# Note: Save the generated admin password!
```

### Step 8: Start the Application
```bash
# Method 1: Direct start (for testing)
source .venv/bin/activate
export PYTHONPATH=/srv/trading/ai-trading-system:/srv/trading/ai-trading-system/shared/python-common:/srv/trading/ai-trading-system/services
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Method 2: Production start with systemd (recommended)
sudo nano /etc/systemd/system/trading-api.service
```

Add this content to the service file:
```ini
[Unit]
Description=AI Trading System API
After=network.target

[Service]
Type=simple
User=nilante
WorkingDirectory=/srv/trading/ai-trading-system
Environment="PYTHONPATH=/srv/trading/ai-trading-system:/srv/trading/ai-trading-system/shared/python-common:/srv/trading/ai-trading-system/services"
ExecStart=/srv/trading/ai-trading-system/.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

Then enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-api
sudo systemctl start trading-api
sudo systemctl status trading-api
```

### Step 9: Set Up Nginx (Optional but Recommended)
```bash
# Install nginx
sudo apt install -y nginx

# Create configuration
sudo nano /etc/nginx/sites-available/trading-api
```

Add this configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain or IP

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/trading-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Step 10: Verify Deployment
```bash
# Check API health
curl http://localhost:8000/health

# Check system status
python health_check.py

# View logs
sudo journalctl -u trading-api -f

# Check Docker containers
docker-compose ps
```

### Step 11: Configure Firewall
```bash
# Allow necessary ports
sudo ufw allow 22     # SSH
sudo ufw allow 80     # HTTP
sudo ufw allow 443    # HTTPS
sudo ufw allow 8000   # API (if not using nginx)
sudo ufw enable
```

### Step 12: Set Up SSL (For Production)
```bash
# Install certbot
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

## Post-Deployment Tasks

### 1. Install AI Models (Optional)
```bash
# Install Ollama for local LLMs
curl -fsSL https://ollama.ai/install.sh | sh

# Download models (requires significant RAM)
ollama pull llama3.3:70b  # 48GB RAM required
ollama pull qwen2.5:72b    # 48GB RAM required
```

### 2. Set Up Monitoring
- Access Grafana: `http://your-server:3000` (admin/admin)
- Access Prometheus: `http://your-server:9090`
- Import dashboards from `infrastructure/grafana/dashboards/`

### 3. Configure Backup
```bash
# Set up automated backups
crontab -e

# Add daily backup at 2 AM
0 2 * * * /srv/trading/ai-trading-system/scripts/backup.sh
```

### 4. Start Trading (Paper Trading First!)
```bash
# Enable paper trading in .env
PAPER_TRADING=true

# Restart service
sudo systemctl restart trading-api

# Monitor performance
tail -f logs/trading.log
```

## Troubleshooting

### If services won't start:
```bash
# Check logs
docker-compose logs -f
sudo journalctl -u trading-api -n 100

# Check Python path
echo $PYTHONPATH

# Test imports
python -c "from api.main import app; print('Success')"
```

### If database connection fails:
```bash
# Check PostgreSQL
docker-compose exec postgres psql -U trading_user -d trading_system

# Check Redis
redis-cli -h localhost ping
```

### If API throws errors:
```bash
# Check dependencies
pip list | grep -E "fastapi|redis|sqlalchemy"

# Reinstall requirements
pip install --force-reinstall -r requirements.txt
```

## Security Checklist
- [ ] Changed all default passwords
- [ ] Configured firewall rules
- [ ] Enabled SSL certificate
- [ ] Set up regular backups
- [ ] Configured log rotation
- [ ] Limited API rate limiting
- [ ] Enabled audit logging

## Maintenance Commands
```bash
# Update code
git pull origin main
pip install -r requirements.txt
alembic upgrade head
sudo systemctl restart trading-api

# View logs
tail -f logs/trading.log
sudo journalctl -u trading-api -f

# Backup database
docker-compose exec postgres pg_dump -U trading_user trading_system > backup.sql

# Clean up old logs
find logs/ -name "*.log" -mtime +30 -delete
```

## Support
For issues, check:
1. Application logs: `/srv/trading/ai-trading-system/logs/`
2. System logs: `sudo journalctl -u trading-api`
3. Docker logs: `docker-compose logs`

## Ready to Trade!
Your AI Trading System is now deployed and ready. Start with paper trading to validate the system before enabling live trading.