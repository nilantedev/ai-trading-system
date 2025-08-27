#!/bin/bash
# AI Trading System - Ollama Setup Script
# Installs and configures Ollama for local AI model serving

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/tmp/ollama-setup.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

info() {
    echo -e "${BLUE}INFO: $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}SUCCESS: $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

# Check system requirements
check_requirements() {
    info "Checking system requirements..."
    
    # Check available memory
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_MEM" -lt 160 ]; then
        error "Insufficient memory. Need at least 160GB for AI models, found ${TOTAL_MEM}GB"
    fi
    success "Memory check passed: ${TOTAL_MEM}GB available"
    
    # Check disk space
    AVAILABLE_SPACE=$(df -BG /mnt/fastdrive 2>/dev/null | awk 'NR==2{print $4}' | sed 's/G//' || echo "0")
    if [ "$AVAILABLE_SPACE" -lt 200 ]; then
        warn "Low disk space on /mnt/fastdrive: ${AVAILABLE_SPACE}GB available"
        warn "Recommend at least 200GB for model storage"
    fi
    
    # Check if running as correct user
    if [ "$EUID" -eq 0 ]; then
        error "Don't run this script as root"
    fi
    
    success "System requirements check completed"
}

# Install Ollama
install_ollama() {
    info "Installing Ollama..."
    
    if command -v ollama >/dev/null 2>&1; then
        info "Ollama already installed: $(ollama --version)"
        return 0
    fi
    
    # Download and install Ollama
    curl -fsSL https://ollama.ai/install.sh | sh
    
    # Verify installation
    if command -v ollama >/dev/null 2>&1; then
        success "Ollama installed successfully: $(ollama --version)"
    else
        error "Ollama installation failed"
    fi
}

# Configure Ollama
configure_ollama() {
    info "Configuring Ollama..."
    
    # Create configuration directory
    mkdir -p ~/.ollama
    
    # Set environment variables for resource management
    cat > ~/.ollama/config << EOF
# Ollama Configuration for AI Trading System
export OLLAMA_HOST=0.0.0.0
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_FLASH_ATTENTION=true
export OLLAMA_KV_CACHE_TYPE=f16

# Memory and performance settings
export OLLAMA_KEEP_ALIVE=24h
export OLLAMA_LOAD_TIMEOUT=10m
export OLLAMA_REQUEST_TIMEOUT=30m

# Model storage paths
export OLLAMA_MODELS=/mnt/fastdrive/trading/models/ollama
EOF
    
    # Source the config
    source ~/.ollama/config
    
    # Create model storage directory
    mkdir -p /mnt/fastdrive/trading/models/ollama
    
    success "Ollama configuration completed"
}

# Start Ollama service
start_ollama_service() {
    info "Starting Ollama service..."
    
    # Source configuration
    source ~/.ollama/config
    
    # Start Ollama in background
    ollama serve > /tmp/ollama-serve.log 2>&1 &
    OLLAMA_PID=$!
    
    # Wait for service to be ready
    sleep 5
    
    # Check if service is running
    if kill -0 $OLLAMA_PID 2>/dev/null; then
        success "Ollama service started successfully (PID: $OLLAMA_PID)"
        echo $OLLAMA_PID > ~/.ollama/ollama.pid
    else
        error "Failed to start Ollama service"
    fi
}

# Download development models (smaller versions for testing)
download_dev_models() {
    info "Downloading development models..."
    
    # Download smaller models for development/testing
    warn "These are development models. For production, use download_production_models()"
    
    info "Downloading Llama 3.2 3B for testing..."
    ollama pull llama3.2:3b
    
    info "Downloading Qwen2.5 7B for testing..."  
    ollama pull qwen2.5:7b
    
    info "Downloading Phi-3 for lightweight testing..."
    ollama pull phi3:medium
    
    success "Development models downloaded"
}

# Download production models (full-size, high-performance)
download_production_models() {
    info "Downloading production models..."
    warn "This requires significant bandwidth and storage (200GB+)"
    
    # Check available space
    AVAILABLE_SPACE=$(df -BG /mnt/fastdrive 2>/dev/null | awk 'NR==2{print $4}' | sed 's/G//' || echo "0")
    if [ "$AVAILABLE_SPACE" -lt 200 ]; then
        error "Insufficient disk space. Need at least 200GB, found ${AVAILABLE_SPACE}GB"
    fi
    
    # Llama 3.3 70B - Latest and greatest from Meta
    info "Downloading Llama 3.3 70B (40GB)..."
    ollama pull llama3.3:70b-instruct-q4_K_M
    
    # Qwen 2.5 72B - Excellent for financial analysis
    info "Downloading Qwen 2.5 72B (40GB)..."
    ollama pull qwen2.5:72b-instruct-q4_K_M
    
    # DeepSeek V3 - State-of-the-art reasoning (if available)
    info "Checking for DeepSeek V3..."
    if ollama list | grep -q "deepseek-v3"; then
        info "DeepSeek V3 available, downloading..."
        ollama pull deepseek-v3:latest
    else
        warn "DeepSeek V3 not yet available in Ollama, check later"
    fi
    
    # Mixtral 8x22B - MoE for efficiency
    info "Downloading Mixtral 8x22B MoE (80GB)..."
    ollama pull mixtral:8x22b-instruct-v0.1-q4_K_M
    
    # Command-R - Good for retrieval
    info "Downloading Command-R 35B (20GB)..."
    ollama pull command-r:35b-v0.1-q4_K_M
    
    # Solar 10.7B - Efficient and capable
    info "Downloading Solar 10.7B (6GB)..."
    ollama pull solar:10.7b-instruct-v1-q5_K_M
    
    success "Production models downloaded"
    info "Total models available:"
    ollama list
}

# Create systemd service (optional)
create_systemd_service() {
    info "Creating systemd service for Ollama..."
    
    # Check if we can create systemd service
    if [ ! -d /etc/systemd/system ]; then
        warn "Systemd not available, skipping service creation"
        return 0
    fi
    
    # This would require sudo, so just create the service file content
    cat > /tmp/ollama.service << EOF
[Unit]
Description=Ollama AI Model Server
After=network.target

[Service]
Type=simple
User=$USER
EnvironmentFile=$HOME/.ollama/config
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    info "Systemd service file created at /tmp/ollama.service"
    info "To install: sudo cp /tmp/ollama.service /etc/systemd/system/"
    info "Then: sudo systemctl daemon-reload && sudo systemctl enable ollama"
}

# Test installation
test_installation() {
    info "Testing Ollama installation..."
    
    # Test basic functionality
    if ollama list >/dev/null 2>&1; then
        success "Ollama is responding to commands"
    else
        error "Ollama is not responding"
    fi
    
    # Test model loading (if models are available)
    AVAILABLE_MODELS=$(ollama list | grep -v "NAME" | wc -l)
    info "Available models: $AVAILABLE_MODELS"
    
    success "Installation test completed"
}

# Main execution
main() {
    info "Starting Ollama setup for AI Trading System"
    info "Log file: $LOG_FILE"
    
    # Parse arguments
    PRODUCTION_MODE=false
    if [ "$1" == "--production" ]; then
        PRODUCTION_MODE=true
        info "Running in PRODUCTION mode"
    fi
    
    check_requirements
    install_ollama
    configure_ollama
    start_ollama_service
    
    # Download appropriate models
    if [ "$PRODUCTION_MODE" == true ]; then
        download_production_models
    else
        download_dev_models
    fi
    
    create_systemd_service
    test_installation
    
    success "Ollama setup completed successfully!"
    info "Next steps:"
    
    if [ "$PRODUCTION_MODE" == true ]; then
        info "1. Production models are ready for use"
        info "2. Configure vLLM or TGI for optimized inference"
        info "3. Set up model routing in production-models.yaml"
        info "4. Monitor GPU memory and performance metrics"
    else
        info "1. Development models are ready for testing"
        info "2. For production, run: $0 --production"
        info "3. Configure model-server service to use Ollama endpoints"
        info "4. Monitor resource usage during model loading"
    fi
}

# Execute main function
main "$@"