// Admin God-Mode Control Panel
// Comprehensive system management and intelligence tweaking

class SystemControlPanel {
    constructor() {
        this.services = [];
        this.metrics = {};
        this.init();
    }

    init() {
        this.loadServices();
        this.setupRealtimeMonitoring();
        this.bindControls();
        this.startHealthChecks();
    }

    async loadServices() {
        try {
            const response = await fetch('/admin/api/services/status');
            const data = await response.json();
            this.services = data.services;
            this.renderServices();
        } catch (error) {
            console.error('Failed to load services:', error);
        }
    }

    renderServices() {
        const container = document.getElementById('services-grid');
        if (!container) return;
        
        const html = `
            <div class="services-control-grid">
                ${this.services.map(service => `
                    <div class="service-card ${service.status}">
                        <div class="service-header">
                            <h4>${service.name}</h4>
                            <span class="status-badge ${service.status}">${service.status}</span>
                        </div>
                        <div class="service-metrics">
                            <div class="metric">
                                <span class="label">CPU:</span>
                                <span class="value">${service.cpu}%</span>
                            </div>
                            <div class="metric">
                                <span class="label">Memory:</span>
                                <span class="value">${service.memory}MB</span>
                            </div>
                            <div class="metric">
                                <span class="label">Requests:</span>
                                <span class="value">${service.requests}/s</span>
                            </div>
                        </div>
                        <div class="service-controls">
                            <button class="btn-sm" onclick="systemControl.restartService('${service.name}')">Restart</button>
                            <button class="btn-sm" onclick="systemControl.viewLogs('${service.name}')">Logs</button>
                            <button class="btn-sm" onclick="systemControl.scaleService('${service.name}')">Scale</button>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        
        container.innerHTML = html;
    }

    async restartService(serviceName) {
        if (!confirm(`Restart ${serviceName}?`)) return;
        
        try {
            await fetch(`/admin/api/services/${serviceName}/restart`, {
                method: 'POST',
                headers: this.getHeaders()
            });
            this.showNotification(`${serviceName} restart initiated`, 'success');
            setTimeout(() => this.loadServices(), 3000);
        } catch (error) {
            this.showNotification(`Restart failed: ${error.message}`, 'error');
        }
    }

    viewLogs(serviceName) {
        window.open(`/admin/logs/${serviceName}`, '_blank');
    }

    async scaleService(serviceName) {
        const replicas = prompt(`Number of replicas for ${serviceName}:`, '1');
        if (!replicas) return;
        
        try {
            await fetch(`/admin/api/services/${serviceName}/scale`, {
                method: 'POST',
                headers: this.getHeaders(),
                body: JSON.stringify({ replicas: parseInt(replicas) })
            });
            this.showNotification(`${serviceName} scaling to ${replicas} replicas`, 'success');
        } catch (error) {
            this.showNotification(`Scaling failed: ${error.message}`, 'error');
        }
    }

    setupRealtimeMonitoring() {
        const es = new EventSource('/admin/api/metrics/stream');
        es.addEventListener('metrics', (event) => {
            const metrics = JSON.parse(event.data);
            this.updateMetrics(metrics);
        });
    }

    updateMetrics(metrics) {
        this.metrics = metrics;
        
        // Update system-wide metrics
        this.updateElement('system-cpu', `${metrics.system.cpu}%`);
        this.updateElement('system-memory', `${metrics.system.memory}GB`);
        this.updateElement('system-disk', `${metrics.system.disk}%`);
        this.updateElement('system-network', `${metrics.system.network_mbps}Mbps`);
        
        // Update throughput
        this.updateElement('requests-per-sec', metrics.throughput.requests);
        this.updateElement('data-ingested', `${(metrics.throughput.data_mb / 1024).toFixed(2)}GB/hr`);
        this.updateElement('signals-generated', metrics.throughput.signals);
    }

    updateElement(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    }

    bindControls() {
        // ML Model Controls
        this.bindButton('promote-all-shadows', () => this.promoteAllShadows());
        this.bindButton('force-model-reload', () => this.forceModelReload());
        this.bindButton('adjust-risk-limits', () => this.showRiskAdjuster());
        
        // Backfill Controls
        this.bindButton('trigger-equity-backfill', () => this.triggerBackfill('equity'));
        this.bindButton('trigger-news-backfill', () => this.triggerBackfill('news'));
        this.bindButton('trigger-options-backfill', () => this.triggerBackfill('options'));
        this.bindButton('trigger-social-backfill', () => this.triggerBackfill('social'));
        
        // System Controls
        this.bindButton('kill-switch', () => this.activateKillSwitch());
        this.bindButton('reset-circuit-breakers', () => this.resetCircuitBreakers());
        this.bindButton('force-gc', () => this.forceGarbageCollection());
    }

    bindButton(id, handler) {
        const btn = document.getElementById(id);
        if (btn) btn.addEventListener('click', handler);
    }

    async promoteAllShadows() {
        if (!confirm('Promote ALL shadow models to production?')) return;
        
        try {
            const response = await fetch('/admin/api/models/promote-all-shadows', {
                method: 'POST',
                headers: this.getHeaders()
            });
            const result = await response.json();
            this.showNotification(`Promoted ${result.promoted} models`, 'success');
        } catch (error) {
            this.showNotification(`Promotion failed: ${error.message}`, 'error');
        }
    }

    async forceModelReload() {
        if (!confirm('Force reload all ML models?')) return;
        
        try {
            await fetch('/admin/api/models/force-reload', {
                method: 'POST',
                headers: this.getHeaders()
            });
            this.showNotification('Model reload initiated', 'success');
        } catch (error) {
            this.showNotification(`Reload failed: ${error.message}`, 'error');
        }
    }

    showRiskAdjuster() {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <h2>Adjust Risk Limits</h2>
                <form id="risk-form">
                    <label>Max Position Size ($): <input type="number" name="max_position" value="100000"></label>
                    <label>Max Daily Loss ($): <input type="number" name="max_daily_loss" value="10000"></label>
                    <label>Max Leverage: <input type="number" step="0.1" name="max_leverage" value="2.0"></label>
                    <label>VaR Limit (95%): <input type="number" step="0.01" name="var_limit" value="0.05"></label>
                    <div class="modal-actions">
                        <button type="submit" class="button">Apply</button>
                        <button type="button" class="button secondary" onclick="this.closest('.modal-overlay').remove()">Cancel</button>
                    </div>
                </form>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        modal.querySelector('form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const limits = Object.fromEntries(formData);
            
            try {
                await fetch('/admin/api/risk/update-limits', {
                    method: 'POST',
                    headers: this.getHeaders(),
                    body: JSON.stringify(limits)
                });
                this.showNotification('Risk limits updated', 'success');
                modal.remove();
            } catch (error) {
                this.showNotification(`Update failed: ${error.message}`, 'error');
            }
        });
    }

    async triggerBackfill(dataType) {
        const years = prompt(`Backfill ${dataType} data for how many years?`, dataType === 'equity' ? '20' : '5');
        if (!years) return;
        
        const symbols = prompt('Number of symbols (blank = all from watchlist):', '2000');
        
        try {
            await fetch(`/admin/api/backfill/${dataType}`, {
                method: 'POST',
                headers: this.getHeaders(),
                body: JSON.stringify({
                    years: parseInt(years),
                    max_symbols: symbols ? parseInt(symbols) : null
                })
            });
            this.showNotification(`${dataType} backfill started (${years}yr)`, 'success');
        } catch (error) {
            this.showNotification(`Backfill failed: ${error.message}`, 'error');
        }
    }

    async activateKillSwitch() {
        const confirmation = prompt('Type "KILL SWITCH" to halt all trading:');
        if (confirmation !== 'KILL SWITCH') return;
        
        try {
            await fetch('/admin/api/emergency/kill-switch', {
                method: 'POST',
                headers: this.getHeaders()
            });
            this.showNotification('🚨 KILL SWITCH ACTIVATED - All trading halted', 'error');
        } catch (error) {
            this.showNotification(`Kill switch failed: ${error.message}`, 'error');
        }
    }

    async resetCircuitBreakers() {
        try {
            await fetch('/admin/api/circuit-breakers/reset', {
                method: 'POST',
                headers: this.getHeaders()
            });
            this.showNotification('Circuit breakers reset', 'success');
        } catch (error) {
            this.showNotification(`Reset failed: ${error.message}`, 'error');
        }
    }

    async forceGarbageCollection() {
        try {
            await fetch('/admin/api/system/force-gc', {
                method: 'POST',
                headers: this.getHeaders()
            });
            this.showNotification('Garbage collection triggered', 'success');
        } catch (error) {
            this.showNotification(`GC failed: ${error.message}`, 'error');
        }
    }

    startHealthChecks() {
        setInterval(() => {
            this.checkSystemHealth();
        }, 5000);
    }

    async checkSystemHealth() {
        try {
            const response = await fetch('/health/full');
            const health = await response.json();
            
            // Update health indicators
            document.querySelectorAll('[data-health-component]').forEach(el => {
                const component = el.dataset.healthComponent;
                const status = health.components[component]?.status || 'unknown';
                el.className = `health-indicator ${status}`;
            });
        } catch (error) {
            console.error('Health check failed:', error);
        }
    }

    getHeaders() {
        const csrf = document.cookie.split(';').find(c => c.trim().startsWith('csrf_token='));
        return {
            'Content-Type': 'application/json',
            'X-CSRF-Token': csrf ? decodeURIComponent(csrf.split('=')[1]) : ''
        };
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('fade-out');
            setTimeout(() => notification.remove(), 500);
        }, 3000);
    }
}

// Intelligence Configuration Panel
class IntelligenceConfigurator {
    constructor() {
        this.models = [];
        this.factors = [];
        this.init();
    }

    async init() {
        await this.loadModels();
        await this.loadFactors();
        this.renderConfig();
    }

    async loadModels() {
        const response = await fetch('/admin/api/ml/models');
        const data = await response.json();
        this.models = data.models;
    }

    async loadFactors() {
        const response = await fetch('/admin/api/factors/list');
        const data = await response.json();
        this.factors = data.factors;
    }

    renderConfig() {
        const container = document.getElementById('intelligence-config');
        if (!container) return;
        
        const html = `
            <div class="config-panel">
                <h3>Model Weights & Ensembles</h3>
                <div class="model-weights">
                    ${this.models.map(model => `
                        <div class="weight-control">
                            <label>${model.name}</label>
                            <input type="range" min="0" max="1" step="0.01" value="${model.weight}"
                                   onchange="intelligenceConfig.updateModelWeight('${model.id}', this.value)">
                            <span class="weight-value">${(model.weight * 100).toFixed(0)}%</span>
                        </div>
                    `).join('')}
                </div>
                
                <h3>Factor Exposures</h3>
                <div class="factor-controls">
                    ${this.factors.map(factor => `
                        <div class="factor-control">
                            <label>${factor.name}</label>
                            <button class="btn-sm ${factor.enabled ? 'active' : ''}"
                                    onclick="intelligenceConfig.toggleFactor('${factor.id}')">
                                ${factor.enabled ? 'Enabled' : 'Disabled'}
                            </button>
                            <input type="number" step="0.1" value="${factor.multiplier}"
                                   onchange="intelligenceConfig.updateFactorMultiplier('${factor.id}', this.value)">
                        </div>
                    `).join('')}
                </div>
                
                <button class="button primary" onclick="intelligenceConfig.saveConfig()">
                    Save Configuration
                </button>
            </div>
        `;
        
        container.innerHTML = html;
    }

    async updateModelWeight(modelId, weight) {
        await fetch(`/admin/api/ml/models/${modelId}/weight`, {
            method: 'PUT',
            headers: window.systemControl.getHeaders(),
            body: JSON.stringify({ weight: parseFloat(weight) })
        });
    }

    async toggleFactor(factorId) {
        const factor = this.factors.find(f => f.id === factorId);
        factor.enabled = !factor.enabled;
        
        await fetch(`/admin/api/factors/${factorId}/toggle`, {
            method: 'POST',
            headers: window.systemControl.getHeaders()
        });
        
        this.renderConfig();
    }

    async updateFactorMultiplier(factorId, multiplier) {
        await fetch(`/admin/api/factors/${factorId}/multiplier`, {
            method: 'PUT',
            headers: window.systemControl.getHeaders(),
            body: JSON.stringify({ multiplier: parseFloat(multiplier) })
        });
    }

    async saveConfig() {
        await fetch('/admin/api/intelligence/save-config', {
            method: 'POST',
            headers: window.systemControl.getHeaders(),
            body: JSON.stringify({
                models: this.models,
                factors: this.factors
            })
        });
        
        window.systemControl.showNotification('Intelligence configuration saved', 'success');
    }
}

// Initialize God-Mode panels
window.addEventListener('DOMContentLoaded', () => {
    window.systemControl = new SystemControlPanel();
    window.intelligenceConfig = new IntelligenceConfigurator();
});
