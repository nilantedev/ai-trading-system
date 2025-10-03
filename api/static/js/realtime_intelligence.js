// Real-Time Symbol Analysis Engine
// PhD-level live trading intelligence for business dashboard

class SymbolAnalysisEngine {
    constructor() {
        this.ws = null;
        this.subscribers = new Map();
        this.activeSymbol = null;
        this.cache = new Map();
    }

    // Connect to real-time data stream
    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${protocol}//${window.location.host}/api/ws/market-data`);
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleUpdate(data);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket closed, reconnecting...');
            setTimeout(() => this.connect(), 3000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    // Subscribe to symbol updates
    subscribe(symbol, callback) {
        if (!this.subscribers.has(symbol)) {
            this.subscribers.set(symbol, new Set());
        }
        this.subscribers.get(symbol).add(callback);
        
        // Request subscription via WebSocket
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                action: 'subscribe',
                symbols: [symbol]
            }));
        }
    }

    unsubscribe(symbol, callback) {
        if (this.subscribers.has(symbol)) {
            this.subscribers.get(symbol).delete(callback);
            if (this.subscribers.get(symbol).size === 0) {
                this.subscribers.delete(symbol);
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({
                        action: 'unsubscribe',
                        symbols: [symbol]
                    }));
                }
            }
        }
    }

    handleUpdate(data) {
        const { symbol, type, payload } = data;
        
        // Update cache
        if (!this.cache.has(symbol)) {
            this.cache.set(symbol, {});
        }
        Object.assign(this.cache.get(symbol), payload);
        
        // Notify subscribers
        if (this.subscribers.has(symbol)) {
            this.subscribers.get(symbol).forEach(callback => {
                callback(symbol, type, payload);
            });
        }
    }

    async getFullAnalysis(symbol) {
        const response = await fetch(`/business/api/company/${symbol}/full-analysis`);
        if (!response.ok) throw new Error(`Analysis failed: ${response.status}`);
        return response.json();
    }
}

// Real-Time Options Flow Visualizer
class OptionsFlowViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.flows = [];
        this.maxFlows = 50;
    }

    addFlow(flow) {
        this.flows.unshift(flow);
        if (this.flows.length > this.maxFlows) {
            this.flows.pop();
        }
        this.render();
    }

    render() {
        if (!this.container) return;
        
        const html = `
            <table class="flow-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Type</th>
                        <th>Strike</th>
                        <th>Exp</th>
                        <th>Size</th>
                        <th>Premium</th>
                        <th>Sentiment</th>
                    </tr>
                </thead>
                <tbody>
                    ${this.flows.map(f => `
                        <tr class="${this.getRowClass(f)}">
                            <td>${new Date(f.timestamp).toLocaleTimeString()}</td>
                            <td><strong>${f.symbol}</strong></td>
                            <td class="${f.type === 'call' ? 'bullish' : 'bearish'}">${f.type.toUpperCase()}</td>
                            <td>$${f.strike}</td>
                            <td>${f.expiration}</td>
                            <td>${f.size.toLocaleString()}</td>
                            <td>$${(f.premium / 1000).toFixed(1)}k</td>
                            <td><span class="badge ${this.getSentimentClass(f.sentiment)}">${f.sentiment}</span></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
        
        this.container.innerHTML = html;
    }

    getRowClass(flow) {
        if (flow.premium > 1000000) return 'whale-flow';
        if (flow.premium > 100000) return 'large-flow';
        return '';
    }

    getSentimentClass(sentiment) {
        if (sentiment === 'bullish') return 'success';
        if (sentiment === 'bearish') return 'danger';
        return 'neutral';
    }
}

// Live Intelligence Insights Panel
class IntelligencePanel {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.insights = [];
        this.engine = new SymbolAnalysisEngine();
    }

    async analyzeSymbol(symbol) {
        const loading = this.showLoading();
        
        try {
            const analysis = await this.engine.getFullAnalysis(symbol);
            this.displayAnalysis(symbol, analysis);
        } catch (error) {
            this.showError(error.message);
        } finally {
            loading.remove();
        }
    }

    showLoading() {
        const loader = document.createElement('div');
        loader.className = 'intelligence-loading';
        loader.innerHTML = '<div class="spinner"></div><p>Running PhD-level analysis...</p>';
        this.container.appendChild(loader);
        return loader;
    }

    displayAnalysis(symbol, analysis) {
        const html = `
            <div class="intelligence-result">
                <h3>${symbol} - Comprehensive Analysis</h3>
                
                <div class="analysis-grid">
                    <div class="analysis-card">
                        <h4>ML Forecast</h4>
                        <div class="forecast-value ${analysis.forecast.direction}">
                            ${analysis.forecast.prediction > 0 ? '+' : ''}${(analysis.forecast.prediction * 100).toFixed(2)}%
                        </div>
                        <div class="confidence">Confidence: ${(analysis.forecast.confidence * 100).toFixed(1)}%</div>
                    </div>
                    
                    <div class="analysis-card">
                        <h4>Factor Exposures</h4>
                        <ul class="factor-list">
                            ${Object.entries(analysis.factors).map(([factor, value]) => `
                                <li>
                                    <span class="factor-name">${factor}</span>
                                    <span class="factor-value">${value.toFixed(3)}</span>
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                    
                    <div class="analysis-card">
                        <h4>Risk Metrics</h4>
                        <div class="risk-grid">
                            <div>VaR (95%): <strong>${analysis.risk.var_95.toFixed(2)}%</strong></div>
                            <div>Beta: <strong>${analysis.risk.beta.toFixed(2)}</strong></div>
                            <div>Volatility: <strong>${(analysis.risk.volatility * 100).toFixed(1)}%</strong></div>
                        </div>
                    </div>
                    
                    <div class="analysis-card">
                        <h4>Options Signal</h4>
                        <div class="options-signal ${analysis.options.signal}">
                            ${analysis.options.signal.toUpperCase()}
                        </div>
                        <div>IV Rank: ${(analysis.options.iv_rank * 100).toFixed(0)}%</div>
                        <div>Put/Call: ${analysis.options.put_call_ratio.toFixed(2)}</div>
                    </div>
                </div>
                
                <div class="news-sentiment">
                    <h4>Recent News Sentiment</h4>
                    <div class="sentiment-bar">
                        <div class="sentiment-fill ${this.getSentimentClass(analysis.news.sentiment)}" 
                             style="width: ${Math.abs(analysis.news.sentiment * 50) + 50}%"></div>
                    </div>
                    <p>${analysis.news.summary}</p>
                </div>
                
                <div class="recommendation">
                    <h4>Trading Recommendation</h4>
                    <div class="rec-badge ${analysis.recommendation.action}">
                        ${analysis.recommendation.action.toUpperCase()}
                    </div>
                    <p>${analysis.recommendation.reason}</p>
                    <div class="rec-details">
                        <span>Entry: $${analysis.recommendation.entry}</span>
                        <span>Target: $${analysis.recommendation.target}</span>
                        <span>Stop: $${analysis.recommendation.stop}</span>
                    </div>
                </div>
            </div>
        `;
        
        this.container.innerHTML = html;
    }

    getSentimentClass(score) {
        if (score > 0.3) return 'positive';
        if (score < -0.3) return 'negative';
        return 'neutral';
    }

    showError(message) {
        this.container.innerHTML = `<div class="error-message">Analysis failed: ${message}</div>`;
    }
}

// Live Market Heatmap
class MarketHeatmap {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.data = [];
    }

    async load() {
        try {
            const response = await fetch('/business/api/market/heatmap');
            const data = await response.json();
            this.data = data.sectors;
            this.render();
        } catch (error) {
            console.error('Heatmap load error:', error);
        }
    }

    render() {
        if (!this.container || !this.data.length) return;
        
        const html = `
            <div class="heatmap-grid">
                ${this.data.map(sector => `
                    <div class="heatmap-sector">
                        <h4>${sector.name}</h4>
                        <div class="sector-stocks">
                            ${sector.stocks.map(stock => `
                                <div class="stock-tile ${this.getPerfClass(stock.change)}" 
                                     style="flex: ${stock.marketCap / 1e9}">
                                    <div class="stock-symbol">${stock.symbol}</div>
                                    <div class="stock-change">${stock.change > 0 ? '+' : ''}${stock.change.toFixed(2)}%</div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        
        this.container.innerHTML = html;
    }

    getPerfClass(change) {
        if (change > 5) return 'gain-strong';
        if (change > 2) return 'gain-moderate';
        if (change > 0) return 'gain-slight';
        if (change > -2) return 'loss-slight';
        if (change > -5) return 'loss-moderate';
        return 'loss-strong';
    }
}

// Initialize real-time components
window.addEventListener('DOMContentLoaded', () => {
    // Initialize analysis engine
    window.symbolEngine = new SymbolAnalysisEngine();
    window.symbolEngine.connect();
    
    // Initialize options flow viewer
    if (document.getElementById('options-flow')) {
        window.optionsFlow = new OptionsFlowViewer('options-flow');
        
        // Subscribe to options flow updates
        const eventSource = new EventSource('/business/api/options/flow/stream');
        eventSource.onmessage = (event) => {
            const flow = JSON.parse(event.data);
            window.optionsFlow.addFlow(flow);
        };
    }
    
    // Initialize intelligence panel
    if (document.getElementById('intelligence-panel')) {
        window.intelligencePanel = new IntelligencePanel('intelligence-panel');
        
        // Bind symbol search
        const searchBtn = document.getElementById('intel-run');
        const searchInput = document.getElementById('intel-symbol');
        if (searchBtn && searchInput) {
            searchBtn.addEventListener('click', () => {
                const symbol = searchInput.value.trim().toUpperCase();
                if (symbol) {
                    window.intelligencePanel.analyzeSymbol(symbol);
                }
            });
            
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    searchBtn.click();
                }
            });
        }
    }
    
    // Initialize market heatmap
    if (document.getElementById('market-heatmap')) {
        window.marketHeatmap = new MarketHeatmap('market-heatmap');
        window.marketHeatmap.load();
        setInterval(() => window.marketHeatmap.load(), 30000);
    }
});
