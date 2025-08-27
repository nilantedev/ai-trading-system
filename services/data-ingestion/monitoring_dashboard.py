#!/usr/bin/env python3
"""
Data Ingestion Monitoring Dashboard
Provides real-time monitoring of data frequency, latency, and system health.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from collections import deque, defaultdict

import aioredis
from prometheus_client import Counter, Gauge, Histogram, Summary, Info
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
import uvicorn

logger = logging.getLogger(__name__)

# Comprehensive Metrics
data_ingestion_rate = Gauge('data_ingestion_rate_per_second', 'Data ingestion rate', ['source', 'data_type'])
data_latency_summary = Summary('data_processing_latency_seconds', 'Data processing latency', ['pipeline_stage'])
api_health_gauge = Gauge('api_health_status', 'API health status (1=healthy, 0=unhealthy)', ['provider'])
strategy_performance_gauge = Gauge('strategy_performance_score', 'Strategy performance score', ['strategy'])
system_resource_gauge = Gauge('system_resource_usage_percent', 'System resource usage', ['resource_type'])
data_quality_score = Gauge('data_quality_score', 'Data quality score (0-100)', ['source'])
alert_counter = Counter('monitoring_alerts_total', 'Total monitoring alerts', ['severity', 'category'])

# System info
system_info = Info('data_ingestion_system', 'Data ingestion system information')


class MonitoringDashboard:
    """Real-time monitoring dashboard for data ingestion system."""
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        """Initialize monitoring dashboard."""
        self.redis = redis_client
        self.app = FastAPI(title="Data Ingestion Monitor")
        
        # Performance tracking
        self.latency_buffer = defaultdict(lambda: deque(maxlen=100))
        self.ingestion_rates = defaultdict(lambda: deque(maxlen=60))
        self.error_buffer = deque(maxlen=100)
        self.alert_buffer = deque(maxlen=50)
        
        # Health tracking
        self.api_health = {}
        self.last_data_times = {}
        
        # Thresholds
        self.thresholds = {
            'max_latency_ms': 1000,
            'min_ingestion_rate': 0.1,
            'max_error_rate': 0.05,
            'stale_data_seconds': 300
        }
        
        self._setup_routes()
        self._initialize_metrics()
        
        logger.info("Monitoring Dashboard initialized")
    
    def _initialize_metrics(self):
        """Initialize system metrics."""
        system_info.info({
            'version': os.getenv('SERVICE_VERSION', '1.0.0'),
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'multi_strategy_enabled': os.getenv('FEATURE_MULTI_STRATEGY_SUPPORT', 'true'),
            'realtime_enabled': os.getenv('FEATURE_REALTIME_DATA_ENABLED', 'true')
        })
    
    def _setup_routes(self):
        """Setup FastAPI routes for monitoring."""
        
        @self.app.get("/metrics")
        async def prometheus_metrics():
            """Expose Prometheus metrics."""
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
        
        @self.app.get("/health")
        async def health_check():
            """System health check endpoint."""
            health_status = await self.get_system_health()
            status_code = 200 if health_status['healthy'] else 503
            return Response(content=str(health_status), status_code=status_code)
        
        @self.app.get("/dashboard", response_class=HTMLResponse)
        async def dashboard():
            """HTML monitoring dashboard."""
            return await self.generate_dashboard_html()
        
        @self.app.get("/api/stats")
        async def api_stats():
            """Get current system statistics."""
            return await self.get_current_stats()
        
        @self.app.get("/api/alerts")
        async def recent_alerts():
            """Get recent system alerts."""
            return list(self.alert_buffer)
    
    async def track_data_ingestion(
        self,
        source: str,
        data_type: str,
        count: int = 1,
        latency_ms: float = 0
    ):
        """Track data ingestion metrics."""
        # Update ingestion rate
        rate_key = f"{source}:{data_type}"
        self.ingestion_rates[rate_key].append({
            'timestamp': datetime.utcnow(),
            'count': count
        })
        
        # Calculate rate per second
        if len(self.ingestion_rates[rate_key]) > 1:
            recent = self.ingestion_rates[rate_key][-10:]
            time_span = (recent[-1]['timestamp'] - recent[0]['timestamp']).total_seconds()
            if time_span > 0:
                rate = sum(r['count'] for r in recent) / time_span
                data_ingestion_rate.labels(source=source, data_type=data_type).set(rate)
        
        # Track latency
        if latency_ms > 0:
            self.latency_buffer[rate_key].append(latency_ms)
            data_latency_summary.labels(pipeline_stage=f"{source}_{data_type}").observe(latency_ms / 1000)
            
            # Check for high latency
            if latency_ms > self.thresholds['max_latency_ms']:
                await self.create_alert(
                    'warning',
                    'latency',
                    f"High latency detected for {source}:{data_type}: {latency_ms:.1f}ms"
                )
        
        # Update last data time
        self.last_data_times[rate_key] = datetime.utcnow()
    
    async def track_api_health(self, provider: str, is_healthy: bool, response_time_ms: float = 0):
        """Track API provider health."""
        self.api_health[provider] = {
            'healthy': is_healthy,
            'last_check': datetime.utcnow(),
            'response_time': response_time_ms
        }
        
        api_health_gauge.labels(provider=provider).set(1 if is_healthy else 0)
        
        if not is_healthy:
            await self.create_alert(
                'error',
                'api_health',
                f"API provider {provider} is unhealthy"
            )
    
    async def track_strategy_performance(self, strategy: str, score: float):
        """Track trading strategy performance."""
        strategy_performance_gauge.labels(strategy=strategy).set(score)
    
    async def track_system_resources(self):
        """Track system resource usage."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            system_resource_gauge.labels(resource_type='cpu').set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            system_resource_gauge.labels(resource_type='memory').set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            system_resource_gauge.labels(resource_type='disk').set(disk.percent)
            
            # Check for resource alerts
            if cpu_percent > 80:
                await self.create_alert('warning', 'resources', f"High CPU usage: {cpu_percent}%")
            if memory.percent > 90:
                await self.create_alert('critical', 'resources', f"High memory usage: {memory.percent}%")
            
        except ImportError:
            logger.warning("psutil not installed, resource monitoring disabled")
    
    async def track_data_quality(self, source: str, quality_metrics: Dict[str, float]):
        """Track data quality metrics."""
        # Calculate overall quality score (0-100)
        score = 100.0
        
        if 'completeness' in quality_metrics:
            score *= quality_metrics['completeness']
        if 'accuracy' in quality_metrics:
            score *= quality_metrics['accuracy']
        if 'timeliness' in quality_metrics:
            score *= quality_metrics['timeliness']
        
        data_quality_score.labels(source=source).set(score)
        
        if score < 50:
            await self.create_alert(
                'warning',
                'data_quality',
                f"Low data quality for {source}: {score:.1f}%"
            )
    
    async def create_alert(self, severity: str, category: str, message: str):
        """Create and store an alert."""
        alert = {
            'timestamp': datetime.utcnow().isoformat(),
            'severity': severity,
            'category': category,
            'message': message
        }
        
        self.alert_buffer.append(alert)
        alert_counter.labels(severity=severity, category=category).inc()
        
        # Store in Redis for persistence
        if self.redis:
            await self.redis.lpush('monitoring:alerts', str(alert))
            await self.redis.ltrim('monitoring:alerts', 0, 999)  # Keep last 1000
        
        logger.warning(f"Alert: [{severity}] {category} - {message}")
    
    async def check_stale_data(self):
        """Check for stale data sources."""
        now = datetime.utcnow()
        stale_threshold = timedelta(seconds=self.thresholds['stale_data_seconds'])
        
        for source, last_time in self.last_data_times.items():
            if now - last_time > stale_threshold:
                await self.create_alert(
                    'warning',
                    'stale_data',
                    f"No data from {source} for {(now - last_time).total_seconds():.0f} seconds"
                )
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        health_checks = {
            'apis_healthy': all(h.get('healthy', False) for h in self.api_health.values()),
            'data_fresh': await self._check_data_freshness(),
            'error_rate_ok': await self._check_error_rate(),
            'resources_ok': await self._check_resources()
        }
        
        return {
            'healthy': all(health_checks.values()),
            'timestamp': datetime.utcnow().isoformat(),
            'checks': health_checks,
            'details': {
                'api_health': self.api_health,
                'recent_errors': len(self.error_buffer),
                'recent_alerts': len(self.alert_buffer)
            }
        }
    
    async def _check_data_freshness(self) -> bool:
        """Check if data is fresh."""
        if not self.last_data_times:
            return True
        
        now = datetime.utcnow()
        stale_threshold = timedelta(seconds=self.thresholds['stale_data_seconds'])
        
        return all(
            now - last_time < stale_threshold
            for last_time in self.last_data_times.values()
        )
    
    async def _check_error_rate(self) -> bool:
        """Check if error rate is acceptable."""
        if not self.error_buffer:
            return True
        
        recent_errors = [
            e for e in self.error_buffer
            if datetime.fromisoformat(e['timestamp']) > datetime.utcnow() - timedelta(minutes=5)
        ]
        
        # Less than 5% error rate
        return len(recent_errors) < 50
    
    async def _check_resources(self) -> bool:
        """Check system resources."""
        try:
            import psutil
            return psutil.cpu_percent() < 90 and psutil.virtual_memory().percent < 95
        except:
            return True
    
    async def get_current_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'ingestion_rates': {},
            'latencies': {},
            'api_health': self.api_health,
            'alerts': {
                'total': len(self.alert_buffer),
                'recent': list(self.alert_buffer)[-5:]
            }
        }
        
        # Calculate average ingestion rates
        for key, rates in self.ingestion_rates.items():
            if rates:
                recent = rates[-10:]
                if len(recent) > 1:
                    time_span = (recent[-1]['timestamp'] - recent[0]['timestamp']).total_seconds()
                    if time_span > 0:
                        stats['ingestion_rates'][key] = sum(r['count'] for r in recent) / time_span
        
        # Calculate average latencies
        for key, latencies in self.latency_buffer.items():
            if latencies:
                stats['latencies'][key] = {
                    'avg': sum(latencies) / len(latencies),
                    'max': max(latencies),
                    'min': min(latencies)
                }
        
        return stats
    
    async def generate_dashboard_html(self) -> str:
        """Generate HTML dashboard."""
        stats = await self.get_current_stats()
        health = await self.get_system_health()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Ingestion Monitor</title>
            <meta http-equiv="refresh" content="5">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .status {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .healthy {{ background: #4CAF50; color: white; }}
                .unhealthy {{ background: #f44336; color: white; }}
                .warning {{ background: #ff9800; color: white; }}
                .metric {{ background: white; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                h1, h2 {{ color: #333; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Data Ingestion Monitoring Dashboard</h1>
                
                <div class="status {'healthy' if health['healthy'] else 'unhealthy'}">
                    System Status: {'✅ HEALTHY' if health['healthy'] else '❌ UNHEALTHY'}
                    - Last Update: {stats['timestamp']}
                </div>
                
                <div class="grid">
                    <div class="metric">
                        <h2>📈 Ingestion Rates (msg/sec)</h2>
                        <table>
                            {''.join(f"<tr><td>{k}</td><td>{v:.2f}</td></tr>" for k, v in stats['ingestion_rates'].items())}
                        </table>
                    </div>
                    
                    <div class="metric">
                        <h2>⚡ Latencies (ms)</h2>
                        <table>
                            {''.join(f"<tr><td>{k}</td><td>Avg: {v['avg']:.1f} / Max: {v['max']:.1f}</td></tr>" for k, v in stats['latencies'].items())}
                        </table>
                    </div>
                    
                    <div class="metric">
                        <h2>🔌 API Health</h2>
                        <table>
                            {''.join(f"<tr><td>{k}</td><td>{'✅' if v.get('healthy') else '❌'}</td></tr>" for k, v in self.api_health.items())}
                        </table>
                    </div>
                    
                    <div class="metric">
                        <h2>🚨 Recent Alerts ({len(self.alert_buffer)})</h2>
                        <table>
                            {''.join(f"<tr><td class='{a['severity']}'>{a['category']}: {a['message']}</td></tr>" for a in list(self.alert_buffer)[-5:])}
                        </table>
                    </div>
                </div>
                
                <div class="metric">
                    <h2>📊 Prometheus Metrics</h2>
                    <a href="/metrics" target="_blank">View Raw Metrics</a>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    async def run_monitoring_tasks(self):
        """Run periodic monitoring tasks."""
        while True:
            try:
                # Check system resources every 30 seconds
                await self.track_system_resources()
                
                # Check for stale data every minute
                await self.check_stale_data()
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in monitoring tasks: {e}")
                await asyncio.sleep(60)
    
    async def start(self, port: int = 8080):
        """Start the monitoring dashboard server."""
        # Start monitoring tasks
        asyncio.create_task(self.run_monitoring_tasks())
        
        # Start web server
        config = uvicorn.Config(
            app=self.app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


# Example usage
async def main():
    """Example usage of monitoring dashboard."""
    dashboard = MonitoringDashboard()
    
    # Simulate some data tracking
    await dashboard.track_data_ingestion('polygon', 'trades', count=100, latency_ms=5.2)
    await dashboard.track_data_ingestion('newsapi', 'news', count=10, latency_ms=125.5)
    
    await dashboard.track_api_health('polygon', True, 12.5)
    await dashboard.track_api_health('newsapi', True, 95.3)
    
    await dashboard.track_strategy_performance('day_trading', 0.85)
    await dashboard.track_strategy_performance('swing_trading', 0.72)
    
    # Start dashboard
    await dashboard.start(port=8080)


if __name__ == "__main__":
    asyncio.run(main())