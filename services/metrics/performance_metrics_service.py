#!/usr/bin/env python3
"""Performance Metrics Service - Real-time performance monitoring and analytics."""

import asyncio
import json
import logging
import statistics
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque, defaultdict

from trading_common import get_settings, get_logger
from trading_common.cache import get_trading_cache
from trading_common.messaging import get_message_consumer, get_message_producer

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    timestamp: datetime
    service_name: str
    
    # Throughput metrics
    messages_per_second: float
    requests_per_second: float
    
    # Latency metrics (milliseconds)
    avg_latency: float
    p95_latency: float
    p99_latency: float
    max_latency: float
    
    # Resource metrics
    memory_usage_mb: float
    cpu_usage_percent: float
    
    # Error metrics
    error_rate: float
    success_rate: float
    
    # Business metrics
    total_processed: int
    total_errors: int
    uptime_seconds: float


@dataclass
class TradingPerformanceMetrics:
    """Trading-specific performance metrics."""
    timestamp: datetime
    
    # Data pipeline metrics
    data_ingestion_rate: float  # Records/second
    data_processing_latency: float  # Milliseconds
    data_quality_score: float  # 0-1
    
    # Signal generation metrics
    signals_generated_per_minute: float
    signal_accuracy: float  # 0-1
    signal_latency: float  # Milliseconds from data to signal
    
    # Risk monitoring metrics
    risk_checks_per_second: float
    alerts_generated_per_hour: float
    risk_calculation_latency: float  # Milliseconds
    
    # Overall system health
    system_health_score: float  # 0-1
    active_services: int
    failed_services: int


@dataclass
class AlertThreshold:
    """Performance alert threshold configuration."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison: str  # 'gt', 'lt', 'eq'


class PerformanceMetricsService:
    """Service for collecting and analyzing performance metrics."""
    
    def __init__(self):
        self.consumer = None
        self.producer = None
        self.cache = None
        
        self.is_running = False
        self.start_time = datetime.utcnow()
        
        # Metrics collection
        self.service_metrics = {}  # service_name -> PerformanceMetrics
        self.latency_samples = defaultdict(lambda: deque(maxlen=1000))  # service_name -> latencies
        self.throughput_samples = defaultdict(lambda: deque(maxlen=100))  # service_name -> throughput
        self.error_counts = defaultdict(int)
        self.success_counts = defaultdict(int)
        
        # Trading metrics tracking
        self.trading_metrics_history = deque(maxlen=1440)  # 24 hours of minute samples
        
        # Alert thresholds
        self.alert_thresholds = {
            'latency_ms': AlertThreshold('latency_ms', 100.0, 500.0, 'gt'),
            'error_rate': AlertThreshold('error_rate', 0.05, 0.20, 'gt'),
            'memory_usage_mb': AlertThreshold('memory_usage_mb', 500.0, 1000.0, 'gt'),
            'cpu_usage_percent': AlertThreshold('cpu_usage_percent', 70.0, 90.0, 'gt'),
            'throughput_rps': AlertThreshold('throughput_rps', 10.0, 1.0, 'lt')  # Low throughput alert
        }
        
        # Performance counters
        self.metrics_collected = 0
        self.alerts_generated = 0
        self.reports_generated = 0
        
        # Message queues
        self.metrics_queue = asyncio.Queue(maxsize=10000)
        
    async def start(self):
        """Initialize and start performance metrics service."""
        logger.info("Starting Performance Metrics Service")
        
        try:
            # Initialize connections
            self.consumer = await get_message_consumer()
            self.producer = await get_message_producer()
            self.cache = get_trading_cache()
            
            # Subscribe to metrics streams
            await self._setup_subscriptions()
            
            # Start metrics collection tasks
            self.is_running = True
            
            tasks = [
                asyncio.create_task(self._process_metrics_queue()),
                asyncio.create_task(self._consume_messages()),
                asyncio.create_task(self._collect_system_metrics()),
                asyncio.create_task(self._analyze_performance_trends()),
                asyncio.create_task(self._generate_periodic_reports()),
                asyncio.create_task(self._cleanup_old_data())
            ]
            
            logger.info("Performance metrics service started with 6 concurrent tasks")
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Failed to start performance metrics service: {e}")
            raise
    
    async def stop(self):
        """Stop performance metrics service gracefully."""
        logger.info("Stopping Performance Metrics Service")
        self.is_running = False
        
        if self.consumer:
            await self.consumer.close()
        if self.producer:
            await self.producer.close()
        
        logger.info("Performance Metrics Service stopped")
    
    async def _setup_subscriptions(self):
        """Subscribe to service health and metrics streams."""
        try:
            await self.consumer.subscribe_service_metrics(
                self._handle_service_metrics_message,
                subscription_name="performance-metrics-service"
            )
            
            await self.consumer.subscribe_system_health(
                self._handle_system_health_message,
                subscription_name="performance-metrics-health"
            )
            
            logger.info("Subscribed to metrics and health streams")
        except Exception as e:
            logger.warning(f"Metrics subscription setup failed: {e}")
    
    async def _consume_messages(self):
        """Consume messages from subscribed topics."""
        try:
            await self.consumer.start_consuming()
        except Exception as e:
            logger.error(f"Message consumption failed: {e}")
    
    async def _handle_service_metrics_message(self, message):
        """Handle incoming service metrics."""
        try:
            metrics_data = json.loads(message) if isinstance(message, str) else message
            await self.metrics_queue.put(metrics_data)
        except Exception as e:
            logger.error(f"Failed to handle service metrics message: {e}")
    
    async def _handle_system_health_message(self, message):
        """Handle system health status updates."""
        try:
            health_data = json.loads(message) if isinstance(message, str) else message
            await self._update_service_health(health_data)
        except Exception as e:
            logger.error(f"Failed to handle system health message: {e}")
    
    async def _process_metrics_queue(self):
        """Process incoming metrics data."""
        while self.is_running:
            try:
                # Wait for metrics data
                metrics_data = await asyncio.wait_for(
                    self.metrics_queue.get(),
                    timeout=1.0
                )
                
                service_name = metrics_data.get('service', 'unknown')
                
                # Record latency samples
                if 'latency_ms' in metrics_data:
                    self.latency_samples[service_name].append(metrics_data['latency_ms'])
                
                # Record throughput samples
                if 'throughput_rps' in metrics_data:
                    self.throughput_samples[service_name].append(metrics_data['throughput_rps'])
                
                # Update error/success counts
                if 'errors' in metrics_data:
                    self.error_counts[service_name] += metrics_data['errors']
                
                if 'successes' in metrics_data:
                    self.success_counts[service_name] += metrics_data['successes']
                
                # Calculate and cache current metrics
                await self._calculate_service_metrics(service_name)
                
                # Check for alert conditions
                await self._check_performance_alerts(service_name, metrics_data)
                
                self.metrics_collected += 1
                self.metrics_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Metrics processing error: {e}")
    
    async def _calculate_service_metrics(self, service_name: str):
        """Calculate comprehensive metrics for a service."""
        try:
            current_time = datetime.utcnow()
            
            # Calculate latency metrics
            latencies = list(self.latency_samples[service_name])
            avg_latency = statistics.mean(latencies) if latencies else 0.0
            p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else avg_latency
            p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else avg_latency
            max_latency = max(latencies) if latencies else 0.0
            
            # Calculate throughput
            throughputs = list(self.throughput_samples[service_name])
            avg_throughput = statistics.mean(throughputs) if throughputs else 0.0
            
            # Calculate error rates
            total_errors = self.error_counts[service_name]
            total_successes = self.success_counts[service_name]
            total_requests = total_errors + total_successes
            
            error_rate = total_errors / max(total_requests, 1)
            success_rate = total_successes / max(total_requests, 1)
            
            # System resource metrics (simplified - would integrate with system monitoring)
            memory_usage_mb = 100.0  # Would get from actual system metrics
            cpu_usage_percent = 25.0  # Would get from actual system metrics
            
            uptime = (current_time - self.start_time).total_seconds()
            
            metrics = PerformanceMetrics(
                timestamp=current_time,
                service_name=service_name,
                messages_per_second=avg_throughput,
                requests_per_second=avg_throughput,
                avg_latency=avg_latency,
                p95_latency=p95_latency,
                p99_latency=p99_latency,
                max_latency=max_latency,
                memory_usage_mb=memory_usage_mb,
                cpu_usage_percent=cpu_usage_percent,
                error_rate=error_rate,
                success_rate=success_rate,
                total_processed=total_successes,
                total_errors=total_errors,
                uptime_seconds=uptime
            )
            
            # Store metrics
            self.service_metrics[service_name] = metrics
            
            # Cache metrics
            await self._cache_service_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics for {service_name}: {e}")
    
    async def _check_performance_alerts(self, service_name: str, metrics_data: Dict[str, Any]):
        """Check metrics against alert thresholds."""
        try:
            alerts = []
            
            for metric_name, threshold in self.alert_thresholds.items():
                if metric_name in metrics_data:
                    value = metrics_data[metric_name]
                    
                    # Check warning threshold
                    if self._exceeds_threshold(value, threshold.warning_threshold, threshold.comparison):
                        severity = "critical" if self._exceeds_threshold(value, threshold.critical_threshold, threshold.comparison) else "warning"
                        
                        alert = {
                            'alert_id': f"perf_alert_{service_name}_{metric_name}_{datetime.utcnow().timestamp()}",
                            'service': service_name,
                            'metric_name': metric_name,
                            'current_value': value,
                            'threshold': threshold.critical_threshold if severity == "critical" else threshold.warning_threshold,
                            'severity': severity,
                            'timestamp': datetime.utcnow().isoformat(),
                            'message': f"{service_name} {metric_name} is {value} (threshold: {threshold.warning_threshold})"
                        }
                        
                        alerts.append(alert)
                        await self._publish_performance_alert(alert)
            
            if alerts:
                self.alerts_generated += len(alerts)
                logger.warning(f"Generated {len(alerts)} performance alerts for {service_name}")
            
        except Exception as e:
            logger.error(f"Performance alert check failed: {e}")
    
    def _exceeds_threshold(self, value: float, threshold: float, comparison: str) -> bool:
        """Check if value exceeds threshold based on comparison type."""
        if comparison == 'gt':
            return value > threshold
        elif comparison == 'lt':
            return value < threshold
        elif comparison == 'eq':
            return value == threshold
        return False
    
    async def _collect_system_metrics(self):
        """Collect system-wide performance metrics."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Collect every minute
                
                current_time = datetime.utcnow()
                
                # Calculate trading performance metrics
                trading_metrics = await self._calculate_trading_metrics()
                self.trading_metrics_history.append(trading_metrics)
                
                # Cache trading metrics
                await self._cache_trading_metrics(trading_metrics)
                
                # Log system health summary
                logger.debug(f"System health score: {trading_metrics.system_health_score:.2f}")
                
            except Exception as e:
                logger.warning(f"System metrics collection error: {e}")
    
    async def _calculate_trading_metrics(self) -> TradingPerformanceMetrics:
        """Calculate trading-specific performance metrics."""
        try:
            current_time = datetime.utcnow()
            
            # Get service metrics for calculations
            ingestion_metrics = self.service_metrics.get('data_ingestion_service')
            processor_metrics = self.service_metrics.get('stream_processor_service')
            indicator_metrics = self.service_metrics.get('indicator_service')
            signal_metrics = self.service_metrics.get('signal_service')
            risk_metrics = self.service_metrics.get('risk_service')
            
            # Data pipeline metrics
            data_ingestion_rate = ingestion_metrics.messages_per_second if ingestion_metrics else 0.0
            data_processing_latency = processor_metrics.avg_latency if processor_metrics else 0.0
            data_quality_score = 0.95  # Would calculate from actual data quality checks
            
            # Signal generation metrics
            signals_generated_per_minute = (signal_metrics.messages_per_second * 60) if signal_metrics else 0.0
            signal_accuracy = 0.75  # Would calculate from actual signal performance
            signal_latency = (indicator_metrics.avg_latency + signal_metrics.avg_latency) / 2 if indicator_metrics and signal_metrics else 0.0
            
            # Risk monitoring metrics
            risk_checks_per_second = risk_metrics.messages_per_second if risk_metrics else 0.0
            alerts_generated_per_hour = self.alerts_generated / max((current_time - self.start_time).total_seconds() / 3600, 1)
            risk_calculation_latency = risk_metrics.avg_latency if risk_metrics else 0.0
            
            # System health calculation
            active_services = len([m for m in self.service_metrics.values() if m.success_rate > 0.5])
            failed_services = len([m for m in self.service_metrics.values() if m.error_rate > 0.5])
            
            # Overall health score (weighted average of key metrics)
            health_factors = [
                data_quality_score * 0.25,
                min(data_ingestion_rate / 100, 1.0) * 0.20,  # Normalize to 0-1
                max(1 - (data_processing_latency / 1000), 0) * 0.20,  # Lower latency = better
                min(signals_generated_per_minute / 10, 1.0) * 0.15,  # Normalize to 0-1
                max(1 - (risk_calculation_latency / 500), 0) * 0.10,  # Lower latency = better
                min(active_services / 10, 1.0) * 0.10  # More active services = better
            ]
            
            system_health_score = sum(health_factors)
            
            return TradingPerformanceMetrics(
                timestamp=current_time,
                data_ingestion_rate=data_ingestion_rate,
                data_processing_latency=data_processing_latency,
                data_quality_score=data_quality_score,
                signals_generated_per_minute=signals_generated_per_minute,
                signal_accuracy=signal_accuracy,
                signal_latency=signal_latency,
                risk_checks_per_second=risk_checks_per_second,
                alerts_generated_per_hour=alerts_generated_per_hour,
                risk_calculation_latency=risk_calculation_latency,
                system_health_score=system_health_score,
                active_services=active_services,
                failed_services=failed_services
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate trading metrics: {e}")
            return TradingPerformanceMetrics(
                timestamp=datetime.utcnow(),
                data_ingestion_rate=0.0,
                data_processing_latency=0.0,
                data_quality_score=0.0,
                signals_generated_per_minute=0.0,
                signal_accuracy=0.0,
                signal_latency=0.0,
                risk_checks_per_second=0.0,
                alerts_generated_per_hour=0.0,
                risk_calculation_latency=0.0,
                system_health_score=0.0,
                active_services=0,
                failed_services=0
            )
    
    async def _analyze_performance_trends(self):
        """Analyze performance trends and patterns."""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
                # Analyze latency trends
                for service_name, latencies in self.latency_samples.items():
                    if len(latencies) >= 10:
                        recent_avg = statistics.mean(list(latencies)[-10:])
                        historical_avg = statistics.mean(list(latencies)[:-10]) if len(latencies) > 10 else recent_avg
                        
                        if recent_avg > historical_avg * 1.5:  # 50% increase
                            logger.warning(f"Latency trend alert: {service_name} latency increased by {((recent_avg/historical_avg - 1) * 100):.1f}%")
                
                # Analyze throughput trends
                for service_name, throughputs in self.throughput_samples.items():
                    if len(throughputs) >= 10:
                        recent_avg = statistics.mean(list(throughputs)[-10:])
                        historical_avg = statistics.mean(list(throughputs)[:-10]) if len(throughputs) > 10 else recent_avg
                        
                        if recent_avg < historical_avg * 0.5:  # 50% decrease
                            logger.warning(f"Throughput trend alert: {service_name} throughput decreased by {((1 - recent_avg/historical_avg) * 100):.1f}%")
                
            except Exception as e:
                logger.warning(f"Performance trend analysis error: {e}")
    
    async def _generate_periodic_reports(self):
        """Generate periodic performance reports."""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Generate hourly reports
                
                report = await self._generate_performance_report()
                await self._cache_performance_report(report)
                
                self.reports_generated += 1
                logger.info("Generated hourly performance report")
                
            except Exception as e:
                logger.warning(f"Performance report generation error: {e}")
    
    async def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            current_time = datetime.utcnow()
            
            # Service performance summary
            service_summary = {}
            for service_name, metrics in self.service_metrics.items():
                service_summary[service_name] = {
                    'avg_latency_ms': metrics.avg_latency,
                    'throughput_rps': metrics.messages_per_second,
                    'error_rate': metrics.error_rate,
                    'uptime_hours': metrics.uptime_seconds / 3600,
                    'health_status': 'healthy' if metrics.error_rate < 0.1 and metrics.avg_latency < 200 else 'degraded'
                }
            
            # Trading metrics summary
            latest_trading_metrics = self.trading_metrics_history[-1] if self.trading_metrics_history else None
            trading_summary = {}
            
            if latest_trading_metrics:
                trading_summary = {
                    'system_health_score': latest_trading_metrics.system_health_score,
                    'data_pipeline_health': 'good' if latest_trading_metrics.data_processing_latency < 100 else 'poor',
                    'signal_generation_rate': latest_trading_metrics.signals_generated_per_minute,
                    'risk_monitoring_active': latest_trading_metrics.risk_checks_per_second > 0
                }
            
            # System-wide metrics
            total_messages_processed = sum(m.total_processed for m in self.service_metrics.values())
            total_errors = sum(m.total_errors for m in self.service_metrics.values())
            overall_error_rate = total_errors / max(total_messages_processed + total_errors, 1)
            
            report = {
                'report_id': f"perf_report_{current_time.strftime('%Y%m%d_%H%M')}",
                'timestamp': current_time.isoformat(),
                'period': 'hourly',
                'system_overview': {
                    'total_messages_processed': total_messages_processed,
                    'overall_error_rate': overall_error_rate,
                    'active_services': len(self.service_metrics),
                    'alerts_generated': self.alerts_generated
                },
                'service_performance': service_summary,
                'trading_metrics': trading_summary,
                'top_performers': self._get_top_performing_services(),
                'performance_concerns': self._get_performance_concerns()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
    
    def _get_top_performing_services(self) -> List[Dict[str, Any]]:
        """Get top performing services by various metrics."""
        try:
            services_by_throughput = sorted(
                self.service_metrics.values(),
                key=lambda m: m.messages_per_second,
                reverse=True
            )[:3]
            
            return [
                {
                    'service': m.service_name,
                    'metric': 'throughput',
                    'value': m.messages_per_second,
                    'unit': 'messages/sec'
                }
                for m in services_by_throughput
            ]
        except Exception:
            return []
    
    def _get_performance_concerns(self) -> List[Dict[str, Any]]:
        """Get services with performance concerns."""
        try:
            concerns = []
            
            for metrics in self.service_metrics.values():
                if metrics.error_rate > 0.1:
                    concerns.append({
                        'service': metrics.service_name,
                        'issue': 'high_error_rate',
                        'value': metrics.error_rate,
                        'severity': 'high' if metrics.error_rate > 0.2 else 'medium'
                    })
                
                if metrics.avg_latency > 500:
                    concerns.append({
                        'service': metrics.service_name,
                        'issue': 'high_latency',
                        'value': metrics.avg_latency,
                        'severity': 'high' if metrics.avg_latency > 1000 else 'medium'
                    })
                
                if metrics.messages_per_second < 1:
                    concerns.append({
                        'service': metrics.service_name,
                        'issue': 'low_throughput',
                        'value': metrics.messages_per_second,
                        'severity': 'medium'
                    })
            
            return concerns
        except Exception:
            return []
    
    async def _cleanup_old_data(self):
        """Clean up old metrics data."""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Clean up hourly
                
                # Keep only recent latency samples
                for service_name in list(self.latency_samples.keys()):
                    if len(self.latency_samples[service_name]) == 0:
                        del self.latency_samples[service_name]
                
                # Keep only recent throughput samples
                for service_name in list(self.throughput_samples.keys()):
                    if len(self.throughput_samples[service_name]) == 0:
                        del self.throughput_samples[service_name]
                
                logger.debug("Cleaned up old performance data")
                
            except Exception as e:
                logger.warning(f"Performance data cleanup error: {e}")
    
    async def _cache_service_metrics(self, metrics: PerformanceMetrics):
        """Cache service metrics."""
        try:
            if self.cache:
                cache_key = f"service_metrics:{metrics.service_name}:latest"
                metrics_data = asdict(metrics)
                metrics_data['timestamp'] = metrics.timestamp.isoformat()
                
                await self.cache.set_json(cache_key, metrics_data, ttl=300)  # 5 minutes
        except Exception as e:
            logger.warning(f"Failed to cache service metrics: {e}")
    
    async def _cache_trading_metrics(self, metrics: TradingPerformanceMetrics):
        """Cache trading metrics."""
        try:
            if self.cache:
                cache_key = "trading_metrics:latest"
                metrics_data = asdict(metrics)
                metrics_data['timestamp'] = metrics.timestamp.isoformat()
                
                await self.cache.set_json(cache_key, metrics_data, ttl=300)  # 5 minutes
        except Exception as e:
            logger.warning(f"Failed to cache trading metrics: {e}")
    
    async def _cache_performance_report(self, report: Dict[str, Any]):
        """Cache performance report."""
        try:
            if self.cache:
                cache_key = f"performance_report:{report.get('report_id', 'unknown')}"
                await self.cache.set_json(cache_key, report, ttl=86400)  # 24 hours
        except Exception as e:
            logger.warning(f"Failed to cache performance report: {e}")
    
    async def _publish_performance_alert(self, alert: Dict[str, Any]):
        """Publish performance alert."""
        try:
            if self.producer:
                # Would publish to performance alerts topic
                logger.debug(f"Publishing performance alert: {alert['message']}")
        except Exception as e:
            logger.warning(f"Failed to publish performance alert: {e}")
    
    async def _update_service_health(self, health_data: Dict[str, Any]):
        """Update service health information."""
        try:
            service_name = health_data.get('service', 'unknown')
            
            # Update service status in metrics
            if service_name in self.service_metrics:
                # Would update service health status
                pass
                
        except Exception as e:
            logger.warning(f"Failed to update service health: {e}")
    
    async def get_service_metrics(self, service_name: str) -> Optional[PerformanceMetrics]:
        """Get current metrics for a service."""
        return self.service_metrics.get(service_name)
    
    async def get_trading_metrics(self) -> Optional[TradingPerformanceMetrics]:
        """Get latest trading performance metrics."""
        return self.trading_metrics_history[-1] if self.trading_metrics_history else None
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all services."""
        try:
            total_services = len(self.service_metrics)
            healthy_services = len([m for m in self.service_metrics.values() if m.error_rate < 0.1])
            
            avg_latency = statistics.mean([m.avg_latency for m in self.service_metrics.values()]) if self.service_metrics else 0.0
            total_throughput = sum([m.messages_per_second for m in self.service_metrics.values()])
            overall_error_rate = statistics.mean([m.error_rate for m in self.service_metrics.values()]) if self.service_metrics else 0.0
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'total_services': total_services,
                'healthy_services': healthy_services,
                'health_percentage': (healthy_services / max(total_services, 1)) * 100,
                'avg_latency_ms': avg_latency,
                'total_throughput_rps': total_throughput,
                'overall_error_rate': overall_error_rate,
                'alerts_generated': self.alerts_generated,
                'uptime_hours': (datetime.utcnow() - self.start_time).total_seconds() / 3600
            }
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {'error': str(e)}
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get service health status."""
        return {
            'service': 'performance_metrics_service',
            'status': 'healthy' if self.is_running else 'stopped',
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {
                'metrics_collected': self.metrics_collected,
                'alerts_generated': self.alerts_generated,
                'reports_generated': self.reports_generated,
                'monitored_services': len(self.service_metrics)
            },
            'connections': {
                'consumer': self.consumer is not None,
                'producer': self.producer is not None,
                'cache': self.cache is not None
            }
        }


# Global service instance
metrics_service: Optional[PerformanceMetricsService] = None


async def get_metrics_service() -> PerformanceMetricsService:
    """Get or create performance metrics service instance."""
    global metrics_service
    if metrics_service is None:
        metrics_service = PerformanceMetricsService()
    return metrics_service