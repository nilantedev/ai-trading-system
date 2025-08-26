#!/usr/bin/env python3
"""
Centralized resilience monitoring and management system.
Provides health checks, metrics, and management for all resilience patterns.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from .resilience import get_all_circuit_breakers
from .logging import get_logger

logger = get_logger(__name__)


class ResilienceHealthStatus(Enum):
    """Health status for resilience components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class ResilienceMetrics:
    """Aggregated resilience metrics."""
    timestamp: datetime
    total_circuit_breakers: int
    open_circuit_breakers: int
    half_open_circuit_breakers: int
    total_requests: int = 0
    failed_requests: int = 0
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    rate_limited_requests: int = 0
    bulkhead_rejections: int = 0


@dataclass
class ResilienceAlert:
    """Alert for resilience issues."""
    alert_id: str
    component: str
    severity: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


class ResilienceMonitor:
    """Centralized resilience monitoring system."""
    
    def __init__(self):
        self.is_running = False
        self.check_interval = 30  # seconds
        self.metrics_history: List[ResilienceMetrics] = []
        self.active_alerts: Dict[str, ResilienceAlert] = {}
        self.alert_history: List[ResilienceAlert] = []
        self.alert_thresholds = {
            'circuit_breaker_open_rate': 0.3,  # Alert if >30% CBs are open
            'failure_rate': 0.1,  # Alert if >10% requests failing
            'response_time_threshold': 5.0,  # Alert if avg response > 5s
        }
        
        # Component registry for tracking external services
        self.tracked_components: Dict[str, Any] = {}
    
    def register_component(self, name: str, component: Any):
        """Register a component for monitoring (e.g., API clients)."""
        self.tracked_components[name] = component
        logger.info(f"Registered component for resilience monitoring: {name}")
    
    async def start_monitoring(self):
        """Start the resilience monitoring loop."""
        if self.is_running:
            logger.warning("Resilience monitor is already running")
            return
        
        self.is_running = True
        logger.info("Starting resilience monitoring")
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop the resilience monitoring loop."""
        self.is_running = False
        logger.info("Stopping resilience monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                await self._collect_metrics()
                await self._check_alerts()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in resilience monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _collect_metrics(self):
        """Collect metrics from all resilience components."""
        try:
            # Get circuit breaker states
            cb_states = get_all_circuit_breakers()
            
            # Count circuit breaker states
            total_cbs = len(cb_states)
            open_cbs = sum(1 for state in cb_states.values() if state['state'] == 'open')
            half_open_cbs = sum(1 for state in cb_states.values() if state['state'] == 'half_open')
            
            # Collect component metrics
            total_requests = 0
            failed_requests = 0
            response_times = []
            rate_limited = 0
            bulkhead_rejections = 0
            
            for name, component in self.tracked_components.items():
                try:
                    if hasattr(component, 'get_resilience_stats'):
                        stats = component.get_resilience_stats()
                        total_requests += stats.get('requests_total', 0)
                        failed_requests += stats.get('errors_total', 0)
                        rate_limited += stats.get('rate_limit_hits', 0)
                        
                        # Get bulkhead rejections if available
                        bulkhead = stats.get('bulkhead', {})
                        if isinstance(bulkhead, dict):
                            bulkhead_rejections += bulkhead.get('rejections', 0)
                    
                    elif hasattr(component, 'get_stats'):
                        stats = component.get_stats()
                        total_requests += stats.get('stats', {}).get('total_requests', 0)
                        failed_requests += stats.get('stats', {}).get('failed_requests', 0)
                        
                except Exception as e:
                    logger.warning(f"Failed to collect metrics from component {name}: {e}")
            
            # Calculate derived metrics
            success_rate = 1.0 - (failed_requests / max(total_requests, 1))
            avg_response_time = sum(response_times) / max(len(response_times), 1) if response_times else 0.0
            
            # Create metrics object
            metrics = ResilienceMetrics(
                timestamp=datetime.utcnow(),
                total_circuit_breakers=total_cbs,
                open_circuit_breakers=open_cbs,
                half_open_circuit_breakers=half_open_cbs,
                total_requests=total_requests,
                failed_requests=failed_requests,
                success_rate=success_rate,
                avg_response_time=avg_response_time,
                rate_limited_requests=rate_limited,
                bulkhead_rejections=bulkhead_rejections
            )
            
            # Store metrics (keep last 100 entries)
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 100:
                self.metrics_history.pop(0)
            
            logger.debug(f"Collected resilience metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Failed to collect resilience metrics: {e}")
    
    async def _check_alerts(self):
        """Check for alert conditions and manage alerts."""
        if not self.metrics_history:
            return
        
        latest_metrics = self.metrics_history[-1]
        
        # Check circuit breaker open rate
        if latest_metrics.total_circuit_breakers > 0:
            cb_open_rate = latest_metrics.open_circuit_breakers / latest_metrics.total_circuit_breakers
            alert_id = "high_circuit_breaker_open_rate"
            
            if cb_open_rate > self.alert_thresholds['circuit_breaker_open_rate']:
                if alert_id not in self.active_alerts:
                    alert = ResilienceAlert(
                        alert_id=alert_id,
                        component="circuit_breakers",
                        severity="warning",
                        message=f"High circuit breaker open rate: {cb_open_rate:.1%} "
                               f"({latest_metrics.open_circuit_breakers}/{latest_metrics.total_circuit_breakers})",
                        timestamp=datetime.utcnow()
                    )
                    self.active_alerts[alert_id] = alert
                    self.alert_history.append(alert)
                    logger.warning(f"ALERT: {alert.message}")
            else:
                # Resolve alert if it exists
                if alert_id in self.active_alerts:
                    self.active_alerts[alert_id].resolved = True
                    self.active_alerts[alert_id].resolution_timestamp = datetime.utcnow()
                    del self.active_alerts[alert_id]
                    logger.info(f"RESOLVED: Circuit breaker open rate alert")
        
        # Check failure rate
        if latest_metrics.total_requests > 100:  # Only check if we have significant traffic
            failure_rate = latest_metrics.failed_requests / latest_metrics.total_requests
            alert_id = "high_failure_rate"
            
            if failure_rate > self.alert_thresholds['failure_rate']:
                if alert_id not in self.active_alerts:
                    alert = ResilienceAlert(
                        alert_id=alert_id,
                        component="api_calls",
                        severity="critical",
                        message=f"High API failure rate: {failure_rate:.1%} "
                               f"({latest_metrics.failed_requests}/{latest_metrics.total_requests})",
                        timestamp=datetime.utcnow()
                    )
                    self.active_alerts[alert_id] = alert
                    self.alert_history.append(alert)
                    logger.error(f"CRITICAL ALERT: {alert.message}")
            else:
                if alert_id in self.active_alerts:
                    self.active_alerts[alert_id].resolved = True
                    self.active_alerts[alert_id].resolution_timestamp = datetime.utcnow()
                    del self.active_alerts[alert_id]
                    logger.info(f"RESOLVED: High failure rate alert")
        
        # Check response time
        if latest_metrics.avg_response_time > self.alert_thresholds['response_time_threshold']:
            alert_id = "high_response_time"
            
            if alert_id not in self.active_alerts:
                alert = ResilienceAlert(
                    alert_id=alert_id,
                    component="api_calls",
                    severity="warning",
                    message=f"High average response time: {latest_metrics.avg_response_time:.2f}s",
                    timestamp=datetime.utcnow()
                )
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                logger.warning(f"ALERT: {alert.message}")
        else:
            alert_id = "high_response_time"
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].resolved = True
                self.active_alerts[alert_id].resolution_timestamp = datetime.utcnow()
                del self.active_alerts[alert_id]
                logger.info(f"RESOLVED: High response time alert")
    
    def get_health_status(self) -> ResilienceHealthStatus:
        """Get overall resilience health status."""
        if not self.metrics_history:
            return ResilienceHealthStatus.UNHEALTHY
        
        latest_metrics = self.metrics_history[-1]
        
        # Check for critical conditions
        critical_alerts = [a for a in self.active_alerts.values() if a.severity == "critical"]
        if critical_alerts:
            return ResilienceHealthStatus.CRITICAL
        
        # Check circuit breakers
        if latest_metrics.total_circuit_breakers > 0:
            cb_open_rate = latest_metrics.open_circuit_breakers / latest_metrics.total_circuit_breakers
            if cb_open_rate > 0.5:  # More than half open
                return ResilienceHealthStatus.UNHEALTHY
            elif cb_open_rate > 0.3:  # More than 30% open
                return ResilienceHealthStatus.DEGRADED
        
        # Check active warnings
        warning_alerts = [a for a in self.active_alerts.values() if a.severity == "warning"]
        if warning_alerts:
            return ResilienceHealthStatus.DEGRADED
        
        return ResilienceHealthStatus.HEALTHY
    
    def get_current_metrics(self) -> Optional[ResilienceMetrics]:
        """Get the most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of recent metrics."""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        
        # Calculate trends from last 10 entries
        recent_metrics = self.metrics_history[-10:]
        if len(recent_metrics) > 1:
            old_success_rate = recent_metrics[0].success_rate
            success_rate_trend = latest.success_rate - old_success_rate
            
            old_response_time = recent_metrics[0].avg_response_time
            response_time_trend = latest.avg_response_time - old_response_time
        else:
            success_rate_trend = 0.0
            response_time_trend = 0.0
        
        return {
            'timestamp': latest.timestamp.isoformat(),
            'health_status': self.get_health_status().value,
            'circuit_breakers': {
                'total': latest.total_circuit_breakers,
                'open': latest.open_circuit_breakers,
                'half_open': latest.half_open_circuit_breakers,
                'closed': latest.total_circuit_breakers - latest.open_circuit_breakers - latest.half_open_circuit_breakers
            },
            'requests': {
                'total': latest.total_requests,
                'failed': latest.failed_requests,
                'success_rate': latest.success_rate,
                'success_rate_trend': success_rate_trend
            },
            'performance': {
                'avg_response_time': latest.avg_response_time,
                'response_time_trend': response_time_trend
            },
            'protection': {
                'rate_limited': latest.rate_limited_requests,
                'bulkhead_rejections': latest.bulkhead_rejections
            },
            'active_alerts': len(self.active_alerts),
            'alerts': [
                {
                    'id': alert.alert_id,
                    'component': alert.component,
                    'severity': alert.severity,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in self.active_alerts.values()
            ]
        }
    
    def get_circuit_breaker_details(self) -> Dict[str, Any]:
        """Get detailed circuit breaker information."""
        return get_all_circuit_breakers()
    
    def get_component_stats(self) -> Dict[str, Any]:
        """Get statistics from all tracked components."""
        stats = {}
        
        for name, component in self.tracked_components.items():
            try:
                if hasattr(component, 'get_resilience_stats'):
                    stats[name] = component.get_resilience_stats()
                elif hasattr(component, 'get_stats'):
                    stats[name] = component.get_stats()
                else:
                    stats[name] = {'status': 'no_stats_available'}
            except Exception as e:
                stats[name] = {'error': str(e)}
        
        return stats


# Global resilience monitor instance
_resilience_monitor: Optional[ResilienceMonitor] = None


def get_resilience_monitor() -> ResilienceMonitor:
    """Get or create the global resilience monitor."""
    global _resilience_monitor
    if _resilience_monitor is None:
        _resilience_monitor = ResilienceMonitor()
    return _resilience_monitor


async def start_resilience_monitoring():
    """Start global resilience monitoring."""
    monitor = get_resilience_monitor()
    await monitor.start_monitoring()


async def stop_resilience_monitoring():
    """Stop global resilience monitoring."""
    monitor = get_resilience_monitor()
    await monitor.stop_monitoring()