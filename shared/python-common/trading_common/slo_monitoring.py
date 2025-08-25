#!/usr/bin/env python3
"""
Service Level Objectives (SLO) monitoring and alerting for AI Trading System.
Defines SLOs, monitors compliance, and triggers alerts when thresholds are breached.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging
import statistics
from collections import defaultdict, deque

try:
    from prometheus_client import Gauge, Counter, Histogram, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from .logging import get_logger
from .metrics import get_metrics_registry
from .database import get_redis_client

logger = get_logger(__name__)


class SLOType(str, Enum):
    """Types of Service Level Objectives."""
    AVAILABILITY = "availability"
    LATENCY = "latency" 
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    FRESHNESS = "freshness"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertState(str, Enum):
    """Alert states."""
    FIRING = "firing"
    PENDING = "pending"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class SLOTarget:
    """SLO target definition."""
    slo_id: str
    name: str
    description: str
    slo_type: SLOType
    target_value: float  # Target value (e.g., 99.9 for 99.9% availability)
    measurement_window: timedelta  # Window for measuring SLO
    evaluation_interval: timedelta  # How often to evaluate
    service: str
    component: Optional[str] = None
    
    # Alert configuration
    warning_threshold: float = 0.95  # Warn at 95% of target
    critical_threshold: float = 0.90  # Critical at 90% of target
    alert_enabled: bool = True
    
    # Metadata
    owner: Optional[str] = None
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class SLOMetric:
    """SLO measurement result."""
    slo_id: str
    timestamp: datetime
    current_value: float
    target_value: float
    compliance_ratio: float  # current_value / target_value
    window_start: datetime
    window_end: datetime
    sample_count: int
    is_compliant: bool
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Alert:
    """Alert definition and state."""
    alert_id: str
    slo_id: str
    title: str
    description: str
    severity: AlertSeverity
    state: AlertState
    created_at: datetime
    updated_at: datetime
    
    # Alert details
    current_value: float
    threshold_value: float
    impact: Optional[str] = None
    runbook_url: Optional[str] = None
    
    # Resolution tracking
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    
    # Metadata
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class SLOEvaluator:
    """Evaluates SLO compliance based on metrics."""
    
    def __init__(self):
        """Initialize SLO evaluator."""
        self.metrics_registry = get_metrics_registry()
        self.redis = get_redis_client()
        
        # Data storage for SLO calculations
        self._metric_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._last_evaluations: Dict[str, datetime] = {}
        
        # Prometheus metrics for SLO monitoring
        if PROMETHEUS_AVAILABLE:
            self.slo_compliance_ratio = Gauge(
                'slo_compliance_ratio',
                'SLO compliance ratio (0-1)',
                ['slo_id', 'slo_name', 'service', 'component']
            )
            
            self.slo_error_budget_remaining = Gauge(
                'slo_error_budget_remaining',
                'Remaining error budget (0-1)',
                ['slo_id', 'slo_name', 'service']
            )
            
            self.slo_evaluations_total = Counter(
                'slo_evaluations_total',
                'Total SLO evaluations',
                ['slo_id', 'result']
            )
    
    async def evaluate_availability_slo(self, slo: SLOTarget) -> SLOMetric:
        """Evaluate availability SLO."""
        now = datetime.utcnow()
        window_start = now - slo.measurement_window
        
        # Query success/failure metrics from the measurement window
        success_key = f"slo_metrics:{slo.service}:success"
        failure_key = f"slo_metrics:{slo.service}:failure"
        
        # Get metrics from Redis (stored by other components)
        success_count = int(await self.redis.get(success_key) or 0)
        failure_count = int(await self.redis.get(failure_key) or 0)
        
        total_requests = success_count + failure_count
        if total_requests == 0:
            availability = 100.0  # No requests = 100% available
        else:
            availability = (success_count / total_requests) * 100
        
        compliance_ratio = availability / slo.target_value if slo.target_value > 0 else 0
        is_compliant = availability >= slo.target_value
        
        metric = SLOMetric(
            slo_id=slo.slo_id,
            timestamp=now,
            current_value=availability,
            target_value=slo.target_value,
            compliance_ratio=compliance_ratio,
            window_start=window_start,
            window_end=now,
            sample_count=total_requests,
            is_compliant=is_compliant,
            metadata={
                'success_count': success_count,
                'failure_count': failure_count
            }
        )
        
        # Record Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            self.slo_compliance_ratio.labels(
                slo_id=slo.slo_id,
                slo_name=slo.name,
                service=slo.service,
                component=slo.component or 'default'
            ).set(compliance_ratio)
            
            error_budget = 1.0 - compliance_ratio
            self.slo_error_budget_remaining.labels(
                slo_id=slo.slo_id,
                slo_name=slo.name,
                service=slo.service
            ).set(max(0, error_budget))
        
        return metric
    
    async def evaluate_latency_slo(self, slo: SLOTarget) -> SLOMetric:
        """Evaluate latency SLO (e.g., P95 < 100ms)."""
        now = datetime.utcnow()
        window_start = now - slo.measurement_window
        
        # Get latency measurements from Redis
        latency_key = f"slo_metrics:{slo.service}:latencies"
        latency_data = await self.redis.lrange(latency_key, 0, -1)
        
        if not latency_data:
            # No data available
            return SLOMetric(
                slo_id=slo.slo_id,
                timestamp=now,
                current_value=0.0,
                target_value=slo.target_value,
                compliance_ratio=1.0,
                window_start=window_start,
                window_end=now,
                sample_count=0,
                is_compliant=True
            )
        
        # Convert to float and calculate percentile
        latencies = [float(x) for x in latency_data]
        if len(latencies) > 0:
            # Calculate P95 latency
            p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        else:
            p95_latency = 0.0
        
        compliance_ratio = slo.target_value / p95_latency if p95_latency > 0 else 1.0
        is_compliant = p95_latency <= slo.target_value
        
        metric = SLOMetric(
            slo_id=slo.slo_id,
            timestamp=now,
            current_value=p95_latency,
            target_value=slo.target_value,
            compliance_ratio=compliance_ratio,
            window_start=window_start,
            window_end=now,
            sample_count=len(latencies),
            is_compliant=is_compliant,
            metadata={
                'p95_latency_ms': p95_latency,
                'sample_count': len(latencies)
            }
        )
        
        if PROMETHEUS_AVAILABLE:
            self.slo_compliance_ratio.labels(
                slo_id=slo.slo_id,
                slo_name=slo.name,
                service=slo.service,
                component=slo.component or 'default'
            ).set(compliance_ratio)
        
        return metric
    
    async def evaluate_error_rate_slo(self, slo: SLOTarget) -> SLOMetric:
        """Evaluate error rate SLO (e.g., <1% error rate)."""
        now = datetime.utcnow()
        window_start = now - slo.measurement_window
        
        # Get error metrics
        error_key = f"slo_metrics:{slo.service}:errors"
        total_key = f"slo_metrics:{slo.service}:requests"
        
        error_count = int(await self.redis.get(error_key) or 0)
        total_count = int(await self.redis.get(total_key) or 0)
        
        if total_count == 0:
            error_rate = 0.0
        else:
            error_rate = (error_count / total_count) * 100
        
        # For error rate, compliance means being UNDER the target
        compliance_ratio = slo.target_value / error_rate if error_rate > 0 else 1.0
        is_compliant = error_rate <= slo.target_value
        
        metric = SLOMetric(
            slo_id=slo.slo_id,
            timestamp=now,
            current_value=error_rate,
            target_value=slo.target_value,
            compliance_ratio=compliance_ratio,
            window_start=window_start,
            window_end=now,
            sample_count=total_count,
            is_compliant=is_compliant,
            metadata={
                'error_count': error_count,
                'total_count': total_count
            }
        )
        
        if PROMETHEUS_AVAILABLE:
            self.slo_compliance_ratio.labels(
                slo_id=slo.slo_id,
                slo_name=slo.name,
                service=slo.service,
                component=slo.component or 'default'
            ).set(compliance_ratio)
        
        return metric
    
    async def evaluate_slo(self, slo: SLOTarget) -> SLOMetric:
        """Evaluate an SLO based on its type."""
        try:
            if slo.slo_type == SLOType.AVAILABILITY:
                result = await self.evaluate_availability_slo(slo)
            elif slo.slo_type == SLOType.LATENCY:
                result = await self.evaluate_latency_slo(slo)
            elif slo.slo_type == SLOType.ERROR_RATE:
                result = await self.evaluate_error_rate_slo(slo)
            else:
                raise ValueError(f"Unsupported SLO type: {slo.slo_type}")
            
            # Record evaluation
            if PROMETHEUS_AVAILABLE:
                evaluation_result = "compliant" if result.is_compliant else "non_compliant"
                self.slo_evaluations_total.labels(
                    slo_id=slo.slo_id,
                    result=evaluation_result
                ).inc()
            
            # Store result in Redis for historical tracking
            result_key = f"slo_results:{slo.slo_id}"
            await self.redis.lpush(result_key, json.dumps(asdict(result), default=str))
            await self.redis.ltrim(result_key, 0, 999)  # Keep last 1000 results
            
            self._last_evaluations[slo.slo_id] = datetime.utcnow()
            
            logger.debug(f"Evaluated SLO {slo.slo_id}: {result.current_value}/{result.target_value} "
                        f"(compliant: {result.is_compliant})")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to evaluate SLO {slo.slo_id}: {e}")
            raise


class AlertManager:
    """Manages SLO-based alerts and notifications."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.redis = get_redis_client()
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable] = []
        
        # Prometheus metrics for alerting
        if PROMETHEUS_AVAILABLE:
            self.active_alerts_total = Gauge(
                'active_alerts_total',
                'Total active alerts',
                ['severity', 'service']
            )
            
            self.alerts_fired_total = Counter(
                'alerts_fired_total',
                'Total alerts fired',
                ['alert_type', 'severity', 'service']
            )
    
    def register_alert_handler(self, handler: Callable):
        """Register a handler for alert notifications."""
        self.alert_handlers.append(handler)
    
    async def evaluate_alert_conditions(self, slo: SLOTarget, metric: SLOMetric) -> List[Alert]:
        """Evaluate if alerts should be fired based on SLO metric."""
        alerts = []
        now = datetime.utcnow()
        
        # Check critical threshold
        if slo.alert_enabled and metric.compliance_ratio <= slo.critical_threshold:
            alert_id = f"{slo.slo_id}_critical"
            
            if alert_id not in self.active_alerts:
                alert = Alert(
                    alert_id=alert_id,
                    slo_id=slo.slo_id,
                    title=f"CRITICAL: {slo.name} SLO breach",
                    description=f"{slo.name} is at {metric.current_value:.2f}, "
                               f"below critical threshold of {slo.target_value * slo.critical_threshold:.2f}",
                    severity=AlertSeverity.CRITICAL,
                    state=AlertState.FIRING,
                    created_at=now,
                    updated_at=now,
                    current_value=metric.current_value,
                    threshold_value=slo.target_value * slo.critical_threshold,
                    impact="Service may be severely degraded",
                    tags={
                        'service': slo.service,
                        'slo_type': slo.slo_type.value,
                        'component': slo.component or 'default'
                    }
                )
                
                self.active_alerts[alert_id] = alert
                alerts.append(alert)
                
                await self._fire_alert(alert)
        
        # Check warning threshold
        elif (slo.alert_enabled and 
              slo.critical_threshold < metric.compliance_ratio <= slo.warning_threshold):
            
            alert_id = f"{slo.slo_id}_warning"
            
            if alert_id not in self.active_alerts:
                alert = Alert(
                    alert_id=alert_id,
                    slo_id=slo.slo_id,
                    title=f"WARNING: {slo.name} SLO approaching breach",
                    description=f"{slo.name} is at {metric.current_value:.2f}, "
                               f"below warning threshold of {slo.target_value * slo.warning_threshold:.2f}",
                    severity=AlertSeverity.HIGH,
                    state=AlertState.FIRING,
                    created_at=now,
                    updated_at=now,
                    current_value=metric.current_value,
                    threshold_value=slo.target_value * slo.warning_threshold,
                    impact="Service performance may be degrading",
                    tags={
                        'service': slo.service,
                        'slo_type': slo.slo_type.value,
                        'component': slo.component or 'default'
                    }
                )
                
                self.active_alerts[alert_id] = alert
                alerts.append(alert)
                
                await self._fire_alert(alert)
        
        # Check if alerts should be resolved
        else:
            # Resolve any active alerts for this SLO if conditions are met
            alerts_to_resolve = [
                alert for alert_id, alert in self.active_alerts.items()
                if alert.slo_id == slo.slo_id and alert.state == AlertState.FIRING
            ]
            
            for alert in alerts_to_resolve:
                await self.resolve_alert(alert.alert_id, "SLO compliance restored")
        
        return alerts
    
    async def _fire_alert(self, alert: Alert):
        """Fire an alert and notify handlers."""
        try:
            # Store alert in Redis
            alert_key = f"alert:{alert.alert_id}"
            await self.redis.setex(
                alert_key,
                int(timedelta(days=7).total_seconds()),  # Keep for 7 days
                json.dumps(asdict(alert), default=str)
            )
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                self.alerts_fired_total.labels(
                    alert_type='slo_breach',
                    severity=alert.severity.value,
                    service=alert.tags.get('service', 'unknown')
                ).inc()
            
            # Notify alert handlers
            for handler in self.alert_handlers:
                try:
                    await handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")
            
            logger.warning(f"ALERT FIRED: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to fire alert {alert.alert_id}: {e}")
    
    async def resolve_alert(self, alert_id: str, resolution_notes: str, resolved_by: str = "system"):
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.state = AlertState.RESOLVED
            alert.resolved_at = datetime.utcnow()
            alert.resolved_by = resolved_by
            alert.resolution_notes = resolution_notes
            alert.updated_at = datetime.utcnow()
            
            # Update in Redis
            alert_key = f"alert:{alert_id}"
            await self.redis.setex(
                alert_key,
                int(timedelta(days=7).total_seconds()),
                json.dumps(asdict(alert), default=str)
            )
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            # Notify handlers of resolution
            for handler in self.alert_handlers:
                try:
                    await handler(alert)
                except Exception as e:
                    logger.error(f"Alert resolution handler failed: {e}")
            
            logger.info(f"ALERT RESOLVED: {alert.title} - {resolution_notes}")
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get all currently active alerts."""
        return list(self.active_alerts.values())


class SLOManager:
    """Main SLO monitoring and management system."""
    
    def __init__(self):
        """Initialize SLO manager."""
        self.evaluator = SLOEvaluator()
        self.alert_manager = AlertManager()
        self.slo_definitions: Dict[str, SLOTarget] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Load default SLOs
        self._load_default_slos()
    
    def _load_default_slos(self):
        """Load default SLO definitions for the trading system."""
        
        # API Availability SLO
        self.slo_definitions['api_availability'] = SLOTarget(
            slo_id='api_availability',
            name='API Availability',
            description='API should be available 99.9% of the time',
            slo_type=SLOType.AVAILABILITY,
            target_value=99.9,
            measurement_window=timedelta(hours=1),
            evaluation_interval=timedelta(minutes=5),
            service='api',
            warning_threshold=0.99,
            critical_threshold=0.95,
            owner='platform-team'
        )
        
        # API Latency SLO
        self.slo_definitions['api_latency_p95'] = SLOTarget(
            slo_id='api_latency_p95',
            name='API P95 Latency',
            description='95% of API requests should complete within 500ms',
            slo_type=SLOType.LATENCY,
            target_value=500.0,  # milliseconds
            measurement_window=timedelta(minutes=30),
            evaluation_interval=timedelta(minutes=5),
            service='api',
            warning_threshold=0.95,
            critical_threshold=0.90,
            owner='platform-team'
        )
        
        # Trading Order Error Rate SLO
        self.slo_definitions['trading_error_rate'] = SLOTarget(
            slo_id='trading_error_rate',
            name='Trading Order Error Rate',
            description='Trading order error rate should be below 0.1%',
            slo_type=SLOType.ERROR_RATE,
            target_value=0.1,  # percentage
            measurement_window=timedelta(hours=4),
            evaluation_interval=timedelta(minutes=10),
            service='trading',
            component='order_execution',
            warning_threshold=0.95,
            critical_threshold=0.80,
            owner='trading-team'
        )
        
        # Market Data Freshness SLO
        self.slo_definitions['market_data_freshness'] = SLOTarget(
            slo_id='market_data_freshness',
            name='Market Data Freshness',
            description='Market data should be no more than 5 seconds old',
            slo_type=SLOType.FRESHNESS,
            target_value=5000.0,  # milliseconds
            measurement_window=timedelta(minutes=15),
            evaluation_interval=timedelta(minutes=2),
            service='market_data',
            warning_threshold=0.95,
            critical_threshold=0.90,
            owner='data-team'
        )
        
        # Model Accuracy SLO
        self.slo_definitions['model_accuracy'] = SLOTarget(
            slo_id='model_accuracy',
            name='ML Model Accuracy',
            description='ML model accuracy should be above 75%',
            slo_type=SLOType.ACCURACY,
            target_value=75.0,  # percentage
            measurement_window=timedelta(hours=24),
            evaluation_interval=timedelta(hours=1),
            service='ml_models',
            warning_threshold=0.95,
            critical_threshold=0.85,
            owner='ml-team'
        )
    
    async def start_monitoring(self):
        """Start SLO monitoring background task."""
        if self.monitoring_task and not self.monitoring_task.done():
            return
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("SLO monitoring started")
    
    async def stop_monitoring(self):
        """Stop SLO monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("SLO monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop that evaluates SLOs."""
        while True:
            try:
                current_time = datetime.utcnow()
                
                for slo_id, slo in self.slo_definitions.items():
                    # Check if it's time to evaluate this SLO
                    last_eval = self.evaluator._last_evaluations.get(slo_id)
                    
                    if (last_eval is None or 
                        current_time - last_eval >= slo.evaluation_interval):
                        
                        try:
                            # Evaluate SLO
                            metric = await self.evaluator.evaluate_slo(slo)
                            
                            # Check alert conditions
                            alerts = await self.alert_manager.evaluate_alert_conditions(slo, metric)
                            
                            if alerts:
                                logger.warning(f"SLO {slo_id} generated {len(alerts)} alerts")
                                
                        except Exception as e:
                            logger.error(f"Failed to evaluate SLO {slo_id}: {e}")
                
                # Wait before next evaluation cycle
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in SLO monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait a bit longer on error
    
    def register_slo(self, slo: SLOTarget):
        """Register a new SLO for monitoring."""
        self.slo_definitions[slo.slo_id] = slo
        logger.info(f"Registered SLO: {slo.name}")
    
    def get_slo_status(self) -> Dict[str, Any]:
        """Get current status of all SLOs."""
        status = {
            'slos': {},
            'active_alerts': len(self.alert_manager.active_alerts),
            'monitoring_active': self.monitoring_task is not None and not self.monitoring_task.done()
        }
        
        for slo_id, slo in self.slo_definitions.items():
            status['slos'][slo_id] = {
                'name': slo.name,
                'type': slo.slo_type.value,
                'target': slo.target_value,
                'service': slo.service,
                'last_evaluation': self.evaluator._last_evaluations.get(slo_id)
            }
        
        return status


# Global SLO manager instance
_slo_manager: Optional[SLOManager] = None


def get_slo_manager() -> SLOManager:
    """Get or create global SLO manager instance."""
    global _slo_manager
    if _slo_manager is None:
        _slo_manager = SLOManager()
    return _slo_manager


async def record_slo_metric(service: str, metric_type: str, value: float):
    """Record a metric for SLO evaluation."""
    try:
        redis = get_redis_client()
        
        if metric_type == 'success':
            await redis.incr(f"slo_metrics:{service}:success")
        elif metric_type == 'failure':
            await redis.incr(f"slo_metrics:{service}:failure")
        elif metric_type == 'error':
            await redis.incr(f"slo_metrics:{service}:errors")
        elif metric_type == 'request':
            await redis.incr(f"slo_metrics:{service}:requests")
        elif metric_type == 'latency':
            await redis.lpush(f"slo_metrics:{service}:latencies", str(value))
            await redis.ltrim(f"slo_metrics:{service}:latencies", 0, 9999)  # Keep last 10k
        
    except Exception as e:
        logger.error(f"Failed to record SLO metric: {e}")


# Example alert handler
async def log_alert_handler(alert: Alert):
    """Simple alert handler that logs alerts."""
    if alert.state == AlertState.FIRING:
        logger.warning(f"🚨 ALERT: {alert.title} - {alert.description}")
    elif alert.state == AlertState.RESOLVED:
        logger.info(f"✅ RESOLVED: {alert.title} - {alert.resolution_notes}")


# Initialize default alert handler
async def init_slo_monitoring():
    """Initialize SLO monitoring with default configuration."""
    slo_manager = get_slo_manager()
    slo_manager.alert_manager.register_alert_handler(log_alert_handler)
    await slo_manager.start_monitoring()
    
    logger.info("SLO monitoring initialized with default configuration")