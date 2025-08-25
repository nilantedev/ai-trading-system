#!/usr/bin/env python3
"""
System API Router - REST endpoints for system monitoring and management
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.models import (
    SystemHealthResponse, SystemHealth, ServiceHealth,
    RiskResponse, RiskMetrics, RiskAlert, RiskLevel,
    NewsResponse, NewsArticle, SentimentResponse, SentimentAnalysis,
    BaseResponse, ErrorResponse, PaginationParams
)
from api.main import verify_token, optional_auth, APIException
from trading_common import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=SystemHealthResponse,
    summary="Get system health status",
    description="Get comprehensive system health including all services"
)
async def get_system_health(
    detailed: bool = Query(True, description="Include detailed service metrics"),
    user: Optional[Dict[str, Any]] = Depends(optional_auth)
):
    """Get comprehensive system health status."""
    try:
        # Import all services for health checking
        services = {}
        service_errors = {}
        
        # Market Data Service
        try:
            from services.data_ingestion.market_data_service import get_market_data_service
            service = await get_market_data_service()
            if hasattr(service, 'get_service_health'):
                health = await service.get_service_health()
                services['market_data_service'] = ServiceHealth(**health)
        except Exception as e:
            service_errors['market_data_service'] = str(e)
        
        # Stream Processor
        try:
            from services.stream_processor.stream_processing_service import get_stream_processor
            service = await get_stream_processor()
            if hasattr(service, 'get_service_health'):
                health = await service.get_service_health()
                services['stream_processor'] = ServiceHealth(**health)
        except Exception as e:
            service_errors['stream_processor'] = str(e)
        
        # Indicator Service
        try:
            from services.indicator_engine.indicator_service import get_indicator_service
            service = await get_indicator_service()
            if hasattr(service, 'get_service_health'):
                health = await service.get_service_health()
                services['indicator_service'] = ServiceHealth(**health)
        except Exception as e:
            service_errors['indicator_service'] = str(e)
        
        # Signal Service
        try:
            from services.signal_generator.signal_generation_service import get_signal_service
            service = await get_signal_service()
            if hasattr(service, 'get_service_health'):
                health = await service.get_service_health()
                services['signal_service'] = ServiceHealth(**health)
        except Exception as e:
            service_errors['signal_service'] = str(e)
        
        # Risk Service
        try:
            from services.risk_monitor.risk_monitoring_service import get_risk_service
            service = await get_risk_service()
            if hasattr(service, 'get_service_health'):
                health = await service.get_service_health()
                services['risk_service'] = ServiceHealth(**health)
        except Exception as e:
            service_errors['risk_service'] = str(e)
        
        # Add mock services for demonstration
        mock_services = {
            'api_gateway': ServiceHealth(
                service='api_gateway',
                status='healthy',
                timestamp=datetime.utcnow(),
                metrics={
                    'requests_per_minute': 45,
                    'avg_response_time': 85,
                    'error_rate': 0.01,
                    'active_connections': 12
                },
                connections={
                    'database': True,
                    'cache': True,
                    'message_queue': True
                }
            ),
            'database': ServiceHealth(
                service='database',
                status='healthy',
                timestamp=datetime.utcnow(),
                metrics={
                    'connections': 8,
                    'queries_per_second': 25,
                    'avg_query_time': 12,
                    'cache_hit_rate': 0.85
                },
                connections={
                    'primary': True,
                    'replica': True
                }
            ),
            'message_queue': ServiceHealth(
                service='message_queue',
                status='healthy',
                timestamp=datetime.utcnow(),
                metrics={
                    'messages_per_second': 150,
                    'queue_depth': 25,
                    'consumer_lag': 2.5,
                    'producer_rate': 155
                },
                connections={
                    'brokers': True,
                    'zookeeper': True
                }
            )
        }
        
        # Merge mock services with real services
        all_services = {**services, **mock_services}
        
        # Add error services
        for service_name, error in service_errors.items():
            all_services[service_name] = ServiceHealth(
                service=service_name,
                status='error',
                timestamp=datetime.utcnow(),
                metrics={'error': error},
                connections={}
            )
        
        # Calculate summary statistics
        healthy_services = sum(1 for s in all_services.values() if s.status == 'healthy')
        error_services = sum(1 for s in all_services.values() if s.status == 'error')
        total_services = len(all_services)
        
        # Determine overall status
        if error_services == 0:
            overall_status = "healthy"
        elif error_services < total_services / 2:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        system_health = SystemHealth(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            uptime=86400.0,  # Mock uptime
            services=all_services,
            summary={
                'total_services': total_services,
                'healthy_services': healthy_services,
                'error_services': error_services
            }
        )
        
        return SystemHealthResponse(
            health=system_health,
            message=f"System status: {overall_status}"
        )
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise APIException(
            status_code=500,
            detail="Failed to get system health",
            error_code="HEALTH_CHECK_ERROR",
            context={"error": str(e)}
        )


@router.get(
    "/services",
    response_model=Dict[str, Any],
    summary="Get services status",
    description="Get status information for all system services"
)
async def get_services_status(
    service_name: Optional[str] = Query(None, description="Filter by specific service"),
    status_filter: Optional[str] = Query(None, description="Filter by status (healthy, degraded, error)"),
    user: Dict[str, Any] = Depends(verify_token)
):
    """Get detailed status for all services."""
    try:
        # This would integrate with actual service discovery/monitoring
        # For now, return mock data
        
        services_status = [
            {
                "name": "market_data_service",
                "status": "healthy",
                "uptime": "2d 14h 32m",
                "version": "1.0.0",
                "cpu_usage": 15.2,
                "memory_usage": 245.6,
                "requests_per_minute": 120,
                "error_rate": 0.001,
                "last_error": None,
                "dependencies": ["data_provider_service", "cache", "message_queue"]
            },
            {
                "name": "signal_service", 
                "status": "healthy",
                "uptime": "2d 14h 28m",
                "version": "1.0.0",
                "cpu_usage": 22.8,
                "memory_usage": 512.3,
                "requests_per_minute": 45,
                "error_rate": 0.002,
                "last_error": None,
                "dependencies": ["indicator_service", "market_data_service"]
            },
            {
                "name": "order_management_system",
                "status": "degraded",
                "uptime": "1d 6h 15m",
                "version": "1.0.0",
                "cpu_usage": 8.5,
                "memory_usage": 128.9,
                "requests_per_minute": 15,
                "error_rate": 0.05,
                "last_error": "2025-08-25T14:30:00Z",
                "dependencies": ["broker_service", "risk_service"]
            },
            {
                "name": "risk_service",
                "status": "healthy",
                "uptime": "2d 14h 30m",
                "version": "1.0.0",
                "cpu_usage": 18.7,
                "memory_usage": 342.1,
                "requests_per_minute": 200,
                "error_rate": 0.003,
                "last_error": None,
                "dependencies": ["market_data_service", "portfolio_service"]
            }
        ]
        
        # Apply filters
        filtered_services = services_status
        
        if service_name:
            filtered_services = [s for s in filtered_services if s["name"] == service_name]
        
        if status_filter:
            filtered_services = [s for s in filtered_services if s["status"] == status_filter]
        
        return {
            "success": True,
            "timestamp": datetime.utcnow(),
            "services": filtered_services,
            "count": len(filtered_services),
            "summary": {
                "healthy": len([s for s in services_status if s["status"] == "healthy"]),
                "degraded": len([s for s in services_status if s["status"] == "degraded"]),
                "error": len([s for s in services_status if s["status"] == "error"])
            },
            "message": f"Retrieved status for {len(filtered_services)} services"
        }
        
    except Exception as e:
        logger.error(f"Error getting services status: {e}")
        raise APIException(
            status_code=500,
            detail="Failed to get services status",
            error_code="SERVICES_STATUS_ERROR",
            context={"error": str(e)}
        )


@router.get(
    "/alerts",
    response_model=Dict[str, Any],
    summary="Get system alerts",
    description="Get current system alerts and notifications"
)
async def get_system_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity (low, medium, high, critical)"),
    service: Optional[str] = Query(None, description="Filter by service name"),
    hours: int = Query(24, ge=1, le=168, description="Hours of alert history"),
    resolved: bool = Query(False, description="Include resolved alerts"),
    pagination: PaginationParams = Depends(),
    user: Dict[str, Any] = Depends(verify_token)
):
    """Get system alerts and notifications."""
    try:
        # Import risk service for alerts
        from services.risk_monitor.risk_monitoring_service import get_risk_service
        
        try:
            risk_service = await get_risk_service()
            # In production: alerts = await risk_service.get_active_alerts()
        except Exception as e:
            logger.warning(f"Failed to get risk service: {e}")
        
        # Mock alerts data
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        mock_alerts = [
            {
                "alert_id": "ALT_001",
                "type": "SYSTEM_PERFORMANCE",
                "severity": "medium",
                "service": "market_data_service",
                "title": "High API Response Time",
                "description": "Average API response time exceeds 200ms threshold",
                "timestamp": datetime.utcnow() - timedelta(minutes=15),
                "resolved": False,
                "metrics": {
                    "current_response_time": 245.6,
                    "threshold": 200.0,
                    "duration_minutes": 15
                }
            },
            {
                "alert_id": "ALT_002",
                "type": "TRADING_RISK",
                "severity": "high",
                "service": "risk_service",
                "title": "Portfolio Risk Limit Exceeded",
                "description": "Portfolio risk score exceeds high threshold",
                "timestamp": datetime.utcnow() - timedelta(hours=1),
                "resolved": True,
                "metrics": {
                    "risk_score": 78.5,
                    "threshold": 75.0,
                    "affected_positions": ["TSLA", "NFLX"]
                }
            },
            {
                "alert_id": "ALT_003",
                "type": "MARKET_DATA",
                "severity": "critical",
                "service": "data_provider_service",
                "title": "Data Feed Interruption",
                "description": "Primary data feed interrupted for 5+ minutes",
                "timestamp": datetime.utcnow() - timedelta(hours=6),
                "resolved": True,
                "metrics": {
                    "interruption_duration": 7.5,
                    "affected_symbols": ["AAPL", "GOOGL", "TSLA"],
                    "backup_feed_active": True
                }
            },
            {
                "alert_id": "ALT_004",
                "type": "ORDER_EXECUTION",
                "severity": "low",
                "service": "order_management_system",
                "title": "Order Fill Delay",
                "description": "Market order fill time exceeded normal range",
                "timestamp": datetime.utcnow() - timedelta(hours=3),
                "resolved": False,
                "metrics": {
                    "fill_time_seconds": 12.5,
                    "normal_range": 5.0,
                    "order_id": "ORD_12345"
                }
            }
        ]
        
        # Apply filters
        filtered_alerts = mock_alerts
        
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a["severity"] == severity]
        
        if service:
            filtered_alerts = [a for a in filtered_alerts if a["service"] == service]
        
        if not resolved:
            filtered_alerts = [a for a in filtered_alerts if not a["resolved"]]
        
        # Filter by time
        filtered_alerts = [
            a for a in filtered_alerts 
            if a["timestamp"] >= cutoff_time
        ]
        
        # Apply pagination
        start_idx = pagination.offset
        end_idx = start_idx + pagination.size
        paginated_alerts = filtered_alerts[start_idx:end_idx]
        
        # Convert datetime objects to ISO strings
        for alert in paginated_alerts:
            alert["timestamp"] = alert["timestamp"].isoformat()
        
        return {
            "success": True,
            "timestamp": datetime.utcnow(),
            "alerts": paginated_alerts,
            "count": len(paginated_alerts),
            "total_alerts": len(filtered_alerts),
            "summary": {
                "critical": len([a for a in filtered_alerts if a["severity"] == "critical"]),
                "high": len([a for a in filtered_alerts if a["severity"] == "high"]),
                "medium": len([a for a in filtered_alerts if a["severity"] == "medium"]),
                "low": len([a for a in filtered_alerts if a["severity"] == "low"]),
                "unresolved": len([a for a in filtered_alerts if not a["resolved"]])
            },
            "message": f"Retrieved {len(paginated_alerts)} alerts"
        }
        
    except Exception as e:
        logger.error(f"Error getting system alerts: {e}")
        raise APIException(
            status_code=500,
            detail="Failed to get system alerts",
            error_code="ALERTS_FETCH_ERROR",
            context={"error": str(e)}
        )


@router.get(
    "/metrics",
    response_model=Dict[str, Any],
    summary="Get system metrics",
    description="Get comprehensive system performance metrics"
)
async def get_system_metrics(
    metric_type: str = Query("all", description="Metric type (performance, trading, risk, all)"),
    period: str = Query("1h", description="Time period (5m, 15m, 1h, 4h, 1d)"),
    user: Dict[str, Any] = Depends(verify_token)
):
    """Get comprehensive system metrics."""
    try:
        # Parse time period
        period_minutes = {
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "1d": 1440
        }
        
        minutes = period_minutes.get(period, 60)
        
        # Mock system metrics
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "period": period,
            "performance_metrics": {
                "api_requests_per_minute": 125.5,
                "avg_response_time_ms": 85.2,
                "error_rate": 0.0025,
                "throughput_rps": 45.8,
                "cpu_usage_percent": 18.7,
                "memory_usage_mb": 2048.6,
                "disk_usage_percent": 45.2,
                "network_io_mbps": 12.5
            },
            "trading_metrics": {
                "signals_generated_per_minute": 3.2,
                "orders_placed_per_minute": 1.8,
                "fill_rate": 0.95,
                "avg_fill_time_ms": 450.5,
                "portfolio_value": 125000.00,
                "daily_pnl": 125.50,
                "positions_count": 4,
                "cash_balance": 25000.00
            },
            "risk_metrics": {
                "portfolio_risk_score": 45.8,
                "active_alerts": 2,
                "var_1d": -2850.00,
                "max_drawdown": -2.85,
                "correlation_risk": 0.25,
                "concentration_risk": 0.35,
                "leverage_ratio": 1.2
            },
            "data_metrics": {
                "market_data_points_per_second": 250.5,
                "data_quality_score": 0.98,
                "feed_uptime": 0.999,
                "cache_hit_rate": 0.85,
                "storage_utilization": 0.62
            }
        }
        
        # Filter metrics based on type
        if metric_type != "all":
            key = f"{metric_type}_metrics"
            if key in metrics:
                filtered_metrics = {
                    "timestamp": metrics["timestamp"],
                    "period": metrics["period"],
                    key: metrics[key]
                }
            else:
                raise APIException(
                    status_code=400,
                    detail=f"Invalid metric type: {metric_type}",
                    error_code="INVALID_METRIC_TYPE"
                )
        else:
            filtered_metrics = metrics
        
        return {
            "success": True,
            "timestamp": datetime.utcnow(),
            "data": filtered_metrics,
            "message": f"Retrieved {metric_type} metrics for {period} period"
        }
        
    except APIException:
        raise
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise APIException(
            status_code=500,
            detail="Failed to get system metrics",
            error_code="METRICS_FETCH_ERROR",
            context={"error": str(e)}
        )


@router.get(
    "/config",
    response_model=Dict[str, Any],
    summary="Get system configuration",
    description="Get current system configuration settings"
)
async def get_system_config(
    section: Optional[str] = Query(None, description="Configuration section filter"),
    user: Dict[str, Any] = Depends(verify_token)
):
    """Get system configuration settings."""
    try:
        # Mock configuration data
        config = {
            "trading": {
                "max_position_size": 0.1,
                "max_order_value": 50000.0,
                "risk_limits": {
                    "max_portfolio_risk": 75.0,
                    "var_limit": -5000.0,
                    "max_drawdown": -0.05
                },
                "default_timeframe": "1min",
                "paper_trading": True
            },
            "market_data": {
                "primary_provider": "alpha_vantage",
                "backup_provider": "yahoo_finance",
                "update_frequency": 60,
                "cache_ttl": 300,
                "symbols_tracked": ["AAPL", "GOOGL", "TSLA", "SPY", "QQQ"]
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "rate_limit": 100,
                "cors_origins": ["*"],
                "authentication_required": True
            },
            "alerts": {
                "email_notifications": True,
                "slack_notifications": False,
                "severity_threshold": "medium",
                "max_alerts_per_hour": 10
            },
            "system": {
                "log_level": "INFO",
                "metrics_collection": True,
                "health_check_interval": 30,
                "backup_enabled": True,
                "maintenance_window": "02:00-04:00 UTC"
            }
        }
        
        # Filter by section if specified
        if section:
            if section in config:
                filtered_config = {section: config[section]}
            else:
                raise APIException(
                    status_code=400,
                    detail=f"Invalid configuration section: {section}",
                    error_code="INVALID_CONFIG_SECTION"
                )
        else:
            filtered_config = config
        
        return {
            "success": True,
            "timestamp": datetime.utcnow(),
            "configuration": filtered_config,
            "message": f"Retrieved system configuration{' for ' + section if section else ''}"
        }
        
    except APIException:
        raise
    except Exception as e:
        logger.error(f"Error getting system configuration: {e}")
        raise APIException(
            status_code=500,
            detail="Failed to get system configuration",
            error_code="CONFIG_FETCH_ERROR",
            context={"error": str(e)}
        )


@router.get(
    "/logs",
    response_model=Dict[str, Any],
    summary="Get system logs",
    description="Get recent system logs with filtering options"
)
async def get_system_logs(
    level: str = Query("INFO", description="Log level filter (DEBUG, INFO, WARNING, ERROR)"),
    service: Optional[str] = Query(None, description="Filter by service name"),
    hours: int = Query(1, ge=1, le=24, description="Hours of log history"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of log entries"),
    user: Dict[str, Any] = Depends(verify_token)
):
    """Get recent system logs."""
    try:
        # In production, this would read from actual log files or log aggregation system
        # For now, return mock log data
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        mock_logs = []
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        services = ["api", "market_data_service", "signal_service", "risk_service", "oms"]
        
        # Generate mock log entries
        for i in range(min(limit, 200)):
            log_entry = {
                "timestamp": (datetime.utcnow() - timedelta(minutes=i * 2)).isoformat(),
                "level": log_levels[i % len(log_levels)],
                "service": services[i % len(services)],
                "message": f"Sample log message {i + 1}",
                "details": {
                    "thread": f"Thread-{i % 10 + 1}",
                    "module": f"module_{i % 5 + 1}.py",
                    "line": 100 + (i % 50)
                }
            }
            mock_logs.append(log_entry)
        
        # Apply filters
        filtered_logs = mock_logs
        
        if level != "ALL":
            # Filter by level and above (ERROR > WARNING > INFO > DEBUG)
            level_hierarchy = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
            min_level = level_hierarchy.get(level, 1)
            filtered_logs = [
                log for log in filtered_logs 
                if level_hierarchy.get(log["level"], 0) >= min_level
            ]
        
        if service:
            filtered_logs = [log for log in filtered_logs if log["service"] == service]
        
        # Apply limit
        filtered_logs = filtered_logs[:limit]
        
        return {
            "success": True,
            "timestamp": datetime.utcnow(),
            "logs": filtered_logs,
            "count": len(filtered_logs),
            "filters": {
                "level": level,
                "service": service,
                "hours": hours
            },
            "message": f"Retrieved {len(filtered_logs)} log entries"
        }
        
    except Exception as e:
        logger.error(f"Error getting system logs: {e}")
        raise APIException(
            status_code=500,
            detail="Failed to get system logs",
            error_code="LOGS_FETCH_ERROR",
            context={"error": str(e)}
        )