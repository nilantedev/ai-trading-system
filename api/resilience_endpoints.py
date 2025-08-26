#!/usr/bin/env python3
"""
API endpoints for resilience monitoring and management.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add shared directory to path
shared_dir = Path(__file__).parent.parent / "shared" / "python-common"
sys.path.append(str(shared_dir))

try:
    from trading_common.resilience_monitor import get_resilience_monitor
    from trading_common.resilience import get_all_circuit_breakers
    from trading_common.logging import get_logger
except ImportError:
    # Fallback imports
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    
    def get_resilience_monitor():
        return None
    
    def get_all_circuit_breakers():
        return {}

from api.auth import get_current_user, require_role, UserRole, User

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/resilience", tags=["resilience"])


@router.get("/health")
async def get_resilience_health(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get overall resilience health status.
    Requires authentication.
    """
    try:
        monitor = get_resilience_monitor()
        if not monitor:
            return {
                "status": "unavailable",
                "message": "Resilience monitoring not available",
                "timestamp": None
            }
        
        health_status = monitor.get_health_status()
        current_metrics = monitor.get_current_metrics()
        
        return {
            "status": health_status.value,
            "timestamp": current_metrics.timestamp.isoformat() if current_metrics else None,
            "message": f"Resilience system is {health_status.value}",
            "active_alerts": len(monitor.active_alerts)
        }
        
    except Exception as e:
        logger.error(f"Error getting resilience health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get resilience health status")


@router.get("/metrics")
async def get_resilience_metrics(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get comprehensive resilience metrics.
    Requires authentication.
    """
    try:
        monitor = get_resilience_monitor()
        if not monitor:
            return {
                "error": "Resilience monitoring not available",
                "metrics": {}
            }
        
        return monitor.get_metrics_summary()
        
    except Exception as e:
        logger.error(f"Error getting resilience metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get resilience metrics")


@router.get("/circuit-breakers")
async def get_circuit_breakers(
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """
    Get detailed circuit breaker states.
    Requires admin role.
    """
    try:
        circuit_breakers = get_all_circuit_breakers()
        
        # Add summary statistics
        total_cbs = len(circuit_breakers)
        open_cbs = sum(1 for state in circuit_breakers.values() if state['state'] == 'open')
        half_open_cbs = sum(1 for state in circuit_breakers.values() if state['state'] == 'half_open')
        closed_cbs = total_cbs - open_cbs - half_open_cbs
        
        return {
            "summary": {
                "total": total_cbs,
                "open": open_cbs,
                "half_open": half_open_cbs,
                "closed": closed_cbs
            },
            "circuit_breakers": circuit_breakers
        }
        
    except Exception as e:
        logger.error(f"Error getting circuit breakers: {e}")
        raise HTTPException(status_code=500, detail="Failed to get circuit breaker states")


@router.get("/components")
async def get_component_stats(
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """
    Get statistics from all tracked resilience components.
    Requires admin role.
    """
    try:
        monitor = get_resilience_monitor()
        if not monitor:
            return {
                "error": "Resilience monitoring not available",
                "components": {}
            }
        
        return {
            "components": monitor.get_component_stats(),
            "tracked_count": len(monitor.tracked_components)
        }
        
    except Exception as e:
        logger.error(f"Error getting component stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get component statistics")


@router.get("/alerts")
async def get_resilience_alerts(
    include_resolved: bool = False,
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """
    Get resilience alerts (active and optionally resolved).
    Requires admin role.
    """
    try:
        monitor = get_resilience_monitor()
        if not monitor:
            return {
                "error": "Resilience monitoring not available",
                "alerts": []
            }
        
        active_alerts = [
            {
                "id": alert.alert_id,
                "component": alert.component,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved
            }
            for alert in monitor.active_alerts.values()
        ]
        
        result = {
            "active_alerts": active_alerts,
            "active_count": len(active_alerts)
        }
        
        if include_resolved:
            resolved_alerts = [
                {
                    "id": alert.alert_id,
                    "component": alert.component,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved,
                    "resolution_timestamp": alert.resolution_timestamp.isoformat() if alert.resolution_timestamp else None
                }
                for alert in monitor.alert_history[-50:]  # Last 50 alerts
                if alert.resolved
            ]
            result["resolved_alerts"] = resolved_alerts
            result["resolved_count"] = len(resolved_alerts)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting resilience alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get resilience alerts")


@router.post("/circuit-breakers/{breaker_name}/reset")
async def reset_circuit_breaker(
    breaker_name: str,
    current_user: User = Depends(require_role(UserRole.SUPER_ADMIN))
) -> Dict[str, Any]:
    """
    Reset a specific circuit breaker (force close).
    Requires super admin role.
    """
    try:
        # This would require additional functionality in the resilience module
        # to allow manual circuit breaker control
        return {
            "message": f"Circuit breaker reset functionality not yet implemented for {breaker_name}",
            "success": False
        }
        
    except Exception as e:
        logger.error(f"Error resetting circuit breaker {breaker_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset circuit breaker")


@router.get("/dashboard")
async def get_resilience_dashboard(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get comprehensive resilience dashboard data.
    Combines metrics, alerts, and circuit breaker status.
    """
    try:
        monitor = get_resilience_monitor()
        if not monitor:
            return {
                "error": "Resilience monitoring not available",
                "dashboard": {}
            }
        
        # Get all the data needed for a dashboard
        metrics_summary = monitor.get_metrics_summary()
        circuit_breakers = get_all_circuit_breakers()
        
        # Add component health indicators
        component_health = {}
        for name, component in monitor.tracked_components.items():
            try:
                if hasattr(component, 'get_resilience_stats'):
                    stats = component.get_resilience_stats()
                    error_rate = 0.0
                    if stats.get('requests_total', 0) > 0:
                        error_rate = stats.get('errors_total', 0) / stats['requests_total']
                    
                    if error_rate > 0.1:
                        health = "unhealthy"
                    elif error_rate > 0.05:
                        health = "degraded"
                    else:
                        health = "healthy"
                    
                    component_health[name] = {
                        "health": health,
                        "error_rate": error_rate,
                        "total_requests": stats.get('requests_total', 0),
                        "last_success": stats.get('last_success')
                    }
                else:
                    component_health[name] = {"health": "unknown"}
                    
            except Exception as e:
                component_health[name] = {
                    "health": "error",
                    "error": str(e)
                }
        
        dashboard_data = {
            **metrics_summary,
            "component_health": component_health,
            "circuit_breaker_details": circuit_breakers
        }
        
        return {
            "dashboard": dashboard_data,
            "generated_at": monitor.get_current_metrics().timestamp.isoformat() if monitor.get_current_metrics() else None
        }
        
    except Exception as e:
        logger.error(f"Error getting resilience dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to get resilience dashboard data")


# Health check endpoint that doesn't require authentication
@router.get("/ping")
async def resilience_ping() -> Dict[str, Any]:
    """
    Simple ping endpoint for resilience monitoring health.
    No authentication required.
    """
    try:
        monitor = get_resilience_monitor()
        available = monitor is not None
        
        return {
            "status": "ok",
            "resilience_monitoring": "available" if available else "unavailable",
            "timestamp": monitor.get_current_metrics().timestamp.isoformat() if available and monitor.get_current_metrics() else None
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "resilience_monitoring": "error"
        }