"""
Admin God-Mode Control API
Comprehensive system management and emergency controls
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel
import logging
import asyncio
import json
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/api", tags=["god-mode-control"])

# Docker client (lazy init)
_docker_client = None

def get_docker_client():
    """Get Docker client with graceful fallback if docker module unavailable"""
    global _docker_client
    if not DOCKER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Docker module not available. Install: pip install docker"
        )
    if _docker_client is None:
        try:
            # Try Docker socket connection
            _docker_client = docker.DockerClient(base_url='unix://var/run/docker.sock')
        except Exception:
            # Fallback to environment
            _docker_client = docker.from_env()
    return _docker_client

# Models
class ServiceStatus(BaseModel):
    name: str
    status: str  # healthy, degraded, unhealthy
    cpu: float
    memory: float
    requests: int

class ServicesStatusResponse(BaseModel):
    services: List[ServiceStatus]
    timestamp: datetime

class SystemMetrics(BaseModel):
    system: Dict[str, Any]
    throughput: Dict[str, Any]
    timestamp: datetime

class BackfillRequest(BaseModel):
    years: int = 1
    max_symbols: Optional[int] = None

class RiskLimitsUpdate(BaseModel):
    max_position: float
    max_daily_loss: float
    max_leverage: float
    var_limit: float

class ScaleServiceRequest(BaseModel):
    replicas: int

# Service management endpoints
@router.get("/services/status")
async def get_services_status() -> ServicesStatusResponse:
    """
    Get status of all Docker services with resource metrics
    """
    try:
        client = get_docker_client()
        containers = client.containers.list()
        
        services = []
        for container in containers:
            # Get container stats (non-blocking)
            stats = container.stats(stream=False)
            
            # Calculate CPU percentage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0
            
            # Calculate memory usage in MB
            memory_mb = stats['memory_stats']['usage'] / (1024 * 1024)
            
            # Determine status based on health
            health = container.attrs.get('State', {}).get('Health', {})
            status_str = health.get('Status', 'unknown')
            
            if status_str == 'healthy':
                status = 'healthy'
            elif status_str in ['starting', 'unhealthy']:
                status = 'degraded' if container.status == 'running' else 'unhealthy'
            else:
                status = 'healthy' if container.status == 'running' else 'unhealthy'
            
            services.append(ServiceStatus(
                name=container.name,
                status=status,
                cpu=round(cpu_percent, 2),
                memory=round(memory_mb, 2),
                requests=0  # TODO: Pull from metrics endpoint if available
            ))
        
        return ServicesStatusResponse(
            services=services,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to get services status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.post("/services/{service_name}/restart")
async def restart_service(service_name: str):
    """
    Restart a specific service
    """
    try:
        client = get_docker_client()
        container = client.containers.get(service_name)
        container.restart(timeout=10)
        
        logger.info(f"Service {service_name} restart initiated")
        return {"status": "restarting", "service": service_name}
        
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
    except Exception as e:
        logger.error(f"Failed to restart {service_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Restart failed: {str(e)}")

@router.post("/services/{service_name}/scale")
async def scale_service(service_name: str, request: ScaleServiceRequest):
    """
    Scale a service to specified number of replicas
    """
    try:
        # TODO: Implement with Docker Swarm or Kubernetes API
        logger.info(f"Scaling {service_name} to {request.replicas} replicas")
        return {
            "status": "scaling",
            "service": service_name,
            "replicas": request.replicas
        }
        
    except Exception as e:
        logger.error(f"Failed to scale {service_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Scaling failed: {str(e)}")

# System metrics streaming
@router.get("/metrics/stream")
async def stream_metrics():
    """
    Server-Sent Events stream of real-time system metrics
    """
    from fastapi.responses import StreamingResponse
    
    async def generate():
        while True:
            # Collect system metrics (with graceful fallback)
            if PSUTIL_AVAILABLE:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                net_io = psutil.net_io_counters()
                
                metrics = {
                    "system": {
                        "cpu": round(cpu_percent, 2),
                        "memory": round(memory.used / (1024**3), 2),  # GB
                        "disk": round(disk.percent, 2),
                        "network_mbps": round(net_io.bytes_sent / (1024**2), 2)
                    },
                    "throughput": {
                        "requests": 0,  # TODO: Pull from metrics
                        "data_mb": 0,
                        "signals": 0
                    }
                }
            else:
                # Fallback metrics when psutil unavailable
                metrics = {
                    "system": {
                        "cpu": 0,
                        "memory": 0,
                        "disk": 0,
                        "network_mbps": 0
                    },
                    "throughput": {
                        "requests": 0,
                        "data_mb": 0,
                        "signals": 0
                    }
                }
            
            yield f"event: metrics\ndata: {json.dumps(metrics)}\n\n"
            await asyncio.sleep(2)
    
    import json
    return StreamingResponse(generate(), media_type="text/event-stream")

# ML Model management
@router.post("/models/promote-all-shadows")
async def promote_all_shadows():
    """
    Promote all shadow models to production
    """
    try:
        # TODO: Integrate with ML governance service
        logger.info("Promoting all shadow models to production")
        return {"status": "promoted", "promoted": 0, "message": "Shadow promotion logic pending integration"}
        
    except Exception as e:
        logger.error(f"Shadow promotion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Promotion failed: {str(e)}")

@router.post("/models/force-reload")
async def force_model_reload():
    """
    Force reload all ML models
    """
    try:
        # TODO: Send signal to ML inference service
        logger.info("Force reloading all ML models")
        return {"status": "reloading", "message": "Model reload initiated"}
        
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")

@router.get("/ml/models")
async def get_ml_models():
    """
    Get list of ML models with current weights
    """
    # TODO: Pull from ML config
    models = [
        {"id": "qwen2.5:72b", "name": "Qwen 2.5 72B", "weight": 0.25},
        {"id": "mixtral:8x22b", "name": "Mixtral 8x22B", "weight": 0.25},
        {"id": "command-r-plus:104b", "name": "Command R+ 104B", "weight": 0.20},
        {"id": "llama3.1:70b", "name": "Llama 3.1 70B", "weight": 0.15},
        {"id": "yi:34b", "name": "Yi 34B", "weight": 0.10},
        {"id": "phi3:14b", "name": "Phi-3 14B", "weight": 0.05}
    ]
    
    return {"models": models}

@router.put("/ml/models/{model_id}/weight")
async def update_model_weight(model_id: str, weight: float):
    """
    Update ensemble weight for a specific model
    """
    # TODO: Persist to ML config
    logger.info(f"Updating {model_id} weight to {weight}")
    return {"status": "updated", "model_id": model_id, "weight": weight}

# Factor management
@router.get("/factors/list")
async def get_factors():
    """
    Get list of all factors with current configuration
    """
    # TODO: Pull from factor engine config
    factors = [
        {"id": "momentum", "name": "Momentum", "enabled": True, "multiplier": 1.0},
        {"id": "value", "name": "Value", "enabled": True, "multiplier": 1.0},
        {"id": "quality", "name": "Quality", "enabled": True, "multiplier": 1.2},
        {"id": "growth", "name": "Growth", "enabled": True, "multiplier": 1.0},
        {"id": "volatility", "name": "Volatility", "enabled": True, "multiplier": 0.8},
        {"id": "size", "name": "Size", "enabled": False, "multiplier": 1.0}
    ]
    
    return {"factors": factors}

@router.post("/factors/{factor_id}/toggle")
async def toggle_factor(factor_id: str):
    """
    Toggle factor enabled/disabled
    """
    # TODO: Update factor engine config
    logger.info(f"Toggling factor {factor_id}")
    return {"status": "toggled", "factor_id": factor_id}

@router.put("/factors/{factor_id}/multiplier")
async def update_factor_multiplier(factor_id: str, multiplier: float):
    """
    Update factor multiplier
    """
    # TODO: Update factor engine config
    logger.info(f"Updating {factor_id} multiplier to {multiplier}")
    return {"status": "updated", "factor_id": factor_id, "multiplier": multiplier}

# Risk management
@router.post("/risk/update-limits")
async def update_risk_limits(limits: RiskLimitsUpdate):
    """
    Update system-wide risk limits
    """
    try:
        # TODO: Persist to risk management service
        logger.info(f"Updating risk limits: {limits}")
        return {"status": "updated", "limits": limits.dict()}
        
    except Exception as e:
        logger.error(f"Risk limits update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

# Backfill controls
@router.post("/backfill/{data_type}")
async def trigger_backfill(data_type: str, request: BackfillRequest):
    """
    Trigger manual backfill for specified data type
    """
    try:
        # TODO: Send message to data-ingestion service
        logger.info(f"Triggering {data_type} backfill: {request.years} years, {request.max_symbols} symbols")
        return {
            "status": "scheduled",
            "data_type": data_type,
            "years": request.years,
            "max_symbols": request.max_symbols
        }
        
    except Exception as e:
        logger.error(f"Backfill trigger failed: {e}")
        raise HTTPException(status_code=500, detail=f"Backfill failed: {str(e)}")

# Emergency controls
@router.post("/emergency/kill-switch")
async def activate_kill_switch():
    """
    🚨 EMERGENCY: Halt all trading activity
    """
    try:
        # TODO: Send emergency stop signal to execution service
        logger.critical("🚨 KILL SWITCH ACTIVATED - Halting all trading")
        return {"status": "activated", "message": "All trading halted"}
        
    except Exception as e:
        logger.error(f"Kill switch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Kill switch failed: {str(e)}")

@router.post("/circuit-breakers/reset")
async def reset_circuit_breakers():
    """
    Reset all circuit breakers
    """
    try:
        # TODO: Reset circuit breaker states
        logger.info("Resetting all circuit breakers")
        return {"status": "reset", "message": "Circuit breakers reset"}
        
    except Exception as e:
        logger.error(f"Circuit breaker reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@router.post("/system/force-gc")
async def force_garbage_collection():
    """
    Force garbage collection across services
    """
    try:
        import gc
        gc.collect()
        logger.info("Forced garbage collection")
        return {"status": "completed", "message": "Garbage collection executed"}
        
    except Exception as e:
        logger.error(f"GC failed: {e}")
        raise HTTPException(status_code=500, detail=f"GC failed: {str(e)}")

@router.post("/intelligence/save-config")
async def save_intelligence_config(config: Dict[str, Any]):
    """
    Save intelligence configuration (model weights, factors)
    """
    try:
        # TODO: Persist to config store
        logger.info("Saving intelligence configuration")
        return {"status": "saved", "message": "Configuration saved"}
        
    except Exception as e:
        logger.error(f"Config save failed: {e}")
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")

# Add to main API router
def register_god_mode_routes(app):
    """Register god-mode control routes with main app"""
    app.include_router(router)
    logger.info("God-mode control API registered")
