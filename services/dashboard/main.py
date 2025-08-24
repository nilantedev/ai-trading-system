"""
Admin Dashboard Service - FastAPI Application
Provides comprehensive control panel for AI Trading System
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from trading_common.config import get_settings
from trading_common.logging import get_logger
from trading_common.services.data_source_manager import get_data_source_manager
from trading_common.database.redis_client import get_redis_client
from trading_common.exceptions import TradingError

logger = get_logger(__name__)
settings = get_settings()

# Security
security = HTTPBearer()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Admin WebSocket connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Admin WebSocket disconnected. Active connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: dict):
        if self.active_connections:
            message_str = json.dumps(message)
            for connection in self.active_connections:
                try:
                    await connection.send_text(message_str)
                except Exception as e:
                    logger.error(f"Failed to send WebSocket message: {e}")

manager = ConnectionManager()

# Pydantic models
class SystemStatus(BaseModel):
    status: str
    uptime: str
    version: str
    environment: str
    active_services: int
    total_services: int

class DataSourceStatus(BaseModel):
    name: str
    type: str
    status: str
    priority: int
    cost: float
    description: str
    last_check: datetime

class EmergencyAction(BaseModel):
    action: str = Field(..., regex="^(emergency_stop|pause_trading|resume_trading|restart_service)$")
    service: Optional[str] = None
    reason: str

class ConfigUpdate(BaseModel):
    key: str
    value: Any
    service: Optional[str] = None

class APIKeyRequest(BaseModel):
    source_name: str
    api_key: str
    additional_keys: Optional[Dict[str, str]] = None
    test_connection: bool = True

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    logger.info("Starting Admin Dashboard Service")
    
    # Initialize data source manager
    try:
        data_manager = get_data_source_manager()
        await data_manager.initialize_available_sources()
        logger.info("Data source manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize data source manager: {e}")
    
    yield
    
    logger.info("Shutting down Admin Dashboard Service")

# Create FastAPI app
app = FastAPI(
    title="AI Trading System - Admin Dashboard",
    description="Comprehensive control panel for AI Trading System management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security dependency
async def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify admin authentication"""
    # In production, implement proper JWT verification
    if not credentials.credentials or len(credentials.credentials) < 10:
        raise HTTPException(status_code=401, detail="Invalid authentication")
    return {"admin": True, "user": "admin"}

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "admin-dashboard"
    }

# System overview endpoints
@app.get("/api/system/status", response_model=SystemStatus)
async def get_system_status(admin: dict = Depends(get_current_admin)):
    """Get comprehensive system status"""
    try:
        redis = get_redis_client()
        
        # Get system metrics
        uptime_info = await redis.get("system:uptime") or "0"
        active_services = await redis.get("system:active_services") or 0
        total_services = await redis.get("system:total_services") or 8
        
        return SystemStatus(
            status="operational",
            uptime=str(timedelta(seconds=int(uptime_info))),
            version=settings.service_version,
            environment=settings.environment,
            active_services=int(active_services),
            total_services=int(total_services)
        )
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data-sources/status")
async def get_data_sources_status(admin: dict = Depends(get_current_admin)):
    """Get all data sources status"""
    try:
        data_manager = get_data_source_manager()
        status_report = data_manager.get_system_status()
        
        sources = []
        for source_name, source_info in status_report.get("available_sources", {}).items():
            sources.append(DataSourceStatus(
                name=source_name,
                type=source_info["type"],
                status=source_info["status"],
                priority=source_info["priority"],
                cost=source_info["cost"],
                description=source_info["description"],
                last_check=datetime.utcnow()
            ))
        
        for source_name, source_info in status_report.get("unavailable_sources", {}).items():
            sources.append(DataSourceStatus(
                name=source_name,
                type=source_info["type"],
                status="unavailable",
                priority=source_info["priority"],
                cost=source_info["cost"],
                description=source_info["description"],
                last_check=datetime.utcnow()
            ))
        
        return {
            "sources": sources,
            "total_monthly_cost": status_report.get("total_monthly_cost", 0),
            "coverage": status_report.get("coverage", {}),
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get data sources status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Control endpoints
@app.post("/api/control/emergency")
async def emergency_action(
    action: EmergencyAction, 
    background_tasks: BackgroundTasks,
    admin: dict = Depends(get_current_admin)
):
    """Execute emergency actions"""
    logger.warning(f"Emergency action requested: {action.action} by admin - Reason: {action.reason}")
    
    try:
        redis = get_redis_client()
        
        if action.action == "emergency_stop":
            # Set emergency stop flag
            await redis.set("system:emergency_stop", "true")
            await redis.set("trading:engine:status", "emergency_stopped")
            message = "Emergency stop activated - All trading halted"
            
        elif action.action == "pause_trading":
            await redis.set("trading:engine:paused", "true")
            message = "Trading paused"
            
        elif action.action == "resume_trading":
            await redis.delete("trading:engine:paused")
            message = "Trading resumed"
            
        elif action.action == "restart_service":
            if not action.service:
                raise HTTPException(status_code=400, detail="Service name required for restart")
            # In production, this would trigger service restart
            message = f"Service {action.service} restart initiated"
        
        # Log action
        await redis.lpush("system:admin_actions", json.dumps({
            "action": action.action,
            "service": action.service,
            "reason": action.reason,
            "timestamp": datetime.utcnow().isoformat(),
            "admin": admin.get("user", "unknown")
        }))
        
        # Broadcast to connected WebSockets
        await manager.broadcast({
            "type": "emergency_action",
            "action": action.action,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {"status": "success", "message": message}
        
    except Exception as e:
        logger.error(f"Emergency action failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data-sources/add-key")
async def add_api_key(
    request: APIKeyRequest, 
    background_tasks: BackgroundTasks,
    admin: dict = Depends(get_current_admin)
):
    """Hot-swap add new API key"""
    try:
        import os
        
        # Set environment variables
        primary_key = f"{request.source_name.upper()}_API_KEY"
        os.environ[primary_key] = request.api_key
        
        if request.additional_keys:
            for key, value in request.additional_keys.items():
                os.environ[key] = value
        
        # Initialize connection
        data_manager = get_data_source_manager()
        
        if request.test_connection:
            await data_manager.add_new_api_key(request.source_name, force_reconnect=True)
        
        # Log the action
        redis = get_redis_client()
        await redis.lpush("system:admin_actions", json.dumps({
            "action": "add_api_key",
            "source": request.source_name,
            "timestamp": datetime.utcnow().isoformat(),
            "admin": admin.get("user", "unknown")
        }))
        
        # Broadcast update
        await manager.broadcast({
            "type": "api_key_added",
            "source": request.source_name,
            "status": "success",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "status": "success",
            "message": f"API key for {request.source_name} added successfully",
            "source": request.source_name
        }
        
    except Exception as e:
        logger.error(f"Failed to add API key for {request.source_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/logs")
async def get_system_logs(
    limit: int = 100,
    level: Optional[str] = None,
    admin: dict = Depends(get_current_admin)
):
    """Get recent system logs"""
    try:
        redis = get_redis_client()
        
        # Get logs from Redis
        logs = await redis.lrange("system:logs", 0, limit - 1)
        
        parsed_logs = []
        for log in logs:
            try:
                log_entry = json.loads(log)
                if level and log_entry.get("level", "").upper() != level.upper():
                    continue
                parsed_logs.append(log_entry)
            except json.JSONDecodeError:
                continue
        
        return {
            "logs": parsed_logs,
            "count": len(parsed_logs),
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/metrics")
async def get_system_metrics(admin: dict = Depends(get_current_admin)):
    """Get real-time system metrics"""
    try:
        redis = get_redis_client()
        
        # Get various metrics
        metrics = {}
        
        # CPU and Memory (mock data - in production, get from node-exporter)
        import psutil
        metrics["cpu_usage"] = psutil.cpu_percent(interval=1)
        metrics["memory_usage"] = psutil.virtual_memory().percent
        metrics["disk_usage"] = psutil.disk_usage('/').percent
        
        # Trading metrics
        metrics["portfolio_value"] = float(await redis.get("trading:portfolio:total_value") or 0)
        metrics["daily_pnl"] = float(await redis.get("trading:pnl:daily") or 0)
        metrics["total_trades"] = int(await redis.get("trading:trades:total_today") or 0)
        
        # Data source metrics
        metrics["active_data_sources"] = int(await redis.get("data_sources:active_count") or 0)
        metrics["data_latency_ms"] = float(await redis.get("data_sources:avg_latency_ms") or 0)
        
        return {
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@app.websocket("/ws/admin")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time admin updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(5)
            
            # Get current metrics
            redis = get_redis_client()
            
            update = {
                "type": "metrics_update",
                "data": {
                    "portfolio_value": float(await redis.get("trading:portfolio:total_value") or 0),
                    "active_trades": int(await redis.get("trading:active_trades") or 0),
                    "system_status": "operational"
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await manager.send_personal_message(json.dumps(update), websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Static files for dashboard UI
@app.get("/", response_class=HTMLResponse)
async def dashboard_home():
    """Serve admin dashboard HTML"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Trading System - Admin Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: #fff; }
            .header { border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }
            .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .card { background: #2a2a2a; padding: 20px; border-radius: 8px; border: 1px solid #444; }
            .metric { font-size: 2em; font-weight: bold; color: #4CAF50; }
            .emergency { background: #d32f2f; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            .status-ok { color: #4CAF50; }
            .status-warning { color: #ff9800; }
            .status-error { color: #f44336; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 AI Trading System - Admin Control Panel</h1>
            <p>Real-time monitoring and control interface</p>
        </div>
        
        <div class="cards">
            <div class="card">
                <h3>System Status</h3>
                <div id="system-status" class="status-ok">● Operational</div>
                <button class="emergency" onclick="emergencyStop()">🛑 Emergency Stop</button>
            </div>
            
            <div class="card">
                <h3>Portfolio Value</h3>
                <div id="portfolio-value" class="metric">$0.00</div>
            </div>
            
            <div class="card">
                <h3>Active Trades</h3>
                <div id="active-trades" class="metric">0</div>
            </div>
            
            <div class="card">
                <h3>Data Sources</h3>
                <div id="data-sources">Loading...</div>
            </div>
        </div>
        
        <script>
            const ws = new WebSocket('ws://localhost:8000/ws/admin');
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'metrics_update') {
                    document.getElementById('portfolio-value').textContent = 
                        '$' + data.data.portfolio_value.toFixed(2);
                    document.getElementById('active-trades').textContent = 
                        data.data.active_trades;
                }
            };
            
            function emergencyStop() {
                if (confirm('Are you sure you want to execute an emergency stop?')) {
                    fetch('/api/control/emergency', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer admin-token-placeholder'
                        },
                        body: JSON.stringify({
                            action: 'emergency_stop',
                            reason: 'Manual admin intervention'
                        })
                    }).then(response => response.json())
                      .then(data => alert(data.message));
                }
            }
            
            // Load data sources on page load
            fetch('/api/data-sources/status', {
                headers: {'Authorization': 'Bearer admin-token-placeholder'}
            }).then(response => response.json())
              .then(data => {
                  document.getElementById('data-sources').innerHTML = 
                      data.sources.length + ' configured';
              });
        </script>
    </body>
    </html>
    '''

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=8000,
        reload=settings.is_development,
        log_level=settings.monitoring.log_level.lower()
    )