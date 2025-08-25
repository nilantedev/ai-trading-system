#!/usr/bin/env python3
"""
AI Trading System API - Main FastAPI Application
Provides REST APIs and WebSocket endpoints for trading system access.
"""

from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
import uvicorn
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_common import get_settings, get_logger

# Initialize logging
logger = get_logger(__name__)
settings = get_settings()


# Create FastAPI app
app = FastAPI(
    title="AI Trading System API",
    description="Comprehensive REST API for AI-powered trading system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware - restrict origins in production
allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Requested-With"],
)

# Trusted host middleware - restrict hosts in production
allowed_hosts = os.getenv("TRUSTED_HOSTS", "localhost,127.0.0.1,0.0.0.0").split(",")
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=allowed_hosts
)


class APIException(HTTPException):
    """Custom API exception with detailed error information."""
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.context = context or {}


# Global exception handler
@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    """Handle custom API exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "code": exc.error_code,
                "context": exc.context,
                "timestamp": datetime.utcnow().isoformat(),
                "path": str(request.url)
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception in API: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "code": "INTERNAL_ERROR",
                "timestamp": datetime.utcnow().isoformat(),
                "path": str(request.url)
            }
        }
    )


# Request logging middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log API requests and responses."""
    start_time = time.time()
    
    # Log request
    logger.info(f"API Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log response
    logger.info(
        f"API Response: {response.status_code} "
        f"({duration:.3f}s) {request.method} {request.url}"
    )
    
    # Add response headers
    response.headers["X-Response-Time"] = f"{duration:.3f}s"
    response.headers["X-API-Version"] = "1.0.0"
    
    return response


# Redis-based rate limiting will be added in startup event


# Import JWT authentication
from api.auth import get_current_active_user, get_optional_user, User


# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    }


# Root endpoint
@app.get("/")
async def root():
    """API root endpoint with basic information."""
    return {
        "name": "AI Trading System API",
        "version": "1.0.0",
        "description": "REST API for AI-powered trading system",
        "timestamp": datetime.utcnow().isoformat(),
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        },
        "endpoints": {
            "health": "/health",
            "api_v1": "/api/v1",
            "websocket": "/ws"
        }
    }


# Service status endpoint
@app.get("/api/v1/health")
async def detailed_health_check():
    """Detailed health check with service status."""
    try:
        # Import services here to avoid circular imports
        from services.ingestion.market_data_service import get_market_data_service
        from services.stream_processor.stream_processing_service import get_stream_processor
        from services.indicator_engine.indicator_service import get_indicator_service
        from services.signal_generator.signal_generation_service import get_signal_service
        from services.risk_monitor.risk_monitoring_service import get_risk_service
        from services.metrics.performance_metrics_service import get_metrics_service
        from services.broker_integration.broker_service import get_broker_service
        from services.data_provider.data_provider_service import get_data_provider_service
        from services.news_integration.news_service import get_news_service
        from services.execution.order_management_system import get_order_management_system
        
        # Check service health
        services = {
            "market_data_service": get_market_data_service,
            "stream_processor": get_stream_processor,
            "indicator_service": get_indicator_service,
            "signal_service": get_signal_service,
            "risk_service": get_risk_service,
            "metrics_service": get_metrics_service,
            "broker_service": get_broker_service,
            "data_provider_service": get_data_provider_service,
            "news_service": get_news_service,
            "order_management_system": get_order_management_system
        }
        
        service_health = {}
        overall_healthy = True
        
        for service_name, get_service_func in services.items():
            try:
                service = await get_service_func()
                if hasattr(service, 'get_service_health'):
                    health = await service.get_service_health()
                    service_health[service_name] = health
                    if health.get('status') != 'healthy':
                        overall_healthy = False
                else:
                    service_health[service_name] = {
                        "status": "unknown",
                        "message": "Health check not implemented"
                    }
            except Exception as e:
                service_health[service_name] = {
                    "status": "error",
                    "error": str(e)
                }
                overall_healthy = False
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
            "services": service_health,
            "summary": {
                "total_services": len(services),
                "healthy_services": sum(1 for s in service_health.values() if s.get('status') == 'healthy'),
                "error_services": sum(1 for s in service_health.values() if s.get('status') == 'error')
            }
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    app.state.start_time = time.time()
    logger.info("AI Trading System API starting up...")
    
    try:
        # Initialize Prometheus metrics
        from api.metrics import create_metrics_middleware, get_metrics_handler
        metrics_middleware = await create_metrics_middleware()
        app.middleware("http")(metrics_middleware)
        
        # Add metrics endpoint
        metrics_handler = get_metrics_handler()
        app.get("/metrics")(metrics_handler)
        logger.info("Prometheus metrics initialized")
        
        # Initialize rate limiting
        from api.rate_limiter import create_rate_limit_middleware
        rate_limiter_middleware = await create_rate_limit_middleware()
        app.middleware("http")(rate_limiter_middleware)
        logger.info("Redis-based rate limiting initialized")
        
        # Initialize WebSocket streaming
        from api.websocket_manager import start_websocket_streaming
        app.state.websocket_task = await start_websocket_streaming()
        logger.info("WebSocket streaming initialized")
        
        # Initialize core services
        logger.info("Initializing trading system services...")
        
        # Services will be initialized on first request to avoid startup delays
        # This is a lazy loading approach for better startup performance
        
        logger.info("API startup complete")
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("AI Trading System API shutting down...")
    
    try:
        # Stop WebSocket streaming
        from api.websocket_manager import stop_websocket_streaming
        try:
            await stop_websocket_streaming()
            # Cancel the WebSocket task if it exists
            if hasattr(app.state, 'websocket_task') and app.state.websocket_task:
                app.state.websocket_task.cancel()
                try:
                    await app.state.websocket_task
                except asyncio.CancelledError:
                    pass
            logger.info("WebSocket streaming stopped")
        except Exception as e:
            logger.warning(f"Error stopping WebSocket streaming: {e}")
        
        # Cleanup rate limiter
        from api.rate_limiter import get_rate_limiter
        try:
            limiter = await get_rate_limiter()
            await limiter.close()
            logger.info("Rate limiter closed")
        except Exception as e:
            logger.warning(f"Error closing rate limiter: {e}")
        
        # Cleanup services if needed
        logger.info("API shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Mount API routers
from api.routers import auth, market_data, trading, portfolio, system, websocket

app.include_router(auth.router, prefix="/api/v1", tags=["Authentication"])
app.include_router(market_data.router, prefix="/api/v1", tags=["Market Data"])
app.include_router(trading.router, prefix="/api/v1", tags=["Trading"])
app.include_router(portfolio.router, prefix="/api/v1", tags=["Portfolio"])
app.include_router(system.router, prefix="/api/v1", tags=["System"])
app.include_router(websocket.router, tags=["WebSocket"])


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host=settings.get("api", {}).get("host", "0.0.0.0"),
        port=settings.get("api", {}).get("port", 8000),
        reload=True,
        log_level="info"
    )