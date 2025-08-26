#!/usr/bin/env python3
"""
AI Trading System API - Main FastAPI Application (FIXED VERSION)
Consolidated startup/shutdown events and proper middleware ordering.
"""

from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
import uvicorn
import logging
import time
import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
import asyncio
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_common import get_settings, get_logger
from trading_common.resilience import get_all_circuit_breakers

# Initialize logging
logger = get_logger(__name__)
settings = get_settings()

# Validate production configuration on startup
try:
    settings.enforce_production_security()
    logger.info("Configuration validation passed")
except ValueError as e:
    logger.error(f"Configuration validation failed: {e}")
    if settings.is_production:
        raise


# Create FastAPI app
app = FastAPI(
    title="AI Trading System API",
    description="Comprehensive REST API for AI-powered trading system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware - use unified settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=settings.security.cors_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Requested-With"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.security.trusted_hosts,
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    # HSTS for production
    if settings.is_production:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # FIXED: Content Security Policy (removed unsafe-inline)
    csp = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none'"
    )
    response.headers["Content-Security-Policy"] = csp
    
    return response


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


# Request logging middleware with correlation IDs
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log API requests and responses with correlation ID."""
    start_time = time.time()
    
    # Generate correlation ID for request tracking
    correlation_id = str(uuid.uuid4())[:8]
    
    # Store in request state
    request.state.correlation_id = correlation_id
    
    # Log request with correlation ID
    logger.info(
        f"[{correlation_id}] API Request: {request.method} {request.url}",
        extra={
            "correlation_id": correlation_id, 
            "method": request.method, 
            "url": str(request.url)
        }
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log response with correlation ID
    logger.info(
        f"[{correlation_id}] API Response: {response.status_code} "
        f"({duration:.3f}s) {request.method} {request.url}",
        extra={
            "correlation_id": correlation_id,
            "status_code": response.status_code,
            "duration_ms": int(duration * 1000),
            "method": request.method,
            "url": str(request.url)
        }
    )
    
    # Add response headers
    response.headers["X-Response-Time"] = f"{duration:.3f}s"
    response.headers["X-API-Version"] = "1.0.0"
    response.headers["X-Correlation-ID"] = correlation_id
    
    return response


# ==========================================
# CONSOLIDATED STARTUP EVENT - FIXED
# ==========================================

@app.on_event("startup")
async def consolidated_startup():
    """Initialize services with proper ordering and error handling."""
    app.state.start_time = time.time()
    logger.info("🚀 AI Trading System API starting up...")
    
    # PHASE 1: Security validation FIRST - fail fast on security issues
    try:
        logger.info("Phase 1: Security validation")
        from trading_common.security_validator import validate_deployment_security
        environment = settings.environment
        
        if not validate_deployment_security(environment):
            logger.critical("❌ Security validation failed - cannot start application")
            raise SystemExit(1)
        
        logger.info(f"✅ Security validation passed for {environment} environment")
    except ImportError:
        logger.warning("⚠️ Security validator not found - skipping validation")
    except SystemExit:
        raise  # Re-raise to actually exit
    except Exception as e:
        logger.critical(f"❌ Security validation error: {e}")
        if settings.environment != "development":
            raise SystemExit(1)
    
    # PHASE 2: Initialize persistent security store
    try:
        logger.info("Phase 2: Initializing persistent security store")
        from trading_common.security_store import get_security_store
        await get_security_store()
        logger.info("✅ Persistent security store initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize security store: {e}")
        if settings.environment in ["production", "staging"]:
            raise  # Required in production
        logger.warning("⚠️ Continuing without persistent security store in development")
    
    # PHASE 3: Initialize middleware in correct order
    # NOTE: Middleware added via decorators is applied in REVERSE order
    # So we add: Metrics -> Rate Limiting -> Auth (will be applied as Auth -> Rate -> Metrics)
    try:
        logger.info("Phase 3: Initializing middleware stack")
        
        # Add metrics middleware (applied last)
        try:
            from api.metrics import create_metrics_middleware, get_metrics_handler
            metrics_middleware = await create_metrics_middleware()
            app.middleware("http")(metrics_middleware)
            
            # Add metrics endpoint
            metrics_handler = get_metrics_handler()
            app.get("/metrics")(metrics_handler)
            logger.info("✅ Prometheus metrics middleware initialized")
        except ImportError:
            logger.warning("⚠️ Metrics module not found - continuing without metrics")
        
        # Add rate limiting middleware (applied second)
        from api.rate_limiter import create_rate_limit_middleware
        rate_limiter_middleware = await create_rate_limit_middleware()
        app.middleware("http")(rate_limiter_middleware)
        logger.info("✅ Rate limiting middleware initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize middleware: {e}")
        if settings.environment in ["production", "staging"]:
            raise
        logger.warning("⚠️ Continuing with limited middleware in development")
    
    # PHASE 4: Initialize WebSocket streaming
    try:
        logger.info("Phase 4: Initializing WebSocket streaming")
        from api.websocket_manager import start_websocket_streaming
        app.state.websocket_task = await start_websocket_streaming()
        logger.info("✅ WebSocket streaming initialized")
    except ImportError:
        logger.warning("⚠️ WebSocket manager not found - continuing without WebSocket")
    except Exception as e:
        logger.error(f"❌ Failed to initialize WebSocket streaming: {e}")
        if settings.environment == "production":
            logger.warning("WebSocket streaming failed but not blocking production start")
    
    # PHASE 5: Initialize core trading services (lazy loading) + ML wiring
    try:
        logger.info("Phase 5: Preparing trading system services (lazy loading)")
        # Services will be initialized on first request to avoid startup delays
        # This is a lazy loading approach for better startup performance
        logger.info("✅ Trading services ready for lazy initialization")
        # Start drift monitoring background task & register production models
        try:
            from trading_common.drift_scheduler import start_drift_monitor
            from trading_common.model_registry import get_model_registry, ModelState
            registry = await get_model_registry()
            rows = await registry.db.fetch_all("SELECT model_name, version FROM model_registry WHERE state='PRODUCTION'")
            models = [(r['model_name'], r['version']) for r in rows]
            interval = int(os.getenv('DRIFT_INTERVAL_SECONDS', '3600'))
            app.state.drift_task = await start_drift_monitor(interval_seconds=interval, models=models)
            logger.info("📈 Drift monitor started (models=%d interval=%ds)", len(models), interval)
        except Exception as e:
            logger.warning(f"Drift monitor not started: {e}")

        # Register default feature view if missing
        try:
            from trading_common.feature_store import get_feature_store
            store = await get_feature_store()
            default_view = os.getenv('DEFAULT_FEATURE_VIEW', 'core_technical')
            row = await store.db.fetch_one("SELECT 1 FROM feature_views WHERE view_name=%s AND version=%s", [default_view, '1'])
            if not row:
                core_features = ['sma_20', 'rsi_14']  # minimal set; extend later
                await store.register_feature_view(default_view, core_features, description='Core technical indicators v1')
                await store.materialize_feature_view(default_view, entity_ids=['AAPL','MSFT','GOOGL'], as_of=datetime.utcnow())
                logger.info("🧱 Registered default feature view '%s' with %d features", default_view, len(core_features))
        except Exception as e:
            logger.warning(f"Feature view registration skipped: {e}")

        # Log startup banner summary
        try:
            from trading_common.feature_store import get_feature_store
            store = await get_feature_store()
            feature_count = len(store.feature_definitions)
            fv_count_row = await store.db.fetch_one("SELECT COUNT(*) AS c FROM feature_views")
            fv_count = fv_count_row['c'] if fv_count_row else 0
            logger.info("🔧 Startup ML Summary: features=%d feature_views=%d drift_monitor=%s experiment_tracking=enabled", feature_count, fv_count, 'on' if hasattr(app.state,'drift_task') else 'off')
        except Exception as e:
            logger.debug(f"Startup ML summary logging failed: {e}")
    except Exception as e:
        logger.error(f"Service initialization setup failed: {e}")
    
    startup_duration = time.time() - app.state.start_time
    logger.info(f"🎉 API startup complete - {startup_duration:.2f}s")


# ==========================================
# CONSOLIDATED SHUTDOWN EVENT - FIXED
# ==========================================

@app.on_event("shutdown")
async def consolidated_shutdown():
    """Graceful shutdown with proper cleanup order."""
    logger.info("🛑 AI Trading System API shutting down...")
    shutdown_start = time.time()
    
    # PHASE 1: Stop accepting new connections and cancel background tasks
    try:
        logger.info("Phase 1: Stopping background services")
        
        # Stop WebSocket streaming
        try:
            from api.websocket_manager import stop_websocket_streaming
            await stop_websocket_streaming()
            
            # Cancel the WebSocket task if it exists
            if hasattr(app.state, 'websocket_task') and app.state.websocket_task:
                app.state.websocket_task.cancel()
                try:
                    await app.state.websocket_task
                except asyncio.CancelledError:
                    pass
            logger.info("✅ WebSocket streaming stopped")
        except ImportError:
            logger.info("ℹ️ WebSocket manager not loaded")
        except Exception as e:
            logger.warning(f"⚠️ Error stopping WebSocket streaming: {e}")
    except Exception as e:
        logger.error(f"❌ Error in shutdown phase 1: {e}")
    
    # PHASE 2: Close external connections (rate limiter, databases)
    try:
        logger.info("Phase 2: Closing external connections")
        
        # Close rate limiter connections
        try:
            from api.rate_limiter import get_rate_limiter
            limiter = await get_rate_limiter()
            await limiter.close()
            logger.info("✅ Rate limiter connections closed")
        except ImportError:
            logger.info("ℹ️ Rate limiter not loaded")
        except Exception as e:
            logger.warning(f"⚠️ Error closing rate limiter: {e}")

        # Stop drift monitoring task
        try:
            if hasattr(app.state, 'drift_task'):
                from trading_common.drift_scheduler import stop_drift_monitor
                await stop_drift_monitor(app.state.drift_task)
        except Exception as e:
            logger.warning(f"⚠️ Error stopping drift monitor: {e}")
        
        # Close security store connections
        try:
            from trading_common.security_store import get_security_store
            store = await get_security_store()
            await store.close()
            logger.info("✅ Security store connections closed")
        except Exception as e:
            logger.warning(f"⚠️ Error closing security store: {e}")
            
    except Exception as e:
        logger.error(f"❌ Error in shutdown phase 2: {e}")
    
    # PHASE 3: Final cleanup and logging
    try:
        logger.info("Phase 3: Final cleanup")
        
        # Any additional cleanup here
        uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        shutdown_time = time.time() - shutdown_start
        
        logger.info("✅ Services cleanup completed")
        logger.info(f"📊 Uptime: {uptime:.2f}s, Shutdown time: {shutdown_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Error in shutdown phase 3: {e}")
    
    logger.info("🏁 API shutdown complete")


# Import JWT authentication
from api.auth import get_current_active_user, get_optional_user, User

# Create compatibility aliases for legacy router imports
verify_token = get_current_active_user
optional_auth = get_optional_user


# Health check endpoint with circuit breaker status
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    circuit_breakers = get_all_circuit_breakers()
    
    # Check if any circuit breakers are open
    unhealthy_breakers = [
        name for name, state in circuit_breakers.items() 
        if state.get("state") == "open"
    ]
    
    health_status = "healthy" if not unhealthy_breakers else "degraded"
    
    # Check security store health
    security_store_health = "unknown"
    try:
        from trading_common.security_store import get_security_store
        store = await get_security_store()
        store_health = await store.get_store_health()
        security_store_health = store_health.get("status", "unknown")
    except Exception:
        security_store_health = "unavailable"
    
    return {
        "status": health_status,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
        "circuit_breakers": circuit_breakers,
        "unhealthy_breakers": unhealthy_breakers,
        "security_store": security_store_health
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
            "websocket": "/ws",
            "metrics": "/metrics"
        }
    }


# Service status endpoint with proper error handling
@app.get("/api/v1/health")
async def detailed_health_check():
    """Detailed health check with service status."""
    try:
        # Gather service health in parallel to prevent blocking
        service_health = {}
        overall_healthy = True
        
        # List of services to check (only if they exist)
        services_to_check = [
            ("market_data_service", "services.data_ingestion.market_data_service", "get_market_data_service"),
            ("stream_processor", "services.stream_processor.stream_processing_service", "get_stream_processor"),
            ("indicator_service", "services.indicator_engine.indicator_service", "get_indicator_service"),
            ("signal_service", "services.signal_generator.signal_generation_service", "get_signal_service"),
            ("risk_service", "services.risk_monitor.risk_monitoring_service", "get_risk_service"),
            ("metrics_service", "services.metrics.performance_metrics_service", "get_metrics_service"),
            ("broker_service", "services.broker_integration.broker_service", "get_broker_service"),
            ("data_provider_service", "services.data_provider.data_provider_service", "get_data_provider_service"),
            ("news_service", "services.news_integration.news_service", "get_news_service"),
            ("order_management_system", "services.execution.order_management_system", "get_order_management_system"),
        ]
        
        async def check_service_health(service_name, module_path, function_name):
            try:
                # Dynamic import with timeout
                module = __import__(module_path, fromlist=[function_name])
                get_service_func = getattr(module, function_name)
                
                # Get service with timeout
                service = await asyncio.wait_for(get_service_func(), timeout=5.0)
                
                if hasattr(service, 'get_service_health'):
                    health = await asyncio.wait_for(service.get_service_health(), timeout=3.0)
                    return service_name, health
                else:
                    return service_name, {"status": "available", "health_check": False}
                    
            except asyncio.TimeoutError:
                return service_name, {"status": "timeout", "error": "Health check timed out"}
            except ImportError:
                return service_name, {"status": "not_available", "error": "Service not loaded"}
            except Exception as e:
                return service_name, {"status": "error", "error": str(e)}
        
        # Run service health checks concurrently
        health_tasks = [
            check_service_health(name, module, func) 
            for name, module, func in services_to_check
        ]
        
        # Wait for all health checks with overall timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*health_tasks, return_exceptions=True),
                timeout=10.0
            )
            
            for result in results:
                if isinstance(result, tuple):
                    service_name, health = result
                    service_health[service_name] = health
                    if health.get("status") not in ["healthy", "available"]:
                        overall_healthy = False
                else:
                    # Exception occurred
                    overall_healthy = False
                    
        except asyncio.TimeoutError:
            service_health["_error"] = "Health check timeout"
            overall_healthy = False
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "services": service_health,
            "overall_healthy": overall_healthy
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


# Mount API routers (only if they exist)
try:
    from api.routers import auth, market_data, trading, portfolio, system, websocket
    
    app.include_router(auth.router, prefix="/api/v1", tags=["Authentication"])
    app.include_router(market_data.router, prefix="/api/v1", tags=["Market Data"])
    app.include_router(trading.router, prefix="/api/v1", tags=["Trading"])
    app.include_router(portfolio.router, prefix="/api/v1", tags=["Portfolio"])
    app.include_router(system.router, prefix="/api/v1", tags=["System"])
    app.include_router(websocket.router, tags=["WebSocket"])
    
    logger.info("✅ All API routers loaded")
except ImportError as e:
    logger.warning(f"⚠️ Some API routers not available: {e}")
    # Continue without missing routers


if __name__ == "__main__":
    import uvicorn
    
    # Production-ready configuration
    config = {
        "host": "0.0.0.0",
        "port": int(os.getenv("PORT", 8000)),
        "log_level": "info",
        "access_log": True,
        "server_header": False,  # Security: hide server info
        "date_header": False,   # Security: hide date info
    }
    
    # Development vs production settings
    if settings.environment == "development":
        config.update({
            "reload": True,
            "reload_dirs": ["./api", "./shared"],
        })
    else:
        config.update({
            "workers": int(os.getenv("WORKERS", 4)),
            "loop": "uvloop" if "uvloop" in str(uvicorn) else "asyncio",
            "http": "httptools" if "httptools" in str(uvicorn) else "h11",
        })
    
    logger.info(f"Starting server with config: {config}")
    uvicorn.run("main:app", **config)

# =====================
# ML STATUS ENDPOINT
# =====================
@app.get("/api/v1/ml/status")
async def ml_status():
    """Return status of ML/feature infrastructure (lightweight)."""
    info: Dict[str, Any] = {}
    # Feature store
    try:
        from trading_common.feature_store import get_feature_store
        fs = await get_feature_store()
        info['features_registered'] = len(fs.feature_definitions)
        row = await fs.db.fetch_one("SELECT COUNT(*) AS c FROM feature_views")
        info['feature_views'] = row['c'] if row else 0
    except Exception as e:
        info['feature_store_error'] = str(e)
    # Drift monitor
    try:
        from trading_common.drift_scheduler import start_drift_monitor
        # access singleton if already started
        monitor = getattr(start_drift_monitor, '_MONITOR', None)  # may not exist
        if hasattr(app.state, 'drift_task'):
            info['drift_monitor'] = 'running'
        else:
            info['drift_monitor'] = 'stopped'
    except Exception as e:
        info['drift_monitor_error'] = str(e)
    # Experiment tracker
    try:
        from trading_common.experiment_tracking import get_experiment_tracker
        tracker = await get_experiment_tracker()
        info['experiment_tracking'] = 'available'
    except Exception as e:
        info['experiment_tracking'] = f'error: {e}'
    info['timestamp'] = datetime.utcnow().isoformat()
    return info