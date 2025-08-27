#!/usr/bin/env python3
"""
AI Trading System API - Main FastAPI Application (FIXED VERSION)
Consolidated startup/shutdown events and proper middleware ordering.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time
import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
import asyncio
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_common import get_settings  # type: ignore[import-not-found]
from shared.logging_config import configure_logging, get_logger, correlation_id_var
from trading_common.resilience import get_all_circuit_breakers  # type: ignore[import-not-found]

# Common tuple of transient/expected infrastructure errors we tolerate during
# phased startup/shutdown without treating them as fatal in non-prod envs.
TRANSIENT_ERRORS = (ConnectionError, TimeoutError, asyncio.TimeoutError, OSError)

# Initialize structured logging (configure once)
configure_logging()
logger = get_logger(__name__)
settings = get_settings()

# Validate production configuration on startup
try:
    settings.enforce_production_security()
    logger.info("Configuration validation passed")
except ValueError as e:
    logger.error("Configuration validation failed", error=str(e))
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
    token = correlation_id_var.set(correlation_id)
    request.state.correlation_id = correlation_id
    logger.info("API Request", method=request.method, url=str(request.url))
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log response with correlation ID
    logger.info("API Response", status_code=response.status_code, duration_ms=int(duration*1000), method=request.method, url=str(request.url))
    
    # Add response headers
    response.headers["X-Response-Time"] = f"{duration:.3f}s"
    response.headers["X-API-Version"] = "1.0.0"
    response.headers["X-Correlation-ID"] = correlation_id
    
    try:
        return response
    finally:
        # Reset correlation id contextvar
        correlation_id_var.reset(token)


# ==========================================
# CONSOLIDATED STARTUP EVENT - FIXED
# ==========================================

@app.on_event("startup")
async def consolidated_startup():
    """Initialize services with proper ordering and error handling."""
    app.state.start_time = time.time()
    logger.info("API starting up", phase="startup", emoji="🚀")
    
    # PHASE 1: Security validation FIRST - fail fast on security issues
    try:
        logger.info("Security validation phase", phase=1)
        from trading_common.security_validator import validate_deployment_security
        environment = settings.environment
        
        if not validate_deployment_security(environment):
            logger.critical("Security validation failed - cannot start application", phase=1, result="fail")
            raise SystemExit(1)

        logger.info("Security validation passed", phase=1, environment=environment, result="success")
    except ImportError:
        logger.warning("Security validator not found - skipping validation", phase=1, skipped=True)
    except Exception as e:  # noqa: BLE001 - Security validation may raise diverse runtime errors; we fail fast except in dev
        logger.critical("Security validation error", error=str(e), phase=1)
        if settings.environment != "development":
            raise SystemExit(1) from e
    
    # PHASE 2: Initialize persistent security store
    try:
        logger.info("Initializing persistent security store", phase=2)
        from trading_common.security_store import get_security_store
        await get_security_store()
        logger.info("Persistent security store initialized", phase=2, result="success")
    except ImportError as e:
        logger.warning("Security store module missing", error=str(e), phase=2, skipped=True)
    except TRANSIENT_ERRORS as e:
        logger.error("Transient error initializing security store", error=str(e), phase=2)
        if settings.environment in ["production", "staging"]:
            raise
        logger.warning("Continuing without persistent security store in development", phase=2, degraded=True)
    except Exception as e:  # noqa: BLE001 - Unexpected; continue in dev, raise in prod/staging
        logger.error("Unexpected error initializing security store", error=str(e), phase=2)
        if settings.environment in ["production", "staging"]:
            raise
        logger.warning("Continuing after unexpected security store error in development", phase=2, degraded=True)
    
    # PHASE 3: Initialize middleware in correct order
    # NOTE: Middleware added via decorators is applied in REVERSE order
    # So we add: Metrics -> Rate Limiting -> Auth (will be applied as Auth -> Rate -> Metrics)
    try:
        logger.info("Initializing middleware stack", phase=3)
        
        # Add audit middleware early (after logging middleware already defined by decorator)
        try:
            from api.audit_middleware import AuditMiddleware
            app.add_middleware(AuditMiddleware)
            logger.info("Audit middleware initialized", phase=3, component="audit", result="success")
        except ImportError:
            logger.warning("Audit middleware not found - continuing without audit logging", phase=3, component="audit", skipped=True)

        # Add metrics middleware (applied last)
        try:
            from api.metrics import create_metrics_middleware, get_metrics_handler
            metrics_middleware = await create_metrics_middleware()
            app.middleware("http")(metrics_middleware)
            
            # Add metrics endpoint
            metrics_handler = get_metrics_handler()
            app.get("/metrics")(metrics_handler)
            logger.info("Prometheus metrics middleware initialized", phase=3, component="metrics", result="success")
        except ImportError:
            logger.warning("Metrics module not found - continuing without metrics", phase=3, component="metrics", skipped=True)
        
        # Add rate limiting middleware (applied second)
        from api.rate_limiter import create_rate_limit_middleware
        rate_limiter_middleware = await create_rate_limit_middleware()
        app.middleware("http")(rate_limiter_middleware)
        logger.info("Rate limiting middleware initialized", phase=3, component="rate_limiter", result="success")

    except Exception as e:  # noqa: BLE001 - Middleware init failure grouped; allow partial startup in dev
        logger.error("Failed to initialize middleware", error=str(e), phase=3)
        if settings.environment in ["production", "staging"]:
            raise
        logger.warning("Continuing with limited middleware in development", phase=3, degraded=True)
    
    # PHASE 4: Initialize WebSocket streaming
    try:
        logger.info("Initializing WebSocket streaming", phase=4)
        from api.websocket_manager import start_websocket_streaming
        app.state.websocket_task = await start_websocket_streaming()
        logger.info("WebSocket streaming initialized", phase=4, result="success")
    except ImportError:
        logger.warning("WebSocket manager not found - continuing without WebSocket", phase=4, skipped=True)
    except TRANSIENT_ERRORS as e:
        logger.error("Transient WebSocket streaming init error", error=str(e), phase=4)
        if settings.environment == "production":
            logger.warning("WebSocket streaming transient failure - proceeding")
    except Exception as e:  # noqa: BLE001 - Unexpected streaming init error; non-fatal
        logger.error("Unexpected WebSocket streaming init error", error=str(e), phase=4)
        if settings.environment == "production":
            logger.warning("WebSocket streaming failed but not blocking production start")
    
    # PHASE 5: Initialize core trading services (lazy loading) + ML wiring
    try:
        logger.info("Preparing trading system services (lazy loading)", phase=5)
        # Services will be initialized on first request to avoid startup delays
        # This is a lazy loading approach for better startup performance
        logger.info("Trading services ready for lazy initialization", phase=5, result="pending_lazy")
        # Start drift monitoring background task & register production models
        try:
            from trading_common.drift_scheduler import start_drift_monitor  # type: ignore[import-not-found]
            from trading_common.model_registry import get_model_registry  # type: ignore[import-not-found]
            registry = await get_model_registry()
            rows = await registry.db.fetch_all("SELECT model_name, version FROM model_registry WHERE state='PRODUCTION'")
            models = [(r['model_name'], r['version']) for r in rows]
            interval = int(os.getenv('DRIFT_INTERVAL_SECONDS', '3600'))
            app.state.drift_task = await start_drift_monitor(interval_seconds=interval, models=models)
            logger.info("Drift monitor started", phase=5, models=len(models), interval_seconds=interval)
        except TRANSIENT_ERRORS as e:
            logger.warning("Transient drift monitor start issue", error=str(e), phase=5, degraded=True)
        except Exception as e:  # noqa: BLE001 - Drift monitor optional; unexpected error tolerated
            logger.warning("Drift monitor not started due to unexpected error", error=str(e), phase=5, degraded=True)

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
                logger.info("Registered default feature view", feature_view=default_view, feature_count=len(core_features), phase=5)
        except TRANSIENT_ERRORS as e:
            logger.warning("Transient issue registering feature view", error=str(e), phase=5)
        except Exception as e:  # noqa: BLE001 - Non-fatal feature registration problem
            logger.warning("Feature view registration skipped", error=str(e), phase=5)

        # Log startup banner summary
        try:
            from trading_common.feature_store import get_feature_store
            store = await get_feature_store()
            feature_count = len(store.feature_definitions)
            fv_count_row = await store.db.fetch_one("SELECT COUNT(*) AS c FROM feature_views")
            fv_count = fv_count_row['c'] if fv_count_row else 0
            logger.info("Startup ML Summary", features=feature_count, feature_views=fv_count, drift_monitor=('on' if hasattr(app.state,'drift_task') else 'off'), experiment_tracking="enabled", phase=5)
        except TRANSIENT_ERRORS as e:
            logger.debug("Startup ML summary logging transient issue", error=str(e), phase=5)
        except Exception as e:  # noqa: BLE001 - Summary logging failure is non-critical
            logger.debug("Startup ML summary logging failed", error=str(e), phase=5)
    except Exception as e:  # noqa: BLE001 - Final phase grouping to ensure startup proceeds when safe
        logger.error("Service initialization setup failed", error=str(e), phase=5)
    
    startup_duration = time.time() - app.state.start_time
    logger.info("API startup complete", startup_seconds=round(startup_duration,2), emoji="🎉")


# ==========================================
# CONSOLIDATED SHUTDOWN EVENT - FIXED
# ==========================================

@app.on_event("shutdown")
async def consolidated_shutdown():
    """Graceful shutdown with proper cleanup order."""
    logger.info("API shutting down", phase="shutdown", emoji="🛑")
    shutdown_start = time.time()
    
    # PHASE 1: Stop accepting new connections and cancel background tasks
    try:
        logger.info("Stopping background services", phase="shutdown-1")
        
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
            logger.info("WebSocket streaming stopped", phase="shutdown-1", result="success")
        except ImportError:
            logger.info("WebSocket manager not loaded", phase="shutdown-1", skipped=True)
        except TRANSIENT_ERRORS as e:
            logger.warning("Transient error stopping WebSocket streaming", error=str(e), phase="shutdown-1")
        except Exception as e:  # noqa: BLE001 - Unexpected; continue shutdown
            logger.warning("Error stopping WebSocket streaming", error=str(e), phase="shutdown-1")
    except Exception as e:  # noqa: BLE001 - Grouping unexpected shutdown phase errors
        logger.error("Error in shutdown phase 1", error=str(e), phase="shutdown-1")
    
    # PHASE 2: Close external connections (rate limiter, databases)
    try:
        logger.info("Closing external connections", phase="shutdown-2")

        # Close rate limiter connections
        try:
            from api.rate_limiter import get_rate_limiter
            limiter = await get_rate_limiter()
            await limiter.close()
            logger.info("Rate limiter connections closed", phase="shutdown-2")
        except ImportError:
            logger.info("Rate limiter not loaded", phase="shutdown-2", skipped=True)
        except TRANSIENT_ERRORS as e:
            logger.warning("Transient error closing rate limiter", error=str(e), phase="shutdown-2")
        except Exception as e:  # noqa: BLE001 - Non-fatal unexpected close error
            logger.warning("Error closing rate limiter", error=str(e), phase="shutdown-2")

        # Stop drift monitoring task
        try:
            if hasattr(app.state, 'drift_task'):
                from trading_common.drift_scheduler import stop_drift_monitor
                await stop_drift_monitor(app.state.drift_task)
        except TRANSIENT_ERRORS as e:
            logger.warning(f"⚠️ Transient error stopping drift monitor: {e}")
        except Exception as e:  # noqa: BLE001 - Unexpected; continue
            logger.warning(f"⚠️ Error stopping drift monitor: {e}")

        # Close security store connections
        try:
            from trading_common.security_store import get_security_store
            store = await get_security_store()
            await store.close()
            logger.info("Security store connections closed", phase="shutdown-2")
        except TRANSIENT_ERRORS as e:
            logger.warning("Transient error closing security store", error=str(e), phase="shutdown-2")
        except Exception as e:  # noqa: BLE001 - Unexpected; continue
            logger.warning("Error closing security store", error=str(e), phase="shutdown-2")

    except Exception as e:  # noqa: BLE001 - Grouping external connection shutdown errors
        logger.error("Error in shutdown phase 2", error=str(e), phase="shutdown-2")
    
    # PHASE 3: Final cleanup and logging
    try:
        logger.info("Final cleanup", phase="shutdown-3")
        # Any additional cleanup here
        uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        shutdown_time = time.time() - shutdown_start
        logger.info("Services cleanup completed", phase="shutdown-3")
        logger.info("Uptime & shutdown metrics", uptime_seconds=round(uptime,2), shutdown_seconds=round(shutdown_time,2))
    except Exception as e:  # noqa: BLE001 - Final cleanup resilience
        logger.error("Error in shutdown phase 3", error=str(e), phase="shutdown-3")
    
    logger.info("API shutdown complete", phase="shutdown", emoji="🏁")


# Import JWT authentication
from api.auth import get_current_active_user, get_optional_user  # type: ignore[import-not-found]

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
    except Exception:  # Broad catch: security store health non-critical
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


# Readiness endpoint (internal operational readiness)
@app.get("/ready")
async def readiness_check():
    """Aggregated readiness probe for orchestration / load balancers.
    Returns 200 only if critical subsystems are available.
    """
    components = {}
    overall_ok = True

    # Security store
    try:
        from trading_common.security_store import get_security_store
        store = await get_security_store()
        health = await store.get_store_health()
        components['security_store'] = health.get('status', 'unknown')
        if components['security_store'] != 'healthy':
            overall_ok = False
    except Exception as e:  # noqa: BLE001 - Component readiness failure captured
        components['security_store'] = f'error: {e}'
        overall_ok = False

    # Rate limiter
    try:
        from api.rate_limiter import get_rate_limiter
        rl = await get_rate_limiter()
        components['rate_limiter'] = 'connected' if rl.connected else 'degraded'
        if not rl.connected:
            overall_ok = False
    except Exception as e:  # noqa: BLE001 - Rate limiter readiness failure captured
        components['rate_limiter'] = f'error: {e}'
        overall_ok = False

    # Feature store & model registry summary (best-effort)
    try:
        from trading_common.feature_store import get_feature_store
        fs = await get_feature_store()
        components['feature_store'] = 'available' if fs else 'unavailable'
    except Exception as e:  # noqa: BLE001 - Feature store optional
        components['feature_store'] = f'error: {e}'
    # Model registry
    try:
        from trading_common.model_registry import get_model_registry
        reg = await get_model_registry()
        components['model_registry'] = 'available' if reg else 'unavailable'
    except Exception as e:  # noqa: BLE001 - Model registry optional
        components['model_registry'] = f'error: {e}'

    # Drift monitor
    components['drift_monitor'] = 'running' if hasattr(app.state, 'drift_task') else 'stopped'

    # Circuit breakers
    try:
        breakers = get_all_circuit_breakers()
        open_breakers = [n for n,s in breakers.items() if s.get('state') == 'open']
        components['circuit_breakers_open'] = len(open_breakers)
        if open_breakers:
            overall_ok = False
    except Exception as e:  # noqa: BLE001 - Circuit breaker enumeration tolerance
        components['circuit_breakers'] = f'error: {e}'

    status_code = 200 if overall_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={
            'status': 'ready' if overall_ok else 'degraded',
            'timestamp': datetime.utcnow().isoformat(),
            'components': components
        }
    )


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
            except TRANSIENT_ERRORS as e:
                return service_name, {"status": "transient_error", "error": str(e)}
            except Exception as e:  # noqa: BLE001 - Unexpected service health error
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
        
    except Exception as e:  # noqa: BLE001 - Aggregate health assembly safeguard
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
    import uvicorn  # Local import to avoid dependency at module import time
    
    # Production-ready configuration
    # Use string defaults for os.getenv to avoid type warnings (then cast)
    config = {
        "host": "0.0.0.0",
        "port": int(os.getenv("PORT", "8000")),
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
            "workers": int(os.getenv("WORKERS", "4")),
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
    except TRANSIENT_ERRORS as e:
        info['feature_store_error'] = f'transient: {e}'
    except Exception as e:  # noqa: BLE001 - Unexpected feature store error
        info['feature_store_error'] = str(e)
    # Drift monitor
    try:
        # Drift monitor status inferred from app state (stubbed scheduler)
        from trading_common.drift_scheduler import start_drift_monitor  # noqa: F401
        info['drift_monitor'] = 'running' if hasattr(app.state, 'drift_task') else 'stopped'
    except TRANSIENT_ERRORS as e:
        info['drift_monitor_error'] = f'transient: {e}'
    except Exception as e:  # noqa: BLE001 - Unexpected drift monitor error
        info['drift_monitor_error'] = str(e)
    # Experiment tracker
    try:
        from trading_common.experiment_tracking import get_experiment_tracker
        await get_experiment_tracker()
        info['experiment_tracking'] = 'available'
    except TRANSIENT_ERRORS as e:
        info['experiment_tracking'] = f'transient: {e}'
    except Exception as e:  # noqa: BLE001 - Unexpected experiment tracking error
        info['experiment_tracking'] = f'error: {e}'
    info['timestamp'] = datetime.utcnow().isoformat()
    return info