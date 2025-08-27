#!/usr/bin/env python3
"""
Audit logging middleware for FastAPI.
Automatically logs all API requests and responses for compliance and security monitoring.
"""

import time
import json
import uuid
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import sys
from pathlib import Path

# Add shared directory to path
shared_dir = Path(__file__).parent.parent / "shared" / "python-common"
sys.path.append(str(shared_dir))

try:
    from trading_common.audit_logger import (  # type: ignore[import-not-found]
        get_audit_logger, AuditEventType, AuditSeverity, AuditContext, log_audit_event
    )
    from trading_common.logging import get_logger  # type: ignore[import-not-found]
except ImportError:
    # Fallback
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    
    # Mock audit functions
    async def log_audit_event(*args, **kwargs):
        pass
    
    class AuditEventType:
        DATA_ACCESSED = "data_accessed"
        PERMISSION_DENIED = "permission_denied"
        ORDER_CREATED = "order_created"
    
    class AuditSeverity:
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"
        CRITICAL = "CRITICAL"
    
    class AuditContext:
        pass

logger = get_logger(__name__)


class AuditMiddleware:
    """Middleware for comprehensive API request/response auditing."""
    
    def __init__(self, app):
        self.app = app
        
        # Configuration
        self.audit_all_requests = True
        self.audit_request_bodies = True
        self.audit_response_bodies = False  # Can contain sensitive data
        self.max_body_size = 10000  # Max bytes to log
        
        # Sensitive endpoints that require special handling
        self.sensitive_endpoints = {
            '/api/v1/auth/login',
            '/api/v1/auth/register', 
            '/api/v1/users/password',
            '/api/v1/orders',
            '/api/v1/portfolio',
            '/api/v1/positions'
        }
        
        # Endpoints that should not log request bodies (contain secrets)
        self.no_body_log_endpoints = {
            '/api/v1/auth/login',
            '/api/v1/auth/refresh',
            '/api/v1/users/password'
        }
        
        # Map HTTP methods to audit event types
        self.method_event_mapping = {
            'GET': AuditEventType.DATA_ACCESSED,
            'POST': AuditEventType.ORDER_CREATED,  # Default, can be refined
            'PUT': 'resource_updated',
            'DELETE': 'resource_deleted',
            'PATCH': 'resource_modified'
        }
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Process request through audit middleware."""
        # Reuse existing correlation ID if already set by upstream middleware (e.g., logging)
        existing_id = getattr(request.state, 'correlation_id', None)
        correlation_id = existing_id or str(uuid.uuid4())
        # Only set if not previously defined to avoid mismatches across layers
        if existing_id is None:
            request.state.correlation_id = correlation_id
        
        # Start timing
        start_time = time.time()
        
        # Extract request information
        request_info = await self._extract_request_info(request)
        
        # Create audit context
        context = AuditContext(
            request_id=correlation_id,
            correlation_id=correlation_id,
            ip_address=self._get_client_ip(request),
            user_agent=request.headers.get('user-agent'),
            api_endpoint=str(request.url.path),
            http_method=request.method
        )
        
        # Get user context if available
        await self._enrich_user_context(request, context)
        
        response = None
        error = None
        
        try:
            # Process request
            response = await call_next(request)
            
            # Log successful request
            await self._log_request_success(request_info, context, response, start_time)
            
        except Exception as e:
            error = e
            # Log failed request
            await self._log_request_error(request_info, context, e, start_time)
            
            # Create error response
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        
        return response
    
    async def _extract_request_info(self, request: Request) -> Dict[str, Any]:
        """Extract relevant information from the request."""
        info = {
            'method': request.method,
            'url': str(request.url),
            'path': request.url.path,
            'query_params': dict(request.query_params),
            'headers': dict(request.headers),
            'body': None
        }
        
        # Extract request body if needed and safe
        if (self.audit_request_bodies and 
            request.url.path not in self.no_body_log_endpoints and
            request.method in ['POST', 'PUT', 'PATCH']):
            
            try:
                body = await request.body()
                if body and len(body) < self.max_body_size:
                    # Try to parse as JSON
                    try:
                        info['body'] = json.loads(body.decode('utf-8'))
                        # Remove sensitive fields
                        info['body'] = self._sanitize_body(info['body'])
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        info['body'] = f"<binary_data_length:{len(body)}>"
                else:
                    info['body'] = f"<large_body_length:{len(body)}>" if body else None
            except Exception as e:
                logger.warning(f"Failed to extract request body: {e}")
                info['body'] = "<extraction_failed>"
        
        return info
    
    def _sanitize_body(self, body: Any) -> Any:
        """Remove sensitive information from request body."""
        if not isinstance(body, dict):
            return body
        
        sensitive_keys = {
            'password', 'secret', 'token', 'api_key', 'private_key',
            'ssn', 'social_security', 'credit_card', 'cvv'
        }
        
        sanitized = {}
        for key, value in body.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                sanitized[key] = "<redacted>"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_body(value)
            elif isinstance(value, list):
                sanitized[key] = [self._sanitize_body(item) for item in value]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        # Fallback to client address
        if hasattr(request, 'client') and request.client:
            return request.client.host
        
        return 'unknown'
    
    async def _enrich_user_context(self, request: Request, context: AuditContext):
        """Add user information to audit context if available."""
        try:
            # Check if user is available in request state (set by auth middleware)
            if hasattr(request.state, 'user'):
                user = request.state.user
                context.user_id = getattr(user, 'user_id', None)
                context.username = getattr(user, 'username', None)
                
                # Try to get session ID from auth header or session
                auth_header = request.headers.get('authorization')
                if auth_header and auth_header.startswith('Bearer '):
                    # Could extract session info from JWT token if needed
                    pass
        except Exception as e:
            logger.debug(f"Could not extract user context: {e}")
    
    async def _log_request_success(
        self, 
        request_info: Dict[str, Any], 
        context: AuditContext,
        response: Response,
        start_time: float
    ):
        """Log successful request."""
        elapsed_time = time.time() - start_time
        
        # Determine audit event type based on endpoint and method
        event_type = self._determine_event_type(request_info)
        
        # Create audit message
        message = f"API Request: {request_info['method']} {request_info['path']}"
        
        # Determine severity based on endpoint sensitivity
        severity = AuditSeverity.INFO
        if request_info['path'] in self.sensitive_endpoints:
            severity = AuditSeverity.WARNING
        
        # Build audit details
        details = {
            'request': {
                'method': request_info['method'],
                'path': request_info['path'],
                'query_params': request_info['query_params'],
                'body': request_info['body']
            },
            'response': {
                'status_code': response.status_code,
                'headers': dict(response.headers)
            },
            'performance': {
                'elapsed_time_ms': round(elapsed_time * 1000, 2)
            }
        }
        
        # Add compliance tags for sensitive operations
        tags = []
        if request_info['path'] in self.sensitive_endpoints:
            tags.append('sensitive_operation')
        if request_info['method'] in ['POST', 'PUT', 'DELETE']:
            tags.append('data_modification')
        
        await log_audit_event(
            event_type=event_type,
            message=message,
            context=context,
            severity=severity,
            details=details,
            tags=tags
        )
    
    async def _log_request_error(
        self,
        request_info: Dict[str, Any],
        context: AuditContext, 
        error: Exception,
        start_time: float
    ):
        """Log failed request."""
        elapsed_time = time.time() - start_time
        
        message = f"API Request Failed: {request_info['method']} {request_info['path']} - {str(error)}"
        
        details = {
            'request': {
                'method': request_info['method'],
                'path': request_info['path'],
                'query_params': request_info['query_params']
            },
            'error': {
                'type': type(error).__name__,
                'message': str(error)
            },
            'performance': {
                'elapsed_time_ms': round(elapsed_time * 1000, 2)
            }
        }
        
        await log_audit_event(
            event_type=AuditEventType.PERMISSION_DENIED,  # Or appropriate error type
            message=message,
            context=context,
            severity=AuditSeverity.ERROR,
            details=details,
            tags=['api_error', 'system_error']
        )
    
    def _determine_event_type(self, request_info: Dict[str, Any]) -> AuditEventType:
        """Determine appropriate audit event type based on request."""
        path = request_info['path']
        method = request_info['method']
        
        # Map specific endpoints to event types
        if '/auth/login' in path:
            return AuditEventType.USER_LOGIN
        elif '/auth/logout' in path:
            return AuditEventType.USER_LOGOUT
        elif '/orders' in path and method == 'POST':
            return AuditEventType.ORDER_CREATED
        elif '/orders' in path and method == 'DELETE':
            return AuditEventType.ORDER_CANCELLED
        elif '/portfolio' in path:
            return AuditEventType.PORTFOLIO_VIEWED
        elif '/users' in path and method == 'POST':
            return AuditEventType.USER_CREATED
        elif '/config' in path:
            return AuditEventType.CONFIG_CHANGED
        elif '/api-keys' in path and method == 'POST':
            return AuditEventType.API_KEY_CREATED
        elif method == 'GET':
            return AuditEventType.DATA_ACCESSED
        else:
            # Default based on method
            return self.method_event_mapping.get(method, AuditEventType.DATA_ACCESSED)


def create_audit_middleware():
    """Factory function to create audit middleware."""
    return AuditMiddleware


# Context manager for temporary audit context
class AuditContextManager:
    """Context manager for setting audit context in request processing."""
    
    def __init__(self, **context_data):
        self.context_data = context_data
        self.previous_context = None
    
    async def __aenter__(self):
        # Store current context and set new one
        # This would integrate with request-local storage
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        pass


def audit_context(**context_data):
    """Decorator to set audit context for a function."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with AuditContextManager(**context_data):
                return await func(*args, **kwargs)
        return wrapper
    return decorator