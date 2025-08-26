#!/usr/bin/env python3
"""
Comprehensive audit logging and compliance system.
Provides centralized audit logging for all system activities with compliance features.
"""

import asyncio
import json
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging

from .logging import get_logger
from .user_models import UserAuditLogs, Users
from .user_repository import UserRepository
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)


class AuditEventType(Enum):
    """Audit event types for comprehensive system tracking."""
    
    # Authentication & Authorization
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    LOGIN_FAILED = "login_failed"
    PASSWORD_CHANGED = "password_changed"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    
    # User Management
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    USER_ACTIVATED = "user_activated"
    USER_DEACTIVATED = "user_deactivated"
    ROLE_CHANGED = "role_changed"
    
    # Trading Activities
    ORDER_CREATED = "order_created"
    ORDER_EXECUTED = "order_executed"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_MODIFIED = "order_modified"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    
    # Portfolio Management
    PORTFOLIO_VIEWED = "portfolio_viewed"
    PORTFOLIO_REBALANCED = "portfolio_rebalanced"
    RISK_LIMIT_CHANGED = "risk_limit_changed"
    RISK_LIMIT_EXCEEDED = "risk_limit_exceeded"
    
    # System Configuration
    CONFIG_CHANGED = "config_changed"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    SECRET_ACCESSED = "secret_accessed"
    
    # Data Access
    DATA_ACCESSED = "data_accessed"
    DATA_EXPORTED = "data_exported"
    SENSITIVE_DATA_VIEWED = "sensitive_data_viewed"
    
    # ML & AI
    MODEL_TRAINED = "model_trained"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_PREDICTION = "model_prediction"
    FEATURE_ACCESSED = "feature_accessed"
    
    # System Events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SERVICE_FAILURE = "service_failure"
    SERVICE_RECOVERY = "service_recovery"
    
    # Compliance & Regulatory
    COMPLIANCE_CHECK = "compliance_check"
    REGULATORY_REPORT = "regulatory_report"
    AUDIT_EXPORT = "audit_export"
    DATA_RETENTION_POLICY = "data_retention_policy"


class AuditSeverity(Enum):
    """Audit event severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class AuditContext:
    """Context information for audit events."""
    user_id: Optional[str] = None
    username: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    service_name: Optional[str] = None
    api_endpoint: Optional[str] = None
    http_method: Optional[str] = None


@dataclass 
class AuditEvent:
    """Comprehensive audit event structure."""
    event_id: str
    event_type: AuditEventType
    event_timestamp: datetime
    severity: AuditSeverity
    message: str
    context: AuditContext
    details: Dict[str, Any] = field(default_factory=dict)
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    resource_id: Optional[str] = None
    resource_type: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class ComplianceRequirement(Enum):
    """Different compliance requirements."""
    SOX = "sox"  # Sarbanes-Oxley
    GDPR = "gdpr"  # General Data Protection Regulation
    FINRA = "finra"  # Financial Industry Regulatory Authority
    SEC = "sec"  # Securities and Exchange Commission
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act


@dataclass
class ComplianceRule:
    """Compliance rule configuration."""
    requirement: ComplianceRequirement
    rule_name: str
    description: str
    event_types: List[AuditEventType]
    retention_days: int
    require_approval: bool = False
    encrypt_data: bool = False
    immutable: bool = True


class AuditLogger:
    """Centralized audit logging system with compliance features."""
    
    def __init__(self, session_factory=None):
        self.session_factory = session_factory
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.audit_queue: asyncio.Queue = asyncio.Queue()
        self.is_processing = False
        self.buffer: List[AuditEvent] = []
        self.buffer_size = 100
        self.flush_interval = 30  # seconds
        
        # Initialize default compliance rules
        self._init_compliance_rules()
    
    def _init_compliance_rules(self):
        """Initialize default compliance rules."""
        # SOX compliance for financial data
        self.compliance_rules["sox_financial"] = ComplianceRule(
            requirement=ComplianceRequirement.SOX,
            rule_name="Financial Transaction Logging",
            description="All financial transactions must be logged for SOX compliance",
            event_types=[
                AuditEventType.ORDER_CREATED,
                AuditEventType.ORDER_EXECUTED,
                AuditEventType.ORDER_CANCELLED,
                AuditEventType.POSITION_OPENED,
                AuditEventType.POSITION_CLOSED,
                AuditEventType.PORTFOLIO_REBALANCED
            ],
            retention_days=2555,  # 7 years
            require_approval=True,
            encrypt_data=True,
            immutable=True
        )
        
        # FINRA compliance for trading activities
        self.compliance_rules["finra_trading"] = ComplianceRule(
            requirement=ComplianceRequirement.FINRA,
            rule_name="Trading Activity Monitoring",
            description="All trading activities must be monitored per FINRA requirements",
            event_types=[
                AuditEventType.ORDER_CREATED,
                AuditEventType.ORDER_EXECUTED,
                AuditEventType.ORDER_MODIFIED,
                AuditEventType.RISK_LIMIT_EXCEEDED,
                AuditEventType.MODEL_PREDICTION
            ],
            retention_days=2190,  # 6 years
            immutable=True
        )
        
        # GDPR compliance for user data
        self.compliance_rules["gdpr_privacy"] = ComplianceRule(
            requirement=ComplianceRequirement.GDPR,
            rule_name="User Data Protection",
            description="User data access and processing must be logged for GDPR compliance",
            event_types=[
                AuditEventType.USER_CREATED,
                AuditEventType.USER_UPDATED,
                AuditEventType.USER_DELETED,
                AuditEventType.SENSITIVE_DATA_VIEWED,
                AuditEventType.DATA_EXPORTED
            ],
            retention_days=2190,  # 6 years (or until deletion request)
            encrypt_data=True
        )
    
    async def start_processing(self):
        """Start the audit log processing loop."""
        if self.is_processing:
            return
        
        self.is_processing = True
        logger.info("Starting audit log processing")
        
        # Start background tasks
        asyncio.create_task(self._process_queue())
        asyncio.create_task(self._periodic_flush())
    
    async def stop_processing(self):
        """Stop audit log processing and flush remaining events."""
        self.is_processing = False
        await self._flush_buffer()
        logger.info("Audit log processing stopped")
    
    async def log_event(
        self,
        event_type: AuditEventType,
        message: str,
        context: Optional[AuditContext] = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        details: Optional[Dict[str, Any]] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """Log an audit event."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            event_timestamp=datetime.utcnow(),
            severity=severity,
            message=message,
            context=context or AuditContext(),
            details=details or {},
            old_values=old_values,
            new_values=new_values,
            resource_id=resource_id,
            resource_type=resource_type,
            tags=tags or []
        )
        
        # Add compliance tags based on rules
        await self._apply_compliance_rules(event)
        
        # Queue for processing
        await self.audit_queue.put(event)
        
        # Log to system logger as well
        logger.log(
            self._severity_to_log_level(severity),
            f"AUDIT [{event_type.value}] {message}",
            extra={
                'audit_event_id': event.event_id,
                'user_id': event.context.user_id,
                'ip_address': event.context.ip_address,
                'resource_id': resource_id
            }
        )
    
    async def _apply_compliance_rules(self, event: AuditEvent):
        """Apply compliance rules to the audit event."""
        for rule_name, rule in self.compliance_rules.items():
            if event.event_type in rule.event_types:
                # Add compliance tags
                event.tags.append(f"compliance:{rule.requirement.value}")
                event.tags.append(f"rule:{rule_name}")
                
                # Mark as requiring encryption if needed
                if rule.encrypt_data:
                    event.tags.append("encrypt:true")
                
                # Mark as immutable if required
                if rule.immutable:
                    event.tags.append("immutable:true")
                
                # Add retention information
                event.tags.append(f"retention_days:{rule.retention_days}")
    
    async def _process_queue(self):
        """Process audit events from the queue."""
        while self.is_processing:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(
                    self.audit_queue.get(), 
                    timeout=1.0
                )
                
                # Add to buffer
                self.buffer.append(event)
                
                # Flush if buffer is full
                if len(self.buffer) >= self.buffer_size:
                    await self._flush_buffer()
                    
            except asyncio.TimeoutError:
                # Timeout is normal, continue processing
                continue
            except Exception as e:
                logger.error(f"Error processing audit queue: {e}")
                await asyncio.sleep(1)
    
    async def _periodic_flush(self):
        """Periodically flush the audit buffer."""
        while self.is_processing:
            await asyncio.sleep(self.flush_interval)
            if self.buffer:
                await self._flush_buffer()
    
    async def _flush_buffer(self):
        """Flush audit events from buffer to storage."""
        if not self.buffer or not self.session_factory:
            return
        
        events_to_process = self.buffer.copy()
        self.buffer.clear()
        
        try:
            async with self.session_factory() as session:
                user_repo = UserRepository(session)
                
                for event in events_to_process:
                    await self._store_audit_event(user_repo, event)
                
                await session.commit()
                
            logger.debug(f"Flushed {len(events_to_process)} audit events to storage")
            
        except Exception as e:
            logger.error(f"Failed to flush audit buffer: {e}")
            # Put events back in buffer for retry
            self.buffer.extend(events_to_process)
    
    async def _store_audit_event(self, user_repo: UserRepository, event: AuditEvent):
        """Store a single audit event in the database."""
        try:
            event_data = {
                'audit_id': event.event_id,
                'user_id': event.context.user_id,
                'event_type': event.event_type.value,
                'event_timestamp': event.event_timestamp,
                'session_id': event.context.session_id,
                'ip_address': event.context.ip_address,
                'user_agent': event.context.user_agent,
                'details': {
                    'message': event.message,
                    'context': {
                        'request_id': event.context.request_id,
                        'correlation_id': event.context.correlation_id,
                        'service_name': event.context.service_name,
                        'api_endpoint': event.context.api_endpoint,
                        'http_method': event.context.http_method
                    },
                    'event_details': event.details,
                    'resource_id': event.resource_id,
                    'resource_type': event.resource_type,
                    'tags': event.tags
                },
                'old_values': event.old_values,
                'new_values': event.new_values,
                'severity': event.severity.value
            }
            
            await user_repo.log_audit_event(event_data)
            
        except Exception as e:
            logger.error(f"Failed to store audit event {event.event_id}: {e}")
            raise
    
    def _severity_to_log_level(self, severity: AuditSeverity) -> int:
        """Convert audit severity to Python log level."""
        mapping = {
            AuditSeverity.INFO: logging.INFO,
            AuditSeverity.WARNING: logging.WARNING,
            AuditSeverity.ERROR: logging.ERROR,
            AuditSeverity.CRITICAL: logging.CRITICAL
        }
        return mapping.get(severity, logging.INFO)
    
    async def get_audit_trail(
        self,
        session: AsyncSession,
        user_id: Optional[str] = None,
        event_types: Optional[List[AuditEventType]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit trail for compliance reporting."""
        user_repo = UserRepository(session)
        
        # Build query parameters (simplified - would need actual implementation in user_repo)
        query_params = {}
        if user_id:
            query_params['user_id'] = user_id
        if start_date:
            query_params['start_date'] = start_date
        if end_date:
            query_params['end_date'] = end_date
        
        # This would need to be implemented in UserRepository
        # For now, return empty list
        return []
    
    async def export_compliance_report(
        self,
        session: AsyncSession,
        compliance_requirement: ComplianceRequirement,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Export compliance report for auditors."""
        # Get relevant compliance rules
        relevant_rules = [
            rule for rule in self.compliance_rules.values()
            if rule.requirement == compliance_requirement
        ]
        
        if not relevant_rules:
            raise ValueError(f"No compliance rules found for {compliance_requirement.value}")
        
        # Collect all relevant event types
        event_types = []
        for rule in relevant_rules:
            event_types.extend(rule.event_types)
        
        event_types = list(set(event_types))  # Remove duplicates
        
        # Get audit trail
        audit_events = await self.get_audit_trail(
            session=session,
            event_types=event_types,
            start_date=start_date,
            end_date=end_date,
            limit=10000  # Large limit for compliance reports
        )
        
        # Generate report
        report = {
            'compliance_requirement': compliance_requirement.value,
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'rules_applied': [
                {
                    'rule_name': rule.rule_name,
                    'description': rule.description,
                    'retention_days': rule.retention_days,
                    'event_types': [et.value for et in rule.event_types]
                }
                for rule in relevant_rules
            ],
            'total_events': len(audit_events),
            'events': audit_events,
            'generated_at': datetime.utcnow().isoformat(),
            'report_id': str(uuid.uuid4())
        }
        
        # Log the compliance report export
        await self.log_event(
            event_type=AuditEventType.AUDIT_EXPORT,
            message=f"Compliance report exported for {compliance_requirement.value}",
            severity=AuditSeverity.INFO,
            details={
                'compliance_requirement': compliance_requirement.value,
                'report_period': f"{start_date.isoformat()} to {end_date.isoformat()}",
                'total_events': len(audit_events)
            }
        )
        
        return report


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(session_factory=None) -> AuditLogger:
    """Get or create the global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(session_factory)
    return _audit_logger


async def log_audit_event(
    event_type: AuditEventType,
    message: str,
    context: Optional[AuditContext] = None,
    severity: AuditSeverity = AuditSeverity.INFO,
    **kwargs
):
    """Convenience function for logging audit events."""
    audit_logger = get_audit_logger()
    await audit_logger.log_event(
        event_type=event_type,
        message=message,
        context=context,
        severity=severity,
        **kwargs
    )


# Decorator for automatic audit logging
def audit_action(
    event_type: AuditEventType,
    message_template: str = None,
    severity: AuditSeverity = AuditSeverity.INFO,
    include_args: bool = False,
    include_result: bool = False
):
    """Decorator to automatically audit function calls."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Generate message
            message = message_template or f"{func.__name__} called"
            
            # Build context and details
            details = {}
            if include_args:
                details['args'] = str(args)
                details['kwargs'] = {k: str(v) for k, v in kwargs.items()}
            
            # Get user context from kwargs if available
            context = AuditContext()
            if 'current_user' in kwargs:
                user = kwargs['current_user']
                context.user_id = getattr(user, 'user_id', None)
                context.username = getattr(user, 'username', None)
            
            try:
                result = await func(*args, **kwargs)
                
                if include_result:
                    details['result'] = str(result)
                
                # Log successful execution
                await log_audit_event(
                    event_type=event_type,
                    message=message,
                    context=context,
                    severity=severity,
                    details=details
                )
                
                return result
                
            except Exception as e:
                # Log failed execution
                await log_audit_event(
                    event_type=event_type,
                    message=f"{message} - FAILED: {str(e)}",
                    context=context,
                    severity=AuditSeverity.ERROR,
                    details={**details, 'error': str(e)}
                )
                raise
        
        def sync_wrapper(*args, **kwargs):
            # For sync functions, convert to async call
            import asyncio
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator