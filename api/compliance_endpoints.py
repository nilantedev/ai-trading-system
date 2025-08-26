#!/usr/bin/env python3
"""
Compliance and audit reporting API endpoints.
Provides access to audit trails, compliance reports, and regulatory data.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import sys
from pathlib import Path

# Add shared directory to path
shared_dir = Path(__file__).parent.parent / "shared" / "python-common"
sys.path.append(str(shared_dir))

try:
    from trading_common.audit_logger import (
        get_audit_logger, AuditEventType, AuditSeverity, ComplianceRequirement
    )
    from trading_common.logging import get_logger
    from trading_common.user_management import get_user_manager
except ImportError:
    # Fallback imports
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    
    def get_audit_logger():
        return None
    
    def get_user_manager():
        return None
    
    class AuditEventType:
        pass
    
    class ComplianceRequirement:
        SOX = "sox"
        FINRA = "finra"
        GDPR = "gdpr"

from api.auth import get_current_user, require_role, UserRole, User

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/compliance", tags=["compliance"])


class AuditTrailRequest(BaseModel):
    """Request model for audit trail queries."""
    user_id: Optional[str] = None
    event_types: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    severity: Optional[str] = None
    resource_type: Optional[str] = None
    limit: int = Field(default=100, le=1000)
    offset: int = Field(default=0, ge=0)


class ComplianceReportRequest(BaseModel):
    """Request model for compliance reports."""
    requirement: str = Field(..., description="Compliance requirement (sox, finra, gdpr, etc.)")
    start_date: datetime = Field(..., description="Report start date")
    end_date: datetime = Field(..., description="Report end date")
    format: str = Field(default="json", description="Report format (json, csv, pdf)")
    include_details: bool = Field(default=True, description="Include detailed event data")


class DataRetentionRequest(BaseModel):
    """Request model for data retention operations."""
    retention_policy: str
    cutoff_date: datetime
    dry_run: bool = Field(default=True, description="Preview changes without executing")


@router.get("/audit-trail")
async def get_audit_trail(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    event_types: Optional[str] = Query(None, description="Comma-separated event types"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    limit: int = Query(100, le=1000, description="Maximum records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """
    Get audit trail records with filtering options.
    Requires admin role for access to audit data.
    """
    try:
        audit_logger = get_audit_logger()
        user_manager = get_user_manager()
        
        if not audit_logger or not user_manager:
            raise HTTPException(
                status_code=503,
                detail="Audit logging system not available"
            )
        
        # Parse event types
        parsed_event_types = None
        if event_types:
            parsed_event_types = [
                event_type.strip() 
                for event_type in event_types.split(',')
            ]
        
        # Get audit trail (this would need session from user_manager)
        async with user_manager._session_factory() as session:
            audit_events = await audit_logger.get_audit_trail(
                session=session,
                user_id=user_id,
                event_types=parsed_event_types,
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
        
        return {
            "total_records": len(audit_events),
            "filters_applied": {
                "user_id": user_id,
                "event_types": parsed_event_types,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "severity": severity
            },
            "pagination": {
                "limit": limit,
                "offset": offset
            },
            "audit_events": audit_events
        }
        
    except Exception as e:
        logger.error(f"Error retrieving audit trail: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve audit trail"
        )


@router.get("/audit-trail/summary")
async def get_audit_summary(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """
    Get summary statistics for audit trail.
    Provides overview of audit activity for specified period.
    """
    try:
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        audit_logger = get_audit_logger()
        user_manager = get_user_manager()
        
        if not audit_logger or not user_manager:
            raise HTTPException(
                status_code=503,
                detail="Audit logging system not available"
            )
        
        async with user_manager._session_factory() as session:
            # Get audit events for the period
            audit_events = await audit_logger.get_audit_trail(
                session=session,
                start_date=start_date,
                end_date=end_date,
                limit=10000  # Large limit for summary
            )
        
        # Calculate summary statistics
        total_events = len(audit_events)
        
        # Group by event type
        event_type_counts = {}
        severity_counts = {}
        user_activity = {}
        daily_activity = {}
        
        for event in audit_events:
            # Event type counts
            event_type = event.get('event_type', 'unknown')
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            
            # Severity counts
            severity = event.get('severity', 'INFO')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # User activity
            user_id = event.get('user_id')
            if user_id:
                user_activity[user_id] = user_activity.get(user_id, 0) + 1
            
            # Daily activity
            event_date = event.get('event_timestamp', '').split('T')[0]  # Extract date part
            if event_date:
                daily_activity[event_date] = daily_activity.get(event_date, 0) + 1
        
        return {
            "summary_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": (end_date - start_date).days
            },
            "totals": {
                "total_events": total_events,
                "unique_users": len(user_activity),
                "unique_event_types": len(event_type_counts)
            },
            "event_type_breakdown": event_type_counts,
            "severity_breakdown": severity_counts,
            "top_active_users": dict(
                sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "daily_activity": daily_activity
        }
        
    except Exception as e:
        logger.error(f"Error generating audit summary: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate audit summary"
        )


@router.post("/reports/compliance")
async def generate_compliance_report(
    report_request: ComplianceReportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_role(UserRole.SUPER_ADMIN))
) -> Dict[str, Any]:
    """
    Generate compliance report for regulatory requirements.
    Requires super admin role due to sensitive nature of compliance data.
    """
    try:
        # Validate compliance requirement
        valid_requirements = [req.value for req in ComplianceRequirement]
        if report_request.requirement not in valid_requirements:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid compliance requirement. Must be one of: {valid_requirements}"
            )
        
        audit_logger = get_audit_logger()
        user_manager = get_user_manager()
        
        if not audit_logger or not user_manager:
            raise HTTPException(
                status_code=503,
                detail="Audit logging system not available"
            )
        
        # Generate report ID
        report_id = f"compliance_{report_request.requirement}_{int(datetime.utcnow().timestamp())}"
        
        # For immediate response, return report info and process in background
        if report_request.format in ["csv", "pdf"]:
            # Add background task for file generation
            background_tasks.add_task(
                _generate_compliance_file_report,
                report_id,
                report_request,
                current_user.user_id
            )
            
            return {
                "report_id": report_id,
                "status": "processing",
                "message": f"Compliance report generation started for {report_request.requirement}",
                "estimated_completion": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
                "download_url": f"/api/v1/compliance/reports/{report_id}/download"
            }
        
        # For JSON format, generate immediately
        async with user_manager._session_factory() as session:
            compliance_requirement = ComplianceRequirement(report_request.requirement)
            report = await audit_logger.export_compliance_report(
                session=session,
                compliance_requirement=compliance_requirement,
                start_date=report_request.start_date,
                end_date=report_request.end_date
            )
        
        return {
            "report_id": report_id,
            "status": "completed",
            "format": "json",
            "generated_at": datetime.utcnow().isoformat(),
            "generated_by": current_user.username,
            "report": report
        }
        
    except Exception as e:
        logger.error(f"Error generating compliance report: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate compliance report"
        )


async def _generate_compliance_file_report(
    report_id: str,
    report_request: ComplianceReportRequest,
    user_id: str
):
    """Background task to generate compliance report files."""
    try:
        # This would implement actual file generation
        # For now, just log the request
        logger.info(f"Generating compliance report {report_id} in {report_request.format} format")
        
        # Simulate processing time
        import asyncio
        await asyncio.sleep(30)
        
        # In real implementation:
        # 1. Generate the report data
        # 2. Convert to requested format (CSV/PDF)
        # 3. Store file securely
        # 4. Update report status
        # 5. Notify user (optional)
        
        logger.info(f"Compliance report {report_id} generation completed")
        
    except Exception as e:
        logger.error(f"Failed to generate compliance report {report_id}: {e}")


@router.get("/reports/{report_id}/status")
async def get_report_status(
    report_id: str,
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """
    Get the status of a compliance report generation.
    """
    try:
        # In real implementation, check report status in database
        return {
            "report_id": report_id,
            "status": "completed",  # processing, completed, failed
            "progress": 100,
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "download_url": f"/api/v1/compliance/reports/{report_id}/download"
        }
        
    except Exception as e:
        logger.error(f"Error checking report status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to check report status"
        )


@router.get("/data-retention/policies")
async def get_data_retention_policies(
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, Any]:
    """
    Get configured data retention policies.
    Shows what data will be retained for how long based on compliance requirements.
    """
    try:
        audit_logger = get_audit_logger()
        
        if not audit_logger:
            raise HTTPException(
                status_code=503,
                detail="Audit logging system not available"
            )
        
        # Get compliance rules from audit logger
        policies = []
        for rule_name, rule in audit_logger.compliance_rules.items():
            policies.append({
                "rule_name": rule.rule_name,
                "compliance_requirement": rule.requirement.value,
                "description": rule.description,
                "retention_days": rule.retention_days,
                "retention_years": round(rule.retention_days / 365, 1),
                "event_types": [et.value for et in rule.event_types],
                "immutable": rule.immutable,
                "requires_encryption": rule.encrypt_data
            })
        
        return {
            "data_retention_policies": policies,
            "total_policies": len(policies),
            "policy_types": list(set(p["compliance_requirement"] for p in policies))
        }
        
    except Exception as e:
        logger.error(f"Error retrieving retention policies: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve retention policies"
        )


@router.post("/data-retention/execute")
async def execute_data_retention(
    retention_request: DataRetentionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_role(UserRole.SUPER_ADMIN))
) -> Dict[str, Any]:
    """
    Execute data retention policy (delete old audit records).
    Requires super admin role due to destructive nature.
    """
    try:
        if not retention_request.dry_run:
            # Add background task for actual deletion
            background_tasks.add_task(
                _execute_data_retention_policy,
                retention_request,
                current_user.user_id
            )
        
        # For dry run, calculate what would be deleted
        affected_records = await _estimate_retention_impact(retention_request)
        
        return {
            "operation": "data_retention",
            "dry_run": retention_request.dry_run,
            "policy": retention_request.retention_policy,
            "cutoff_date": retention_request.cutoff_date.isoformat(),
            "estimated_affected_records": affected_records,
            "status": "scheduled" if not retention_request.dry_run else "estimated",
            "executed_by": current_user.username,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error executing data retention: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to execute data retention policy"
        )


async def _estimate_retention_impact(retention_request: DataRetentionRequest) -> int:
    """Estimate how many records would be affected by retention policy."""
    # This would query the database to count records older than cutoff_date
    # For now, return a mock estimate
    return 1000


async def _execute_data_retention_policy(retention_request: DataRetentionRequest, user_id: str):
    """Background task to execute data retention policy."""
    try:
        logger.info(f"Executing data retention policy: {retention_request.retention_policy}")
        
        # In real implementation:
        # 1. Query for records older than cutoff_date
        # 2. Respect compliance rules (some data cannot be deleted)
        # 3. Create backup before deletion (if required)
        # 4. Delete records in batches
        # 5. Log the retention action
        
        await asyncio.sleep(60)  # Simulate processing time
        
        logger.info(f"Data retention policy execution completed")
        
    except Exception as e:
        logger.error(f"Failed to execute data retention policy: {e}")


@router.get("/health")
async def get_compliance_health(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get compliance system health status.
    Shows audit logging status and compliance monitoring health.
    """
    try:
        audit_logger = get_audit_logger()
        user_manager = get_user_manager()
        
        health_status = {
            "status": "healthy",
            "components": {
                "audit_logger": "available" if audit_logger else "unavailable",
                "user_manager": "available" if user_manager else "unavailable",
                "database": "connected",  # Would check actual DB connection
                "compliance_rules": len(audit_logger.compliance_rules) if audit_logger else 0
            },
            "compliance_requirements": [
                req.value for req in ComplianceRequirement
            ] if audit_logger else [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Determine overall health
        if not audit_logger or not user_manager:
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error checking compliance health: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }