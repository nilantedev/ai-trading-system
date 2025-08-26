#!/usr/bin/env python3
"""
Disaster Recovery automation for AI Trading System.
Handles automated backup scheduling, health monitoring, and recovery procedures.
"""

import os
import sys
import asyncio
import logging
import json
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import subprocess
import argparse

# Add shared directory to path
sys.path.append(str(Path(__file__).parent.parent / "shared" / "python-common"))

try:
    from trading_common.backup_manager import EnhancedBackupManager, BackupConfig
    from trading_common.audit_logger import log_audit_event, AuditEventType, AuditSeverity, AuditContext
    from trading_common.logging import get_logger
except ImportError as e:
    logging.error(f"Failed to import trading_common modules: {e}")
    logging.error("Make sure you're running from the correct directory and dependencies are installed")
    sys.exit(1)

logger = get_logger(__name__)


@dataclass
class DisasterRecoveryConfig:
    """Configuration for disaster recovery operations."""
    # Backup scheduling
    backup_schedule: str = "0 2 * * *"  # Daily at 2 AM
    backup_retention_days: int = 30
    
    # Health monitoring
    health_check_interval: int = 300  # 5 minutes
    max_consecutive_failures: int = 3
    
    # Recovery thresholds
    disk_space_threshold: float = 0.85  # 85% full
    database_connection_timeout: int = 30
    service_health_timeout: int = 60
    
    # Notification settings
    alert_webhook_url: Optional[str] = None
    email_recipients: List[str] = None
    
    # Recovery procedures
    auto_recovery_enabled: bool = False
    max_recovery_attempts: int = 3
    recovery_cooldown_minutes: int = 60
    
    # Compliance settings
    audit_all_operations: bool = True
    encrypt_backups: bool = True
    verify_backup_integrity: bool = True
    
    def __post_init__(self):
        if self.email_recipients is None:
            self.email_recipients = []


class DisasterRecoveryManager:
    """Comprehensive disaster recovery and business continuity manager."""
    
    def __init__(self, config: DisasterRecoveryConfig):
        self.config = config
        self.backup_manager = None
        self.last_health_check = None
        self.consecutive_failures = 0
        self.recovery_attempts = {}
        self.system_state = "healthy"
        
        # Initialize backup manager
        backup_config = BackupConfig(
            encrypt_backups=config.encrypt_backups,
            verify_backups=config.verify_backup_integrity,
            enable_compliance_backup=True
        )
        self.backup_manager = EnhancedBackupManager(backup_config)
    
    async def start_monitoring(self):
        """Start continuous disaster recovery monitoring."""
        logger.info("Starting disaster recovery monitoring...")
        
        await self._log_dr_event(
            "DR_MONITORING_STARTED",
            {"health_check_interval": self.config.health_check_interval}
        )
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._health_monitor_loop()),
            asyncio.create_task(self._backup_scheduler_loop()),
            asyncio.create_task(self._recovery_coordinator_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down disaster recovery monitoring...")
            await self._log_dr_event("DR_MONITORING_STOPPED", {})
        except Exception as e:
            logger.error(f"Disaster recovery monitoring failed: {e}")
            await self._log_dr_event(
                "DR_MONITORING_FAILED", 
                {"error": str(e)}, 
                severity="ERROR"
            )
    
    async def _health_monitor_loop(self):
        """Continuous health monitoring loop."""
        while True:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute on error
    
    async def _backup_scheduler_loop(self):
        """Automated backup scheduling loop."""
        while True:
            try:
                # Check if it's time for next backup based on schedule
                if await self._should_run_backup():
                    await self._execute_scheduled_backup()
                
                # Check every hour for backup scheduling
                await asyncio.sleep(3600)
            except Exception as e:
                logger.error(f"Backup scheduler error: {e}")
                await asyncio.sleep(300)  # Retry after 5 minutes
    
    async def _recovery_coordinator_loop(self):
        """Recovery coordination and decision making loop."""
        while True:
            try:
                if self.system_state == "degraded" and self.config.auto_recovery_enabled:
                    await self._attempt_auto_recovery()
                
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Recovery coordinator error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Comprehensive system health check."""
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "components": {}
        }
        
        try:
            # Database connectivity check
            db_health = await self._check_database_health()
            health_status["components"]["database"] = db_health
            
            # Redis connectivity check  
            redis_health = await self._check_redis_health()
            health_status["components"]["redis"] = redis_health
            
            # Disk space check
            disk_health = await self._check_disk_space()
            health_status["components"]["disk_space"] = disk_health
            
            # Service health check
            service_health = await self._check_services_health()
            health_status["components"]["services"] = service_health
            
            # Backup system health
            backup_health = await self._check_backup_system_health()
            health_status["components"]["backup_system"] = backup_health
            
            # Determine overall health
            component_statuses = [
                comp["status"] for comp in health_status["components"].values()
            ]
            
            if "critical" in component_statuses:
                health_status["overall_status"] = "critical"
                self.system_state = "critical"
                self.consecutive_failures += 1
            elif "degraded" in component_statuses:
                health_status["overall_status"] = "degraded"
                self.system_state = "degraded"
                self.consecutive_failures += 1
            else:
                health_status["overall_status"] = "healthy"
                self.system_state = "healthy"
                self.consecutive_failures = 0
            
            self.last_health_check = health_status
            
            # Log health status
            await self._log_dr_event(
                "HEALTH_CHECK_COMPLETED",
                {
                    "overall_status": health_status["overall_status"],
                    "consecutive_failures": self.consecutive_failures
                },
                severity="WARNING" if health_status["overall_status"] != "healthy" else "INFO"
            )
            
            # Trigger alerts if necessary
            if self.consecutive_failures >= self.config.max_consecutive_failures:
                await self._trigger_alert(health_status)
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["overall_status"] = "unknown"
            health_status["error"] = str(e)
            
            await self._log_dr_event(
                "HEALTH_CHECK_FAILED",
                {"error": str(e)},
                severity="ERROR"
            )
        
        return health_status
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check PostgreSQL database health."""
        try:
            # Basic connection test
            postgres_url = os.getenv('DATABASE_URL', 'postgresql://trading_user:trading_password@localhost:5432/trading_db')
            
            # Use asyncpg for async connection test
            cmd = ['pg_isready', '-d', postgres_url]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=self.config.database_connection_timeout
            )
            
            if result.returncode == 0:
                return {
                    "status": "healthy",
                    "response_time_ms": 100,  # Approximate
                    "message": "Database connection successful"
                }
            else:
                return {
                    "status": "critical",
                    "error": result.stderr or "Connection failed",
                    "message": "Database connection failed"
                }
                
        except subprocess.TimeoutExpired:
            return {
                "status": "critical",
                "error": "Connection timeout",
                "timeout_seconds": self.config.database_connection_timeout
            }
        except Exception as e:
            return {
                "status": "critical",
                "error": str(e),
                "message": "Database health check failed"
            }
    
    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis health."""
        try:
            import redis
            from urllib.parse import urlparse
            
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            parsed = urlparse(redis_url)
            
            r = redis.Redis(
                host=parsed.hostname or 'localhost',
                port=parsed.port or 6379,
                db=int(parsed.path.lstrip('/')) if parsed.path else 0,
                password=parsed.password,
                socket_timeout=10
            )
            
            # Test ping
            start_time = datetime.utcnow()
            pong = r.ping()
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if pong:
                return {
                    "status": "healthy",
                    "response_time_ms": round(response_time, 2),
                    "message": "Redis connection successful"
                }
            else:
                return {
                    "status": "critical",
                    "message": "Redis ping failed"
                }
                
        except ImportError:
            return {
                "status": "degraded",
                "message": "Redis module not available",
                "error": "redis-py not installed"
            }
        except Exception as e:
            return {
                "status": "critical",
                "error": str(e),
                "message": "Redis connection failed"
            }
    
    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space usage."""
        try:
            import shutil
            
            # Check backup directory space
            backup_path = Path(self.backup_manager.config.local_backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            total, used, free = shutil.disk_usage(backup_path)
            usage_percent = used / total
            
            if usage_percent >= self.config.disk_space_threshold:
                return {
                    "status": "critical",
                    "usage_percent": round(usage_percent * 100, 2),
                    "threshold_percent": round(self.config.disk_space_threshold * 100, 2),
                    "free_gb": round(free / (1024**3), 2),
                    "message": "Disk space critically low"
                }
            elif usage_percent >= 0.75:  # Warning at 75%
                return {
                    "status": "degraded",
                    "usage_percent": round(usage_percent * 100, 2),
                    "free_gb": round(free / (1024**3), 2),
                    "message": "Disk space warning"
                }
            else:
                return {
                    "status": "healthy",
                    "usage_percent": round(usage_percent * 100, 2),
                    "free_gb": round(free / (1024**3), 2),
                    "message": "Disk space healthy"
                }
                
        except Exception as e:
            return {
                "status": "unknown",
                "error": str(e),
                "message": "Disk space check failed"
            }
    
    async def _check_services_health(self) -> Dict[str, Any]:
        """Check critical services health."""
        services = [
            {"name": "trading-api", "port": 8000},
            {"name": "market-data", "port": 8001},
            {"name": "news-integration", "port": 8002}
        ]
        
        service_statuses = {}
        overall_status = "healthy"
        
        for service in services:
            try:
                # Simple port check
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex(('localhost', service["port"]))
                sock.close()
                
                if result == 0:
                    service_statuses[service["name"]] = {
                        "status": "healthy",
                        "port": service["port"]
                    }
                else:
                    service_statuses[service["name"]] = {
                        "status": "critical",
                        "port": service["port"],
                        "error": "Port not accessible"
                    }
                    overall_status = "critical"
                    
            except Exception as e:
                service_statuses[service["name"]] = {
                    "status": "critical",
                    "port": service["port"],
                    "error": str(e)
                }
                overall_status = "critical"
        
        return {
            "status": overall_status,
            "services": service_statuses,
            "healthy_count": sum(1 for s in service_statuses.values() if s["status"] == "healthy"),
            "total_count": len(services)
        }
    
    async def _check_backup_system_health(self) -> Dict[str, Any]:
        """Check backup system health."""
        try:
            # Check recent backups
            backups = self.backup_manager.list_backups()
            
            if not backups:
                return {
                    "status": "degraded",
                    "message": "No backups found",
                    "backup_count": 0
                }
            
            # Check if latest backup is recent (within 25 hours for daily backups)
            latest_backup = backups[0]
            latest_time = datetime.fromisoformat(latest_backup["timestamp"])
            age_hours = (datetime.utcnow() - latest_time).total_seconds() / 3600
            
            if age_hours > 25:  # Allow 1 hour grace period for daily backups
                return {
                    "status": "degraded",
                    "message": f"Latest backup is {age_hours:.1f} hours old",
                    "latest_backup_age_hours": round(age_hours, 1),
                    "backup_count": len(backups)
                }
            
            # Check backup success rate
            successful_backups = [b for b in backups[:10] if b["status"] == "completed"]
            success_rate = len(successful_backups) / min(10, len(backups))
            
            if success_rate < 0.8:  # Less than 80% success rate
                return {
                    "status": "degraded",
                    "message": f"Backup success rate low: {success_rate:.0%}",
                    "success_rate": success_rate,
                    "backup_count": len(backups)
                }
            
            return {
                "status": "healthy",
                "message": "Backup system healthy",
                "latest_backup_age_hours": round(age_hours, 1),
                "success_rate": success_rate,
                "backup_count": len(backups)
            }
            
        except Exception as e:
            return {
                "status": "critical",
                "error": str(e),
                "message": "Backup system check failed"
            }
    
    async def _should_run_backup(self) -> bool:
        """Check if it's time to run a scheduled backup."""
        # Simple daily backup check - in production, use proper cron parsing
        backups = self.backup_manager.list_backups()
        
        if not backups:
            return True  # No backups exist, create one
        
        latest_backup = backups[0]
        latest_time = datetime.fromisoformat(latest_backup["timestamp"])
        hours_since_last = (datetime.utcnow() - latest_time).total_seconds() / 3600
        
        # Run backup if more than 23 hours since last one
        return hours_since_last >= 23
    
    async def _execute_scheduled_backup(self):
        """Execute a scheduled backup."""
        logger.info("Starting scheduled backup...")
        
        try:
            manifest = await self.backup_manager.create_full_backup("scheduled")
            
            await self._log_dr_event(
                "SCHEDULED_BACKUP_COMPLETED",
                {
                    "backup_id": manifest["backup_id"],
                    "components": list(manifest["components"].keys()),
                    "status": manifest["status"]
                }
            )
            
            logger.info(f"✅ Scheduled backup completed: {manifest['backup_id']}")
            
        except Exception as e:
            logger.error(f"❌ Scheduled backup failed: {e}")
            
            await self._log_dr_event(
                "SCHEDULED_BACKUP_FAILED",
                {"error": str(e)},
                severity="ERROR"
            )
    
    async def _attempt_auto_recovery(self):
        """Attempt automated recovery procedures."""
        recovery_id = f"recovery_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Check recovery cooldown
        last_recovery = self.recovery_attempts.get("last_attempt")
        if last_recovery:
            minutes_since = (datetime.utcnow() - last_recovery).total_seconds() / 60
            if minutes_since < self.config.recovery_cooldown_minutes:
                logger.info(f"Recovery cooldown active, waiting {self.config.recovery_cooldown_minutes - minutes_since:.1f} minutes")
                return
        
        # Check max attempts
        attempt_count = self.recovery_attempts.get("count", 0)
        if attempt_count >= self.config.max_recovery_attempts:
            logger.warning("Maximum recovery attempts reached, manual intervention required")
            return
        
        logger.info(f"Starting automated recovery attempt {attempt_count + 1}/{self.config.max_recovery_attempts}")
        
        try:
            recovery_actions = []
            
            # Analyze health status and plan recovery
            if self.last_health_check:
                for component, status in self.last_health_check["components"].items():
                    if status["status"] in ["critical", "degraded"]:
                        actions = await self._plan_component_recovery(component, status)
                        recovery_actions.extend(actions)
            
            # Execute recovery actions
            recovery_results = []
            for action in recovery_actions:
                try:
                    result = await self._execute_recovery_action(action)
                    recovery_results.append(result)
                except Exception as e:
                    recovery_results.append({
                        "action": action,
                        "status": "failed",
                        "error": str(e)
                    })
            
            # Update recovery tracking
            self.recovery_attempts["count"] = attempt_count + 1
            self.recovery_attempts["last_attempt"] = datetime.utcnow()
            
            await self._log_dr_event(
                "AUTO_RECOVERY_COMPLETED",
                {
                    "recovery_id": recovery_id,
                    "attempt_number": attempt_count + 1,
                    "actions_executed": len(recovery_actions),
                    "results": recovery_results
                }
            )
            
            logger.info(f"Auto recovery attempt completed: {recovery_id}")
            
        except Exception as e:
            logger.error(f"Auto recovery failed: {e}")
            
            await self._log_dr_event(
                "AUTO_RECOVERY_FAILED",
                {
                    "recovery_id": recovery_id,
                    "error": str(e)
                },
                severity="ERROR"
            )
    
    async def _plan_component_recovery(self, component: str, status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan recovery actions for a specific component."""
        actions = []
        
        if component == "database":
            if "timeout" in status.get("error", "").lower():
                actions.append({
                    "type": "restart_service",
                    "target": "postgresql",
                    "priority": "high"
                })
        
        elif component == "redis":
            if status["status"] == "critical":
                actions.append({
                    "type": "restart_service", 
                    "target": "redis",
                    "priority": "medium"
                })
        
        elif component == "disk_space":
            if status["status"] == "critical":
                actions.append({
                    "type": "cleanup_old_files",
                    "target": "logs",
                    "priority": "high"
                })
                actions.append({
                    "type": "cleanup_old_files",
                    "target": "backups",
                    "priority": "medium"
                })
        
        elif component == "services":
            unhealthy_services = [
                name for name, svc in status.get("services", {}).items()
                if svc["status"] != "healthy"
            ]
            for service in unhealthy_services:
                actions.append({
                    "type": "restart_service",
                    "target": service,
                    "priority": "high"
                })
        
        return actions
    
    async def _execute_recovery_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific recovery action."""
        action_type = action["type"]
        target = action["target"]
        
        logger.info(f"Executing recovery action: {action_type} on {target}")
        
        if action_type == "restart_service":
            return await self._restart_service(target)
        elif action_type == "cleanup_old_files":
            return await self._cleanup_old_files(target)
        else:
            return {
                "action": action,
                "status": "unsupported",
                "message": f"Action type {action_type} not implemented"
            }
    
    async def _restart_service(self, service_name: str) -> Dict[str, Any]:
        """Restart a system service."""
        try:
            # This would use systemctl or docker restart in production
            logger.info(f"Simulating restart of service: {service_name}")
            
            # In production:
            # result = subprocess.run(['systemctl', 'restart', service_name], 
            #                        capture_output=True, text=True)
            
            return {
                "action": "restart_service",
                "target": service_name,
                "status": "simulated",
                "message": f"Service {service_name} restart simulated"
            }
            
        except Exception as e:
            return {
                "action": "restart_service",
                "target": service_name,
                "status": "failed",
                "error": str(e)
            }
    
    async def _cleanup_old_files(self, target: str) -> Dict[str, Any]:
        """Clean up old files to free disk space."""
        try:
            if target == "logs":
                # Clean logs older than 7 days
                logs_dir = Path("logs")
                if logs_dir.exists():
                    cutoff_date = datetime.utcnow() - timedelta(days=7)
                    files_removed = 0
                    
                    for log_file in logs_dir.rglob("*.log*"):
                        if log_file.is_file():
                            file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                            if file_time < cutoff_date:
                                log_file.unlink()
                                files_removed += 1
                    
                    return {
                        "action": "cleanup_old_files",
                        "target": target,
                        "status": "success",
                        "files_removed": files_removed
                    }
            
            elif target == "backups":
                # Use backup manager cleanup
                await self.backup_manager._cleanup_old_backups()
                return {
                    "action": "cleanup_old_files",
                    "target": target,
                    "status": "success",
                    "message": "Backup cleanup completed"
                }
            
            return {
                "action": "cleanup_old_files", 
                "target": target,
                "status": "no_action",
                "message": f"No cleanup action defined for {target}"
            }
            
        except Exception as e:
            return {
                "action": "cleanup_old_files",
                "target": target,
                "status": "failed",
                "error": str(e)
            }
    
    async def _trigger_alert(self, health_status: Dict[str, Any]):
        """Trigger alerts for system issues."""
        alert_message = f"🚨 SYSTEM HEALTH ALERT 🚨\n"
        alert_message += f"Status: {health_status['overall_status'].upper()}\n"
        alert_message += f"Consecutive failures: {self.consecutive_failures}\n"
        alert_message += f"Time: {health_status['timestamp']}\n\n"
        
        # Add component details
        for component, status in health_status["components"].items():
            if status["status"] != "healthy":
                alert_message += f"❌ {component.upper()}: {status['status']}\n"
                if "error" in status:
                    alert_message += f"   Error: {status['error']}\n"
                if "message" in status:
                    alert_message += f"   Message: {status['message']}\n"
        
        logger.error(alert_message)
        
        # Log alert
        await self._log_dr_event(
            "ALERT_TRIGGERED",
            {
                "alert_type": "health_degradation",
                "overall_status": health_status["overall_status"],
                "consecutive_failures": self.consecutive_failures,
                "components": health_status["components"]
            },
            severity="CRITICAL"
        )
        
        # In production, send to webhook, email, etc.
        if self.config.alert_webhook_url:
            # Send webhook alert
            pass
        
        if self.config.email_recipients:
            # Send email alerts
            pass
    
    async def _log_dr_event(self, event_type: str, details: Dict[str, Any], severity: str = "INFO"):
        """Log disaster recovery events to audit system."""
        try:
            await log_audit_event(
                event_type=event_type,
                message=f"Disaster Recovery: {event_type}",
                context=AuditContext(
                    component="disaster_recovery",
                    operation="dr_management"
                ),
                severity=severity,
                details=details,
                tags=["disaster_recovery", "system_monitoring", "automation"]
            )
        except Exception as e:
            logger.warning(f"Failed to log DR event: {e}")
    
    async def create_emergency_backup(self) -> Dict[str, Any]:
        """Create an emergency backup immediately."""
        logger.info("Creating emergency backup...")
        
        try:
            manifest = await self.backup_manager.create_full_backup("emergency")
            
            await self._log_dr_event(
                "EMERGENCY_BACKUP_CREATED",
                {
                    "backup_id": manifest["backup_id"],
                    "trigger": "manual_request",
                    "status": manifest["status"]
                }
            )
            
            return manifest
            
        except Exception as e:
            logger.error(f"Emergency backup failed: {e}")
            
            await self._log_dr_event(
                "EMERGENCY_BACKUP_FAILED",
                {"error": str(e)},
                severity="ERROR"
            )
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "system_state": self.system_state,
            "consecutive_failures": self.consecutive_failures,
            "last_health_check": self.last_health_check,
            "recovery_attempts": self.recovery_attempts,
            "monitoring_active": True,
            "auto_recovery_enabled": self.config.auto_recovery_enabled
        }


async def main():
    """Main disaster recovery CLI."""
    parser = argparse.ArgumentParser(description="AI Trading System Disaster Recovery Manager")
    parser.add_argument("--action", 
                       choices=["monitor", "health-check", "backup", "status", "recovery-test"],
                       default="monitor",
                       help="Action to perform")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--auto-recovery", action="store_true", help="Enable auto-recovery")
    
    args = parser.parse_args()
    
    # Load configuration
    config = DisasterRecoveryConfig()
    config.auto_recovery_enabled = args.auto_recovery
    
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                custom_config = yaml.safe_load(f)
            else:
                custom_config = json.load(f)
        
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Create DR manager
    dr_manager = DisasterRecoveryManager(config)
    
    try:
        if args.action == "monitor":
            print("🛡️  Starting disaster recovery monitoring...")
            print(f"   Health checks every {config.health_check_interval} seconds")
            print(f"   Auto-recovery: {'enabled' if config.auto_recovery_enabled else 'disabled'}")
            print("   Press Ctrl+C to stop")
            await dr_manager.start_monitoring()
            
        elif args.action == "health-check":
            print("🔍 Performing system health check...")
            health_status = await dr_manager._perform_health_check()
            
            print(f"\n📊 System Health Report")
            print(f"Overall Status: {health_status['overall_status'].upper()}")
            print(f"Timestamp: {health_status['timestamp']}")
            
            for component, status in health_status['components'].items():
                emoji = "✅" if status['status'] == 'healthy' else "⚠️" if status['status'] == 'degraded' else "❌"
                print(f"{emoji} {component.title()}: {status['status']}")
                if 'message' in status:
                    print(f"   {status['message']}")
            
        elif args.action == "backup":
            print("💾 Creating emergency backup...")
            manifest = await dr_manager.create_emergency_backup()
            print(f"✅ Emergency backup created: {manifest['backup_id']}")
            
        elif args.action == "status":
            status = dr_manager.get_system_status()
            print(f"\n📋 Disaster Recovery Status")
            print(f"System State: {status['system_state']}")
            print(f"Consecutive Failures: {status['consecutive_failures']}")
            print(f"Auto Recovery: {'enabled' if status['auto_recovery_enabled'] else 'disabled'}")
            print(f"Recovery Attempts: {status['recovery_attempts']}")
            
        elif args.action == "recovery-test":
            print("🧪 Running recovery test...")
            # This would run a recovery simulation
            print("Recovery test functionality not yet implemented")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)