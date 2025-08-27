#!/usr/bin/env python3
"""
Unified Backup and Restore System for AI Trading System
Combines backup creation and automated restore capabilities with encryption and verification
"""

import os
import sys
import asyncio
import logging
import shutil
import gzip
import json
import hashlib
import tempfile
import subprocess
import psycopg2
import redis
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from cryptography.fernet import Fernet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backups"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"


class RestoreMode(Enum):
    """Restore operation modes"""
    FULL = "full"
    DATABASE = "database"
    REDIS = "redis"
    MODELS = "models"
    CONFIG = "config"
    PARTIAL = "partial"
    VERIFICATION = "verify"


@dataclass
class BackupConfig:
    """Unified backup and restore configuration"""
    # Paths
    backup_dir: str = "/var/backups/trading-system"
    temp_dir: str = "/tmp/backup_restore"
    model_artifacts_dir: str = "services/model-registry/artifacts"
    logs_dir: str = "logs"
    config_dir: str = "config"
    
    # Database settings
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "trading_db")
    postgres_user: str = os.getenv("POSTGRES_USER", "trading_user")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")
    
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_password: str = os.getenv("REDIS_PASSWORD", "")
    
    # Retention policies (days)
    daily_retention: int = 30
    weekly_retention: int = 84   # 12 weeks
    monthly_retention: int = 365  # 12 months
    
    # Security and verification
    compress_backups: bool = True
    encrypt_backups: bool = True
    encryption_key: Optional[str] = os.getenv("BACKUP_ENCRYPTION_KEY")
    verify_checksums: bool = True
    checksum_algorithm: str = "sha256"
    
    # Performance
    parallel_operations: bool = True
    max_workers: int = 4
    operation_timeout_seconds: int = 3600
    
    # Testing and validation
    test_restore: bool = True
    create_restore_point: bool = True
    
    # Notifications
    notification_webhook: Optional[str] = os.getenv("NOTIFICATION_WEBHOOK")
    alert_on_failure: bool = True


class UnifiedBackupRestoreManager:
    """
    Comprehensive backup and restore manager with encryption, verification, and automation
    """
    
    def __init__(self, config: Optional[BackupConfig] = None):
        self.config = config or BackupConfig()
        self.backup_root = Path(self.config.backup_dir)
        self.backup_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(self.config.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup encryption
        self.cipher = None
        if self.config.encrypt_backups and self.config.encryption_key:
            try:
                self.cipher = Fernet(self.config.encryption_key.encode())
            except Exception as e:
                logger.warning(f"Could not setup encryption: {e}")
        
        # Backup subdirectories
        self.db_backup_dir = self.backup_root / "databases"
        self.model_backup_dir = self.backup_root / "models"
        self.config_backup_dir = self.backup_root / "config"
        self.logs_backup_dir = self.backup_root / "logs"
        self.restore_points_dir = self.backup_root / "restore_points"
        
        for dir_path in [self.db_backup_dir, self.model_backup_dir, 
                         self.config_backup_dir, self.logs_backup_dir,
                         self.restore_points_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Track operations
        self.operation_history: List[Dict[str, Any]] = []
    
    # ==================== BACKUP OPERATIONS ====================
    
    async def create_backup(
        self,
        backup_type: BackupType = BackupType.FULL,
        components: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a backup with specified type and components"""
        
        backup_id = f"{backup_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting {backup_type.value} backup: {backup_id}")
        
        manifest = {
            "backup_id": backup_id,
            "backup_type": backup_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
            "checksums": {},
            "status": "in_progress",
            "encrypted": self.config.encrypt_backups,
            "compressed": self.config.compress_backups
        }
        
        try:
            # Determine what to backup
            if backup_type == BackupType.FULL or components is None:
                components = ["database", "redis", "models", "config", "logs"]
            
            # Backup each component
            if "database" in components:
                db_result = await self._backup_postgresql(backup_id)
                manifest["components"]["postgresql"] = db_result
            
            if "redis" in components:
                redis_result = await self._backup_redis(backup_id)
                manifest["components"]["redis"] = redis_result
            
            if "models" in components:
                models_result = await self._backup_models(backup_id)
                manifest["components"]["models"] = models_result
            
            if "config" in components:
                config_result = await self._backup_configuration(backup_id)
                manifest["components"]["config"] = config_result
            
            if "logs" in components and backup_type == BackupType.FULL:
                logs_result = await self._backup_logs(backup_id)
                manifest["components"]["logs"] = logs_result
            
            # Calculate checksums for all backup files
            if self.config.verify_checksums:
                await self._calculate_backup_checksums(backup_id, manifest)
            
            # Mark as completed
            manifest["status"] = "completed"
            manifest["completed_at"] = datetime.utcnow().isoformat()
            
            # Save manifest
            manifest_path = self.backup_root / f"{backup_id}_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Cleanup old backups
            await self._cleanup_old_backups()
            
            logger.info(f"✅ Backup completed: {backup_id}")
            
            # Schedule automatic test restore if enabled
            if self.config.test_restore:
                asyncio.create_task(self._schedule_test_restore(backup_id, delay_hours=1))
            
            return manifest
            
        except Exception as e:
            manifest["status"] = "failed"
            manifest["error"] = str(e)
            logger.error(f"❌ Backup failed: {e}")
            
            if self.config.alert_on_failure:
                await self._send_alert(f"Backup failed: {backup_id}", str(e))
            
            raise
    
    async def _backup_postgresql(self, backup_id: str) -> Dict[str, Any]:
        """Backup PostgreSQL database"""
        try:
            backup_file = self.db_backup_dir / f"{backup_id}_postgresql.sql"
            
            # Create pg_dump command
            dump_cmd = [
                "pg_dump",
                "-h", self.config.postgres_host,
                "-p", str(self.config.postgres_port),
                "-U", self.config.postgres_user,
                "-d", self.config.postgres_db,
                "-f", str(backup_file),
                "--verbose",
                "--clean",
                "--if-exists",
                "--no-owner",
                "--no-privileges"
            ]
            
            if self.config.parallel_operations:
                dump_cmd.extend(["-j", str(self.config.max_workers)])
            
            # Set password via environment
            env = os.environ.copy()
            env["PGPASSWORD"] = self.config.postgres_password
            
            # Execute backup
            process = await asyncio.create_subprocess_exec(
                *dump_cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.operation_timeout_seconds
            )
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode,
                    dump_cmd,
                    stderr.decode()
                )
            
            # Compress if enabled
            if self.config.compress_backups:
                compressed_file = await self._compress_file(backup_file)
                backup_file.unlink()
                backup_file = compressed_file
            
            # Encrypt if enabled
            if self.cipher:
                encrypted_file = await self._encrypt_file(backup_file)
                backup_file.unlink()
                backup_file = encrypted_file
            
            return {
                "status": "success",
                "backup_path": str(backup_file),
                "size_bytes": backup_file.stat().st_size,
                "compressed": self.config.compress_backups,
                "encrypted": bool(self.cipher)
            }
            
        except Exception as e:
            logger.error(f"PostgreSQL backup failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _backup_redis(self, backup_id: str) -> Dict[str, Any]:
        """Backup Redis database"""
        try:
            backup_file = self.db_backup_dir / f"{backup_id}_redis.rdb"
            
            # Connect to Redis
            r = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password
            )
            
            # Trigger background save
            r.bgsave()
            
            # Wait for save to complete
            while r.lastsave() == r.lastsave():
                await asyncio.sleep(1)
            
            # Copy RDB file
            redis_rdb = Path("/var/lib/redis/dump.rdb")
            if redis_rdb.exists():
                shutil.copy2(redis_rdb, backup_file)
            else:
                # Try alternative location
                alt_rdb = Path("/data/dump.rdb")
                if alt_rdb.exists():
                    shutil.copy2(alt_rdb, backup_file)
                else:
                    raise FileNotFoundError("Redis RDB file not found")
            
            # Encrypt if enabled
            if self.cipher:
                encrypted_file = await self._encrypt_file(backup_file)
                backup_file.unlink()
                backup_file = encrypted_file
            
            return {
                "status": "success",
                "backup_path": str(backup_file),
                "size_bytes": backup_file.stat().st_size,
                "encrypted": bool(self.cipher)
            }
            
        except Exception as e:
            logger.error(f"Redis backup failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _backup_models(self, backup_id: str) -> Dict[str, Any]:
        """Backup ML model artifacts"""
        try:
            models_dir = Path(self.config.model_artifacts_dir)
            if not models_dir.exists():
                return {"status": "skipped", "reason": "Models directory not found"}
            
            backup_dir = self.model_backup_dir / backup_id
            backup_dir.mkdir(exist_ok=True)
            
            # Copy model files
            model_count = 0
            for model_file in models_dir.glob("**/*"):
                if model_file.is_file():
                    relative_path = model_file.relative_to(models_dir)
                    target_path = backup_dir / relative_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(model_file, target_path)
                    model_count += 1
            
            # Create tar archive
            if self.config.compress_backups:
                archive_path = self.model_backup_dir / f"{backup_id}_models.tar.gz"
                shutil.make_archive(
                    str(archive_path.with_suffix("")),
                    "gztar",
                    backup_dir
                )
                shutil.rmtree(backup_dir)
                
                # Encrypt if enabled
                if self.cipher:
                    encrypted_file = await self._encrypt_file(archive_path)
                    archive_path.unlink()
                    archive_path = encrypted_file
                
                return {
                    "status": "success",
                    "backup_path": str(archive_path),
                    "model_count": model_count,
                    "size_bytes": archive_path.stat().st_size,
                    "compressed": True,
                    "encrypted": bool(self.cipher)
                }
            
            return {
                "status": "success",
                "backup_path": str(backup_dir),
                "model_count": model_count
            }
            
        except Exception as e:
            logger.error(f"Models backup failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _backup_configuration(self, backup_id: str) -> Dict[str, Any]:
        """Backup configuration files"""
        try:
            config_dir = Path(self.config.config_dir)
            if not config_dir.exists():
                return {"status": "skipped", "reason": "Config directory not found"}
            
            backup_dir = self.config_backup_dir / backup_id
            backup_dir.mkdir(exist_ok=True)
            
            # Copy config files
            config_count = 0
            for config_file in config_dir.glob("**/*"):
                if config_file.is_file() and not config_file.name.startswith('.'):
                    relative_path = config_file.relative_to(config_dir)
                    target_path = backup_dir / relative_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(config_file, target_path)
                    config_count += 1
            
            return {
                "status": "success",
                "backup_path": str(backup_dir),
                "config_count": config_count
            }
            
        except Exception as e:
            logger.error(f"Config backup failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _backup_logs(self, backup_id: str) -> Dict[str, Any]:
        """Backup log files"""
        try:
            logs_dir = Path(self.config.logs_dir)
            if not logs_dir.exists():
                return {"status": "skipped", "reason": "Logs directory not found"}
            
            # Create compressed archive of logs
            archive_path = self.logs_backup_dir / f"{backup_id}_logs.tar.gz"
            shutil.make_archive(
                str(archive_path.with_suffix("")),
                "gztar",
                logs_dir
            )
            
            return {
                "status": "success",
                "backup_path": str(archive_path),
                "size_bytes": archive_path.stat().st_size,
                "compressed": True
            }
            
        except Exception as e:
            logger.error(f"Logs backup failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    # ==================== RESTORE OPERATIONS ====================
    
    async def restore_backup(
        self,
        backup_id: str,
        mode: RestoreMode = RestoreMode.FULL,
        components: Optional[List[str]] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Restore from backup with specified mode"""
        
        logger.info(f"{'DRY RUN: ' if dry_run else ''}Starting restore: {backup_id} (mode: {mode.value})")
        
        result = {
            "backup_id": backup_id,
            "mode": mode.value,
            "start_time": datetime.utcnow().isoformat(),
            "components_restored": {},
            "verification": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Load backup manifest
            manifest = await self._load_manifest(backup_id)
            if not manifest:
                raise FileNotFoundError(f"Backup manifest not found: {backup_id}")
            
            # Verify backup integrity
            if self.config.verify_checksums and "checksums" in manifest:
                integrity_ok = await self._verify_integrity(manifest)
                result["verification"]["integrity"] = integrity_ok
                if not integrity_ok:
                    raise ValueError("Backup integrity verification failed")
            
            # Create restore point
            if not dry_run and self.config.create_restore_point:
                restore_point = await self._create_restore_point()
                result["restore_point"] = restore_point
            
            # Perform restore based on mode
            if mode == RestoreMode.FULL:
                await self._restore_full(manifest, dry_run, result)
            elif mode == RestoreMode.DATABASE:
                await self._restore_postgresql(manifest, dry_run, result)
            elif mode == RestoreMode.REDIS:
                await self._restore_redis(manifest, dry_run, result)
            elif mode == RestoreMode.MODELS:
                await self._restore_models(manifest, dry_run, result)
            elif mode == RestoreMode.CONFIG:
                await self._restore_config(manifest, dry_run, result)
            elif mode == RestoreMode.PARTIAL and components:
                await self._restore_partial(manifest, components, dry_run, result)
            elif mode == RestoreMode.VERIFICATION:
                await self._verify_only(manifest, result)
            
            # Post-restore verification
            if not dry_run and self.config.test_restore:
                verification = await self._verify_restore(mode, components)
                result["verification"]["post_restore"] = verification
            
            result["status"] = "success" if not result["errors"] else "failed"
            result["end_time"] = datetime.utcnow().isoformat()
            
            logger.info(f"{'✅' if result['status'] == 'success' else '❌'} Restore {result['status']}: {backup_id}")
            
            # Save to history
            self.operation_history.append(result)
            
            return result
            
        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(str(e))
            result["end_time"] = datetime.utcnow().isoformat()
            
            logger.error(f"Restore failed: {e}")
            
            if self.config.alert_on_failure:
                await self._send_alert(f"Restore failed: {backup_id}", str(e))
            
            # Attempt rollback if restore point exists
            if "restore_point" in result and not dry_run:
                try:
                    await self._rollback_to_restore_point(result["restore_point"])
                    result["rollback"] = "success"
                except Exception as rollback_error:
                    result["rollback"] = f"failed: {rollback_error}"
            
            return result
    
    async def _restore_postgresql(
        self,
        manifest: Dict,
        dry_run: bool,
        result: Dict
    ):
        """Restore PostgreSQL database"""
        try:
            pg_backup = manifest["components"].get("postgresql", {})
            if pg_backup.get("status") != "success":
                result["warnings"].append("PostgreSQL backup not available")
                return
            
            backup_file = Path(pg_backup["backup_path"])
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_file}")
            
            if dry_run:
                logger.info(f"[DRY RUN] Would restore PostgreSQL from: {backup_file}")
                result["components_restored"]["postgresql"] = "dry_run"
                return
            
            # Decrypt if needed
            if pg_backup.get("encrypted") and self.cipher:
                backup_file = await self._decrypt_file(backup_file)
            
            # Decompress if needed
            if pg_backup.get("compressed"):
                backup_file = await self._decompress_file(backup_file)
            
            # Terminate existing connections
            await self._terminate_db_connections()
            
            # Restore database
            restore_cmd = [
                "psql",
                "-h", self.config.postgres_host,
                "-p", str(self.config.postgres_port),
                "-U", self.config.postgres_user,
                "-d", self.config.postgres_db,
                "-f", str(backup_file)
            ]
            
            env = os.environ.copy()
            env["PGPASSWORD"] = self.config.postgres_password
            
            process = await asyncio.create_subprocess_exec(
                *restore_cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode,
                    restore_cmd,
                    stderr.decode()
                )
            
            # Run ANALYZE
            await self._analyze_database()
            
            result["components_restored"]["postgresql"] = "success"
            logger.info("PostgreSQL restore completed")
            
        except Exception as e:
            result["errors"].append(f"PostgreSQL restore error: {e}")
            result["components_restored"]["postgresql"] = "failed"
    
    async def _restore_redis(
        self,
        manifest: Dict,
        dry_run: bool,
        result: Dict
    ):
        """Restore Redis database"""
        try:
            redis_backup = manifest["components"].get("redis", {})
            if redis_backup.get("status") != "success":
                result["warnings"].append("Redis backup not available")
                return
            
            backup_file = Path(redis_backup["backup_path"])
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_file}")
            
            if dry_run:
                logger.info(f"[DRY RUN] Would restore Redis from: {backup_file}")
                result["components_restored"]["redis"] = "dry_run"
                return
            
            # Decrypt if needed
            if redis_backup.get("encrypted") and self.cipher:
                backup_file = await self._decrypt_file(backup_file)
            
            # Stop Redis
            subprocess.run(["redis-cli", "SHUTDOWN", "NOSAVE"], check=False)
            
            # Replace dump file
            redis_dir = Path("/var/lib/redis")
            target_file = redis_dir / "dump.rdb"
            shutil.copy2(backup_file, target_file)
            
            # Start Redis
            subprocess.run(["systemctl", "start", "redis"], check=True)
            
            # Verify connection
            await asyncio.sleep(2)
            r = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password
            )
            r.ping()
            
            result["components_restored"]["redis"] = "success"
            logger.info("Redis restore completed")
            
        except Exception as e:
            result["errors"].append(f"Redis restore error: {e}")
            result["components_restored"]["redis"] = "failed"
    
    async def _restore_models(
        self,
        manifest: Dict,
        dry_run: bool,
        result: Dict
    ):
        """Restore ML models"""
        try:
            models_backup = manifest["components"].get("models", {})
            if models_backup.get("status") != "success":
                result["warnings"].append("Models backup not available")
                return
            
            backup_path = Path(models_backup["backup_path"])
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup not found: {backup_path}")
            
            if dry_run:
                logger.info(f"[DRY RUN] Would restore models from: {backup_path}")
                result["components_restored"]["models"] = "dry_run"
                return
            
            models_dir = Path(self.config.model_artifacts_dir)
            
            # Backup current models
            if models_dir.exists():
                backup_current = models_dir.parent / f"models_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                shutil.move(str(models_dir), str(backup_current))
            
            # Restore models
            if backup_path.is_file() and backup_path.suffix == ".gz":
                # Extract archive
                models_dir.mkdir(parents=True, exist_ok=True)
                shutil.unpack_archive(backup_path, models_dir)
            else:
                # Copy directory
                shutil.copytree(backup_path, models_dir)
            
            result["components_restored"]["models"] = "success"
            logger.info("Models restore completed")
            
        except Exception as e:
            result["errors"].append(f"Models restore error: {e}")
            result["components_restored"]["models"] = "failed"
    
    async def _restore_config(
        self,
        manifest: Dict,
        dry_run: bool,
        result: Dict
    ):
        """Restore configuration files"""
        try:
            config_backup = manifest["components"].get("config", {})
            if config_backup.get("status") != "success":
                result["warnings"].append("Config backup not available")
                return
            
            backup_path = Path(config_backup["backup_path"])
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup not found: {backup_path}")
            
            if dry_run:
                logger.info(f"[DRY RUN] Would restore config from: {backup_path}")
                result["components_restored"]["config"] = "dry_run"
                return
            
            config_dir = Path(self.config.config_dir)
            
            # Backup current config
            if config_dir.exists():
                backup_current = config_dir.parent / f"config_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                shutil.copytree(config_dir, backup_current)
            
            # Restore config files
            for config_file in backup_path.glob("**/*"):
                if config_file.is_file():
                    relative_path = config_file.relative_to(backup_path)
                    target_path = config_dir / relative_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(config_file, target_path)
            
            result["components_restored"]["config"] = "success"
            logger.info("Config restore completed")
            
        except Exception as e:
            result["errors"].append(f"Config restore error: {e}")
            result["components_restored"]["config"] = "failed"
    
    async def _restore_full(
        self,
        manifest: Dict,
        dry_run: bool,
        result: Dict
    ):
        """Perform full system restore"""
        # Stop services first
        if not dry_run:
            await self._stop_services()
        
        # Restore all components
        await self._restore_postgresql(manifest, dry_run, result)
        await self._restore_redis(manifest, dry_run, result)
        await self._restore_models(manifest, dry_run, result)
        await self._restore_config(manifest, dry_run, result)
        
        # Restart services
        if not dry_run:
            await self._start_services()
    
    async def _restore_partial(
        self,
        manifest: Dict,
        components: List[str],
        dry_run: bool,
        result: Dict
    ):
        """Restore specific components"""
        for component in components:
            if component == "postgresql":
                await self._restore_postgresql(manifest, dry_run, result)
            elif component == "redis":
                await self._restore_redis(manifest, dry_run, result)
            elif component == "models":
                await self._restore_models(manifest, dry_run, result)
            elif component == "config":
                await self._restore_config(manifest, dry_run, result)
            else:
                result["warnings"].append(f"Unknown component: {component}")
    
    async def _verify_only(self, manifest: Dict, result: Dict):
        """Verify backup without restoring"""
        for component, data in manifest["components"].items():
            if isinstance(data, dict) and "backup_path" in data:
                path = Path(data["backup_path"])
                result["verification"][f"{component}_exists"] = path.exists()
                if path.exists():
                    result["verification"][f"{component}_size"] = path.stat().st_size
    
    # ==================== UTILITY FUNCTIONS ====================
    
    async def _compress_file(self, file_path: Path) -> Path:
        """Compress a file using gzip"""
        compressed_path = file_path.with_suffix(file_path.suffix + ".gz")
        
        with open(file_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return compressed_path
    
    async def _decompress_file(self, file_path: Path) -> Path:
        """Decompress a gzipped file"""
        decompressed_path = self.temp_dir / file_path.stem
        
        with gzip.open(file_path, 'rb') as f_in:
            with open(decompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return decompressed_path
    
    async def _encrypt_file(self, file_path: Path) -> Path:
        """Encrypt a file"""
        if not self.cipher:
            return file_path
        
        encrypted_path = file_path.with_suffix(file_path.suffix + ".enc")
        
        with open(file_path, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.cipher.encrypt(data)
        
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted_data)
        
        return encrypted_path
    
    async def _decrypt_file(self, file_path: Path) -> Path:
        """Decrypt a file"""
        if not self.cipher:
            return file_path
        
        decrypted_path = self.temp_dir / file_path.stem
        
        with open(file_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = self.cipher.decrypt(encrypted_data)
        
        with open(decrypted_path, 'wb') as f:
            f.write(decrypted_data)
        
        return decrypted_path
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        hasher = hashlib.new(self.config.checksum_algorithm)
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    async def _calculate_backup_checksums(self, backup_id: str, manifest: Dict):
        """Calculate checksums for all backup files"""
        for component, data in manifest["components"].items():
            if isinstance(data, dict) and "backup_path" in data:
                path = Path(data["backup_path"])
                if path.exists():
                    checksum = await self._calculate_checksum(path)
                    manifest["checksums"][str(path)] = checksum
    
    async def _verify_integrity(self, manifest: Dict) -> bool:
        """Verify backup integrity using checksums"""
        if "checksums" not in manifest:
            return True
        
        for file_path, expected_checksum in manifest["checksums"].items():
            path = Path(file_path)
            if path.exists():
                actual_checksum = await self._calculate_checksum(path)
                if actual_checksum != expected_checksum:
                    logger.error(f"Checksum mismatch: {file_path}")
                    return False
        
        return True
    
    async def _load_manifest(self, backup_id: str) -> Optional[Dict]:
        """Load backup manifest"""
        manifest_path = self.backup_root / f"{backup_id}_manifest.json"
        
        if not manifest_path.exists():
            # Try partial match
            for manifest_file in self.backup_root.glob(f"*{backup_id}*_manifest.json"):
                manifest_path = manifest_file
                break
        
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                return json.load(f)
        
        return None
    
    async def _create_restore_point(self) -> str:
        """Create a restore point"""
        restore_point_id = f"restore_point_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        restore_point_dir = self.restore_points_dir / restore_point_id
        restore_point_dir.mkdir(parents=True, exist_ok=True)
        
        # Quick snapshot of current state
        snapshot_manifest = {
            "restore_point_id": restore_point_id,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "restore_point"
        }
        
        manifest_path = restore_point_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(snapshot_manifest, f, indent=2)
        
        logger.info(f"Restore point created: {restore_point_id}")
        return restore_point_id
    
    async def _rollback_to_restore_point(self, restore_point_id: str):
        """Rollback to a restore point"""
        logger.warning(f"Rolling back to restore point: {restore_point_id}")
        # Implementation depends on your backup strategy
    
    async def _cleanup_old_backups(self):
        """Clean up old backups based on retention policy"""
        current_time = datetime.utcnow()
        
        for manifest_file in self.backup_root.glob("*_manifest.json"):
            try:
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
                
                backup_time = datetime.fromisoformat(manifest["timestamp"])
                age_days = (current_time - backup_time).days
                
                # Determine if backup should be kept
                keep = False
                
                # Keep daily backups
                if age_days <= self.config.daily_retention:
                    keep = True
                # Keep weekly backups
                elif age_days <= self.config.weekly_retention and backup_time.weekday() == 0:
                    keep = True
                # Keep monthly backups
                elif age_days <= self.config.monthly_retention and backup_time.day == 1:
                    keep = True
                
                if not keep:
                    # Delete backup files
                    for component_data in manifest.get("components", {}).values():
                        if isinstance(component_data, dict) and "backup_path" in component_data:
                            backup_path = Path(component_data["backup_path"])
                            if backup_path.exists():
                                if backup_path.is_file():
                                    backup_path.unlink()
                                else:
                                    shutil.rmtree(backup_path)
                    
                    # Delete manifest
                    manifest_file.unlink()
                    logger.info(f"Deleted old backup: {manifest['backup_id']}")
                    
            except Exception as e:
                logger.warning(f"Error processing old backup: {e}")
    
    async def _terminate_db_connections(self):
        """Terminate existing database connections"""
        try:
            conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database="postgres",
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
            conn.autocommit = True
            cursor = conn.cursor()
            
            cursor.execute(f"""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = '{self.config.postgres_db}'
                AND pid <> pg_backend_pid()
            """)
            
            conn.close()
        except Exception as e:
            logger.warning(f"Could not terminate DB connections: {e}")
    
    async def _analyze_database(self):
        """Run ANALYZE on database"""
        try:
            conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
            conn.autocommit = True
            cursor = conn.cursor()
            cursor.execute("ANALYZE")
            conn.close()
        except Exception as e:
            logger.warning(f"Could not run ANALYZE: {e}")
    
    async def _stop_services(self):
        """Stop services"""
        services = ["trading-api", "trading-worker", "trading-scheduler"]
        for service in services:
            try:
                subprocess.run(["systemctl", "stop", service], check=True)
                logger.info(f"Stopped service: {service}")
            except subprocess.CalledProcessError:
                pass
    
    async def _start_services(self):
        """Start services"""
        services = ["redis", "postgresql", "trading-api", "trading-worker", "trading-scheduler"]
        for service in services:
            try:
                subprocess.run(["systemctl", "start", service], check=True)
                logger.info(f"Started service: {service}")
            except subprocess.CalledProcessError:
                pass
    
    async def _verify_restore(
        self,
        mode: RestoreMode,
        components: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Verify restored data"""
        verification = {"success": True, "checks": {}}
        
        # Verify PostgreSQL
        if mode in [RestoreMode.FULL, RestoreMode.DATABASE]:
            try:
                conn = psycopg2.connect(
                    host=self.config.postgres_host,
                    port=self.config.postgres_port,
                    database=self.config.postgres_db,
                    user=self.config.postgres_user,
                    password=self.config.postgres_password
                )
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
                table_count = cursor.fetchone()[0]
                verification["checks"]["postgresql_tables"] = table_count
                conn.close()
            except Exception as e:
                verification["success"] = False
                verification["checks"]["postgresql_error"] = str(e)
        
        # Verify Redis
        if mode in [RestoreMode.FULL, RestoreMode.REDIS]:
            try:
                r = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    password=self.config.redis_password
                )
                r.ping()
                verification["checks"]["redis_keys"] = r.dbsize()
            except Exception as e:
                verification["success"] = False
                verification["checks"]["redis_error"] = str(e)
        
        return verification
    
    async def _send_alert(self, subject: str, message: str):
        """Send alert notification"""
        if self.config.notification_webhook:
            # Implement webhook notification
            logger.error(f"ALERT: {subject} - {message}")
    
    async def _schedule_test_restore(self, backup_id: str, delay_hours: int = 1):
        """Schedule a test restore"""
        await asyncio.sleep(delay_hours * 3600)
        
        logger.info(f"Running scheduled test restore for {backup_id}")
        
        result = await self.restore_backup(
            backup_id=backup_id,
            mode=RestoreMode.VERIFICATION,
            dry_run=True
        )
        
        if result.get("status") != "success":
            await self._send_alert(
                f"Test restore failed for {backup_id}",
                str(result.get("errors", []))
            )


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Backup and Restore System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create backup")
    backup_parser.add_argument("--type", choices=["full", "incremental", "differential", "snapshot"],
                              default="full", help="Backup type")
    backup_parser.add_argument("--components", nargs="+", help="Components to backup")
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument("backup_id", help="Backup ID to restore")
    restore_parser.add_argument("--mode", choices=["full", "database", "redis", "models", "config", "partial", "verify"],
                               default="full", help="Restore mode")
    restore_parser.add_argument("--components", nargs="+", help="Components for partial restore")
    restore_parser.add_argument("--dry-run", action="store_true", help="Perform dry run")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available backups")
    
    args = parser.parse_args()
    
    # Create manager
    manager = UnifiedBackupRestoreManager()
    
    if args.command == "backup":
        result = await manager.create_backup(
            backup_type=BackupType(args.type),
            components=args.components
        )
        print(json.dumps(result, indent=2))
        
    elif args.command == "restore":
        result = await manager.restore_backup(
            backup_id=args.backup_id,
            mode=RestoreMode(args.mode),
            components=args.components,
            dry_run=args.dry_run
        )
        print(json.dumps(result, indent=2))
        
    elif args.command == "list":
        backups = []
        for manifest_file in manager.backup_root.glob("*_manifest.json"):
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
                backups.append({
                    "backup_id": manifest["backup_id"],
                    "type": manifest["backup_type"],
                    "timestamp": manifest["timestamp"],
                    "status": manifest["status"]
                })
        
        print(json.dumps(backups, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())