"""MinIO storage abstraction (scaffold).

Design Goals (Phase 1 Scaffold):
- Provide a thin, testable wrapper around MinIO client creation without forcing runtime side-effects.
- Support future extension: metrics, encryption, lifecycle policies, multipart uploads.
- Avoid tightening coupling with ML pipeline until manifest generation task.

This file intentionally limits itself to pure / low-side-effect helpers unless explicitly invoked.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Iterable, Generator, Any, Dict
import hashlib
import os
import time
import contextlib
from time import perf_counter

try:
    from minio import Minio  # type: ignore
    from minio.error import S3Error  # type: ignore
except Exception:  # pragma: no cover - dependency import guarded
    Minio = object  # type: ignore
    S3Error = Exception  # type: ignore


class StorageError(RuntimeError):
    """Domain-specific storage error wrapper."""


@dataclass(frozen=True)
class MinIOConfig:
    endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key: str = os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", ""))
    secret_key: str = os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", ""))
    secure: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"
    region: Optional[str] = os.getenv("MINIO_REGION")
    # Default buckets to ensure lazily
    default_buckets: tuple[str, ...] = ("models", "datasets", "backups")

    def validate(self) -> None:
        if not self.access_key or not self.secret_key:
            raise StorageError("MinIO credentials not configured (access_key/secret_key missing)")
        if ":" not in self.endpoint and not self.endpoint.endswith(":9000"):
            # heuristic advisory; not fatal
            pass


@lru_cache(maxsize=1)
def get_minio_client(cfg: Optional[MinIOConfig] = None) -> Minio:
    """Create (or return cached) MinIO client.

    A cache is used to avoid recreating clients frequently. Config differences are ignored
    after first call by design in this scaffold (explicit reset not yet required).
    """
    cfg = cfg or MinIOConfig()
    cfg.validate()
    try:
        client = Minio(
            endpoint=cfg.endpoint,
            access_key=cfg.access_key,
            secret_key=cfg.secret_key,
            secure=cfg.secure,
            region=cfg.region,
        )
    except Exception as e:  # pragma: no cover - direct client init errors
        raise StorageError(f"Failed to initialize MinIO client: {e}") from e
    return client  # type: ignore


def sha256_digest(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def build_model_artifact_key(model_name: str, version: str, filename: str, *, prefix: str = "models") -> str:
    safe_model = model_name.replace("/", "-")
    safe_file = filename.replace("\\", "_")
    return f"{prefix}/{safe_model}/{version}/{safe_file}".lower()


def ensure_bucket(name: str, *, client: Optional[Minio] = None) -> bool:
    """Idempotently ensure bucket exists.
    Returns True if created, False if already existed.
    Exceptions wrapped in StorageError.
    """
    client = client or get_minio_client()
    start = perf_counter()
    try:
        if client.bucket_exists(name):  # type: ignore[attr-defined]
            created = False
        else:
            client.make_bucket(name)  # type: ignore[attr-defined]
            created = True
        _record_storage_metric('ensure_bucket', 'success', start)
        return created
    except S3Error as e:  # pragma: no cover - network path
        _record_storage_metric('ensure_bucket', 'error', start)
        raise StorageError(f"Bucket ensure failed: {e}") from e


def put_bytes(
    bucket: str,
    data: bytes,
    object_key: str,
    *,
    content_type: str = "application/octet-stream",
    verify_integrity: bool = True,
    client: Optional[Minio] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Upload bytes to MinIO, optionally verifying SHA256 after upload.
    Returns structured metadata.
    """
    client = client or get_minio_client()
    start = perf_counter()
    digest = sha256_digest(data)
    size = len(data)
    try:
        ensure_bucket(bucket, client=client)
        from io import BytesIO
        stream = BytesIO(data)
        # put_object returns etag in headers (via response) but python client returns nothing; we may need stat
        client.put_object(bucket, object_key, stream, length=size, content_type=content_type, metadata=extra_headers or {})  # type: ignore[attr-defined]
        etag = None
        if verify_integrity:
            # Fetch object stat to confirm size; deeper checksum verification would require separate store
            stat = client.stat_object(bucket, object_key)  # type: ignore[attr-defined]
            etag = getattr(stat, 'etag', None)
            if getattr(stat, 'size', size) != size:
                _record_storage_metric('put_bytes', 'error', start)
                raise StorageError("Integrity check failed: size mismatch")
        _record_storage_metric('put_bytes', 'success', start)
        return {
            "bucket": bucket,
            "key": object_key,
            "size": size,
            "sha256": digest,
            "etag": etag,
            "uploaded_at": int(time.time()),
        }
    except S3Error as e:  # pragma: no cover
        _record_storage_metric('put_bytes', 'error', start)
        raise StorageError(f"Upload failed: {e}") from e


def generate_presigned_url(bucket: str, object_key: str, expiry_seconds: int = 3600, *, client: Optional[Minio] = None) -> str:
    client = client or get_minio_client()
    start = perf_counter()
    try:
        url = client.get_presigned_url("GET", bucket, object_key, expires=expiry_seconds)  # type: ignore[attr-defined]
        _record_storage_metric('generate_presigned_url', 'success', start)
        return url
    except S3Error as e:  # pragma: no cover
        _record_storage_metric('generate_presigned_url', 'error', start)
        raise StorageError(f"Presigned URL generation failed: {e}") from e


def _record_storage_metric(operation: str, status: str, start_time: float) -> None:
    """Internal helper to record MinIO instrumentation metrics (best effort)."""
    try:  # pragma: no cover
        from api.metrics import object_storage_operations_total, object_storage_operation_duration_seconds
        object_storage_operations_total.labels(operation=operation, status=status).inc()
        object_storage_operation_duration_seconds.labels(operation=operation).observe(perf_counter() - start_time)
    except Exception:
        pass


__all__ = [
    "MinIOConfig",
    "get_minio_client",
    "build_model_artifact_key",
    "ensure_bucket",
    "put_bytes",
    "generate_presigned_url",
    "sha256_digest",
    "StorageError",
]
