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
import base64
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


def _env_minio_secret() -> str:
    """Resolve MinIO secret from env, supporting optional base64 encoding.

    Priority:
      1) MINIO_SECRET_KEY_B64 or MINIO_ROOT_PASSWORD_B64 (base64 decoded)
      2) MINIO_SECRET_KEY
      3) MINIO_ROOT_PASSWORD
    """
    b64 = os.getenv("MINIO_SECRET_KEY_B64") or os.getenv("MINIO_ROOT_PASSWORD_B64")
    if b64:
        try:
            return base64.b64decode(b64).decode("utf-8")
        except Exception:
            # Fall through to non-b64 vars if decoding fails
            pass
    return os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", ""))


@dataclass(frozen=True)
class MinIOConfig:
    endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key: str = os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", ""))
    secret_key: str = _env_minio_secret()
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
    """Create (or return cached) MinIO client with basic network timeouts.

    Uses environment overrides:
      MINIO_HTTP_CLIENT_TIMEOUT (seconds, default 30)
    """
    cfg = cfg or MinIOConfig()
    cfg.validate()
    timeout = float(os.getenv("MINIO_HTTP_CLIENT_TIMEOUT", "30"))
    try:
        # The MinIO Python client allows a custom HTTP client via http_client=, but for
        # simplicity we leverage the internal default and rely on urllib3 timeouts by
        # patching env if needed in future. Placeholder kept for future injection.
        client = Minio(
            endpoint=cfg.endpoint,
            access_key=cfg.access_key,
            secret_key=cfg.secret_key,
            secure=cfg.secure,
            region=cfg.region,
            # No direct timeout param; we will wrap operations with our own timeouts/backoff.
        )
    except Exception as e:  # pragma: no cover
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


def ensure_bucket(name: str, *, client: Optional[Minio] = None, retries: int = 3, backoff: float = 0.5) -> bool:
    """Idempotently ensure bucket exists with simple retry/backoff.

    Returns True if created, False if already existed.
    """
    client = client or get_minio_client()
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        start = perf_counter()
        try:
            if client.bucket_exists(name):  # type: ignore[attr-defined]
                _record_storage_metric('ensure_bucket', 'success', start)
                return False
            client.make_bucket(name)  # type: ignore[attr-defined]
            _record_storage_metric('ensure_bucket', 'success', start)
            return True
        except Exception as e:  # pragma: no cover
            last_error = e
            _record_storage_metric('ensure_bucket', 'error', start)
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
            else:
                raise StorageError(f"Bucket ensure failed after {retries} attempts: {e}") from e
    raise StorageError(f"Bucket ensure failed: {last_error}")


def put_bytes(
    bucket: str,
    data: bytes,
    object_key: str,
    *,
    content_type: str = "application/octet-stream",
    verify_integrity: bool = True,
    client: Optional[Minio] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    retries: int = 3,
    backoff: float = 0.5,
) -> Dict[str, Any]:
    """Upload bytes to MinIO with retry/backoff and optional post-upload integrity check."""
    client = client or get_minio_client()
    digest = sha256_digest(data)
    size = len(data)
    last_error: Optional[Exception] = None
    from io import BytesIO
    for attempt in range(retries):
        start = perf_counter()
        try:
            ensure_bucket(bucket, client=client)
            stream = BytesIO(data)
            client.put_object(
                bucket,
                object_key,
                stream,
                length=size,
                content_type=content_type,
                metadata=extra_headers or {},
            )  # type: ignore[attr-defined]
            etag = None
            if verify_integrity:
                stat = client.stat_object(bucket, object_key)  # type: ignore[attr-defined]
                etag = getattr(stat, 'etag', None)
                if getattr(stat, 'size', size) != size:
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
        except Exception as e:  # pragma: no cover
            last_error = e
            _record_storage_metric('put_bytes', 'error', start)
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
            else:
                raise StorageError(f"Upload failed after {retries} attempts: {e}") from e
    raise StorageError(f"Upload failed: {last_error}")


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
