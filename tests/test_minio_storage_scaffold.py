"""Minimal tests for MinIO storage scaffold (no live server required).

These tests focus on pure helper logic to avoid network dependency in CI.
"""
from shared.storage.minio_storage import (
    build_model_artifact_key,
    sha256_digest,
)


def test_build_model_artifact_key_basic():
    key = build_model_artifact_key("MyModel", "v1", "weights.bin")
    assert key == "models/mymodel/v1/weights.bin"


def test_build_model_artifact_key_sanitization():
    key = build_model_artifact_key("nested/model", "v2", "file\\name.txt")
    assert key == "models/nested-model/v2/file_name.txt"


def test_sha256_digest():
    data = b"hello world"
    digest = sha256_digest(data)
    assert digest == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
