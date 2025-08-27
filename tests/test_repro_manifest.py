import json
from pathlib import Path
import sys

# Add python-common path dynamically (hyphenated directory name)
root = Path(__file__).resolve().parents[1]
python_common = root / "shared" / "python-common"
if str(python_common) not in sys.path:
    sys.path.append(str(python_common))

try:  # runtime import resolution; test will be skipped if unavailable
    from trading_common import registry_utils  # type: ignore
except Exception:  # pragma: no cover
    registry_utils = None  # type: ignore


def test_build_repro_manifest_minimal(tmp_path: Path):
    if registry_utils is None:
        import pytest
        pytest.skip("trading_common not available")
    dummy_artifact = tmp_path / "model.pkl"
    dummy_artifact.write_bytes(b"binarymodeldata")
    manifest = registry_utils.build_repro_manifest(
        model_name="test_model",
        version="0.1.0",
        training_config_hash="abcd1234efgh5678",
        git_commit="deadbeefcafebabefeedface1234567890abcdef",
        config={"model_name": "test_model", "version": "0.1.0", "train_start": "2024-01-01T00:00:00", "train_end": "2024-02-01T00:00:00"},
        artifact_path=str(dummy_artifact),
    )
    assert manifest["schema_version"] == "1.0"
    assert manifest["id"].startswith("test_model:0.1.0:")
    assert manifest["environment"]["git_short"] == "deadbee"
    assert manifest["artifact"]["size_bytes"] == dummy_artifact.stat().st_size
    assert len(manifest["artifact"]["sha256"]) == 64
    assert manifest["data_window"]["train_start"].startswith("2024-01-01")
    # Ensure JSON serializable
    json.dumps(manifest)
