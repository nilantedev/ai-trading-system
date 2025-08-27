import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
python_common = root / "shared" / "python-common"
if str(python_common) not in sys.path:
    sys.path.append(str(python_common))

# Import modules (best effort)
from shared.storage import minio_storage  # type: ignore
from shared.vector import weaviate_schema  # type: ignore


def test_storage_metric_helper_noop():
    # Call internal helper directly to ensure it swallows errors without metrics registry
    minio_storage._record_storage_metric("ensure_bucket", "success", 0.0)  # type: ignore


def test_vector_metric_helper_noop():
    weaviate_schema._record_vector_metric("schema_apply", "success", 0.0)  # type: ignore
