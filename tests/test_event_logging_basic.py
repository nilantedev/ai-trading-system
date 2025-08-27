import logging
from io import StringIO
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
python_common = root / "shared" / "python-common"
if str(python_common) not in sys.path:
    sys.path.append(str(python_common))

from trading_common.event_logging import emit_event  # type: ignore


def test_emit_event_basic():
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger('trading_common.event_logging')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    emit_event('model.test.event', model_name='m1', version='0.0.1', detail='ok')
    handler.flush()
    contents = stream.getvalue()
    assert 'MODEL_EVENT' in contents
    assert 'model.test.event' in contents
    assert '"model_name": "m1"' in contents
    logger.removeHandler(handler)
