import asyncio
import uuid
import pytest

from trading_common.experiment_tracking import get_experiment_tracker, RunStatus

@pytest.mark.asyncio
async def test_minimal_experiment_run_lifecycle():
    tracker = await get_experiment_tracker()
    run_id = str(uuid.uuid4())
    run = await tracker.start_run("quick_exp", run_id, run_name="trial")
    assert run.run_id == run_id
    await tracker.log_param(run_id, "lr", 0.01)
    await tracker.log_metric(run_id, "loss", 0.5, step=0)
    await tracker.log_metric(run_id, "loss", 0.4, step=1)
    await tracker.finish_run(run_id)
    stored = await tracker.get_run(run_id)
    assert stored is not None
    assert stored.status == RunStatus.FINISHED
    metrics = await tracker.get_run_metrics(run_id, "loss")
    assert len(metrics) == 2
