#!/usr/bin/env python3
"""Lightweight Experiment Tracking Module.

Scope: Provide minimal in-repo alternative to MLflow (explicitly out-of-scope) for:
- Experiment Run lifecycle (start / finish / fail)
- Parameter logging (immutable once set)
- Metrics logging (time-series; last value + history)
- Artifact reference registration (paths / hashes)
- Linkage to Model Registry entries (model_name+version)

Non-Goals (see NON_GOALS.md): full MLflow parity, permissioning, external UI.

Tables (created on initialize):
  experiment_runs(
      run_id UUID PK,
      experiment_name VARCHAR,
      run_name VARCHAR,
      status VARCHAR ENUM['RUNNING','FINISHED','FAILED'],
      user_id VARCHAR NULL,
      model_name VARCHAR NULL,
      model_version VARCHAR NULL,
      parent_run_id UUID NULL,
      start_time TIMESTAMP,
      end_time TIMESTAMP NULL,
      tags JSONB,
      notes TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  )
  experiment_run_params(run_id UUID, param_name VARCHAR, param_value TEXT)
  experiment_run_metrics(run_id UUID, metric_name VARCHAR, metric_value DOUBLE PRECISION, step BIGINT, timestamp TIMESTAMP)
  experiment_run_artifacts(run_id UUID, artifact_name VARCHAR, artifact_path TEXT, content_hash VARCHAR(64), metadata JSONB, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
Indexes optimize lookup by experiment_name, model_name+model_version, metric_name.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List, AsyncIterator
import logging

try:  # local project DB util
    from .database_manager import get_database_manager as get_database  # type: ignore
except ImportError:  # pragma: no cover
    async def get_database():  # type: ignore
        raise RuntimeError("database_manager module not available")

logger = logging.getLogger(__name__)

class RunStatus:
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"

@dataclass
class ExperimentRun:
    run_id: str
    experiment_name: str
    run_name: Optional[str]
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    user_id: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    parent_run_id: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

    def to_summary(self) -> Dict[str, Any]:
        return {
            'run_id': self.run_id,
            'experiment_name': self.experiment_name,
            'run_name': self.run_name,
            'status': self.status,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'tags': self.tags or {},
        }

class ExperimentTracker:
    def __init__(self):
        self.db = None

    async def initialize(self):
        self.db = await get_database()
        await self._create_tables()
        logger.info("Experiment tracker initialized")

    async def _create_tables(self):
        await self.db.execute("""
        CREATE TABLE IF NOT EXISTS experiment_runs (
            run_id UUID PRIMARY KEY,
            experiment_name VARCHAR(255) NOT NULL,
            run_name VARCHAR(255),
            status VARCHAR(20) NOT NULL,
            user_id VARCHAR(255),
            model_name VARCHAR(255),
            model_version VARCHAR(50),
            parent_run_id UUID,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            tags JSONB,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        await self.db.execute("""
        CREATE INDEX IF NOT EXISTS idx_experiment_runs_exp ON experiment_runs(experiment_name)
        """)
        await self.db.execute("""
        CREATE INDEX IF NOT EXISTS idx_experiment_runs_model ON experiment_runs(model_name, model_version)
        """)
        await self.db.execute("""
        CREATE TABLE IF NOT EXISTS experiment_run_params (
            run_id UUID REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
            param_name VARCHAR(255) NOT NULL,
            param_value TEXT,
            PRIMARY KEY(run_id, param_name)
        )
        """)
        await self.db.execute("""
        CREATE TABLE IF NOT EXISTS experiment_run_metrics (
            run_id UUID REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
            metric_name VARCHAR(255) NOT NULL,
            metric_value DOUBLE PRECISION,
            step BIGINT DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(run_id, metric_name, step)
        )
        """)
        await self.db.execute("""
        CREATE INDEX IF NOT EXISTS idx_experiment_run_metrics_name ON experiment_run_metrics(metric_name)
        """)
        await self.db.execute("""
        CREATE TABLE IF NOT EXISTS experiment_run_artifacts (
            run_id UUID REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
            artifact_name VARCHAR(255) NOT NULL,
            artifact_path TEXT NOT NULL,
            content_hash VARCHAR(64),
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(run_id, artifact_name)
        )
        """)

    async def start_run(self, experiment_name: str, run_id: str, run_name: Optional[str] = None, *, user_id: Optional[str] = None, parent_run_id: Optional[str] = None, tags: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None, model_version: Optional[str] = None, notes: Optional[str] = None) -> ExperimentRun:
        await self.db.execute("""
        INSERT INTO experiment_runs(run_id, experiment_name, run_name, status, user_id, model_name, model_version, parent_run_id, start_time, tags, notes)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, [run_id, experiment_name, run_name, RunStatus.RUNNING, user_id, model_name, model_version, parent_run_id, datetime.utcnow(), json.dumps(tags or {}), notes])
        return ExperimentRun(run_id=run_id, experiment_name=experiment_name, run_name=run_name, status=RunStatus.RUNNING, start_time=datetime.utcnow(), user_id=user_id, model_name=model_name, model_version=model_version, parent_run_id=parent_run_id, tags=tags or {}, notes=notes)

    async def log_param(self, run_id: str, name: str, value: Any):
        # Immutable param semantics: ignore if exists
        await self.db.execute("""
        INSERT INTO experiment_run_params(run_id, param_name, param_value)
        VALUES (%s,%s,%s) ON CONFLICT DO NOTHING
        """, [run_id, name, str(value)])

    async def log_params(self, run_id: str, params: Dict[str, Any]):
        for k, v in params.items():
            await self.log_param(run_id, k, v)

    async def log_metric(self, run_id: str, name: str, value: float, step: int = 0, timestamp: Optional[datetime] = None):
        ts = timestamp or datetime.utcnow()
        await self.db.execute("""
        INSERT INTO experiment_run_metrics(run_id, metric_name, metric_value, step, timestamp)
        VALUES (%s,%s,%s,%s,%s)
        ON CONFLICT (run_id, metric_name, step) DO UPDATE SET metric_value = EXCLUDED.metric_value, timestamp = EXCLUDED.timestamp
        """, [run_id, name, float(value), step, ts])

    async def log_metrics(self, run_id: str, metrics: Dict[str, float], step: int = 0):
        for k, v in metrics.items():
            await self.log_metric(run_id, k, v, step=step)

    async def log_artifact(self, run_id: str, name: str, path: str, *, content_hash: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        if content_hash is None:
            # Lightweight hash if feasible
            try:
                with open(path, 'rb') as f:
                    content_hash = hashlib.sha256(f.read(1024 * 1024)).hexdigest()  # hash first 1MB
            except FileNotFoundError:
                content_hash = None
        await self.db.execute("""
        INSERT INTO experiment_run_artifacts(run_id, artifact_name, artifact_path, content_hash, metadata)
        VALUES (%s,%s,%s,%s,%s)
        ON CONFLICT (run_id, artifact_name) DO UPDATE SET artifact_path=EXCLUDED.artifact_path, content_hash=EXCLUDED.content_hash, metadata=EXCLUDED.metadata
        """, [run_id, name, path, content_hash, json.dumps(metadata or {})])

    async def finish_run(self, run_id: str, status: str = RunStatus.FINISHED):
        end_time = datetime.utcnow()
        await self.db.execute("""
        UPDATE experiment_runs SET status=%s, end_time=%s, updated_at=CURRENT_TIMESTAMP WHERE run_id=%s
        """, [status, end_time, run_id])

    async def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        row = await self.db.fetch_one("SELECT * FROM experiment_runs WHERE run_id=%s", [run_id])
        if not row:
            return None
        return ExperimentRun(
            run_id=row['run_id'],
            experiment_name=row['experiment_name'],
            run_name=row.get('run_name'),
            status=row['status'],
            start_time=row['start_time'],
            end_time=row.get('end_time'),
            user_id=row.get('user_id'),
            model_name=row.get('model_name'),
            model_version=row.get('model_version'),
            parent_run_id=row.get('parent_run_id'),
            tags=row.get('tags') or {},
            notes=row.get('notes')
        )

    async def list_runs(self, experiment_name: str, limit: int = 50) -> List[ExperimentRun]:
        rows = await self.db.fetch_all("SELECT * FROM experiment_runs WHERE experiment_name=%s ORDER BY start_time DESC LIMIT %s", [experiment_name, limit])
        return [ExperimentRun(
            run_id=r['run_id'], experiment_name=r['experiment_name'], run_name=r.get('run_name'), status=r['status'], start_time=r['start_time'], end_time=r.get('end_time'), user_id=r.get('user_id'), model_name=r.get('model_name'), model_version=r.get('model_version'), parent_run_id=r.get('parent_run_id'), tags=r.get('tags') or {}, notes=r.get('notes')) for r in rows]

    async def get_run_metrics(self, run_id: str, metric_name: Optional[str] = None) -> List[Dict[str, Any]]:
        if metric_name:
            rows = await self.db.fetch_all("SELECT metric_name, metric_value, step, timestamp FROM experiment_run_metrics WHERE run_id=%s AND metric_name=%s ORDER BY step", [run_id, metric_name])
        else:
            rows = await self.db.fetch_all("SELECT metric_name, metric_value, step, timestamp FROM experiment_run_metrics WHERE run_id=%s ORDER BY metric_name, step", [run_id])
        return [dict(r) for r in rows]

    async def get_run_params(self, run_id: str) -> Dict[str, str]:
        rows = await self.db.fetch_all("SELECT param_name, param_value FROM experiment_run_params WHERE run_id=%s", [run_id])
        return {r['param_name']: r['param_value'] for r in rows}

    async def get_run_artifacts(self, run_id: str) -> List[Dict[str, Any]]:
        rows = await self.db.fetch_all("SELECT artifact_name, artifact_path, content_hash, metadata, created_at FROM experiment_run_artifacts WHERE run_id=%s", [run_id])
        return [dict(r) for r in rows]

    async def stream_metrics(self, run_id: str, metric_name: str) -> AsyncIterator[Dict[str, Any]]:
        # Simple polling based stream (could be replaced with LISTEN/NOTIFY or websockets)
        last_step = -1
        while True:
            rows = await self.db.fetch_all("SELECT metric_name, metric_value, step, timestamp FROM experiment_run_metrics WHERE run_id=%s AND metric_name=%s AND step > %s ORDER BY step", [run_id, metric_name, last_step])
            if rows:
                for r in rows:
                    last_step = max(last_step, r['step'])
                    yield dict(r)
            # Stop condition heuristic: no updates for some iterations after run finished
            run = await self.get_run(run_id)
            if run and run.status != RunStatus.RUNNING and last_step >= 0:
                # one final fetch then break
                break


async def get_experiment_tracker() -> ExperimentTracker:
    if getattr(get_experiment_tracker, "_instance", None) is None:  # type: ignore[attr-defined]
        inst = ExperimentTracker()
        await inst.initialize()
        setattr(get_experiment_tracker, "_instance", inst)  # type: ignore[attr-defined]
    return getattr(get_experiment_tracker, "_instance")  # type: ignore[attr-defined]

__all__ = [
    'ExperimentTracker', 'ExperimentRun', 'RunStatus', 'get_experiment_tracker'
]
