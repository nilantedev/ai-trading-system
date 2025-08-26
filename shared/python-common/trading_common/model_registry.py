#!/usr/bin/env python3
"""Extended Model Registry Abstraction.
Wraps DB persistence with state machine, promotion policies, and drift integration hooks.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List

# NOTE: get_database import path may differ; using local lazy import wrapper to avoid attribute errors.
try:  # narrow to ImportError only
    from .database_manager import get_database_manager as get_database  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback placeholder
    async def get_database():  # type: ignore
        raise RuntimeError("get_database import failed; provide trading_common.database_manager.get_database_manager")

class ModelState(str, Enum):
    DRAFT = "DRAFT"
    TRAINED = "TRAINED"
    VALIDATED = "VALIDATED"
    STAGING = "STAGING"
    PRODUCTION = "PRODUCTION"
    RETIRED = "RETIRED"

@dataclass
class RegistryEntry:
    model_name: str
    version: str
    model_type: str
    state: ModelState
    metrics: Dict[str, Any]
    config: Dict[str, Any]
    artifact_path: str
    dataset_hash: str
    feature_graph_hash: str
    training_config_hash: str
    git_commit: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    promotion_history: List[Dict[str, Any]] | None = None
    blocked: bool = False

    def to_row(self) -> Dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        return data

PROMOTION_KEYS = [
    (ModelState.DRAFT, ModelState.TRAINED),
    (ModelState.TRAINED, ModelState.VALIDATED),
    (ModelState.VALIDATED, ModelState.STAGING),
    (ModelState.STAGING, ModelState.PRODUCTION),
]

class PromotionError(Exception):
    pass

class ModelRegistry:
    def __init__(self):
        self.db = None

    async def initialize(self):
        self.db = await get_database()
        # Extend existing table if needed
        await self.db.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            model_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            model_name VARCHAR(255) NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            version VARCHAR(50) NOT NULL,
            state VARCHAR(30) NOT NULL DEFAULT 'DRAFT',
            config JSON NOT NULL,
            training_metrics JSON,
            artifact_path TEXT,
            dataset_hash VARCHAR(64),
            feature_graph_hash VARCHAR(64),
            training_config_hash VARCHAR(64),
            git_commit VARCHAR(64),
            promotion_history JSON,
            blocked BOOLEAN DEFAULT FALSE,
            is_active BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(model_name, version)
        )
        """)
        await self.db.execute("""
        CREATE TABLE IF NOT EXISTS model_drift_reports (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            model_name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            feature_name VARCHAR(255),
            psi FLOAT, ks FLOAT, mean_shift FLOAT, var_shift FLOAT,
            severity VARCHAR(20),
            reference_period JSON,
            live_period JSON
        )
        """)

    async def upsert_entry(self, entry: RegistryEntry):
        await self.db.execute("""
        INSERT INTO model_registry
        (model_name, model_type, version, state, config, training_metrics, artifact_path,
         dataset_hash, feature_graph_hash, training_config_hash, git_commit, promotion_history,
         blocked, is_active)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (model_name, version) DO UPDATE SET
          state = EXCLUDED.state,
          training_metrics = EXCLUDED.training_metrics,
          artifact_path = EXCLUDED.artifact_path,
          dataset_hash = EXCLUDED.dataset_hash,
          feature_graph_hash = EXCLUDED.feature_graph_hash,
          training_config_hash = EXCLUDED.training_config_hash,
          git_commit = EXCLUDED.git_commit,
          promotion_history = EXCLUDED.promotion_history,
          blocked = EXCLUDED.blocked,
          is_active = EXCLUDED.is_active,
          updated_at = CURRENT_TIMESTAMP
        """, [
            entry.model_name,
            entry.model_type,
            entry.version,
            entry.state.value,
            json.dumps(entry.config, default=str),
            json.dumps(entry.metrics, default=str),
            entry.artifact_path,
            entry.dataset_hash,
            entry.feature_graph_hash,
            entry.training_config_hash,
            entry.git_commit,
            json.dumps(entry.promotion_history or []),
            entry.blocked,
            True if entry.state == ModelState.PRODUCTION else False
        ])

    async def get_entry(self, model_name: str, version: Optional[str] = None) -> Optional[RegistryEntry]:
        q = "SELECT * FROM model_registry WHERE model_name = %s"
        params = [model_name]
        if version:
            q += " AND version = %s"
            params.append(version)
        q += " ORDER BY created_at DESC LIMIT 1"
        row = await self.db.fetch_one(q, params)
        if not row:
            return None
        return self._row_to_entry(row)

    def _row_to_entry(self, row: Dict[str, Any]) -> RegistryEntry:
        return RegistryEntry(
            model_name=row['model_name'],
            model_type=row['model_type'],
            version=row['version'],
            state=ModelState(row['state']),
            config=row['config'],
            metrics=row.get('training_metrics') or {},
            artifact_path=row.get('artifact_path') or '',
            dataset_hash=row.get('dataset_hash') or '',
            feature_graph_hash=row.get('feature_graph_hash') or '',
            training_config_hash=row.get('training_config_hash') or '',
            git_commit=row.get('git_commit') or '',
            promotion_history=row.get('promotion_history') or [],
            blocked=row.get('blocked') or False,
            created_at=row.get('created_at') or None,
            updated_at=row.get('updated_at') or None
        )

    async def promote(self, model_name: str, version: str, target: ModelState, reason: str, policy: Dict[str, Any]):
        entry = await self.get_entry(model_name, version)
        if not entry:
            raise PromotionError("Model version not found")
        if (entry.state, target) not in PROMOTION_KEYS and not (entry.state == target):
            raise PromotionError(f"Illegal promotion {entry.state} -> {target}")
        # Policy enforcement (example thresholds)
        metrics = entry.metrics or {}
        if target == ModelState.PRODUCTION:
            min_sharpe = policy.get('min_sharpe')
            max_dd = policy.get('max_drawdown')
            sharpe = metrics.get('sharpe_ratio') or metrics.get('Sharpe Ratio')
            drawdown = metrics.get('max_drawdown') or metrics.get('Max Drawdown')
            if min_sharpe and (sharpe is None or sharpe < min_sharpe):
                raise PromotionError(f"Sharpe {sharpe} < required {min_sharpe}")
            if max_dd and (drawdown is None or drawdown < max_dd):  # drawdown negative
                raise PromotionError(f"Drawdown {drawdown} worse than {max_dd}")
        # Update entry
        entry.state = target
        history = entry.promotion_history or []
        history.append({
            'to': target.value,
            'at': datetime.utcnow().isoformat(),
            'reason': reason
        })
        entry.promotion_history = history
        await self.upsert_entry(entry)
        return entry

_REGISTRY_SINGLETON: Optional[ModelRegistry] = None

async def get_model_registry() -> ModelRegistry:
    """Singleton accessor without explicit global usage in calling scope."""
    if getattr(get_model_registry, "_instance", None) is None:  # type: ignore
        instance = ModelRegistry()
        await instance.initialize()
        setattr(get_model_registry, "_instance", instance)  # type: ignore
    return getattr(get_model_registry, "_instance")  # type: ignore

__all__ = [
    'ModelRegistry', 'get_model_registry', 'RegistryEntry', 'ModelState', 'PromotionError'
]
