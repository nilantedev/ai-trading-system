#!/usr/bin/env python3
"""Model Drift Detection Utilities.

Computes distribution drift metrics (PSI, KS, mean/variance shifts) and persists
results to the model_drift_reports table defined by the model registry.

Design goals:
- Lightweight: operate on pandas DataFrames already materialized by feature store.
- Deterministic: stable ordering / hashing for reproducibility.
- Extensible: severity thresholds configurable via function args (TODO external config).

Public API:
- compute_feature_drift(reference: pd.Series, live: pd.Series) -> dict
- compute_drift_frame(reference_df, live_df, feature_cols) -> List[dict]
- classify_severity(drift_dict) -> str
- build_reference_live_windows(feature_store, symbols, features, ref_period, live_period)
- persist_drift_reports(db, model_name, version, reports)
- run_model_drift_scan(model_name, version=None, live_window_days=7, reference_window_days=30)

NOTE: Database persistence relies on registry DB interface having an async execute method
compatible with parameterized inserts similar to model_registry.ModelRegistry usage.
"""
from __future__ import annotations

import math
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

try:  # pragma: no cover
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
    from scipy import stats  # type: ignore
except ImportError:  # pragma: no cover
    pd = None  # type: ignore
    np = None  # type: ignore
    stats = None  # type: ignore

from .feature_store import get_feature_store
from .model_registry import get_model_registry

logger = logging.getLogger(__name__)

# Default severity thresholds (can be externalized later)
PSI_THRESHOLDS = {
    "none": 0.1,      # PSI < 0.1 no significant drift
    "low": 0.2,       # 0.1 <= PSI < 0.2 low drift
    "medium": 0.3,    # 0.2 <= PSI < 0.3 medium drift
    # PSI >= 0.3 high drift
}

KS_THRESHOLDS = {
    "none": 0.05,
    "low": 0.1,
    "medium": 0.2,
}

MEAN_SHIFT_STD_MULT = 0.25  # mean shift above 0.25 sigma flagged
VAR_SHIFT_MULT = 0.5        # variance change >50% flagged


def _psi(expected: 'pd.Series', actual: 'pd.Series', bins: int = 10) -> float:
    """Population Stability Index between two distributions.
    Bins are derived from expected quantiles.
    """
    if pd is None:
        raise RuntimeError("pandas required for PSI computation")
    expected_clean = expected.dropna()
    actual_clean = actual.dropna()
    if len(expected_clean) == 0 or len(actual_clean) == 0:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = expected_clean.quantile(quantiles).values
    # Ensure uniqueness for pd.cut
    cuts = pd.Series(cuts).drop_duplicates().values
    if len(cuts) < 2:
        return 0.0
    expected_bins = pd.cut(expected_clean, bins=cuts, include_lowest=True)
    actual_bins = pd.cut(actual_clean, bins=cuts, include_lowest=True)
    expected_dist = expected_bins.value_counts(normalize=True)
    actual_dist = actual_bins.value_counts(normalize=True)
    # Align indexes
    dist = pd.DataFrame({'e': expected_dist, 'a': actual_dist}).fillna(1e-8)
    psi = ((dist['e'] - dist['a']) * np.log(dist['e'] / dist['a'])).sum()
    return float(max(psi, 0.0))


def _ks(expected: 'pd.Series', actual: 'pd.Series') -> float:
    if stats is None:
        return 0.0
    expected_clean = expected.dropna()
    actual_clean = actual.dropna()
    if len(expected_clean) == 0 or len(actual_clean) == 0:
        return 0.0
    try:
        ks_stat, _ = stats.ks_2samp(expected_clean.values, actual_clean.values)
        return float(ks_stat)
    except Exception:
        return 0.0


def compute_feature_drift(expected: 'pd.Series', actual: 'pd.Series') -> Dict[str, Any]:
    """Compute drift metrics for a single feature."""
    psi_val = _psi(expected, actual)
    ks_val = _ks(expected, actual)
    exp_mean, act_mean = expected.mean(), actual.mean()
    exp_std, act_std = expected.std(ddof=0), actual.std(ddof=0)
    mean_shift = 0.0 if exp_std == 0 or math.isnan(exp_std) else abs(act_mean - exp_mean) / (exp_std + 1e-12)
    var_shift = 0.0 if exp_std == 0 or math.isnan(exp_std) else abs(act_std - exp_std) / (exp_std + 1e-12)
    drift = {
        'psi': psi_val,
        'ks': ks_val,
        'mean_shift': float(mean_shift),
        'var_shift': float(var_shift),
    }
    drift['severity'] = classify_severity(drift)
    return drift


def classify_severity(drift: Dict[str, Any]) -> str:
    psi = drift['psi']
    ks = drift['ks']
    mean_shift = drift['mean_shift']
    var_shift = drift['var_shift']

    # Determine PSI severity
    if psi < PSI_THRESHOLDS['none']:
        psi_sev = 'NONE'
    elif psi < PSI_THRESHOLDS['low']:
        psi_sev = 'LOW'
    elif psi < PSI_THRESHOLDS['medium']:
        psi_sev = 'MEDIUM'
    else:
        psi_sev = 'HIGH'

    # KS severity
    if ks < KS_THRESHOLDS['none']:
        ks_sev = 'NONE'
    elif ks < KS_THRESHOLDS['low']:
        ks_sev = 'LOW'
    elif ks < KS_THRESHOLDS['medium']:
        ks_sev = 'MEDIUM'
    else:
        ks_sev = 'HIGH'

    # Mean and variance shift flags
    mv_flags = []
    if mean_shift > MEAN_SHIFT_STD_MULT:
        mv_flags.append('MEAN')
    if var_shift > VAR_SHIFT_MULT:
        mv_flags.append('VAR')

    # Aggregate severity (max)
    order = ['NONE', 'LOW', 'MEDIUM', 'HIGH']
    agg = max([psi_sev, ks_sev], key=lambda s: order.index(s))
    if agg != 'NONE' and mv_flags:
        agg = agg  # Could escalate if needed
    return agg


def compute_drift_frame(reference_df: 'pd.DataFrame', live_df: 'pd.DataFrame', feature_cols: List[str]) -> List[Dict[str, Any]]:
    reports = []
    for col in feature_cols:
        if col not in reference_df.columns or col not in live_df.columns:
            continue
        drift = compute_feature_drift(reference_df[col], live_df[col])
        reports.append({
            'feature_name': col,
            **drift
        })
    return reports


async def build_reference_live_windows(feature_store, symbols: List[str], features: List[str], reference_start: datetime, reference_end: datetime, live_start: datetime, live_end: datetime):
    ref_matrix = await feature_store.get_feature_matrix(symbols, features, reference_start, reference_end)
    live_matrix = await feature_store.get_feature_matrix(symbols, features, live_start, live_end)
    return ref_matrix, live_matrix


async def persist_drift_reports(db, model_name: str, version: str, reports: List[Dict[str, Any]], reference_period: Tuple[datetime, datetime], live_period: Tuple[datetime, datetime]):
    if not reports:
        return 0
    inserted = 0
    for r in reports:
        try:
            await db.execute(
                """INSERT INTO model_drift_reports
                (model_name, version, feature_name, psi, ks, mean_shift, var_shift, severity, reference_period, live_period)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                [
                    model_name,
                    version,
                    r['feature_name'],
                    r['psi'],
                    r['ks'],
                    r['mean_shift'],
                    r['var_shift'],
                    r['severity'],
                    {'start': reference_period[0].isoformat(), 'end': reference_period[1].isoformat()},
                    {'start': live_period[0].isoformat(), 'end': live_period[1].isoformat()},
                ]
            )
            inserted += 1
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to persist drift report for %s: %s", r.get('feature_name'), e)
    return inserted


async def run_model_drift_scan(model_name: str, version: Optional[str] = None, live_window_days: int = 7, reference_window_days: int = 30) -> Dict[str, Any]:
    """High-level drift scan for a model.

    Strategy:
    - Load model registry entry (latest if version not provided)
    - Determine feature set & training window from stored config
    - Pull reference window (most recent slice of training period or earlier) and live window (most recent days)
    - Compute per-feature drift metrics
    - Persist and return summary
    """
    registry = await get_model_registry()
    entry = await registry.get_entry(model_name, version)
    if not entry:
        raise ValueError("Model registry entry not found")
    cfg = entry.config or {}
    feature_cols = cfg.get('feature_names') or []
    train_start = datetime.fromisoformat(cfg.get('train_start')) if isinstance(cfg.get('train_start'), str) else cfg.get('train_start')
    train_end = datetime.fromisoformat(cfg.get('train_end')) if isinstance(cfg.get('train_end'), str) else cfg.get('train_end')
    if not (train_start and train_end):
        raise ValueError("Training period missing in config")

    now = datetime.utcnow()
    live_end = now
    live_start = now - timedelta(days=live_window_days)
    reference_end = live_start
    reference_start = max(train_start, reference_end - timedelta(days=reference_window_days))

    feature_store = await get_feature_store()
    symbols = cfg.get('symbols') or ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    ref_df, live_df = await build_reference_live_windows(feature_store, symbols, feature_cols, reference_start, reference_end, live_start, live_end)
    if ref_df.empty or live_df.empty:
        return {"model_name": model_name, "version": entry.version, "status": "insufficient_data"}

    reports = compute_drift_frame(ref_df, live_df, feature_cols)
    inserted = 0
    try:
        db = registry.db  # underlying db manager set in registry.initialize
        if db:
            inserted = await persist_drift_reports(db, model_name, entry.version, reports, (reference_start, reference_end), (live_start, live_end))
    except Exception as e:
        logger.warning("Persistence skipped/failed: %s", e)

    worst_severity = 'NONE'
    order = ['NONE', 'LOW', 'MEDIUM', 'HIGH']
    for r in reports:
        if order.index(r['severity']) > order.index(worst_severity):
            worst_severity = r['severity']

    summary = {
        'model_name': model_name,
        'version': entry.version,
        'features_evaluated': len(reports),
        'worst_severity': worst_severity,
        'inserted': inserted,
        'reference_period': {'start': reference_start.isoformat(), 'end': reference_end.isoformat()},
        'live_period': {'start': live_start.isoformat(), 'end': live_end.isoformat()},
        'reports': reports,
    }
    # Structured drift event emission (best-effort; no hard dependency on metrics)
    try:  # pragma: no cover
        from trading_common.event_logging import emit_event
        emit_event(
            event_type="model.drift.scan", 
            model_name=model_name,
            version=entry.version,
            severity=worst_severity,
            features=len(reports),
            reference_window_days=reference_window_days,
            live_window_days=live_window_days,
            status=summary['status'] if 'status' in summary else 'completed'
        )
    except Exception:
        pass
    return summary

__all__ = [
    'compute_feature_drift',
    'compute_drift_frame',
    'run_model_drift_scan',
    'classify_severity',
]
