# ML Operations & Governance Overview

This document describes the production ML governance architecture implemented in the AI Trading System.

## Lifecycle States
Models progress via a controlled state machine:

States: DRAFT -> TRAINED -> VALIDATED -> STAGING -> PRODUCTION -> RETIRED

Promotion rules enforced in `model_registry` with optional policy thresholds (Sharpe, max drawdown). Promotion attempts outside allowed transitions raise errors.

## Reproducibility & Provenance
Each trained model writes a manifest (`*.manifest.json`) containing:
 - feature_graph_hash (deterministic DAG hash of feature definitions)
 - training_config_hash (stable hash of sorted serialized config)
 - dataset_hash (content hash subset of training dataset)
 - git_commit (source revision)
 - metrics & raw config snapshot

These hashes enable exact run reconstruction and change auditing.

## Feature Graph Enforcement
`feature_graph.py` builds a canonical DAG from feature definitions (name, version, dependencies, logic). The DAG hash is stored with the model. On model load and inference, the active feature store is validated; mismatches block prediction (409) until retrain or override.

## Drift Monitoring
`drift_detection.py` computes per-feature drift metrics: PSI, Kolmogorov–Smirnov, mean shift, variance shift. A severity classifier flags NORMAL / MODERATE / SEVERE. Results persisted to `model_drift_reports`.

`drift_scheduler.py` runs periodic scans (default hourly) over production models, comparing last day vs prior 30-day reference window. Future enhancements: auto-demotion, alerting hooks.

## Promotion Policy
Externalized YAML (`config/ml_promotion_policy.yaml`) provides default and strategy-specific overrides for thresholds (min_sharpe, max_drawdown, etc.). Endpoint `/api/v1/ml/{model}/promote` merges request overrides with policy file. Loader: `promotion_policy.get_policy_for_model`.

## Inference & Management APIs
Routers:
 - `ml_models.py`:
   - POST `/api/v1/models/{model}/predict`
   - GET  `/api/v1/models/{model}/status`
   - GET  `/api/v1/models/{model}/versions`
 - `ml_management.py`:
   - POST `/api/v1/ml/{model}/backtest`
   - POST `/api/v1/ml/{model}/promote`

All inference requests enforce feature graph hash alignment.

## Testing Strategy (Planned/Expand)
Tests to cover:
 - Registry lifecycle transitions & rejection of invalid promotions
 - Hash determinism for feature graph & training config
 - Drift detection classification boundaries
 - Feature graph mismatch blocking inference
 - Promotion policy override precedence
 - Prediction endpoint contract (200/409/404 cases)

## Future Enhancements
 - Alerting integration (Slack/Email) on SEVERE drift
 - Canary deployment strategy for PRODUCTION promotion
 - Automated rollback on performance degradation
 - Data quality integrated into drift severity weighting
 - Model lineage visualization (export DAG)

---
Generated: automated governance implementation summary.
