# Explicit Non-Goals (Current Scope Exclusions)

This document enumerates items **intentionally excluded** from the present implementation phase for the Intelligence / Signal & ML Layer. They may be revisited in future roadmap phases, but are **not** to be implemented now. Keeping these explicit prevents scope creep and clarifies acceptance criteria for this phase.

## 1. Full MLflow Integration
We will not integrate MLflow (tracking server, model registry, artifact store) at this stage. Instead we implement a lightweight in‑repo `experiment_tracking` module with: run metadata, params, metrics, artifacts references. Migration paths will be documented for future MLflow adoption.

Rationale: Reduce infra overhead, avoid premature external service dependency while core pipelines stabilize.

## 2. Automated Retraining Pipeline
No scheduled or event-driven automated model retraining (e.g., Airflow / Prefect / Dagster orchestration) will be added now. Manual / scripted retraining triggers remain acceptable.

Rationale: Focus on observability (drift + experiment telemetry) first; retraining automation requires robust data validation and promotion guardrails not yet complete.

## 3. Distributed Feature Computation Scheduler
We will not build a distributed / scalable feature computation orchestration layer (e.g., Spark, Ray, Flink) or a cron cluster for large-scale materialization. Current implementation limits to synchronous / ad-hoc batch + lightweight background tasks.

Rationale: Current feature volume and latency requirements fit existing DB + async tasks. Scaling concerns deferred until empirical saturation metrics collected.

## 4. Advanced Lineage Graph UI
No interactive lineage visualization (graph database, web UI) will be implemented. Lineage exposure limited to JSON APIs / programmatic queries via `feature_store.get_feature_lineage` and simple metadata.

Rationale: Developer productivity prioritized; visual lineage adds complexity without immediate decision value.

## 5. Full Permissioning Layer Around Experiments
We will not implement role-based access control / ownership enforcement for experiment runs beyond existing application auth. Experiment artifacts and metadata are assumed to be accessed by trusted internal roles.

Rationale: Premature hardening; will reassess alongside broader compliance / audit logging phase.

---
## Summary
Deliverables in scope: Lightweight experiment tracking, drift scheduler, feature views, signal provenance. Everything above is explicitly OUT of scope now.

Document version: 1.0.0  |  Owner: system  |  Last Updated: 2025-08-26
