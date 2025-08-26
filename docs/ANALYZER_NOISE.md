# Analyzer Noise & Development Environment Notes

This project uses strict type and lint settings. Some issues that appear in editor problem panels are environmental rather than code defects.

## Common Sources of Noise

| Category | Symptom | Root Cause | Resolution |
|----------|---------|------------|------------|
| Unresolved import `fastapi` / `uvicorn` | Import errors in Problems tab | Language server not using project venv | Ensure interpreter is set to `.venv/bin/python` |
| Unresolved `trading_common.*` | Missing internal package | Monorepo-internal package provided as stubs only | Real implementation will replace stubs; stubs satisfy interface now |
| Broad exception warnings (BLE001) | "Catching too general exception" | Intentional resilience in phased startup/shutdown | Per-file flake8 ignore configured for `api/main.py` |
| Unused arg in stubs | Lint warning | Minimal stub doesn’t use parameter | Add `_ = arg` or future real implementation |
| SystemExit style suggestion | Suggests `raise SystemExit(1) from e` | Exception chaining clarity | Implemented where applicable |

## Philosophy
We differentiate between: 
- Critical failures (must halt startup in prod) 
- Optional subsystems (log + degrade) 
- Observability/summary logging (never block)

## Broad Exceptions Policy
Broad `except Exception` blocks exist only where: 
1. Boundary between critical and optional subsystem. 
2. During shutdown to avoid masking other cleanup. 
3. In readiness / health probes collecting heterogeneous signals.

All such blocks carry comments and are isolated; flake8 per-file ignore for BLE001 limited to `api/main.py`.

## When Adding New Code
- Prefer specific exceptions (`asyncio.TimeoutError`, `ConnectionError`) first.
- If a truly broad catch is required, justify with a comment and keep block size small.

## Upgrading From Stubs
Replace modules under `trading_common/` with real logic incrementally. Maintain interface shape so dependent code keeps working.

## Verifying Environment
Run these quick checks after activating the venv:
```bash
python -c "import fastapi, uvicorn, structlog; print('ok')"
python -c "import trading_common; print(trading_common.get_settings())"
```
If these pass, runtime imports are healthy even if editor still reports some stale problems.

---
For more setup detail see `docs/DEVELOPMENT_SETUP.md`.
