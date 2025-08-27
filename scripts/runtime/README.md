Runtime Scripts Directory
=========================

This folder contains scripts required at *runtime* or during *deployment packaging*.

Migration Plan (Phase A):
1. Copy runtime-relevant scripts from scripts/ (do NOT delete originals yet).
2. Adjust deploy_production.sh to first look here (already partially implemented for SBOM generation).
3. Run `make preprod-verify` and a dry-run deployment to validate no regressions.
4. After validation, remove the duplicated originals in Phase B.

Candidate scripts to relocate (copy first):
- generate_sbom.py (DONE - moved reference only; file will be copied here next)
- emit_deployment_event.py
- security_scan_local.py (rename to security_scan.py)
- any lightweight health or readiness check helpers

Non-runtime scripts (stay in development tooling area or another folder):
- data seeding utilities
- large one-off research helpers
- heavy ML training orchestration scripts

Notes:
- The deployment.manifest should include scripts/runtime/ but exclude dev-only scripts to minimize artifact size.
- Keep this README until Phase B is complete; it will then be replaced with a concise description.Runtime Scripts
===============
This directory contains only scripts required at runtime / during deployment:
- generate_sbom.py
- emit_deployment_event.py
- disaster_recovery.py
- run_migrations.py
- manage_secrets.py (if used for rotation on server)

Non-runtime / dev scripts have been moved to scripts/dev and are excluded from deployment packaging.