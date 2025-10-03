# ML Service Image Notes

## Overview
This document explains design decisions for the hardened multi-stage ML service image.

## Objectives
- Deterministic builds (pinned versions inside Dockerfile)
- Reduced attack surface (multi-stage, remove build toolchain)
- Faster cold start (FinBERT pre-warmed in build stage)
- Non-root runtime user (`appuser`)
- Clear boundary between heavy ML deps and lighter microservices to avoid dependency bloat across the fleet

## Stages
1. **builder**
   - Based on `python:3.11-slim`
   - Installs build dependencies and compiles TA-Lib 0.4.0 to `/usr`
   - Creates a virtual environment at `/opt/venv`
   - Installs heavy ML libraries (Torch 2.5.1 CPU, Torch Geometric 2.6.1, Transformers 4.46.3, etc.)
   - Pre-downloads the FinBERT model into the huggingface cache (`/root/.cache/huggingface`)
2. **runtime**
   - Copies only the virtual environment, TA-Lib artifacts, and HF cache
   - Adds a minimal set of OS packages (curl + CA certs for health checks)
   - Drops privileges to `appuser`

## Dependency Divergence vs Root `requirements.txt`
The root file pins older versions to keep smaller service images lean:

| Package | Root Pin | ML Image | Rationale |
|---------|----------|----------|-----------|
| torch | 2.1.0 | 2.5.1 | Newer kernel / performance & operator coverage needed for experimental models |
| torch-geometric | 2.4.0 | 2.6.1 | Compatibility with Torch 2.5.x |
| transformers | 4.35.0 | 4.46.3 | Required for newer model architectures & tokenizer bugfixes |
| sentence-transformers | 2.2.2 | 3.3.0 | API and embedding quality improvements |
| xgboost | 2.0.3 | 2.1.2 | Performance updates |
| lightgbm | 4.3.0 | 4.5.0 | Bug fixes and new features |

Strategy: Keep divergence isolated. Future action item: introduce per-service requirements constraints OR uplift root pins after regression tests.

## FinBERT Pre-Warm
The model is downloaded during build to avoid first-request latency. Cache path is copied into runtime stage under `/app/.cache/huggingface` and surfaced via `HF_HOME`.

## Updating Heavy Dependencies
1. Create a feature branch.
2. Increment versions in Dockerfile builder stage.
3. Rebuild locally: `docker build -f services/ml/Dockerfile . -t ml:test`.
4. Run smoke tests & a sample inference to ensure no ABI or serialization regressions.
5. Open PR with changelog section.

## CPU vs GPU
Currently using CPU wheels (index override for Torch). For GPU variant:
- Switch base image to an appropriate CUDA-enabled image (e.g. `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`)
- Remove custom `--index-url` line (or adjust to CUDA wheels)
- Ensure host nodes have matching NVIDIA driver & container runtime (nvidia-container-toolkit)
- Add `RUN nvidia-smi || true` (optional) for diagnostics in builds

## Security & Slimming Opportunities (Future)
- Add `pip install --require-hashes` using a generated `requirements-ml.txt` for stronger supply-chain guarantees
- Run `trivy fs` / `grype` scans in CI to track CVEs
- Strip debug symbols from compiled libs (TA-Lib already small)
- Consider `python -m compileall` then remove `.py` sources for proprietary logic (trade-off: harder debugging)

## Health Check
Uses `/health` endpoint on port 8001. Ensure service exposes this path quickly on startup; model loading is already pre-warmed.

## Environment Variables
- `HF_HOME`: Points to model cache location (`/app/.cache/huggingface`)
- `PYTHONPATH`: Includes `shared/python-common` for cross-service utilities

## Known Limitations
- Larger image size (~GB range) due to ML frameworks; acceptable as isolated service
- Divergent dependency graph; must avoid accidental promotion into lightweight services

## Next Steps (Backlog)
- Hash-locked dependency file for ML only
- Model artifact version pin & integrity hash verification
- Optional quantization / pruning step to cut memory footprint
- Evaluate migrating to faster inference runtime for FinBERT (ONNX Runtime) with pre-conversion layer

---
Maintainers: ML Platform / Quant Engineering
