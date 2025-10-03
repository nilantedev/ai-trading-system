# Tools and Non-Production Utilities

This folder holds supporting scripts and utilities that are not part of the core production runtime.

Guidelines:
- Do not import or execute scripts from this folder in production services.
- If you had muscle memory paths like `scripts/*.sh`, use the notes below to find the new location or symlink.

Proposed relocations (safe, non-destructive):
- scripts/ci_smoke.sh → tools/ci/ci_smoke.sh
- scripts/smoke_hosts.sh → tools/ops/smoke_hosts.sh
- scripts/tests/test_compose_mounts.sh → tools/tests/test_compose_mounts.sh
- scripts/verify_env.sh → tools/ops/verify_env.sh
- scripts/docker_hygiene.sh → tools/ops/docker_hygiene.sh

Why not delete? Some of these are still useful for diagnostics. Relocating out of `scripts/` keeps production cron/automation tidy and avoids accidental runs.

Next steps (optional):
- Add symlinks from the old path to preserve habits, for example:
  ln -s ../../tools/ops/smoke_hosts.sh scripts/smoke_hosts.sh
- Or update any README/runbooks to point to the tools/ path.

