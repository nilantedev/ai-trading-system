#!/usr/bin/env python3
"""
Bootstrap common Postgres indexes (idempotent) and report status as JSON.

Uses trading_common.database_manager to obtain a connection. Creates indexes with
IF NOT EXISTS guards where supported.

Exit codes:
 0 success
 2 error
"""
from __future__ import annotations

import asyncio
import json
from typing import List, Tuple


INDEX_DDL: List[Tuple[str, str]] = [
    ("users_username_idx", "CREATE INDEX IF NOT EXISTS users_username_idx ON users (lower(username))"),
    ("model_registry_state_idx", "CREATE INDEX IF NOT EXISTS model_registry_state_idx ON model_registry (state)"),
    ("drift_reports_detected_at_idx", "CREATE INDEX IF NOT EXISTS drift_reports_detected_at_idx ON drift_reports (detected_at)"),
    ("feature_views_name_ver_idx", "CREATE INDEX IF NOT EXISTS feature_views_name_ver_idx ON feature_views (view_name, version)"),
    ("risk_events_ts_idx", "CREATE INDEX IF NOT EXISTS risk_events_ts_idx ON risk_events (timestamp)"),
    ("trading_signals_ts_strategy_idx", "CREATE INDEX IF NOT EXISTS trading_signals_ts_strategy_idx ON trading_signals (timestamp, strategy_name)"),
    ("option_surface_daily_sym_asof_idx", "CREATE INDEX IF NOT EXISTS option_surface_daily_sym_asof_idx ON option_surface_daily (symbol, as_of)"),
    ("factor_exposures_daily_sym_asof_idx", "CREATE INDEX IF NOT EXISTS factor_exposures_daily_sym_asof_idx ON factor_exposures_daily (symbol, as_of)")
]


async def main() -> int:
    try:
        from trading_common.database_manager import get_database_manager  # type: ignore
        dbm = await get_database_manager()
        created = []
        errors = []
        async with dbm.get_postgres() as sess:  # type: ignore[attr-defined]
            for name, ddl in INDEX_DDL:
                try:
                    await sess.execute(ddl)
                    created.append(name)
                except Exception as e:  # noqa: BLE001
                    errors.append({"index": name, "error": str(e)})
            try:
                await sess.commit()
            except Exception:
                pass
        out = {"status": "ok" if not errors else "partial", "created_or_verified": created, "errors": errors}
        print(json.dumps(out, indent=2))
        return 0 if not errors else 2
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"status": "error", "error": str(e)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
