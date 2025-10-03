-- Recommended Postgres indexes for trading-api (idempotent)
-- Users
CREATE INDEX IF NOT EXISTS users_username_idx ON users (username);

-- Trading signals frequently filtered by timestamp and strategy
CREATE INDEX IF NOT EXISTS trading_signals_ts_strategy_idx ON trading_signals (ts, strategy);

-- Risk events filtered by timestamp
CREATE INDEX IF NOT EXISTS risk_events_ts_idx ON risk_events (ts);

-- Option surface daily lookups by (symbol, as_of)
CREATE INDEX IF NOT EXISTS option_surface_daily_sym_asof_idx ON option_surface_daily (symbol, as_of);

-- Factor exposures daily lookups by (symbol, as_of)
CREATE INDEX IF NOT EXISTS factor_exposures_daily_sym_asof_idx ON factor_exposures_daily (symbol, as_of);
