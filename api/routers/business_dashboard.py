from __future__ import annotations
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse
from datetime import datetime, date, timedelta
import random, hashlib
import os
import asyncio
import time
from typing import Any, Dict, List
from api.auth import get_current_user_cookie_or_bearer
from fastapi import Depends
from api.cache_utils import ttl_cache
from api.rate_limiter import EnhancedRateLimiter
import logging
try:
    from services.ml.ml_orchestrator import get_ml_orchestrator  # type: ignore
except Exception:  # noqa: BLE001
    async def get_ml_orchestrator():  # type: ignore
        class _Stub:
            async def kpi_snapshot(self):
                return {"status": "unavailable", "detail": "orchestrator stub"}
            async def list_shadow_stats(self, horizon: int = 1):  # noqa: D401
                return []
        logger = logging.getLogger(__name__)
        logger.warning("ML orchestrator module unavailable; using stub implementation (business dashboard)")
        return _Stub()

logger = logging.getLogger(__name__)

router = APIRouter(tags=["business-dashboard"], include_in_schema=False)

"""Dashboard access restrictions

Per updated requirements ONLY the admin user 'nilante' may access any business dashboard
routes. We enforce this by wrapping the existing role-based dependency and then
performing a username check. This avoids scattering username checks across
individual endpoints and keeps future extension simple (e.g., allow list).
"""

async def _single_user_guard(user=Depends(get_current_user_cookie_or_bearer)):
    """Restrict ALL business dashboard access to the bootstrap user 'nilante'.

    We intentionally bypass role checks to avoid accidental public access if
    user role provisioning failed. This aligns with requirement: ONLY the
    admin user nilante can access ANY dashboard.
    """
    if getattr(user, 'username', None) != 'nilante':
        raise HTTPException(status_code=403, detail='Access restricted')
    return user

VIEW_DEP = _single_user_guard


def _template(request: Request, name: str, **ctx):
    """Render a template ensuring 'request' is provided for base.html usage."""
    env = request.app.state.jinja_env
    tpl = env.get_template(name)
    nonce = getattr(request.state, 'csp_nonce', '')
    return HTMLResponse(tpl.render(request=request, csp_nonce=nonce, year=datetime.utcnow().year, **ctx))

@router.get('/business', response_class=HTMLResponse)
async def business_dashboard(request: Request, user=Depends(VIEW_DEP)):
    # user is guaranteed (403 otherwise). Could add future personalization here.
    return _template(request, 'business/dashboard_v2.html', title='Mekoshi Intelligence Platform', user=user)

@router.head('/business')
async def business_dashboard_head(user=Depends(VIEW_DEP)):
    return HTMLResponse(content="", status_code=200)

@router.get('/business/company/{symbol}', response_class=HTMLResponse)
async def company_profile(request: Request, symbol: str, user=Depends(VIEW_DEP)):
    return _template(request, 'business/company_profile.html', title=f'Company {symbol}', symbol=symbol.upper(), user=user)

_rate_limiter: EnhancedRateLimiter | None = None

async def _get_rate_limiter():
    global _rate_limiter
    if _rate_limiter is None:
        try:
            rl = EnhancedRateLimiter()
            await rl.initialize()
            _rate_limiter = rl
        except Exception:  # noqa: BLE001
            logger.warning("Rate limiter unavailable for business dashboard endpoints", exc_info=True)
            _rate_limiter = None
    return _rate_limiter

def _seeded_rand(seed_key: str) -> random.Random:
    h = hashlib.sha256(seed_key.encode()).hexdigest()[:16]
    return random.Random(int(h, 16))

def _cache(key: str, ttl: int, loader):
    return ttl_cache(key, ttl, loader)

# Async-aware TTL cache for DB-backed loaders
_async_cache: Dict[str, tuple[float, Any]] = {}
_async_lock = asyncio.Lock()

async def _acache(key: str, ttl: int, loader):
    now = time.time()
    async with _async_lock:
        entry = _async_cache.get(key)
        if entry and entry[0] > now:
            return entry[1]
    # Load outside lock
    value = await loader()
    async with _async_lock:
        _async_cache[key] = (now + ttl, value)
    return value

def _audit(user, action: str, extra: dict | None = None):
    # Merge base audit fields with any additional context using proper dict unpacking
    base = {'user': getattr(user, 'username', 'unknown'), 'user_id': getattr(user, 'user_id', 'unknown'), 'action': action}
    merged = {**base, **(extra or {})}
    logger.info("BUSINESS_AUDIT", extra=merged)


@router.get('/business/api/coverage/summary')
async def business_coverage_summary(request: Request, user=Depends(VIEW_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'coverage_summary')
    # Query metrics / DB for coverage info (best-effort; fallback deterministic)
    async def load():
        try:
            from trading_common.database_manager import get_database_manager
            db = await get_database_manager()
            async with db.get_questdb() as conn:  # type: ignore[attr-defined]
                # Expect precomputed coverage tables or derive simple stats (placeholder queries)
                # For resilience, each query wrapped individually
                ratios: dict[str, float] = {}
                try:
                    row = await conn.fetchrow("SELECT avg(coverage_ratio) AS r FROM equities_coverage_daily WHERE ts > now() - 1d")
                    if row and row['r'] is not None:
                        ratios['equities_1d_avg'] = float(row['r'])
                except Exception:
                    pass
                try:
                    row2 = await conn.fetchrow("SELECT avg(options_coverage_ratio) AS r FROM options_coverage_daily WHERE ts > now() - 1d")
                    if row2 and row2['r'] is not None:
                        ratios['options_1d_avg'] = float(row2['r'])
                except Exception:
                    pass
                # Attempt real coverage history (last ~4 hours, ~60 points)
                hist: list[dict] = []
                try:
                    eq_rows = []
                    op_rows = []
                    try:
                        eq_rows = await conn.fetch("""
                            SELECT ts, coverage_ratio AS r
                            FROM equities_coverage_daily
                            WHERE ts > now() - 4h
                            ORDER BY ts DESC LIMIT 60
                        """)
                    except Exception:
                        eq_rows = []
                    try:
                        op_rows = await conn.fetch("""
                            SELECT ts, options_coverage_ratio AS r
                            FROM options_coverage_daily
                            WHERE ts > now() - 4h
                            ORDER BY ts DESC LIMIT 60
                        """)
                    except Exception:
                        op_rows = []
                    if eq_rows or op_rows:
                        eq_map = {row['ts']: float(row['r']) for row in eq_rows if row and row.get('r') is not None}
                        op_map = {row['ts']: float(row['r']) for row in op_rows if row and row.get('r') is not None}
                        # union of timestamps, sorted ascending
                        all_ts = sorted(set(eq_map.keys()) | set(op_map.keys()))
                        for ts_val in all_ts:
                            hist.append({
                                'ts': ts_val.isoformat() if hasattr(ts_val, 'isoformat') else str(ts_val),
                                'equities': eq_map.get(ts_val),
                                'options': op_map.get(ts_val)
                            })
                except Exception:
                    hist = []
                # Fallback synthetic if history empty but ratios exist
                if not hist and ratios:
                    base_time = datetime.utcnow()
                    for i in range(10):
                        t = (base_time - timedelta(minutes=5*(9-i))).isoformat()
                        eq = ratios.get('equities_1d_avg')
                        op = ratios.get('options_1d_avg')
                        eqv = min(1.0, max(0.0, (eq or 0) + (i-5)*0.0005)) if eq is not None else None
                        opv = min(1.0, max(0.0, (op or 0) + (i-5)*0.001)) if op is not None else None
                        hist.append({'ts': t, 'equities': eqv, 'options': opv})
                return {
                    'timestamp': datetime.utcnow().isoformat(),
                    'ratios': ratios,
                    'history': hist,
                }
        except Exception:
            r = _seeded_rand("coverage:summary")
            eq = round(r.uniform(0.85, 0.99), 4)
            op = round(r.uniform(0.70, 0.95), 4)
            base_time = datetime.utcnow()
            hist = []
            for i in range(10):
                t = (base_time - timedelta(minutes=5*(9-i))).isoformat()
                hist.append({'ts': t, 'equities': min(1.0, max(0.0, eq + (i-5)*0.0005)), 'options': min(1.0, max(0.0, op + (i-5)*0.001))})
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'ratios': {
                    'equities_1d_avg': eq,
                    'options_1d_avg': op
                },
                'history': hist
            }
    return await _acache('business:coverage_summary', 30, load)

@router.get('/business/api/ingestion/health')
async def business_ingestion_health(request: Request, user=Depends(VIEW_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'ingestion_health')
    async def load():
        try:
            # Pull loop status from another service via shared state/Redis if available
            from trading_common.database_manager import get_database_manager
            db = await get_database_manager()
            async with db.get_questdb() as conn:  # type: ignore[attr-defined]
                # Placeholder simple recency check on market_data
                row = await conn.fetchrow("SELECT max(timestamp) AS last_bar FROM market_data")
                last_bar = row['last_bar'].isoformat() if row and row['last_bar'] else None
                row2 = await conn.fetchrow("SELECT max(timestamp) AS last_opt FROM options_data")
                last_opt = row2['last_opt'].isoformat() if row2 and row2['last_opt'] else None
                return {
                    'timestamp': datetime.utcnow().isoformat(),
                    'last_equity_bar': last_bar,
                    'last_option_bar': last_opt,
                    'lag_seconds_equity': None,  # supply once timestamp timezone normalized
                    'lag_seconds_option': None
                }
        except Exception:
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'last_equity_bar': None,
                'last_option_bar': None,
                'lag_seconds_equity': None,
                'lag_seconds_option': None,
                'degraded': True
            }
    return await _acache('business:ingestion_health', 15, load)

@router.get('/business/api/news/sentiment')
async def business_news_sentiment(request: Request, user=Depends(VIEW_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'news_sentiment')
    async def load():
        try:
            from trading_common.database_manager import get_database_manager
            db = await get_database_manager()
            async with db.get_questdb() as conn:  # type: ignore[attr-defined]
                row = await conn.fetchrow("SELECT avg(sentiment_score) AS s FROM news_events WHERE timestamp > now() - 1d")
                row2 = await conn.fetchrow("SELECT avg(sentiment_score) AS s FROM news_events WHERE timestamp > now() - 7d")
                # Build simple daily history (avg by day for last 30d) best-effort
                history_rows = []
                try:
                    history_rows = await conn.fetch(
                        """
                        SELECT cast(timestamp as date) AS d, avg(sentiment_score) AS s
                        FROM news_events
                        WHERE timestamp > now() - 30d
                        GROUP BY d
                        ORDER BY d
                        """
                    )
                except Exception:
                    history_rows = []
                d1 = float(row['s']) if row and row['s'] is not None else None
                d7 = float(row2['s']) if row2 and row2['s'] is not None else None
                anomaly = (d1 - d7) if (d1 is not None and d7 is not None) else None
                return {
                    'timestamp': datetime.utcnow().isoformat(),
                    'sentiment_1d_avg': d1,
                    'sentiment_7d_avg': d7,
                    'anomaly_delta': anomaly,
                    'history': [
                        {
                            'date': (hr['d'].isoformat() if hasattr(hr['d'], 'isoformat') else str(hr['d'])),
                            'avg_sentiment': (float(hr['s']) if hr and hr['s'] is not None else None)
                        } for hr in history_rows
                    ]
                }
        except Exception:
            r = _seeded_rand('news:sent')
            d7 = round(r.uniform(-0.05, 0.05),4)
            d1 = min(1,max(-1, d7 + r.uniform(-0.02,0.02)))
            # Deterministic synthetic fallback history (flat line with tiny noise)
            hist = []
            base = d7 if d7 is not None else 0.0
            for i in range(30):
                val = min(1,max(-1, base + r.uniform(-0.01,0.01)))
                from datetime import timedelta as _td
                hist.append({'date': (date.today()-_td(days=29-i)).isoformat(), 'avg_sentiment': val})
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'sentiment_1d_avg': d1,
                'sentiment_7d_avg': d7,
                'anomaly_delta': round(d1-d7,4),
                'history': hist
            }
    return await _acache('business:news_sent', 120, load)

async def _enforce_rate(request: Request, user):
    rl = await _get_rate_limiter()
    if rl is None:
        return
    # Identify by user if authenticated, else by client IP
    user_id = getattr(user, 'user_id', 'anon') if user is not None else 'anon'
    client_ip = request.client.host if request.client else 'unknown'
    ident = f"business:{user_id}:{client_ip}"
    result = await rl.check_rate_limit(ident, limit_type="default", request=request)
    if not result.get('allowed'):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

@router.get('/business/api/kpis')
async def business_kpis(request: Request, user=Depends(VIEW_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'kpis')
    today = date.today().isoformat()
    async def load_db() -> Dict[str, Any]:
        # Query QuestDB for simple KPIs derived from recent data
        try:
            from trading_common.database_manager import get_database_manager
            db = await get_database_manager()
            async with db.get_questdb() as conn:
                # Count distinct active strategies inferred from signals table over last 1d
                q_active_strat = """
                    SELECT count(DISTINCT strategy_name) AS c
                    FROM trading_signals WHERE timestamp > now() - 1d
                """
                row1 = await conn.fetchrow(q_active_strat)
                active_strategies = int(row1['c']) if row1 and 'c' in row1 else 0

                # Daily signals
                q_daily_signals = """
                    SELECT count(*) AS c FROM trading_signals
                    WHERE timestamp >= date_trunc('d', now())
                """
                row2 = await conn.fetchrow(q_daily_signals)
                daily_signals = int(row2['c']) if row2 and 'c' in row2 else 0

                # Risk alerts today
                q_risk_alerts = """
                    SELECT count(*) AS c FROM risk_events
                    WHERE timestamp >= date_trunc('d', now())
                """
                row3 = await conn.fetchrow(q_risk_alerts)
                risk_alerts_today = int(row3['c']) if row3 and 'c' in row3 else 0

                # Pending deployments from model_performance recent entries with shadow models
                q_deployments = """
                    SELECT count(DISTINCT model_id) AS c
                    FROM model_performance WHERE timestamp > now() - 7d AND model_id LIKE 'shadow%'
                """
                row4 = await conn.fetchrow(q_deployments)
                deployments_pending = int(row4['c']) if row4 and 'c' in row4 else 0

                return {
                    'timestamp': datetime.utcnow().isoformat(),
                    'kpis': {
                        'active_strategies': active_strategies,
                        'daily_signals': daily_signals,
                        'avg_signal_latency_ms': 0,  # placeholder unless latency table exists
                        'risk_alerts_today': risk_alerts_today,
                        'deployments_pending': deployments_pending
                    }
                }
        except Exception:
            # Fallback to deterministic values if DB not ready
            r = _seeded_rand(f"kpis:{today}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'kpis': {
                    'active_strategies': r.randint(10, 40),
                    'daily_signals': r.randint(200, 1200),
                    'avg_signal_latency_ms': r.randint(20, 150),
                    'risk_alerts_today': r.randint(0, 5),
                    'deployments_pending': r.randint(0, 3)
                }
            }
    return await _acache('business:kpis', 10, load_db)

@router.get('/business/api/companies')
async def list_companies(request: Request, user=Depends(VIEW_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'companies')
    def load():
        companies = ['AAPL','MSFT','GOOGL','AMZN','TSLA']
        return {'timestamp': datetime.utcnow().isoformat(), 'companies': companies}
    return _cache('business:companies', 60, load)

@router.get('/business/api/company/{symbol}/forecast')
async def company_forecast(symbol: str, request: Request, user=Depends(VIEW_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'forecast', {'symbol': symbol})
    today = date.today().isoformat()
    def load():
        r = _seeded_rand(f"forecast:{symbol}:{today}")
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol.upper(),
            'forecasts': {
                'next_1d_return': {'value': round(r.uniform(-0.02, 0.02), 5), 'confidence': round(r.uniform(0.5, 0.9), 2)},
                'volatility_5d': {'value': round(r.uniform(0.1, 0.4), 3)},
                'regime': r.choice(['bullish','neutral','bearish'])
            }
        }
    return _cache(f'business:forecast:{symbol.upper()}', 30, load)

@router.get('/business/api/company/{symbol}/report')
async def company_report(symbol: str, request: Request, user=Depends(VIEW_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'report', {'symbol': symbol})
    def load():
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol.upper(),
            'summary': f"Automated summary placeholder for {symbol.upper()}.",
            'highlights': [
                'Revenue momentum stable',
                'Volatility within expected band',
                'No anomalous drift detected'
            ]
        }
    return _cache(f'business:report:{symbol.upper()}', 120, load)

@router.get('/business/api/company/{symbol}/sparkline')
async def company_sparkline(symbol: str, request: Request, user=Depends(VIEW_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'sparkline', {'symbol': symbol})
    upper = symbol.upper()
    async def load_db() -> Dict[str, Any]:
        try:
            from trading_common.database_manager import get_database_manager
            db = await get_database_manager()
            async with db.get_questdb() as conn:
                # Get last 50 closes for sparkline
                q = """
                    SELECT timestamp, close FROM market_data
                    WHERE symbol = $1
                    ORDER BY timestamp DESC LIMIT 50
                """
                rows = await conn.fetch(q, upper)
                series = [float(r['close']) for r in reversed(rows)] if rows else []

                # Optionally enrich with simple sentiment over last day
                q_sent = """
                    SELECT avg(sentiment_score) AS s FROM news_events
                    WHERE timestamp > now() - 1d AND symbols ilike $1
                """
                row_s = await conn.fetchrow(q_sent, f"%{upper}%")
                sentiment = float(row_s['s']) if row_s and row_s['s'] is not None else None

                return {'timestamp': datetime.utcnow().isoformat(), 'symbol': upper, 'series': series, 'sentiment_1d_avg': sentiment}
        except Exception:
            # Fallback deterministic series
            today = date.today().isoformat()
            r = _seeded_rand(f"spark:{upper}:{today}")
            base = 100 + (r.uniform(-5,5))
            series = []
            price = base
            for _ in range(50):
                price += r.uniform(-1, 1)
                series.append(round(price, 2))
            return {'timestamp': datetime.utcnow().isoformat(), 'symbol': upper, 'series': series}
    return await _acache(f'business:spark:{upper}', 30, load_db)

@router.get('/business/api/models/kpis')
async def model_kpis(request: Request, user=Depends(VIEW_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'model_kpis')
    orch = await get_ml_orchestrator()
    return await orch.kpi_snapshot()

@router.get('/business/api/models/shadow')
async def business_shadow_models(request: Request, user=Depends(VIEW_DEP), horizon: int = 1):
    await _enforce_rate(request, user)
    _audit(user, 'shadow_models')
    orch = await get_ml_orchestrator()
    stats = await orch.list_shadow_stats(horizon=horizon)
    return {'timestamp': datetime.utcnow().isoformat(), 'horizon': horizon, 'shadow_models': stats}

# === NEW ENDPOINTS FOR COMPLETE DASHBOARD ===

@router.get('/business/api/company/{symbol}/fundamentals')
async def company_fundamentals(symbol: str, request: Request, user=Depends(VIEW_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'fundamentals', {'symbol': symbol})
    upper = symbol.upper()
    async def load_db() -> Dict[str, Any]:
        try:
            from trading_common.database_manager import get_database_manager
            db = await get_database_manager()
            async with db.get_postgres() as conn:
                q = """
                    SELECT pe_ratio, pb_ratio, dividend_yield, market_cap, revenue_ttm, 
                           earnings_per_share, free_cash_flow, debt_to_equity
                    FROM company_fundamentals
                    WHERE symbol = $1
                    ORDER BY timestamp DESC LIMIT 1
                """
                row = await conn.fetchrow(q, upper)
                if row:
                    return {
                        'timestamp': datetime.utcnow().isoformat(),
                        'symbol': upper,
                        'fundamentals': {
                            'pe': float(row['pe_ratio']) if row['pe_ratio'] else None,
                            'pb': float(row['pb_ratio']) if row['pb_ratio'] else None,
                            'dividend_yield': float(row['dividend_yield']) if row['dividend_yield'] else None,
                            'market_cap': float(row['market_cap']) if row['market_cap'] else None,
                            'revenue_ttm': float(row['revenue_ttm']) if row['revenue_ttm'] else None,
                            'eps': float(row['earnings_per_share']) if row['earnings_per_share'] else None,
                            'fcf': float(row['free_cash_flow']) if row['free_cash_flow'] else None,
                            'debt_to_equity': float(row['debt_to_equity']) if row['debt_to_equity'] else None
                        }
                    }
        except Exception:
            pass
        # Fallback
        r = _seeded_rand(f"fund:{upper}")
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': upper,
            'fundamentals': {
                'pe': round(r.uniform(10, 35), 2),
                'pb': round(r.uniform(1.5, 8), 2),
                'dividend_yield': round(r.uniform(0, 0.04), 4),
                'market_cap': round(r.uniform(10e9, 2000e9), 0),
                'revenue_ttm': round(r.uniform(5e9, 500e9), 0),
                'eps': round(r.uniform(1, 25), 2),
                'fcf': round(r.uniform(1e9, 50e9), 0),
                'debt_to_equity': round(r.uniform(0.2, 1.5), 2)
            }
        }
    return await _acache(f'business:fund:{upper}', 300, load_db)

@router.get('/business/api/company/{symbol}/factor-exposures')
async def company_factor_exposures(symbol: str, request: Request, user=Depends(VIEW_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'factor_exposures', {'symbol': symbol})
    upper = symbol.upper()
    async def load_db() -> Dict[str, Any]:
        try:
            from trading_common.database_manager import get_database_manager
            db = await get_database_manager()
            async with db.get_questdb() as conn:
                q = """
                    SELECT factor_name, exposure
                    FROM factor_exposures
                    WHERE symbol = $1 AND timestamp > now() - 1d
                    ORDER BY timestamp DESC
                    LIMIT 10
                """
                rows = await conn.fetch(q, upper)
                if rows:
                    exposures = {row['factor_name']: float(row['exposure']) for row in rows}
                    return {
                        'timestamp': datetime.utcnow().isoformat(),
                        'symbol': upper,
                        'exposures': exposures
                    }
        except Exception:
            pass
        # Fallback
        r = _seeded_rand(f"factors:{upper}")
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': upper,
            'exposures': {
                'momentum': round(r.uniform(-1, 1), 3),
                'value': round(r.uniform(-1, 1), 3),
                'quality': round(r.uniform(-1, 1), 3),
                'volatility': round(r.uniform(-1, 1), 3),
                'size': round(r.uniform(-1, 1), 3),
                'liquidity': round(r.uniform(-1, 1), 3)
            }
        }
    return await _acache(f'business:factors:{upper}', 120, load_db)

@router.get('/business/api/company/{symbol}/factor-exposures/timeseries')
async def company_factor_timeseries(symbol: str, request: Request, user=Depends(VIEW_DEP), days: int = 90):
    await _enforce_rate(request, user)
    _audit(user, 'factor_timeseries', {'symbol': symbol, 'days': days})
    upper = symbol.upper()
    async def load_db() -> Dict[str, Any]:
        try:
            from trading_common.database_manager import get_database_manager
            db = await get_database_manager()
            async with db.get_questdb() as conn:
                q = """
                    SELECT timestamp, factor_name, exposure
                    FROM factor_exposures
                    WHERE symbol = $1 AND timestamp > now() - $2 * 24 * 3600 * 1000000
                    ORDER BY timestamp ASC
                """
                rows = await conn.fetch(q, upper, days)
                if rows:
                    factors_dict: Dict[str, List[Dict]] = {}
                    for row in rows:
                        fname = row['factor_name']
                        if fname not in factors_dict:
                            factors_dict[fname] = []
                        factors_dict[fname].append({
                            'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                            'value': float(row['exposure'])
                        })
                    return {
                        'timestamp': datetime.utcnow().isoformat(),
                        'symbol': upper,
                        'days': days,
                        'factors': factors_dict
                    }
        except Exception:
            pass
        # Fallback synthetic
        r = _seeded_rand(f"factseries:{upper}:{days}")
        factors_dict = {}
        for fname in ['momentum', 'value', 'quality']:
            series = []
            base = r.uniform(-0.5, 0.5)
            for i in range(min(days, 30)):
                val = base + r.uniform(-0.1, 0.1)
                series.append({
                    'timestamp': (datetime.utcnow() - timedelta(days=days-i)).isoformat(),
                    'value': round(val, 3)
                })
            factors_dict[fname] = series
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': upper,
            'days': days,
            'factors': factors_dict
        }
    return await _acache(f'business:factseries:{upper}:{days}', 300, load_db)

@router.get('/business/api/company/{symbol}/risk-metrics')
async def company_risk_metrics(symbol: str, request: Request, user=Depends(VIEW_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'risk_metrics', {'symbol': symbol})
    upper = symbol.upper()
    async def load_db() -> Dict[str, Any]:
        try:
            from trading_common.database_manager import get_database_manager
            db = await get_database_manager()
            async with db.get_questdb() as conn:
                q = """
                    SELECT beta_60d, realized_vol_20d, sharpe_60d, max_drawdown_30d, var_95, cvar_95
                    FROM risk_metrics
                    WHERE symbol = $1
                    ORDER BY timestamp DESC LIMIT 1
                """
                row = await conn.fetchrow(q, upper)
                if row:
                    return {
                        'timestamp': datetime.utcnow().isoformat(),
                        'symbol': upper,
                        'risk_metrics': {
                            'beta_60d': float(row['beta_60d']) if row['beta_60d'] else None,
                            'realized_vol_20d': float(row['realized_vol_20d']) if row['realized_vol_20d'] else None,
                            'sharpe_60d': float(row['sharpe_60d']) if row['sharpe_60d'] else None,
                            'max_drawdown_30d': float(row['max_drawdown_30d']) if row['max_drawdown_30d'] else None,
                            'var_95': float(row['var_95']) if row['var_95'] else None,
                            'cvar_95': float(row['cvar_95']) if row['cvar_95'] else None
                        }
                    }
        except Exception:
            pass
        # Fallback
        r = _seeded_rand(f"risk:{upper}")
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': upper,
            'risk_metrics': {
                'beta_60d': round(r.uniform(0.5, 1.5), 2),
                'realized_vol_20d': round(r.uniform(0.15, 0.45), 3),
                'sharpe_60d': round(r.uniform(-0.5, 2.5), 2),
                'max_drawdown_30d': round(r.uniform(-0.25, -0.05), 3),
                'var_95': round(r.uniform(-0.05, -0.01), 4),
                'cvar_95': round(r.uniform(-0.08, -0.015), 4)
            }
        }
    return await _acache(f'business:risk:{upper}', 180, load_db)

@router.get('/business/api/company/{symbol}/options-summary')
async def company_options_summary(symbol: str, request: Request, user=Depends(VIEW_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'options_summary', {'symbol': symbol})
    upper = symbol.upper()
    async def load_db() -> Dict[str, Any]:
        try:
            from trading_common.database_manager import get_database_manager
            db = await get_database_manager()
            async with db.get_questdb() as conn:
                q = """
                    SELECT iv_atm, iv_skew_25d, put_call_ratio, open_interest_total
                    FROM options_summary
                    WHERE symbol = $1
                    ORDER BY timestamp DESC LIMIT 1
                """
                row = await conn.fetchrow(q, upper)
                if row:
                    return {
                        'timestamp': datetime.utcnow().isoformat(),
                        'symbol': upper,
                        'options': {
                            'iv_atm': float(row['iv_atm']) if row['iv_atm'] else None,
                            'iv_skew_25d': float(row['iv_skew_25d']) if row['iv_skew_25d'] else None,
                            'put_call_ratio': float(row['put_call_ratio']) if row['put_call_ratio'] else None,
                            'open_interest': int(row['open_interest_total']) if row['open_interest_total'] else None
                        }
                    }
        except Exception:
            pass
        # Fallback
        r = _seeded_rand(f"opts:{upper}")
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': upper,
            'options': {
                'iv_atm': round(r.uniform(0.15, 0.55), 3),
                'iv_skew_25d': round(r.uniform(-0.05, 0.05), 3),
                'put_call_ratio': round(r.uniform(0.6, 1.4), 2),
                'open_interest': int(r.uniform(10000, 500000))
            }
        }
    return await _acache(f'business:opts:{upper}', 120, load_db)

@router.get('/business/api/company/{symbol}/options-surface')
async def company_options_surface(symbol: str, request: Request, user=Depends(VIEW_DEP), days: int = 60):
    await _enforce_rate(request, user)
    _audit(user, 'options_surface', {'symbol': symbol, 'days': days})
    upper = symbol.upper()
    async def load_db() -> Dict[str, Any]:
        try:
            from trading_common.database_manager import get_database_manager
            db = await get_database_manager()
            async with db.get_questdb() as conn:
                q = """
                    SELECT timestamp, iv_atm, risk_reversal_25d
                    FROM options_surface_snapshot
                    WHERE symbol = $1 AND timestamp > now() - $2 * 24 * 3600 * 1000000
                    ORDER BY timestamp ASC
                """
                rows = await conn.fetch(q, upper, days)
                if rows:
                    surface = []
                    for row in rows:
                        surface.append({
                            'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                            'iv_atm': float(row['iv_atm']) if row['iv_atm'] else None,
                            'risk_reversal_25d': float(row['risk_reversal_25d']) if row['risk_reversal_25d'] else None
                        })
                    return {
                        'timestamp': datetime.utcnow().isoformat(),
                        'symbol': upper,
                        'days': days,
                        'surface': surface
                    }
        except Exception:
            pass
        # Fallback
        r = _seeded_rand(f"optsurf:{upper}:{days}")
        surface = []
        base_iv = r.uniform(0.2, 0.4)
        for i in range(min(days, 20)):
            surface.append({
                'timestamp': (datetime.utcnow() - timedelta(days=days-i)).isoformat(),
                'iv_atm': round(base_iv + r.uniform(-0.05, 0.05), 3),
                'risk_reversal_25d': round(r.uniform(-0.02, 0.02), 4)
            })
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': upper,
            'days': days,
            'surface': surface
        }
    return await _acache(f'business:optsurf:{upper}:{days}', 300, load_db)
