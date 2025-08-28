# 🔴 CRITICAL FIXES BEFORE LIVE TRADING
## Must Fix Before Real Money (Currently Paper Trading Only)

---

## 🚨 CRITICAL RISKS TO ADDRESS NOW

### 1. Risk Controls & Kill Switch
**Gap:** No enforced runtime risk gating or emergency stop

**FIX NOW:**
```python
# Create file: services/risk-monitor/kill_switch.py
import asyncio
from typing import Dict
from datetime import datetime
import redis

class TradingKillSwitch:
    """Emergency trading halt mechanism"""
    
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379)
        self.KILL_SWITCH_KEY = "trading:kill_switch:active"
        self.RISK_LIMITS = {
            "max_daily_loss": 10000,  # $10k max daily loss
            "max_position_size": 50000,  # $50k max per position
            "max_total_exposure": 100000,  # $100k total exposure
            "max_order_value": 10000,  # $10k per order
            "max_orders_per_minute": 10
        }
    
    async def emergency_stop(self, reason: str):
        """IMMEDIATE HALT - Stops ALL trading"""
        await self.redis_client.set(self.KILL_SWITCH_KEY, "1", ex=86400)
        await self.redis_client.publish("trading:alerts", f"EMERGENCY STOP: {reason}")
        # Cancel all open orders
        # Close all positions
        # Alert all connected clients
        print(f"🔴 TRADING HALTED: {reason}")
        
    async def check_before_trade(self, order: Dict) -> bool:
        """Must pass before EVERY order"""
        # Check kill switch
        if await self.redis_client.get(self.KILL_SWITCH_KEY):
            return False
            
        # Check limits
        if order['value'] > self.RISK_LIMITS['max_order_value']:
            await self.emergency_stop(f"Order exceeds limit: ${order['value']}")
            return False
            
        return True
```

**ADD TO API NOW:**
```python
# In api/main.py - Add this endpoint
@app.post("/emergency/stop")
async def emergency_stop(reason: str = "Manual stop"):
    """BIG RED BUTTON - Stops everything"""
    kill_switch = TradingKillSwitch()
    await kill_switch.emergency_stop(reason)
    return {"status": "TRADING HALTED", "reason": reason}
```

### 2. Position & Exposure Limits
**Gap:** No hard limits on positions

**FIX NOW:**
```python
# Add to shared/python-common/trading_common/risk_limits.py
class HardLimits:
    """Absolute limits that CANNOT be exceeded"""
    
    # Per-position limits
    MAX_POSITION_SIZE_USD = 10000  # Start small!
    MAX_POSITION_PCT_PORTFOLIO = 0.10  # 10% max per position
    
    # Portfolio limits  
    MAX_TOTAL_EXPOSURE_USD = 50000
    MAX_LEVERAGE = 1.0  # No leverage initially
    MAX_OPEN_POSITIONS = 20
    
    # Order limits
    MAX_ORDER_SIZE_USD = 5000
    MAX_ORDERS_PER_DAY = 100
    MAX_ORDERS_PER_MINUTE = 5
    
    # Loss limits
    MAX_DAILY_LOSS_USD = 2000
    MAX_WEEKLY_LOSS_USD = 5000
    MAX_DRAWDOWN_PCT = 0.15  # 15% max drawdown
    
    @staticmethod
    def validate_order(order, portfolio) -> tuple[bool, str]:
        """Returns (is_valid, rejection_reason)"""
        # Check every limit
        if order.value > HardLimits.MAX_ORDER_SIZE_USD:
            return False, f"Order ${order.value} exceeds max ${HardLimits.MAX_ORDER_SIZE_USD}"
        # ... check all limits
        return True, ""
```

### 3. Data Validation & Quality Checks
**Gap:** No validation of incoming data

**FIX NOW:**
```python
# Create file: services/data_ingestion/data_validator.py
from typing import Dict, List
import numpy as np
from datetime import datetime, timedelta

class MarketDataValidator:
    """Validates all incoming market data"""
    
    def __init__(self):
        self.SANITY_CHECKS = {
            "min_price": 0.01,
            "max_price": 100000,
            "max_price_change_pct": 0.50,  # 50% in one tick
            "max_spread_pct": 0.10,  # 10% bid-ask spread
            "stale_data_seconds": 60
        }
    
    def validate_tick(self, tick: Dict) -> tuple[bool, str]:
        """Validate single price tick"""
        
        # Price sanity
        if tick['price'] <= self.SANITY_CHECKS['min_price']:
            return False, f"Price too low: {tick['price']}"
            
        if tick['price'] > self.SANITY_CHECKS['max_price']:
            return False, f"Price too high: {tick['price']}"
            
        # Timestamp freshness
        age = datetime.now() - tick['timestamp']
        if age.total_seconds() > self.SANITY_CHECKS['stale_data_seconds']:
            return False, f"Stale data: {age.total_seconds()}s old"
            
        # Spread validation
        if 'bid' in tick and 'ask' in tick:
            spread_pct = (tick['ask'] - tick['bid']) / tick['bid']
            if spread_pct > self.SANITY_CHECKS['max_spread_pct']:
                return False, f"Spread too wide: {spread_pct*100:.2f}%"
                
        return True, "OK"
    
    def detect_anomalies(self, prices: List[float]) -> Dict:
        """Detect price anomalies"""
        return {
            "has_gaps": self._detect_gaps(prices),
            "has_spikes": self._detect_spikes(prices),
            "is_flatlined": len(set(prices)) == 1,
            "has_negatives": any(p <= 0 for p in prices)
        }
```

---

## 🟡 HIGH PRIORITY (Fix Week 1 on Server)

### 1. Model Drift Detection
```python
# Add to services/ml/drift_monitor.py
class DriftMonitor:
    def __init__(self):
        self.baseline_stats = {}
        
    async def detect_drift(self, features, predictions):
        """KS test for distribution shift"""
        from scipy.stats import ks_2samp
        
        for feature_name, values in features.items():
            if feature_name in self.baseline_stats:
                ks_stat, p_value = ks_2samp(
                    self.baseline_stats[feature_name],
                    values
                )
                if p_value < 0.01:  # Significant drift
                    await self.alert_drift(feature_name, ks_stat, p_value)
```

### 2. Audit Trail Immutability
```python
# Add to shared/python-common/trading_common/audit.py
import hashlib
import json

class ImmutableAuditLog:
    def __init__(self):
        self.chain = []
        
    def add_event(self, event: Dict):
        """Add event with hash chain"""
        event['timestamp'] = datetime.utcnow().isoformat()
        event['index'] = len(self.chain)
        
        # Chain to previous hash
        if self.chain:
            event['prev_hash'] = self.chain[-1]['hash']
        else:
            event['prev_hash'] = "0"
            
        # Calculate hash
        event_str = json.dumps(event, sort_keys=True)
        event['hash'] = hashlib.sha256(event_str.encode()).hexdigest()
        
        self.chain.append(event)
        
        # Write to append-only log
        with open('/var/log/trading-system/audit.log', 'a') as f:
            f.write(json.dumps(event) + '\n')
```

### 3. Backtesting Realism
```python
# Add to services/ml/realistic_backtest.py
class RealisticBacktest:
    def __init__(self):
        self.SLIPPAGE_MODEL = {
            "base_bps": 5,  # 5 basis points base
            "size_impact": 0.1,  # 10bps per $100k
            "volatility_mult": 1.5
        }
        
    def calculate_slippage(self, order, market_state):
        """Realistic slippage model"""
        base = self.SLIPPAGE_MODEL['base_bps'] / 10000
        size_impact = (order.value / 100000) * self.SLIPPAGE_MODEL['size_impact'] / 10000
        vol_impact = market_state.volatility * self.SLIPPAGE_MODEL['volatility_mult']
        
        total_slippage = base + size_impact + vol_impact
        return order.price * (1 + total_slippage * order.side)
```

---

## 🟢 MEDIUM PRIORITY (Month 1)

### Dependency Scanning
```bash
# Add to Makefile
security-scan:
    pip-audit --fix
    safety check
    bandit -r services/ shared/
    
pin-deps:
    pip-compile requirements.in --generate-hashes
```

### Config Validation
```python
# Add to api/main.py startup
from pydantic import BaseSettings, validator

class TradingConfig(BaseSettings):
    alpaca_api_key: str
    max_position_size: float
    
    @validator('max_position_size')
    def validate_position_size(cls, v):
        if v > 100000:
            raise ValueError("Position size too large for current capital")
        return v
        
# On startup
try:
    config = TradingConfig()
except ValidationError as e:
    logger.error(f"CONFIG VALIDATION FAILED: {e}")
    sys.exit(1)
```

---

## 📊 METRICS TO ADD IMMEDIATELY

```python
# Add to api/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Critical metrics
orders_placed = Counter('trading_orders_total', 'Total orders placed')
orders_rejected = Counter('trading_orders_rejected', 'Orders rejected by risk')
risk_breaches = Counter('trading_risk_breaches', 'Risk limit breaches')
kill_switch_triggers = Counter('trading_kill_switch', 'Kill switch activations')

# Data quality
data_validation_failures = Counter('data_validation_failures', 'Invalid data points')
data_staleness = Histogram('data_staleness_seconds', 'Age of data when processed')

# Model metrics  
model_drift_score = Gauge('model_drift_score', 'Distribution drift score')
prediction_confidence = Histogram('model_prediction_confidence', 'Model confidence')
```

---

## ✅ VALIDATION CHECKLIST BEFORE LIVE

### Must Have Working:
- [ ] Kill switch endpoint tested
- [ ] All risk limits enforced
- [ ] Data validation catching bad ticks
- [ ] Audit trail writing immutable logs
- [ ] Metrics showing in Prometheus

### Must Test:
- [ ] Pull network cable - system stops trading
- [ ] Send $1M order - gets rejected
- [ ] Send 100 orders/second - rate limit works
- [ ] Kill Redis - system goes to safe mode
- [ ] Manual emergency stop - halts in <1 second

### Must Document:
- [ ] Risk limit configuration
- [ ] Emergency procedures
- [ ] Rollback process
- [ ] Incident response plan

---

## 🚀 DEPLOYMENT STRATEGY

### Phase 1: Paper Trading (NOW)
- Deploy with current code
- Paper trade only
- Monitor everything
- Fix issues as they appear

### Phase 2: Small Live (Week 2-4)
- Add all risk controls above
- Start with $1,000 account
- Max $100 positions
- Monitor 24/7

### Phase 3: Scale Up (Month 2+)
- Proven profitable in paper
- All controls tested
- Gradually increase limits
- Add ML models

Remember: **Better to be safe than sorry!** Start small, fail fast, learn constantly.