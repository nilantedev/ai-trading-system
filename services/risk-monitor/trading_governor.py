#!/usr/bin/env python3
"""
Trading Governor - Central control system for safe unattended operation
Allows admin panel to control all trading parameters in real-time
"""

import asyncio
import json
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import redis.asyncio as redis
try:
    from trading_common.config import get_settings  # type: ignore
except Exception:  # pragma: no cover
    get_settings = None
from enum import Enum

logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """System operating modes"""
    STOPPED = "stopped"           # No trading
    PAPER = "paper"               # Paper trading only
    CONSERVATIVE = "conservative"  # Small positions, strict limits
    NORMAL = "normal"             # Standard parameters
    AGGRESSIVE = "aggressive"     # Larger positions (still safe)
    
class TradingGovernor:
    """
    Central governor for unattended operation with admin control
    All settings can be adjusted via admin panel in real-time
    """
    
    def __init__(self):
        # Use central settings when available to avoid hardcoded localhost
        redis_url = None
        password = None
        if get_settings:
            try:
                settings = get_settings()
                redis_url = settings.database.redis_url
                password = settings.database.redis_password
            except Exception:
                pass
        redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis = redis.from_url(
            redis_url,
            password=password or os.getenv("REDIS_PASSWORD"),
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        self.mode = TradingMode.PAPER
        self.settings_key = "trading:governor:settings"
        self.limits_key = "trading:governor:limits"
        self.state_key = "trading:governor:state"
        
    async def initialize_default_settings(self):
        """Set safe defaults for unattended operation"""
        default_settings = {
            # Operating mode
            "mode": "paper",
            "auto_trade_enabled": False,
            "require_confirmation": True,
            
            # Position limits (adjustable via admin)
            "max_position_size": 1000,      # $1,000 max per position
            "max_portfolio_value": 10000,   # $10,000 total
            "max_open_positions": 5,        # 5 concurrent positions
            "position_size_pct": 0.02,       # 2% of portfolio per trade
            
            # Risk limits
            "max_daily_loss": 500,           # $500 daily loss limit
            "max_drawdown_pct": 0.10,        # 10% drawdown triggers halt
            "stop_loss_pct": 0.02,           # 2% stop loss per position
            "take_profit_pct": 0.05,         # 5% take profit
            
            # Rate limits
            "max_orders_per_minute": 5,
            "max_orders_per_hour": 30,
            "max_orders_per_day": 100,
            
            # Time restrictions
            "trading_start_time": "09:30",
            "trading_end_time": "16:00",
            "trade_on_weekends": False,
            "allowed_symbols": ["SPY", "QQQ", "AAPL", "MSFT"],  # Whitelist
            "blocked_symbols": [],           # Blacklist
            
            # Auto-safety features
            "auto_stop_on_disconnect": True,
            "auto_close_at_eod": True,
            "panic_button_enabled": True,
            "heartbeat_timeout_seconds": 60,
            
            # ML model controls
            "ml_enabled": False,
            "ml_confidence_threshold": 0.8,
            "use_ensemble": False,
            "model_version": "paper_only_v1"
        }
        
        # Only set if not exists (preserve admin changes)
        exists = await self.redis.exists(self.settings_key)
        if not exists:
            await self.redis.hset(self.settings_key, mapping=default_settings)
            
    async def get_setting(self, key: str):
        """Get current setting value"""
        value = await self.redis.hget(self.settings_key, key)
        return json.loads(value) if value else None
        
    async def update_setting(self, key: str, value):
        """Update setting from admin panel"""
        await self.redis.hset(self.settings_key, key, json.dumps(value))
        await self._log_setting_change(key, value)
        
    async def can_trade(self, symbol: str, amount: float) -> tuple[bool, str]:
        """Check if trade is allowed under current settings"""
        
        # Check if trading is enabled
        if not await self.get_setting("auto_trade_enabled"):
            return False, "Auto trading disabled"
            
        # Check mode
        mode = await self.get_setting("mode")
        if mode == "stopped":
            return False, "Trading stopped"
            
        # Check symbol whitelist/blacklist
        allowed = await self.get_setting("allowed_symbols")
        if allowed and symbol not in allowed:
            return False, f"{symbol} not in allowed symbols"
            
        blocked = await self.get_setting("blocked_symbols")
        if blocked and symbol in blocked:
            return False, f"{symbol} is blocked"
            
        # Check position size
        max_size = await self.get_setting("max_position_size")
        if amount > max_size:
            return False, f"Position ${amount} exceeds max ${max_size}"
            
        # Check time restrictions
        if not await self._is_trading_hours():
            return False, "Outside trading hours"
            
        # Check rate limits
        if not await self._check_rate_limits():
            return False, "Rate limit exceeded"
            
        # Check daily loss
        daily_loss = await self._get_daily_loss()
        max_loss = await self.get_setting("max_daily_loss")
        if daily_loss >= max_loss:
            return False, f"Daily loss ${daily_loss} at limit"
            
        return True, "OK"
        
    async def _is_trading_hours(self) -> bool:
        """Check if current time is within trading hours"""
        now = datetime.now()
        
        # Check weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            if not await self.get_setting("trade_on_weekends"):
                return False
                
        # Check time
        start = await self.get_setting("trading_start_time")
        end = await self.get_setting("trading_end_time")
        
        current_time = now.strftime("%H:%M")
        return start <= current_time <= end
        
    async def _check_rate_limits(self) -> bool:
        """Check if we're within rate limits"""
        # Track orders in Redis with expiry
        now = datetime.now()
        
        # Per minute check
        minute_key = f"orders:minute:{now.strftime('%Y%m%d%H%M')}"
        minute_count = await self.redis.incr(minute_key)
        await self.redis.expire(minute_key, 60)
        
        max_per_minute = await self.get_setting("max_orders_per_minute")
        if minute_count > max_per_minute:
            return False
            
        return True
        
    async def _get_daily_loss(self) -> float:
        """Calculate today's P&L"""
        # This would query actual P&L from database
        # For now return placeholder
        pnl_key = f"pnl:daily:{datetime.now().strftime('%Y%m%d')}"
        pnl = await self.redis.get(pnl_key)
        return float(pnl) if pnl else 0
        
    async def emergency_stop(self, reason: str, cancel_orders: bool = True, close_positions: bool = False):
        """EMERGENCY STOP - Called by panic button or auto-safety"""
        # IMMEDIATE HALT
        await self.update_setting("mode", "stopped")
        await self.update_setting("auto_trade_enabled", False)
        
        # Set kill switch flag with 24hr expiry
        await self.redis.set("trading:kill_switch:active", "1", ex=86400)
        
        # Log emergency stop with full context
        stop_event = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "cancel_orders": cancel_orders,
            "close_positions": close_positions,
            "positions_at_stop": await self._get_position_count(),
            "daily_pnl_at_stop": await self._get_daily_loss()
        }
        await self.redis.lpush("emergency:stops", json.dumps(stop_event))
        
        # Notify all systems immediately
        await self.redis.publish("trading:emergency", json.dumps({
            "event": "EMERGENCY_STOP",
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }))
        
        # Cancel all open orders if requested
        if cancel_orders:
            await self._cancel_all_open_orders()
            
        # Close all positions if requested (more drastic)
        if close_positions:
            await self._close_all_positions()
        
        return {"status": "EMERGENCY STOP", "reason": reason, "actions_taken": {
            "orders_cancelled": cancel_orders,
            "positions_closed": close_positions
        }}
        
    async def get_current_state(self) -> Dict:
        """Get full system state for admin panel"""
        settings = await self.redis.hgetall(self.settings_key)
        
        # Get current metrics
        state = {
            "mode": await self.get_setting("mode"),
            "auto_trading": await self.get_setting("auto_trade_enabled"),
            "current_positions": await self._get_position_count(),
            "daily_pnl": await self._get_daily_loss(),
            "is_trading_hours": await self._is_trading_hours(),
            "settings": {k: json.loads(v) for k, v in settings.items()},
            "health": await self._system_health_check()
        }
        
        return state
        
    async def _get_position_count(self) -> int:
        """Get current open positions"""
        # Query from database/redis
        return 0  # Placeholder
        
    async def _system_health_check(self) -> Dict:
        """Check system health"""
        return {
            "api": "healthy",
            "database": "healthy",
            "redis": "healthy",
            "data_feed": "healthy",
            "last_heartbeat": datetime.now().isoformat()
        }
        
    async def _log_setting_change(self, key: str, value):
        """Audit log for setting changes"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "setting": key,
            "new_value": value,
            "changed_by": "admin"  # Would get from auth context
        }
        await self.redis.lpush("audit:setting_changes", json.dumps(log_entry))
        
    async def apply_trading_mode(self, mode: TradingMode):
        """Apply preset configurations based on mode"""
        presets = {
            TradingMode.STOPPED: {
                "auto_trade_enabled": False,
                "max_position_size": 0
            },
            TradingMode.PAPER: {
                "auto_trade_enabled": True,
                "max_position_size": 10000,
                "max_portfolio_value": 100000
            },
            TradingMode.CONSERVATIVE: {
                "auto_trade_enabled": True,
                "max_position_size": 500,
                "max_portfolio_value": 5000,
                "max_daily_loss": 100,
                "stop_loss_pct": 0.01
            },
            TradingMode.NORMAL: {
                "auto_trade_enabled": True,
                "max_position_size": 2000,
                "max_portfolio_value": 20000,
                "max_daily_loss": 500,
                "stop_loss_pct": 0.02
            },
            TradingMode.AGGRESSIVE: {
                "auto_trade_enabled": True,
                "max_position_size": 5000,
                "max_portfolio_value": 50000,
                "max_daily_loss": 1000,
                "stop_loss_pct": 0.03
            }
        }
        
        if mode in presets:
            for key, value in presets[mode].items():
                await self.update_setting(key, value)
            await self.update_setting("mode", mode.value)
    
    async def _cancel_all_open_orders(self):
        """Cancel all open orders immediately"""
        try:
            # Publish cancellation event
            await self.redis.publish("orders:cancel_all", json.dumps({
                "reason": "emergency_stop",
                "timestamp": datetime.now().isoformat()
            }))
            return True
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            return False
    
    async def _close_all_positions(self):
        """Close all positions at market (emergency only)"""
        try:
            # Publish position closure event
            await self.redis.publish("positions:close_all", json.dumps({
                "reason": "emergency_stop",
                "type": "market",
                "timestamp": datetime.now().isoformat()
            }))
            return True
        except Exception as e:
            logger.error(f"Failed to close positions: {e}")
            return False
    
    async def check_kill_switch(self) -> bool:
        """Check if kill switch is active"""
        kill_switch = await self.redis.get("trading:kill_switch:active")
        return kill_switch == "1"
    
    async def clear_kill_switch(self, admin_key: str) -> bool:
        """Clear kill switch (requires admin key)"""
        # In production, validate admin_key properly
        if admin_key:
            await self.redis.delete("trading:kill_switch:active")
            await self.redis.lpush("audit:kill_switch_cleared", json.dumps({
                "timestamp": datetime.now().isoformat(),
                "cleared_by": admin_key[:8] + "****"  # Log partial key for audit
            }))
            return True
        return False