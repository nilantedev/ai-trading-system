#!/usr/bin/env python3
"""
Hard Risk Limits - Absolute limits that CANNOT be exceeded
These are the guardrails that prevent catastrophic losses
"""

import logging
from typing import Dict, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level for different trading modes"""
    PAPER = "paper"
    CONSERVATIVE = "conservative"  
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"


@dataclass
class OrderValidation:
    """Result of order validation"""
    is_valid: bool
    reason: str
    risk_score: float = 0.0
    checks_performed: Dict[str, bool] = None


class HardLimits:
    """
    Absolute limits that CANNOT be exceeded under any circumstances.
    These are fail-safe mechanisms to prevent catastrophic losses.
    """
    
    # Position limits - Start VERY conservative for real money
    MAX_POSITION_SIZE_USD = {
        RiskLevel.PAPER: 10000,        # Paper trading
        RiskLevel.CONSERVATIVE: 1000,   # $1k max for conservative
        RiskLevel.NORMAL: 5000,         # $5k for normal
        RiskLevel.AGGRESSIVE: 10000     # $10k for aggressive
    }
    
    MAX_POSITION_PCT_PORTFOLIO = 0.10  # 10% max per position regardless of mode
    
    # Portfolio limits
    MAX_TOTAL_EXPOSURE_USD = {
        RiskLevel.PAPER: 100000,       # Paper trading
        RiskLevel.CONSERVATIVE: 10000,  # $10k conservative
        RiskLevel.NORMAL: 50000,        # $50k normal  
        RiskLevel.AGGRESSIVE: 100000    # $100k aggressive
    }
    
    MAX_LEVERAGE = 1.0  # NO leverage initially (can be increased later)
    MAX_OPEN_POSITIONS = {
        RiskLevel.PAPER: 50,
        RiskLevel.CONSERVATIVE: 5,
        RiskLevel.NORMAL: 20,
        RiskLevel.AGGRESSIVE: 30
    }
    
    # Order limits
    MAX_ORDER_SIZE_USD = {
        RiskLevel.PAPER: 10000,
        RiskLevel.CONSERVATIVE: 500,    # Very small orders to start
        RiskLevel.NORMAL: 2500,
        RiskLevel.AGGRESSIVE: 5000
    }
    
    MAX_ORDERS_PER_DAY = 100
    MAX_ORDERS_PER_MINUTE = 5
    MAX_ORDERS_PER_HOUR = 30
    
    # Loss limits - CRITICAL for capital preservation
    MAX_DAILY_LOSS_USD = {
        RiskLevel.PAPER: 10000,
        RiskLevel.CONSERVATIVE: 200,    # Tight stop at $200
        RiskLevel.NORMAL: 1000,         # $1k daily max
        RiskLevel.AGGRESSIVE: 2000      # $2k for aggressive
    }
    
    MAX_WEEKLY_LOSS_USD = {
        RiskLevel.PAPER: 50000,
        RiskLevel.CONSERVATIVE: 500,
        RiskLevel.NORMAL: 3000,
        RiskLevel.AGGRESSIVE: 7000
    }
    
    MAX_DRAWDOWN_PCT = 0.15  # 15% max drawdown triggers halt
    
    # Minimum account balance to trade
    MIN_ACCOUNT_BALANCE = {
        RiskLevel.PAPER: 0,             # No minimum for paper
        RiskLevel.CONSERVATIVE: 2000,   # Need $2k minimum
        RiskLevel.NORMAL: 10000,        # $10k for normal
        RiskLevel.AGGRESSIVE: 25000     # $25k for aggressive (PDT rule)
    }
    
    @staticmethod
    def validate_order(order: Dict[str, Any], portfolio: Dict[str, Any], 
                      risk_level: RiskLevel = RiskLevel.CONSERVATIVE) -> OrderValidation:
        """
        Validate an order against ALL hard limits.
        Returns (is_valid, rejection_reason, risk_score)
        """
        checks = {}
        
        # Check order size
        order_value = order.get('quantity', 0) * order.get('price', 0)
        max_order_size = HardLimits.MAX_ORDER_SIZE_USD.get(risk_level, 1000)
        
        if order_value > max_order_size:
            return OrderValidation(
                is_valid=False,
                reason=f"Order ${order_value:.2f} exceeds max ${max_order_size}",
                risk_score=1.0,
                checks_performed={"order_size": False}
            )
        checks["order_size"] = True
        
        # Check position size limit
        current_position_value = portfolio.get('positions', {}).get(order['symbol'], {}).get('value', 0)
        new_position_value = current_position_value + order_value
        max_position = HardLimits.MAX_POSITION_SIZE_USD.get(risk_level, 1000)
        
        if new_position_value > max_position:
            return OrderValidation(
                is_valid=False,
                reason=f"Position ${new_position_value:.2f} would exceed max ${max_position}",
                risk_score=0.9,
                checks_performed={**checks, "position_size": False}
            )
        checks["position_size"] = True
        
        # Check portfolio percentage
        portfolio_value = portfolio.get('total_value', 0)
        if portfolio_value > 0:
            position_pct = new_position_value / portfolio_value
            if position_pct > HardLimits.MAX_POSITION_PCT_PORTFOLIO:
                return OrderValidation(
                    is_valid=False,
                    reason=f"Position would be {position_pct*100:.1f}% of portfolio (max {HardLimits.MAX_POSITION_PCT_PORTFOLIO*100}%)",
                    risk_score=0.85,
                    checks_performed={**checks, "position_pct": False}
                )
        checks["position_pct"] = True
        
        # Check total exposure
        current_exposure = portfolio.get('total_exposure', 0)
        new_exposure = current_exposure + order_value
        max_exposure = HardLimits.MAX_TOTAL_EXPOSURE_USD.get(risk_level, 10000)
        
        if new_exposure > max_exposure:
            return OrderValidation(
                is_valid=False,
                reason=f"Total exposure ${new_exposure:.2f} would exceed max ${max_exposure}",
                risk_score=0.95,
                checks_performed={**checks, "total_exposure": False}
            )
        checks["total_exposure"] = True
        
        # Check number of open positions
        open_positions = len(portfolio.get('positions', {}))
        max_positions = HardLimits.MAX_OPEN_POSITIONS.get(risk_level, 5)
        
        if order.get('side') == 'buy' and open_positions >= max_positions:
            return OrderValidation(
                is_valid=False,
                reason=f"Already have {open_positions} positions (max {max_positions})",
                risk_score=0.7,
                checks_performed={**checks, "position_count": False}
            )
        checks["position_count"] = True
        
        # Check daily loss limit
        daily_pnl = portfolio.get('daily_pnl', 0)
        max_daily_loss = HardLimits.MAX_DAILY_LOSS_USD.get(risk_level, 500)
        
        if daily_pnl < -max_daily_loss:
            return OrderValidation(
                is_valid=False,
                reason=f"Daily loss ${-daily_pnl:.2f} exceeds limit ${max_daily_loss}",
                risk_score=1.0,
                checks_performed={**checks, "daily_loss": False}
            )
        checks["daily_loss"] = True
        
        # Check account balance
        account_balance = portfolio.get('cash_balance', 0)
        min_balance = HardLimits.MIN_ACCOUNT_BALANCE.get(risk_level, 0)
        
        if account_balance < min_balance:
            return OrderValidation(
                is_valid=False,
                reason=f"Account balance ${account_balance:.2f} below minimum ${min_balance}",
                risk_score=1.0,
                checks_performed={**checks, "min_balance": False}
            )
        checks["min_balance"] = True
        
        # Calculate risk score (0-1, lower is better)
        risk_score = 0.0
        risk_score += (order_value / max_order_size) * 0.2
        risk_score += (new_position_value / max_position) * 0.2
        risk_score += (new_exposure / max_exposure) * 0.3
        risk_score += (open_positions / max_positions) * 0.1
        if daily_pnl < 0:
            risk_score += (-daily_pnl / max_daily_loss) * 0.2
        
        return OrderValidation(
            is_valid=True,
            reason="All checks passed",
            risk_score=min(risk_score, 1.0),
            checks_performed=checks
        )
    
    @staticmethod
    def get_safe_position_size(portfolio: Dict[str, Any], 
                               risk_level: RiskLevel = RiskLevel.CONSERVATIVE) -> float:
        """Calculate safe position size based on current portfolio state"""
        portfolio_value = portfolio.get('total_value', 0)
        max_position_usd = HardLimits.MAX_POSITION_SIZE_USD.get(risk_level, 1000)
        max_position_pct = HardLimits.MAX_POSITION_PCT_PORTFOLIO
        
        # Take the smaller of percentage-based or absolute limit
        pct_based_limit = portfolio_value * max_position_pct
        safe_size = min(max_position_usd, pct_based_limit)
        
        # Further reduce if we've had losses today
        daily_pnl = portfolio.get('daily_pnl', 0)
        if daily_pnl < 0:
            # Reduce position size proportionally to losses
            loss_reduction = 1.0 - (abs(daily_pnl) / HardLimits.MAX_DAILY_LOSS_USD.get(risk_level, 500))
            safe_size *= max(0.5, loss_reduction)  # At least 50% of normal size
        
        return safe_size
    
    @staticmethod
    def emergency_checks(portfolio: Dict[str, Any], 
                        risk_level: RiskLevel = RiskLevel.CONSERVATIVE) -> Tuple[bool, Optional[str]]:
        """
        Perform emergency checks that should trigger immediate halt
        Returns (should_halt, reason)
        """
        # Check daily loss
        daily_pnl = portfolio.get('daily_pnl', 0)
        max_daily_loss = HardLimits.MAX_DAILY_LOSS_USD.get(risk_level, 500)
        if daily_pnl < -max_daily_loss:
            return True, f"Daily loss ${-daily_pnl:.2f} exceeds limit"
        
        # Check weekly loss
        weekly_pnl = portfolio.get('weekly_pnl', 0)
        max_weekly_loss = HardLimits.MAX_WEEKLY_LOSS_USD.get(risk_level, 1000)
        if weekly_pnl < -max_weekly_loss:
            return True, f"Weekly loss ${-weekly_pnl:.2f} exceeds limit"
        
        # Check drawdown
        peak_value = portfolio.get('peak_value', portfolio.get('total_value', 0))
        current_value = portfolio.get('total_value', 0)
        if peak_value > 0:
            drawdown = (peak_value - current_value) / peak_value
            if drawdown > HardLimits.MAX_DRAWDOWN_PCT:
                return True, f"Drawdown {drawdown*100:.1f}% exceeds limit"
        
        # Check for margin call (if using margin)
        if portfolio.get('margin_call', False):
            return True, "Margin call received"
        
        return False, None