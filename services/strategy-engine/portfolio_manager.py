#!/usr/bin/env python3
"""
Portfolio Manager - Position Sizing and Risk Management
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "../../shared/python-common"))

from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from trading_common import get_logger

logger = get_logger(__name__)


@dataclass
class PortfolioConfig:
    """Portfolio configuration"""
    total_capital: float = 100000.0  # Total account value
    max_position_pct: float = 0.10  # 10% max per position
    max_portfolio_risk: float = 0.20  # 20% max total exposure
    max_positions: int = 10  # Maximum concurrent positions
    reserve_cash_pct: float = 0.20  # 20% cash reserve
    max_leverage: float = 1.0  # No leverage by default
    
    # Risk limits
    max_daily_loss_pct: float = 0.05  # 5% max daily loss
    max_drawdown_pct: float = 0.10  # 10% max drawdown
    
    # Position sizing
    min_position_size: float = 100.0  # Minimum position value
    use_volatility_scaling: bool = True
    target_volatility: float = 0.15  # 15% target volatility


class PortfolioManager:
    """
    Manages portfolio positions and sizing
    - Calculates position sizes based on account value
    - Enforces risk limits
    - Tracks available capital
    - Prevents over-allocation
    """
    
    def __init__(self, config: PortfolioConfig = None):
        self.config = config or PortfolioConfig()
        self.positions = {}  # symbol -> position_size
        self.cash_used = 0.0
        self.daily_pnl = 0.0
        self.peak_value = self.config.total_capital
        
        logger.info(f"Portfolio Manager initialized: ${self.config.total_capital:,.2f}")
    
    @property
    def total_capital(self) -> float:
        """Get total portfolio value"""
        return self.config.total_capital + self.daily_pnl
    
    @property
    def available_capital(self) -> float:
        """Get available capital for new positions"""
        reserve = self.config.total_capital * self.config.reserve_cash_pct
        return max(0, self.total_capital - self.cash_used - reserve)
    
    @property
    def max_position_value(self) -> float:
        """Maximum value per position"""
        return self.total_capital * self.config.max_position_pct
    
    @property
    def portfolio_utilization(self) -> float:
        """Percentage of capital in use"""
        return self.cash_used / self.total_capital if self.total_capital > 0 else 0
    
    def calculate_position_size(self, symbol: str, price: float, 
                                confidence: float = 0.5,
                                volatility: float = None) -> Dict:
        """
        Calculate position size for a symbol
        
        Args:
            symbol: Trading symbol
            price: Current price
            confidence: Strategy confidence (0-1)
            volatility: Symbol volatility (optional)
        
        Returns:
            Dict with shares, value, reason
        """
        # Check if we can take new position
        if len(self.positions) >= self.config.max_positions:
            return {
                'shares': 0,
                'value': 0.0,
                'reason': f'Max positions reached ({self.config.max_positions})'
            }
        
        # Check if symbol already has position
        if symbol in self.positions:
            return {
                'shares': 0,
                'value': 0.0,
                'reason': f'Position already exists for {symbol}'
            }
        
        # Check available capital
        if self.available_capital < self.config.min_position_size:
            return {
                'shares': 0,
                'value': 0.0,
                'reason': f'Insufficient capital (available: ${self.available_capital:,.2f})'
            }
        
        # Base position size (% of portfolio)
        base_size = self.total_capital * self.config.max_position_pct
        
        # Scale by confidence
        confidence_scaled = base_size * confidence
        
        # Scale by volatility if enabled
        if self.config.use_volatility_scaling and volatility:
            # Lower volatility = larger position
            vol_scalar = self.config.target_volatility / max(volatility, 0.01)
            vol_scalar = min(vol_scalar, 2.0)  # Cap at 2x
            position_value = confidence_scaled * vol_scalar
        else:
            position_value = confidence_scaled
        
        # Apply limits
        position_value = min(position_value, self.max_position_value)
        position_value = min(position_value, self.available_capital)
        position_value = max(position_value, self.config.min_position_size)
        
        # Calculate shares
        shares = int(position_value / price)
        actual_value = shares * price
        
        if shares == 0:
            return {
                'shares': 0,
                'value': 0.0,
                'reason': f'Price too high (${price:.2f}) for available capital'
            }
        
        return {
            'shares': shares,
            'value': actual_value,
            'pct_of_portfolio': actual_value / self.total_capital,
            'confidence': confidence,
            'reason': 'OK'
        }
    
    def add_position(self, symbol: str, shares: int, price: float) -> bool:
        """
        Add a new position
        
        Args:
            symbol: Trading symbol
            shares: Number of shares
            price: Entry price
        
        Returns:
            True if added successfully
        """
        value = shares * price
        
        # Validate
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}")
            return False
        
        if value > self.available_capital:
            logger.warning(f"Insufficient capital for {symbol}: ${value:.2f} > ${self.available_capital:.2f}")
            return False
        
        # Add position
        self.positions[symbol] = {
            'shares': shares,
            'entry_price': price,
            'entry_value': value,
            'entry_time': datetime.utcnow().isoformat(),
            'current_price': price,
            'current_value': value,
            'pnl': 0.0
        }
        
        self.cash_used += value
        
        logger.info(f"Added position: {shares} shares {symbol} @ ${price:.2f} (${value:,.2f})")
        return True
    
    def update_position(self, symbol: str, current_price: float) -> Optional[Dict]:
        """Update position with current price"""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        pos['current_price'] = current_price
        pos['current_value'] = pos['shares'] * current_price
        pos['pnl'] = pos['current_value'] - pos['entry_value']
        
        return pos
    
    def close_position(self, symbol: str, exit_price: float) -> Optional[Dict]:
        """
        Close a position
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
        
        Returns:
            Position details with P&L
        """
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return None
        
        pos = self.positions[symbol]
        exit_value = pos['shares'] * exit_price
        pnl = exit_value - pos['entry_value']
        
        # Update tracking
        self.cash_used -= pos['entry_value']
        self.daily_pnl += pnl
        
        # Update peak for drawdown calc
        current_value = self.total_capital
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        result = {
            'symbol': symbol,
            'shares': pos['shares'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'entry_value': pos['entry_value'],
            'exit_value': exit_value,
            'pnl': pnl,
            'pnl_pct': pnl / pos['entry_value'] if pos['entry_value'] > 0 else 0,
            'hold_time': datetime.utcnow().isoformat()
        }
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Closed position: {symbol} P&L=${pnl:,.2f} ({result['pnl_pct']*100:.2f}%)")
        return result
    
    def check_risk_limits(self) -> Dict:
        """
        Check if any risk limits are breached
        
        Returns:
            Dict with limit checks
        """
        current_value = self.total_capital
        drawdown = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0
        daily_loss_pct = abs(self.daily_pnl) / self.config.total_capital if self.daily_pnl < 0 else 0
        
        breaches = []
        
        # Check drawdown
        if drawdown > self.config.max_drawdown_pct:
            breaches.append(f"Max drawdown exceeded: {drawdown*100:.2f}% > {self.config.max_drawdown_pct*100:.2f}%")
        
        # Check daily loss
        if daily_loss_pct > self.config.max_daily_loss_pct:
            breaches.append(f"Daily loss limit exceeded: {daily_loss_pct*100:.2f}% > {self.config.max_daily_loss_pct*100:.2f}%")
        
        # Check position count
        if len(self.positions) > self.config.max_positions:
            breaches.append(f"Too many positions: {len(self.positions)} > {self.config.max_positions}")
        
        # Check portfolio risk
        if self.portfolio_utilization > self.config.max_portfolio_risk:
            breaches.append(f"Portfolio risk exceeded: {self.portfolio_utilization*100:.2f}% > {self.config.max_portfolio_risk*100:.2f}%")
        
        return {
            'ok': len(breaches) == 0,
            'breaches': breaches,
            'drawdown': drawdown,
            'daily_loss_pct': daily_loss_pct,
            'position_count': len(self.positions),
            'utilization': self.portfolio_utilization
        }
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        total_pnl = sum(p['pnl'] for p in self.positions.values())
        
        return {
            'total_capital': self.total_capital,
            'cash_used': self.cash_used,
            'available_capital': self.available_capital,
            'position_count': len(self.positions),
            'positions': self.positions,
            'total_pnl': total_pnl + self.daily_pnl,
            'unrealized_pnl': total_pnl,
            'realized_pnl': self.daily_pnl,
            'utilization': self.portfolio_utilization,
            'peak_value': self.peak_value
        }


if __name__ == "__main__":
    # Test portfolio manager
    config = PortfolioConfig(
        total_capital=100000.0,
        max_position_pct=0.10,  # 10% per position
        max_positions=10
    )
    
    pm = PortfolioManager(config)
    
    print("=== Portfolio Manager ===\n")
    print(f"Total Capital: ${pm.total_capital:,.2f}")
    print(f"Available: ${pm.available_capital:,.2f}")
    print(f"Max per position: ${pm.max_position_value:,.2f}")
    
    # Test position sizing
    print("\nTest Position Sizing:")
    
    # AAPL at $220
    size = pm.calculate_position_size('AAPL', 220.0, confidence=0.8)
    print(f"AAPL: {size['shares']} shares = ${size['value']:,.2f} ({size.get('pct_of_portfolio', 0)*100:.1f}%)")
    
    # Add position
    if size['shares'] > 0:
        pm.add_position('AAPL', size['shares'], 220.0)
    
    # Check limits
    limits = pm.check_risk_limits()
    print(f"\nRisk Check: {'✓ OK' if limits['ok'] else '✗ BREACH'}")
    
    # Portfolio summary
    summary = pm.get_portfolio_summary()
    print(f"\nPortfolio: {summary['position_count']} positions, ${summary['cash_used']:,.2f} used ({summary['utilization']*100:.1f}%)")
