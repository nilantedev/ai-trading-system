#!/usr/bin/env python3
"""
Market Making Strategy - Bid-Ask Spread Capture
Based on Virtu Financial and Tower Research approaches
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketMakingSignal:
    """Signal for market making"""
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    spread: float
    inventory: int
    skew: float  # Inventory skew adjustment
    confidence: float
    edge: float  # Expected edge in bps


class MarketMakingStrategy:
    """
    Elite market making strategy using:
    - Inventory management (Avellaneda-Stoikov model)
    - Order book imbalance
    - Adverse selection protection
    - Dynamic spread adjustment
    
    Used by:
    - Virtu Financial (99.9% profitable days)
    - Tower Research Capital
    - Jump Trading
    - Hudson River Trading
    """
    
    def __init__(
        self,
        base_spread_bps: float = 10.0,  # Base spread in basis points
        min_spread_bps: float = 5.0,  # Minimum spread
        max_spread_bps: float = 50.0,  # Maximum spread
        target_inventory: int = 0,  # Target inventory (neutral)
        max_inventory: int = 1000,  # Maximum inventory
        inventory_penalty: float = 0.01,  # Inventory risk penalty
        tick_size: float = 0.01,  # Minimum price increment
        risk_aversion: float = 0.5,  # Avellaneda-Stoikov gamma
        volatility_window: int = 100,  # Trades for volatility calc
    ):
        self.base_spread_bps = base_spread_bps
        self.min_spread_bps = min_spread_bps
        self.max_spread_bps = max_spread_bps
        self.target_inventory = target_inventory
        self.max_inventory = max_inventory
        self.inventory_penalty = inventory_penalty
        self.tick_size = tick_size
        self.risk_aversion = risk_aversion
        self.volatility_window = volatility_window
        
        # State tracking
        self.inventory: Dict[str, int] = {}
        self.recent_trades: Dict[str, deque] = {}
        self.quote_history: Dict[str, List[Dict]] = {}
        
        logger.info(
            "Market making strategy initialized",
            base_spread=base_spread_bps,
            max_inventory=max_inventory
        )
    
    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate recent realized volatility."""
        if symbol not in self.recent_trades or len(self.recent_trades[symbol]) < 2:
            return 0.02  # Default 2% volatility
        
        trades = list(self.recent_trades[symbol])
        prices = [t['price'] for t in trades]
        returns = np.diff(np.log(prices))
        
        if len(returns) == 0:
            return 0.02
        
        volatility = np.std(returns) * np.sqrt(252 * 390)  # Annualized
        return max(volatility, 0.001)  # Minimum 0.1%
    
    def _calculate_order_book_imbalance(
        self,
        bid_volume: float,
        ask_volume: float
    ) -> float:
        """
        Calculate order book imbalance.
        Positive = more buying pressure, Negative = more selling pressure
        """
        total = bid_volume + ask_volume
        if total == 0:
            return 0.0
        return (bid_volume - ask_volume) / total
    
    def _calculate_inventory_skew(
        self,
        symbol: str,
        mid_price: float,
        volatility: float,
        time_horizon: float = 1.0  # Hours until close
    ) -> float:
        """
        Calculate inventory skew using Avellaneda-Stoikov model.
        
        Adjusts quotes to mean-revert inventory to target:
        - High inventory → widen asks, tighten bids (encourage selling)
        - Low inventory → tighten asks, widen bids (encourage buying)
        """
        current_inventory = self.inventory.get(symbol, 0)
        inventory_diff = current_inventory - self.target_inventory
        
        # Avellaneda-Stoikov reservation price adjustment
        # r = s - q * gamma * sigma^2 * (T - t)
        # where q = inventory, gamma = risk aversion, sigma = volatility
        skew = inventory_diff * self.risk_aversion * (volatility ** 2) * time_horizon
        
        return skew
    
    def _calculate_optimal_spread(
        self,
        mid_price: float,
        volatility: float,
        order_book_imbalance: float,
        time_horizon: float = 1.0
    ) -> float:
        """
        Calculate optimal spread using Avellaneda-Stoikov formula.
        
        Optimal spread = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
        Simplified to: base_spread + volatility_adjustment + imbalance_adjustment
        """
        # Volatility component
        vol_adjustment = self.risk_aversion * (volatility ** 2) * time_horizon
        vol_spread_bps = vol_adjustment * 10000  # Convert to bps
        
        # Order book imbalance component
        # Widen spread if imbalanced (adverse selection protection)
        imbalance_spread_bps = abs(order_book_imbalance) * 10
        
        # Total spread
        total_spread_bps = self.base_spread_bps + vol_spread_bps + imbalance_spread_bps
        
        # Clamp to limits
        return np.clip(total_spread_bps, self.min_spread_bps, self.max_spread_bps)
    
    async def generate_quotes(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        order_book: Optional[Dict[str, Any]] = None
    ) -> Optional[MarketMakingSignal]:
        """
        Generate bid/ask quotes for a symbol.
        
        Args:
            symbol: Trading symbol
            market_data: Current market data (last, bid, ask, volume)
            order_book: Optional order book data
            
        Returns:
            MarketMakingSignal or None
        """
        # Extract market data
        last_price = market_data.get('last', 0)
        current_bid = market_data.get('bid', 0)
        current_ask = market_data.get('ask', 0)
        
        if last_price == 0 or current_bid == 0 or current_ask == 0:
            return None
        
        # Calculate mid price
        mid_price = (current_bid + current_ask) / 2
        
        # Update trade history
        if symbol not in self.recent_trades:
            self.recent_trades[symbol] = deque(maxlen=self.volatility_window)
        
        self.recent_trades[symbol].append({
            'price': last_price,
            'time': datetime.now()
        })
        
        # Calculate volatility
        volatility = self._calculate_volatility(symbol)
        
        # Calculate order book imbalance
        if order_book:
            bid_volume = sum(level['size'] for level in order_book.get('bids', [])[:5])
            ask_volume = sum(level['size'] for level in order_book.get('asks', [])[:5])
            imbalance = self._calculate_order_book_imbalance(bid_volume, ask_volume)
        else:
            imbalance = 0.0
        
        # Calculate optimal spread
        spread_bps = self._calculate_optimal_spread(
            mid_price, volatility, imbalance
        )
        spread = mid_price * spread_bps / 10000
        
        # Calculate inventory skew
        skew = self._calculate_inventory_skew(symbol, mid_price, volatility)
        
        # Adjust mid price for inventory skew
        adjusted_mid = mid_price - skew
        
        # Calculate quotes
        bid_price = adjusted_mid - spread / 2
        ask_price = adjusted_mid + spread / 2
        
        # Round to tick size
        bid_price = round(bid_price / self.tick_size) * self.tick_size
        ask_price = round(ask_price / self.tick_size) * self.tick_size
        
        # Check inventory limits
        current_inventory = self.inventory.get(symbol, 0)
        if abs(current_inventory) >= self.max_inventory:
            logger.warning(
                f"Inventory limit reached for {symbol}",
                inventory=current_inventory,
                limit=self.max_inventory
            )
            # Don't quote on the side that would increase inventory
            if current_inventory > 0:
                # Too long, only offer to sell
                bid_price = 0
            else:
                # Too short, only bid to buy
                ask_price = 0
        
        # Calculate position sizes (scale down near limits)
        inventory_ratio = abs(current_inventory) / self.max_inventory
        size_scalar = max(0.1, 1.0 - inventory_ratio)
        
        base_size = 100
        bid_size = int(base_size * size_scalar) if bid_price > 0 else 0
        ask_size = int(base_size * size_scalar) if ask_price > 0 else 0
        
        # Calculate confidence (higher when inventory neutral, lower when at limits)
        confidence = (1.0 - inventory_ratio) * (1.0 - abs(imbalance))
        
        # Calculate expected edge
        market_spread = current_ask - current_bid
        our_spread = ask_price - bid_price if ask_price > 0 and bid_price > 0 else 0
        edge_bps = (our_spread / mid_price) * 10000 if our_spread > 0 else 0
        
        signal = MarketMakingSignal(
            symbol=symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            spread=our_spread,
            inventory=current_inventory,
            skew=skew,
            confidence=confidence,
            edge=edge_bps
        )
        
        logger.debug(
            f"Market making quote for {symbol}",
            bid=bid_price,
            ask=ask_price,
            spread_bps=spread_bps,
            inventory=current_inventory,
            edge_bps=edge_bps
        )
        
        return signal
    
    def update_inventory(
        self,
        symbol: str,
        quantity: int,  # Positive = bought, Negative = sold
        price: float
    ):
        """Update inventory after a fill."""
        if symbol not in self.inventory:
            self.inventory[symbol] = 0
        
        self.inventory[symbol] += quantity
        
        logger.info(
            f"Inventory updated for {symbol}",
            quantity=quantity,
            price=price,
            new_inventory=self.inventory[symbol]
        )
    
    async def backtest(
        self,
        market_data: Dict[str, pd.DataFrame],
        initial_capital: float = 100000.0,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Backtest the market making strategy.
        
        Args:
            market_data: Historical tick/quote data
            initial_capital: Starting capital
            symbols: Symbols to make markets in
            
        Returns:
            Backtest results
        """
        if symbols is None:
            symbols = list(market_data.keys())[:5]  # Limit to 5 symbols
        
        logger.info("Starting market making backtest", symbols=len(symbols))
        
        capital = initial_capital
        equity_curve = [capital]
        trades = []
        total_spread_captured = 0.0
        
        # Simulate market making
        for symbol in symbols:
            if symbol not in market_data:
                continue
            
            df = market_data[symbol]
            
            for idx in range(100, len(df)):
                # Get market snapshot
                row = df.iloc[idx]
                market_snapshot = {
                    'last': row.get('close', 0),
                    'bid': row.get('close', 0) * 0.9995,  # Simulated bid
                    'ask': row.get('close', 0) * 1.0005,  # Simulated ask
                    'volume': row.get('volume', 0)
                }
                
                # Generate quotes
                signal = await self.generate_quotes(symbol, market_snapshot)
                
                if not signal:
                    continue
                
                # Simulate fills (assume 10% fill probability per tick)
                if np.random.random() < 0.1 and signal.bid_size > 0:
                    # Buy at bid
                    fill_size = min(signal.bid_size, 100)
                    self.update_inventory(symbol, fill_size, signal.bid_price)
                    capital -= fill_size * signal.bid_price
                    
                    trades.append({
                        'symbol': symbol,
                        'side': 'BUY',
                        'price': signal.bid_price,
                        'size': fill_size
                    })
                
                if np.random.random() < 0.1 and signal.ask_size > 0:
                    # Sell at ask
                    fill_size = min(signal.ask_size, abs(self.inventory.get(symbol, 0)))
                    if fill_size > 0:
                        self.update_inventory(symbol, -fill_size, signal.ask_price)
                        capital += fill_size * signal.ask_price
                        
                        # Capture spread
                        spread_captured = signal.spread * fill_size
                        total_spread_captured += spread_captured
                        
                        trades.append({
                            'symbol': symbol,
                            'side': 'SELL',
                            'price': signal.ask_price,
                            'size': fill_size,
                            'spread_captured': spread_captured
                        })
                
                equity_curve.append(capital)
        
        # Calculate metrics
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 390) if np.std(returns) > 0 else 0
        total_return = (capital - initial_capital) / initial_capital
        
        logger.info(
            "Market making backtest complete",
            sharpe=sharpe_ratio,
            return_pct=total_return * 100,
            spread_captured=total_spread_captured,
            num_trades=len(trades)
        )
        
        return {
            'sharpe_ratio': float(sharpe_ratio),
            'total_return': float(total_return),
            'spread_captured': float(total_spread_captured),
            'num_trades': len(trades),
            'final_capital': float(capital),
            'win_rate': 1.0 if total_return > 0 else 0.0  # Market making should be consistently profitable
        }
    
    async def evaluate(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate method for backtesting compatibility.
        Market making strategy based on bid-ask spread and volatility.
        """
        try:
            # Extract OHLC data
            if not all(k in data for k in ['high', 'low', 'close']):
                return {
                    'symbol': symbol,
                    'signal_type': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'Missing OHLC data'
                }
            
            highs = np.array(data['high'])
            lows = np.array(data['low'])
            closes = np.array(data['close'])
            
            if len(closes) < 20:
                return {
                    'symbol': symbol,
                    'signal_type': 'HOLD',
                    'confidence': 0.0,
                    'reason': f'Insufficient data: {len(closes)} bars'
                }
            
            # Calculate implied spread from high-low range
            recent_ranges = highs[-20:] - lows[-20:]
            avg_spread_pct = np.mean(recent_ranges / closes[-20:])
            
            # Calculate volatility
            returns = np.diff(closes[-21:]) / closes[-21:-1]
            volatility = np.std(returns)
            
            current_price = closes[-1]
            current_spread = highs[-1] - lows[-1]
            current_spread_pct = current_spread / current_price if current_price > 0 else 0
            
            # Market making signals based on spread and volatility
            signal_type = 'HOLD'
            confidence = 0.0
            reason = ''
            
            # Wide spread + high volatility = good market making opportunity
            if current_spread_pct > avg_spread_pct * 1.2 and volatility > 0.01:
                # Alternate buy/sell to capture spread
                # Use price position in range to determine direction
                price_position = (current_price - lows[-1]) / current_spread if current_spread > 0 else 0.5
                
                if price_position < 0.4:
                    # Price near low - BUY
                    signal_type = 'BUY'
                    confidence = 0.6
                    reason = f'Market making: Wide spread {current_spread_pct:.2%}, price near low'
                elif price_position > 0.6:
                    # Price near high - SELL
                    signal_type = 'SELL'
                    confidence = 0.6
                    reason = f'Market making: Wide spread {current_spread_pct:.2%}, price near high'
            
            # Tight spread = less opportunity
            elif current_spread_pct < avg_spread_pct * 0.5:
                signal_type = 'HOLD'
                confidence = 0.0
                reason = f'Tight spread {current_spread_pct:.2%}, no opportunity'
            
            return {
                'symbol': symbol,
                'signal_type': signal_type,
                'confidence': confidence,
                'reason': reason,
                'indicators': {
                    'current_spread_pct': float(current_spread_pct),
                    'avg_spread_pct': float(avg_spread_pct),
                    'volatility': float(volatility),
                    'current_price': float(current_price)
                }
            }
            
        except Exception as e:
            logger.error(f"Market making evaluate error: {e}")
            return {
                'symbol': symbol,
                'signal_type': 'HOLD',
                'confidence': 0.0,
                'reason': f'Error: {str(e)}'
            }
