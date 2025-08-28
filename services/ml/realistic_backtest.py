#!/usr/bin/env python3
"""
Realistic Backtesting Engine - Simulates real market conditions including slippage,
market impact, and transaction costs
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """Market volatility conditions"""
    LOW_VOL = "low_volatility"
    NORMAL = "normal"
    HIGH_VOL = "high_volatility"
    EXTREME = "extreme_volatility"


@dataclass
class SlippageModel:
    """Realistic slippage and market impact model"""
    base_bps: float = 5.0           # Base slippage in basis points
    size_impact_bps: float = 0.1    # BPS per $100k traded
    volatility_multiplier: float = 1.5  # Multiplier for volatility
    urgency_multiplier: float = 1.2    # Multiplier for urgent orders
    
    # Time of day factors (US market hours)
    time_factors = {
        "open": 2.0,      # 9:30-10:00 AM - Higher slippage
        "mid_morning": 1.2,  # 10:00-11:30 AM
        "lunch": 1.5,     # 11:30-1:30 PM - Lower liquidity
        "afternoon": 1.0,  # 1:30-3:30 PM - Normal
        "close": 2.5      # 3:30-4:00 PM - Higher slippage
    }


@dataclass
class TransactionCosts:
    """Realistic transaction cost model"""
    commission_per_share: float = 0.005  # $0.005 per share
    commission_minimum: float = 1.0      # $1 minimum
    sec_fee_rate: float = 0.0000278      # SEC fee for sells
    taf_fee_rate: float = 0.000119       # FINRA TAF
    ecn_rebate: float = -0.002            # ECN rebate for adding liquidity


class RealisticBacktest:
    """
    Realistic backtesting engine that simulates actual market conditions
    """
    
    def __init__(self):
        self.slippage_model = SlippageModel()
        self.transaction_costs = TransactionCosts()
        self.order_history = []
        self.execution_history = []
        
    def calculate_slippage(self, 
                          order: Dict[str, Any], 
                          market_state: Dict[str, Any]) -> float:
        """
        Calculate realistic slippage based on order and market conditions
        
        Args:
            order: Order details (size, urgency, type)
            market_state: Current market conditions (volatility, spread, volume)
        
        Returns:
            Slippage in price terms
        """
        base_price = order.get('price', 0)
        order_value = order.get('quantity', 0) * base_price
        side = order.get('side', 'buy')
        
        # Base slippage
        slippage_bps = self.slippage_model.base_bps
        
        # Size impact - larger orders move the market more
        size_impact = (order_value / 100000) * self.slippage_model.size_impact_bps
        slippage_bps += size_impact
        
        # Volatility impact
        volatility = market_state.get('volatility', 0.01)
        vol_factor = 1.0 + (volatility - 0.01) * self.slippage_model.volatility_multiplier
        slippage_bps *= vol_factor
        
        # Time of day impact
        time_period = self._get_time_period(market_state.get('timestamp'))
        time_factor = self.slippage_model.time_factors.get(time_period, 1.0)
        slippage_bps *= time_factor
        
        # Spread impact
        spread_bps = market_state.get('spread_bps', 5)
        slippage_bps += spread_bps / 2  # Cross half the spread
        
        # Urgency impact (market orders have more slippage)
        if order.get('order_type') == 'market':
            slippage_bps *= self.slippage_model.urgency_multiplier
        
        # Volume impact - thin markets have more slippage
        avg_volume = market_state.get('avg_volume', 1000000)
        order_volume = order.get('quantity', 0)
        if order_volume > avg_volume * 0.01:  # Order is >1% of avg volume
            volume_impact = (order_volume / avg_volume) * 100  # BPS
            slippage_bps += volume_impact
        
        # Convert to price impact
        slippage_price = base_price * (slippage_bps / 10000)
        
        # Apply directionally (buy orders get worse prices, sell orders get worse prices)
        if side == 'buy':
            return slippage_price  # Pay more
        else:
            return -slippage_price  # Receive less
    
    def calculate_transaction_costs(self, 
                                   order: Dict[str, Any],
                                   execution_price: float) -> float:
        """
        Calculate realistic transaction costs including commissions and fees
        """
        quantity = order.get('quantity', 0)
        side = order.get('side', 'buy')
        order_value = quantity * execution_price
        
        # Commission
        commission = max(
            quantity * self.transaction_costs.commission_per_share,
            self.transaction_costs.commission_minimum
        )
        
        # SEC fee (sells only)
        sec_fee = 0
        if side == 'sell':
            sec_fee = order_value * self.transaction_costs.sec_fee_rate
        
        # FINRA TAF
        taf_fee = quantity * self.transaction_costs.taf_fee_rate
        
        # ECN rebate for limit orders that add liquidity
        ecn_fee = 0
        if order.get('order_type') == 'limit':
            ecn_fee = quantity * self.transaction_costs.ecn_rebate  # Negative = rebate
        
        total_costs = commission + sec_fee + taf_fee + ecn_fee
        
        return total_costs
    
    def simulate_order_execution(self,
                                order: Dict[str, Any],
                                market_data: pd.DataFrame,
                                timestamp: datetime) -> Dict[str, Any]:
        """
        Simulate realistic order execution with partial fills and time priority
        """
        symbol = order['symbol']
        quantity = order['quantity']
        order_type = order.get('order_type', 'market')
        
        # Get market state at order time
        market_state = self._get_market_state(market_data, timestamp)
        
        execution = {
            'order_id': order['order_id'],
            'symbol': symbol,
            'quantity_ordered': quantity,
            'quantity_filled': 0,
            'average_price': 0,
            'slippage': 0,
            'transaction_costs': 0,
            'execution_time': timestamp,
            'fills': []
        }
        
        # Simulate fills (large orders might get partial fills)
        remaining = quantity
        fills = []
        current_price = market_state['price']
        
        while remaining > 0:
            # Determine fill size based on available liquidity
            available_liquidity = market_state.get('bid_size' if order['side'] == 'sell' else 'ask_size', quantity)
            fill_size = min(remaining, available_liquidity)
            
            # Calculate slippage for this fill
            fill_order = {**order, 'quantity': fill_size}
            slippage = self.calculate_slippage(fill_order, market_state)
            
            # Calculate execution price
            if order['side'] == 'buy':
                exec_price = current_price + slippage
            else:
                exec_price = current_price + slippage  # Slippage is negative for sells
            
            # Record fill
            fills.append({
                'quantity': fill_size,
                'price': exec_price,
                'timestamp': timestamp + timedelta(seconds=len(fills))
            })
            
            remaining -= fill_size
            
            # Update market impact for next fill
            market_state['price'] = exec_price
            
            # Break if market order or fully filled
            if order_type == 'market' or remaining == 0:
                break
            
            # For limit orders, check if price still acceptable
            if order_type == 'limit':
                limit_price = order.get('limit_price')
                if order['side'] == 'buy' and exec_price > limit_price:
                    break  # Can't fill rest at limit
                elif order['side'] == 'sell' and exec_price < limit_price:
                    break
        
        # Calculate results
        if fills:
            total_quantity = sum(f['quantity'] for f in fills)
            weighted_price = sum(f['quantity'] * f['price'] for f in fills) / total_quantity
            
            execution['quantity_filled'] = total_quantity
            execution['average_price'] = weighted_price
            execution['fills'] = fills
            
            # Calculate total slippage
            base_price = market_state.get('mid_price', current_price)
            if order['side'] == 'buy':
                execution['slippage'] = weighted_price - base_price
            else:
                execution['slippage'] = base_price - weighted_price
            
            # Calculate transaction costs
            execution['transaction_costs'] = self.calculate_transaction_costs(
                {'quantity': total_quantity, 'side': order['side']},
                weighted_price
            )
        
        return execution
    
    def run_backtest(self,
                    strategy_signals: List[Dict],
                    market_data: pd.DataFrame,
                    initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Run realistic backtest with proper order execution simulation
        """
        portfolio = {
            'cash': initial_capital,
            'positions': {},
            'total_value': initial_capital,
            'trades': [],
            'equity_curve': []
        }
        
        for signal in strategy_signals:
            timestamp = signal['timestamp']
            
            # Create order from signal
            order = self._signal_to_order(signal, portfolio)
            
            if order:
                # Simulate execution
                execution = self.simulate_order_execution(order, market_data, timestamp)
                
                # Update portfolio
                self._update_portfolio(portfolio, execution, market_data, timestamp)
                
                # Record trade
                portfolio['trades'].append(execution)
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(portfolio, market_data, timestamp)
            portfolio['total_value'] = portfolio_value
            portfolio['equity_curve'].append({
                'timestamp': timestamp,
                'value': portfolio_value
            })
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(portfolio)
        
        return {
            'portfolio': portfolio,
            'metrics': metrics,
            'execution_analysis': self._analyze_executions(portfolio['trades'])
        }
    
    def _get_market_state(self, market_data: pd.DataFrame, timestamp: datetime) -> Dict[str, Any]:
        """Extract market state at given timestamp"""
        # Find nearest market data point
        idx = market_data.index.get_loc(timestamp, method='nearest')
        row = market_data.iloc[idx]
        
        return {
            'timestamp': timestamp,
            'price': row.get('close', row.get('price', 0)),
            'mid_price': (row.get('bid', 0) + row.get('ask', 0)) / 2 if 'bid' in row else row.get('close', 0),
            'bid': row.get('bid', 0),
            'ask': row.get('ask', 0),
            'spread_bps': ((row.get('ask', 0) - row.get('bid', 0)) / row.get('bid', 1)) * 10000 if 'bid' in row else 5,
            'volume': row.get('volume', 0),
            'avg_volume': market_data['volume'].rolling(20).mean().iloc[idx] if 'volume' in market_data else 1000000,
            'volatility': market_data['close'].pct_change().rolling(20).std().iloc[idx] if 'close' in market_data else 0.01,
            'bid_size': row.get('bid_size', 1000),
            'ask_size': row.get('ask_size', 1000)
        }
    
    def _get_time_period(self, timestamp: Optional[datetime]) -> str:
        """Determine market period for time-based slippage"""
        if not timestamp:
            return "afternoon"
        
        hour = timestamp.hour
        minute = timestamp.minute
        
        if hour == 9 and minute >= 30:
            return "open"
        elif hour == 10 or (hour == 11 and minute < 30):
            return "mid_morning"
        elif (hour == 11 and minute >= 30) or hour == 12 or (hour == 13 and minute < 30):
            return "lunch"
        elif hour >= 13 and hour < 15 or (hour == 15 and minute < 30):
            return "afternoon"
        else:
            return "close"
    
    def _signal_to_order(self, signal: Dict, portfolio: Dict) -> Optional[Dict]:
        """Convert trading signal to order"""
        # Check if we have enough capital
        available_cash = portfolio['cash']
        signal_value = signal.get('quantity', 0) * signal.get('price', 0)
        
        if signal['side'] == 'buy' and signal_value > available_cash:
            # Reduce order size to fit available capital
            max_quantity = int(available_cash / signal['price'] * 0.95)  # Leave some buffer
            if max_quantity <= 0:
                return None
            signal['quantity'] = max_quantity
        
        return {
            'order_id': f"ORD_{datetime.now().timestamp()}",
            'symbol': signal['symbol'],
            'side': signal['side'],
            'quantity': signal['quantity'],
            'price': signal['price'],
            'order_type': signal.get('order_type', 'market'),
            'limit_price': signal.get('limit_price')
        }
    
    def _update_portfolio(self, portfolio: Dict, execution: Dict, market_data: pd.DataFrame, timestamp: datetime):
        """Update portfolio after execution"""
        if execution['quantity_filled'] > 0:
            symbol = execution['symbol']
            side = execution.get('side', 'buy')
            quantity = execution['quantity_filled']
            price = execution['average_price']
            costs = execution['transaction_costs']
            
            if side == 'buy':
                # Deduct cash
                portfolio['cash'] -= (quantity * price + costs)
                # Add position
                if symbol not in portfolio['positions']:
                    portfolio['positions'][symbol] = {'quantity': 0, 'avg_price': 0}
                
                pos = portfolio['positions'][symbol]
                new_quantity = pos['quantity'] + quantity
                pos['avg_price'] = ((pos['quantity'] * pos['avg_price']) + (quantity * price)) / new_quantity
                pos['quantity'] = new_quantity
                
            else:  # sell
                # Add cash
                portfolio['cash'] += (quantity * price - costs)
                # Reduce position
                if symbol in portfolio['positions']:
                    portfolio['positions'][symbol]['quantity'] -= quantity
                    if portfolio['positions'][symbol]['quantity'] <= 0:
                        del portfolio['positions'][symbol]
    
    def _calculate_portfolio_value(self, portfolio: Dict, market_data: pd.DataFrame, timestamp: datetime) -> float:
        """Calculate total portfolio value"""
        value = portfolio['cash']
        
        for symbol, position in portfolio['positions'].items():
            # Get current price
            market_state = self._get_market_state(market_data, timestamp)
            current_price = market_state['price']
            value += position['quantity'] * current_price
        
        return value
    
    def _calculate_performance_metrics(self, portfolio: Dict) -> Dict[str, float]:
        """Calculate performance metrics"""
        equity_curve = pd.DataFrame(portfolio['equity_curve'])
        if equity_curve.empty:
            return {}
        
        equity_curve['returns'] = equity_curve['value'].pct_change()
        
        # Calculate metrics
        total_return = (equity_curve['value'].iloc[-1] / equity_curve['value'].iloc[0]) - 1
        sharpe_ratio = equity_curve['returns'].mean() / equity_curve['returns'].std() * np.sqrt(252) if equity_curve['returns'].std() > 0 else 0
        
        # Maximum drawdown
        rolling_max = equity_curve['value'].expanding().max()
        drawdown = (equity_curve['value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        trades = portfolio['trades']
        profitable_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        win_rate = profitable_trades / len(trades) if trades else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'total_slippage': sum(t.get('slippage', 0) * t.get('quantity_filled', 0) for t in trades),
            'total_costs': sum(t.get('transaction_costs', 0) for t in trades)
        }
    
    def _analyze_executions(self, trades: List[Dict]) -> Dict[str, Any]:
        """Analyze execution quality"""
        if not trades:
            return {}
        
        return {
            'avg_slippage_bps': np.mean([t.get('slippage', 0) / t.get('average_price', 1) * 10000 for t in trades]),
            'total_slippage_cost': sum(t.get('slippage', 0) * t.get('quantity_filled', 0) for t in trades),
            'avg_fill_rate': np.mean([t.get('quantity_filled', 0) / t.get('quantity_ordered', 1) for t in trades]),
            'total_transaction_costs': sum(t.get('transaction_costs', 0) for t in trades)
        }