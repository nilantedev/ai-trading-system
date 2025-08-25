#!/usr/bin/env python3
"""
Comprehensive Backtesting Framework for AI Trading System.
Provides realistic backtesting with transaction costs, slippage, and risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
from abc import ABC, abstractmethod

from .models import MarketData, Order, Position, Trade
from .logging import get_logger
from .metrics import get_metrics_registry, track_api_metrics

logger = get_logger(__name__)
metrics = get_metrics_registry()


class OrderType(str, Enum):
    """Order types for backtesting."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(str, Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class BacktestOrder:
    """Order representation for backtesting."""
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    order_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate order ID if not provided."""
        if self.order_id is None:
            import uuid
            self.order_id = str(uuid.uuid4())[:8]


@dataclass
class BacktestTrade:
    """Trade execution record for backtesting."""
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float
    order_id: str
    trade_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate trade ID if not provided."""
        if self.trade_id is None:
            import uuid
            self.trade_id = str(uuid.uuid4())[:8]
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value of trade."""
        return self.quantity * self.price
    
    @property
    def net_proceeds(self) -> float:
        """Calculate net proceeds after costs."""
        gross = self.notional_value
        costs = self.commission + abs(self.slippage * self.quantity)
        return gross - costs if self.side == OrderSide.SELL else -(gross + costs)


@dataclass
class BacktestPosition:
    """Position tracking for backtesting."""
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    market_value: float = 0.0
    cost_basis: float = 0.0
    
    def update_market_value(self, current_price: float):
        """Update market value and unrealized PnL."""
        if self.quantity != 0:
            self.market_value = self.quantity * current_price
            self.unrealized_pnl = self.market_value - self.cost_basis
        else:
            self.market_value = 0.0
            self.unrealized_pnl = 0.0
    
    def add_trade(self, trade: BacktestTrade):
        """Add a trade to the position."""
        old_quantity = self.quantity
        new_quantity = trade.quantity if trade.side == OrderSide.BUY else -trade.quantity
        
        if old_quantity == 0:
            # Opening new position
            self.quantity = new_quantity
            self.average_price = trade.price
            self.cost_basis = abs(new_quantity * trade.price)
        elif (old_quantity > 0 and new_quantity > 0) or (old_quantity < 0 and new_quantity < 0):
            # Adding to existing position
            total_cost = (abs(old_quantity) * self.average_price) + (abs(new_quantity) * trade.price)
            self.quantity += new_quantity
            if self.quantity != 0:
                self.average_price = total_cost / abs(self.quantity)
                self.cost_basis = total_cost
        else:
            # Reducing or closing position
            if abs(new_quantity) >= abs(old_quantity):
                # Closing and possibly reversing position
                closing_quantity = abs(old_quantity)
                closing_pnl = closing_quantity * (trade.price - self.average_price)
                if old_quantity < 0:
                    closing_pnl = -closing_pnl
                
                self.realized_pnl += closing_pnl
                
                # Remaining quantity after closing
                remaining_quantity = abs(new_quantity) - closing_quantity
                if remaining_quantity > 0:
                    self.quantity = remaining_quantity if trade.side == OrderSide.BUY else -remaining_quantity
                    self.average_price = trade.price
                    self.cost_basis = remaining_quantity * trade.price
                else:
                    self.quantity = 0.0
                    self.average_price = 0.0
                    self.cost_basis = 0.0
            else:
                # Partial close
                closing_quantity = abs(new_quantity)
                closing_pnl = closing_quantity * (trade.price - self.average_price)
                if old_quantity < 0:
                    closing_pnl = -closing_pnl
                
                self.realized_pnl += closing_pnl
                self.quantity += new_quantity
                self.cost_basis = abs(self.quantity) * self.average_price


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1% per trade
    slippage_rate: float = 0.0005   # 0.05% slippage
    margin_requirement: float = 1.0  # 100% margin (no leverage)
    max_position_size: float = 0.2   # 20% of portfolio per position
    risk_free_rate: float = 0.02     # 2% annual risk-free rate
    
    # Risk management
    max_daily_loss: float = 0.05     # 5% max daily loss
    max_portfolio_loss: float = 0.20  # 20% max total loss
    
    # Transaction costs
    min_commission: float = 1.0      # Minimum commission per trade
    borrowing_cost_rate: float = 0.03  # 3% annual borrowing cost for shorts


@dataclass
class BacktestMetrics:
    """Comprehensive backtesting performance metrics."""
    # Basic performance
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Drawdown metrics
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    calmar_ratio: float = 0.0
    
    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics
    value_at_risk_95: float = 0.0
    conditional_var_95: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    
    # Additional metrics
    total_fees: float = 0.0
    total_slippage: float = 0.0
    avg_holding_period: float = 0.0
    turnover_rate: float = 0.0


class TradingStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    @abstractmethod
    def initialize(self, backtest_context: 'BacktestContext'):
        """Initialize strategy with backtest context."""
        pass
    
    @abstractmethod
    def generate_signals(
        self, 
        current_data: MarketData, 
        historical_data: List[MarketData],
        context: 'BacktestContext'
    ) -> List[BacktestOrder]:
        """Generate trading signals based on market data."""
        pass
    
    @abstractmethod
    def risk_management(
        self,
        orders: List[BacktestOrder],
        context: 'BacktestContext'
    ) -> List[BacktestOrder]:
        """Apply risk management rules to orders."""
        pass


class BacktestContext:
    """Context object passed to strategies during backtesting."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.current_time: Optional[datetime] = None
        self.portfolio_value: float = config.initial_capital
        self.cash: float = config.initial_capital
        self.positions: Dict[str, BacktestPosition] = {}
        self.pending_orders: List[BacktestOrder] = []
        self.trade_history: List[BacktestTrade] = []
        self.portfolio_history: List[Tuple[datetime, float]] = []
        
        # Performance tracking
        self.daily_returns: List[float] = []
        self.high_water_mark: float = config.initial_capital
        self.current_drawdown: float = 0.0
        self.max_drawdown: float = 0.0
        self.drawdown_start: Optional[datetime] = None
        
    def get_position(self, symbol: str) -> BacktestPosition:
        """Get position for a symbol."""
        if symbol not in self.positions:
            self.positions[symbol] = BacktestPosition(symbol=symbol)
        return self.positions[symbol]
    
    def update_portfolio_value(self, market_data: Dict[str, MarketData]):
        """Update portfolio value based on current market prices."""
        total_market_value = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in market_data and position.quantity != 0:
                current_price = market_data[symbol].close
                position.update_market_value(current_price)
                total_market_value += position.market_value
        
        self.portfolio_value = self.cash + total_market_value
        
        # Update drawdown tracking
        if self.portfolio_value > self.high_water_mark:
            self.high_water_mark = self.portfolio_value
            self.current_drawdown = 0.0
            self.drawdown_start = None
        else:
            self.current_drawdown = (self.high_water_mark - self.portfolio_value) / self.high_water_mark
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
            if self.drawdown_start is None:
                self.drawdown_start = self.current_time
        
        # Record portfolio history
        if self.current_time:
            self.portfolio_history.append((self.current_time, self.portfolio_value))
    
    def can_place_order(self, order: BacktestOrder, current_price: float) -> Tuple[bool, str]:
        """Check if an order can be placed given current constraints."""
        position = self.get_position(order.symbol)
        
        # Calculate required capital
        if order.side == OrderSide.BUY:
            required_capital = order.quantity * current_price * self.config.margin_requirement
            if required_capital > self.cash:
                return False, "Insufficient cash"
        
        # Check position size limits
        if order.side == OrderSide.BUY:
            new_position_value = (position.quantity + order.quantity) * current_price
        else:
            new_position_value = (position.quantity - order.quantity) * current_price
        
        position_weight = abs(new_position_value) / self.portfolio_value
        if position_weight > self.config.max_position_size:
            return False, f"Position size limit exceeded: {position_weight:.2%}"
        
        # Check daily loss limit
        if self.current_drawdown > self.config.max_daily_loss:
            return False, "Daily loss limit exceeded"
        
        return True, "OK"


class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(self, config: BacktestConfig = None):
        """Initialize backtesting engine."""
        self.config = config or BacktestConfig()
        self.strategies: List[TradingStrategy] = []
        self.market_data: Dict[str, List[MarketData]] = {}
        
    def add_strategy(self, strategy: TradingStrategy):
        """Add a trading strategy to the backtest."""
        self.strategies.append(strategy)
    
    def add_market_data(self, symbol: str, data: List[MarketData]):
        """Add market data for a symbol."""
        self.market_data[symbol] = sorted(data, key=lambda x: x.timestamp)
    
    @track_api_metrics("backtesting")
    async def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        benchmark_symbol: Optional[str] = None
    ) -> Tuple[BacktestMetrics, BacktestContext]:
        """Run the complete backtest."""
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Initialize context
        context = BacktestContext(self.config)
        
        # Initialize strategies
        for strategy in self.strategies:
            strategy.initialize(context)
        
        # Get all unique timestamps
        all_timestamps = set()
        for data_list in self.market_data.values():
            for md in data_list:
                if start_date <= md.timestamp <= end_date:
                    all_timestamps.add(md.timestamp)
        
        sorted_timestamps = sorted(all_timestamps)
        
        if not sorted_timestamps:
            raise ValueError("No market data found in the specified date range")
        
        logger.info(f"Processing {len(sorted_timestamps)} time steps")
        
        # Main backtest loop
        for i, timestamp in enumerate(sorted_timestamps):
            context.current_time = timestamp
            
            # Get current market data
            current_market_data = {}
            historical_market_data = {}
            
            for symbol, data_list in self.market_data.items():
                # Current data
                current_data = [md for md in data_list if md.timestamp == timestamp]
                if current_data:
                    current_market_data[symbol] = current_data[0]
                
                # Historical data up to current point
                historical_data = [md for md in data_list if md.timestamp <= timestamp]
                historical_market_data[symbol] = historical_data
            
            # Update portfolio value
            context.update_portfolio_value(current_market_data)
            
            # Process pending orders (fills, cancellations)
            await self._process_pending_orders(context, current_market_data)
            
            # Generate new signals from strategies
            all_orders = []
            for strategy in self.strategies:
                for symbol, current_data in current_market_data.items():
                    historical_data = historical_market_data.get(symbol, [])
                    if len(historical_data) >= 2:  # Need at least some history
                        try:
                            orders = strategy.generate_signals(current_data, historical_data, context)
                            if orders:
                                # Apply risk management
                                filtered_orders = strategy.risk_management(orders, context)
                                all_orders.extend(filtered_orders)
                        except Exception as e:
                            logger.warning(f"Strategy error at {timestamp}: {e}")
            
            # Execute orders
            for order in all_orders:
                if order.symbol in current_market_data:
                    current_price = current_market_data[order.symbol].close
                    can_place, reason = context.can_place_order(order, current_price)
                    
                    if can_place:
                        await self._execute_order(context, order, current_market_data[order.symbol])
                    else:
                        logger.debug(f"Order rejected: {reason}")
            
            # Log progress
            if i % 1000 == 0 or i == len(sorted_timestamps) - 1:
                logger.info(f"Progress: {i+1}/{len(sorted_timestamps)} "
                          f"({(i+1)/len(sorted_timestamps)*100:.1f}%) "
                          f"Portfolio: ${context.portfolio_value:,.2f}")
        
        # Calculate final metrics
        metrics = await self._calculate_metrics(context, benchmark_symbol)
        
        logger.info(f"Backtest completed. Total return: {metrics.total_return:.2%}, "
                   f"Sharpe ratio: {metrics.sharpe_ratio:.2f}, "
                   f"Max drawdown: {metrics.max_drawdown:.2%}")
        
        return metrics, context
    
    async def _process_pending_orders(self, context: BacktestContext, market_data: Dict[str, MarketData]):
        """Process pending limit and stop orders."""
        filled_orders = []
        
        for order in context.pending_orders:
            if order.symbol not in market_data:
                continue
            
            current_md = market_data[order.symbol]
            should_fill = False
            fill_price = order.price
            
            # Check if order should be filled
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and current_md.low <= order.price:
                    should_fill = True
                    fill_price = min(order.price, current_md.open)
                elif order.side == OrderSide.SELL and current_md.high >= order.price:
                    should_fill = True
                    fill_price = max(order.price, current_md.open)
            
            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and current_md.high >= order.stop_price:
                    should_fill = True
                    fill_price = max(order.stop_price, current_md.open)
                elif order.side == OrderSide.SELL and current_md.low <= order.stop_price:
                    should_fill = True
                    fill_price = min(order.stop_price, current_md.open)
            
            if should_fill:
                # Execute the order
                trade = await self._create_trade(order, fill_price, current_md.timestamp)
                context.trade_history.append(trade)
                
                # Update position
                position = context.get_position(order.symbol)
                position.add_trade(trade)
                
                # Update cash
                context.cash += trade.net_proceeds
                
                filled_orders.append(order)
        
        # Remove filled orders from pending
        for order in filled_orders:
            context.pending_orders.remove(order)
    
    async def _execute_order(self, context: BacktestContext, order: BacktestOrder, market_data: MarketData):
        """Execute an order immediately or add to pending."""
        if order.order_type == OrderType.MARKET:
            # Execute immediately at current price
            fill_price = market_data.close
            trade = await self._create_trade(order, fill_price, market_data.timestamp)
            context.trade_history.append(trade)
            
            # Update position
            position = context.get_position(order.symbol)
            position.add_trade(trade)
            
            # Update cash
            context.cash += trade.net_proceeds
            
        else:
            # Add to pending orders
            context.pending_orders.append(order)
    
    async def _create_trade(self, order: BacktestOrder, fill_price: float, timestamp: datetime) -> BacktestTrade:
        """Create a trade from an executed order."""
        # Calculate commission
        commission = max(
            order.quantity * fill_price * self.config.commission_rate,
            self.config.min_commission
        )
        
        # Calculate slippage (adverse price movement)
        slippage = order.quantity * fill_price * self.config.slippage_rate
        if order.side == OrderSide.BUY:
            fill_price += slippage / order.quantity  # Higher price for buys
        else:
            fill_price -= slippage / order.quantity  # Lower price for sells
        
        return BacktestTrade(
            timestamp=timestamp,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            slippage=slippage,
            order_id=order.order_id
        )
    
    async def _calculate_metrics(
        self, 
        context: BacktestContext, 
        benchmark_symbol: Optional[str] = None
    ) -> BacktestMetrics:
        """Calculate comprehensive performance metrics."""
        if not context.portfolio_history:
            return BacktestMetrics()
        
        # Convert portfolio history to returns
        portfolio_values = [pv for _, pv in context.portfolio_history]
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Basic performance metrics
        total_return = (portfolio_values[-1] - self.config.initial_capital) / self.config.initial_capital
        
        # Annualized metrics
        days = (context.portfolio_history[-1][0] - context.portfolio_history[0][0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
        
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
        
        # Risk-adjusted metrics
        excess_returns = returns - (self.config.risk_free_rate / 252)
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 1 else 0.0
        sortino_ratio = np.mean(excess_returns) * np.sqrt(252) / downside_deviation if downside_deviation > 0 else 0.0
        
        # Drawdown metrics
        calmar_ratio = annualized_return / context.max_drawdown if context.max_drawdown > 0 else 0.0
        
        # Trading metrics
        trades = context.trade_history
        total_trades = len(trades)
        
        if total_trades > 0:
            trade_pnl = []
            for trade in trades:
                # Simplified PnL calculation for trade metrics
                position = context.get_position(trade.symbol)
                pnl = trade.net_proceeds if trade.side == OrderSide.SELL else -trade.net_proceeds
                trade_pnl.append(pnl)
            
            winning_trades = len([pnl for pnl in trade_pnl if pnl > 0])
            losing_trades = len([pnl for pnl in trade_pnl if pnl < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            wins = [pnl for pnl in trade_pnl if pnl > 0]
            losses = [pnl for pnl in trade_pnl if pnl < 0]
            
            avg_win = np.mean(wins) if wins else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
            
            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        else:
            winning_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = profit_factor = 0.0
        
        # Risk metrics
        value_at_risk_95 = np.percentile(returns, 5) if len(returns) > 0 else 0.0
        conditional_var_95 = np.mean(returns[returns <= value_at_risk_95]) if len(returns) > 0 else 0.0
        
        # Cost analysis
        total_fees = sum(trade.commission for trade in trades)
        total_slippage = sum(abs(trade.slippage) for trade in trades)
        
        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=context.max_drawdown,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            value_at_risk_95=value_at_risk_95,
            conditional_var_95=conditional_var_95,
            total_fees=total_fees,
            total_slippage=total_slippage
        )