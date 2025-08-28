#!/usr/bin/env python3
"""
Advanced Backtesting Engine with Walk-Forward Analysis
Implements professional-grade backtesting with realistic market conditions,
slippage modeling, and comprehensive performance analytics.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import vectorbt as vbt
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import quantstats as qs
from empyrical import (
    sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio,
    annual_return, max_drawdown, value_at_risk, conditional_value_at_risk
)

from trading_common import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    position_size: float = 0.02  # 2% per trade
    max_positions: int = 10
    
    # Risk management
    stop_loss: float = 0.02  # 2% stop loss
    take_profit: float = 0.05  # 5% take profit
    max_drawdown_limit: float = 0.15  # 15% max drawdown
    
    # Market simulation
    commission: float = 0.001  # 0.1% commission
    slippage: float = 0.0005  # 0.05% slippage
    bid_ask_spread: float = 0.0002  # 0.02% spread
    
    # Walk-forward parameters
    in_sample_periods: int = 252  # 1 year
    out_sample_periods: int = 63  # 3 months
    reoptimization_frequency: int = 21  # Reoptimize monthly
    
    # Monte Carlo simulation
    n_simulations: int = 1000
    confidence_level: float = 0.95


@dataclass
class TradeSignal:
    """Trading signal with confidence and metadata."""
    timestamp: datetime
    symbol: str
    action: str  # 'long', 'short', 'close'
    confidence: float  # 0-1
    predicted_return: float
    predicted_volatility: float
    stop_loss: float
    take_profit: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Comprehensive backtest results."""
    # Performance metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    
    # Risk metrics
    value_at_risk: float
    conditional_var: float
    downside_deviation: float
    ulcer_index: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Time metrics
    avg_holding_period: float
    longest_drawdown_days: int
    recovery_factor: float
    
    # Monte Carlo results
    confidence_interval: Tuple[float, float]
    probability_of_profit: float
    expected_shortfall: float
    
    # Detailed results
    equity_curve: pd.Series
    trades: pd.DataFrame
    daily_returns: pd.Series
    monthly_returns: pd.Series


class AdvancedBacktestingEngine:
    """
    Professional backtesting engine with realistic market simulation.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.results_cache = {}
        self.optimization_history = []
        
    async def backtest_strategy(
        self,
        strategy: Callable,
        data: pd.DataFrame,
        signals: List[TradeSignal],
        walk_forward: bool = True
    ) -> BacktestResult:
        """
        Run comprehensive backtest with walk-forward analysis.
        """
        logger.info("Starting advanced backtest")
        
        if walk_forward:
            results = await self._walk_forward_backtest(strategy, data, signals)
        else:
            results = await self._single_backtest(strategy, data, signals)
        
        # Run Monte Carlo simulation
        monte_carlo_results = self._monte_carlo_simulation(results)
        
        # Calculate comprehensive metrics
        final_results = self._calculate_metrics(results, monte_carlo_results)
        
        # Generate report
        self._generate_report(final_results)
        
        return final_results
    
    async def _walk_forward_backtest(
        self,
        strategy: Callable,
        data: pd.DataFrame,
        signals: List[TradeSignal]
    ) -> Dict[str, Any]:
        """
        Walk-forward analysis for robust testing.
        """
        results = []
        
        # Split data into windows
        windows = self._create_walk_forward_windows(data)
        
        for i, (train_data, test_data) in enumerate(windows):
            logger.info(f"Walk-forward window {i+1}/{len(windows)}")
            
            # Optimize on training data
            optimal_params = await self._optimize_parameters(
                strategy, train_data
            )
            
            # Test on out-of-sample data
            window_results = await self._single_backtest(
                strategy, test_data, signals, optimal_params
            )
            
            results.append(window_results)
            
            # Store optimization history
            self.optimization_history.append({
                'window': i,
                'params': optimal_params,
                'in_sample_sharpe': window_results['in_sample_sharpe'],
                'out_sample_sharpe': window_results['out_sample_sharpe']
            })
        
        # Combine results
        return self._combine_walk_forward_results(results)
    
    async def _single_backtest(
        self,
        strategy: Callable,
        data: pd.DataFrame,
        signals: List[TradeSignal],
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Single backtest run with realistic execution.
        """
        # Initialize portfolio
        portfolio = self._initialize_portfolio()
        
        # Convert signals to DataFrame
        signals_df = self._signals_to_dataframe(signals)
        
        # Merge with price data
        data = data.merge(signals_df, left_index=True, right_index=True, how='left')
        
        # Run backtest with realistic execution
        results = self._execute_backtest(data, portfolio, strategy, params)
        
        return results
    
    def _execute_backtest(
        self,
        data: pd.DataFrame,
        portfolio: Dict,
        strategy: Callable,
        params: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Execute backtest with realistic market conditions.
        """
        equity_curve = [self.config.initial_capital]
        trades = []
        positions = {}
        
        for i in range(1, len(data)):
            current_time = data.index[i]
            current_price = data.iloc[i]
            
            # Update existing positions
            for symbol, position in list(positions.items()):
                # Check stop loss and take profit
                if self._check_exit_conditions(position, current_price):
                    trade = self._close_position(
                        position, current_price, current_time
                    )
                    trades.append(trade)
                    del positions[symbol]
            
            # Check for new signals
            if hasattr(data.iloc[i], 'signal') and data.iloc[i].signal:
                signal = data.iloc[i].signal
                
                # Risk management checks
                if self._passes_risk_checks(portfolio, positions):
                    # Calculate position size with Kelly Criterion
                    position_size = self._calculate_position_size(
                        signal, portfolio['equity']
                    )
                    
                    # Execute trade with slippage and commission
                    entry_price = self._calculate_entry_price(
                        current_price.close, signal.action
                    )
                    
                    position = self._open_position(
                        signal, entry_price, position_size, current_time
                    )
                    positions[signal.symbol] = position
            
            # Update portfolio equity
            portfolio['equity'] = self._calculate_equity(
                self.config.initial_capital, positions, current_price
            )
            equity_curve.append(portfolio['equity'])
            
            # Check for margin calls or drawdown limits
            if self._check_portfolio_limits(portfolio):
                logger.warning("Portfolio limits exceeded, closing all positions")
                for position in positions.values():
                    trade = self._close_position(
                        position, current_price, current_time
                    )
                    trades.append(trade)
                positions.clear()
        
        return {
            'equity_curve': pd.Series(equity_curve, index=data.index),
            'trades': pd.DataFrame(trades),
            'final_equity': equity_curve[-1],
            'total_return': (equity_curve[-1] - self.config.initial_capital) / 
                          self.config.initial_capital
        }
    
    def _calculate_position_size(
        self,
        signal: TradeSignal,
        portfolio_equity: float
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        """
        # Kelly fraction
        win_prob = signal.confidence
        win_loss_ratio = signal.predicted_return / self.config.stop_loss
        
        kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Adjust for portfolio constraints
        max_position = portfolio_equity * self.config.position_size
        position_size = portfolio_equity * kelly_fraction
        
        return min(position_size, max_position)
    
    def _calculate_entry_price(
        self,
        price: float,
        action: str
    ) -> float:
        """
        Calculate realistic entry price with slippage and spread.
        """
        slippage = price * self.config.slippage
        spread = price * self.config.bid_ask_spread
        
        if action == 'long':
            # Buy at ask + slippage
            entry_price = price + spread/2 + slippage
        else:
            # Sell at bid - slippage
            entry_price = price - spread/2 - slippage
        
        return entry_price
    
    def _monte_carlo_simulation(
        self,
        backtest_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Monte Carlo simulation for confidence intervals.
        """
        returns = backtest_results['equity_curve'].pct_change().dropna()
        
        simulated_results = []
        for _ in range(self.config.n_simulations):
            # Bootstrap returns
            simulated_returns = np.random.choice(
                returns, size=len(returns), replace=True
            )
            
            # Calculate equity curve
            simulated_equity = self.config.initial_capital * \
                              (1 + simulated_returns).cumprod()
            
            final_return = (simulated_equity[-1] - self.config.initial_capital) / \
                          self.config.initial_capital
            simulated_results.append(final_return)
        
        # Calculate confidence intervals
        simulated_results = np.array(simulated_results)
        lower_ci = np.percentile(simulated_results, (1 - self.config.confidence_level) * 50)
        upper_ci = np.percentile(simulated_results, 
                               100 - (1 - self.config.confidence_level) * 50)
        
        return {
            'confidence_interval': (lower_ci, upper_ci),
            'probability_of_profit': np.mean(simulated_results > 0),
            'expected_shortfall': np.mean(
                simulated_results[simulated_results < np.percentile(simulated_results, 5)]
            ),
            'median_return': np.median(simulated_results)
        }
    
    def _calculate_metrics(
        self,
        results: Dict[str, Any],
        monte_carlo: Dict[str, Any]
    ) -> BacktestResult:
        """
        Calculate comprehensive performance metrics.
        """
        equity_curve = results['equity_curve']
        trades = results['trades']
        returns = equity_curve.pct_change().dropna()
        
        # Performance metrics
        total_return = results['total_return']
        annual_return_val = annual_return(returns)
        sharpe = sharpe_ratio(returns)
        sortino = sortino_ratio(returns)
        calmar = calmar_ratio(returns)
        max_dd = max_drawdown(returns)
        
        # Risk metrics
        var = value_at_risk(returns, alpha=0.05)
        cvar = conditional_value_at_risk(returns, alpha=0.05)
        downside_dev = returns[returns < 0].std() * np.sqrt(252)
        
        # Trade statistics
        if len(trades) > 0:
            winning_trades = trades[trades['pnl'] > 0]
            losing_trades = trades[trades['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades)
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
            profit_factor = winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum()) \
                          if len(losing_trades) > 0 else float('inf')
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return_val,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            value_at_risk=var,
            conditional_var=cvar,
            downside_deviation=downside_dev,
            ulcer_index=self._calculate_ulcer_index(equity_curve),
            total_trades=len(trades),
            winning_trades=len(winning_trades) if len(trades) > 0 else 0,
            losing_trades=len(losing_trades) if len(trades) > 0 else 0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=trades['pnl'].max() if len(trades) > 0 else 0,
            largest_loss=trades['pnl'].min() if len(trades) > 0 else 0,
            avg_holding_period=trades['holding_period'].mean() if len(trades) > 0 else 0,
            longest_drawdown_days=self._calculate_max_dd_duration(equity_curve),
            recovery_factor=total_return / abs(max_dd) if max_dd != 0 else 0,
            confidence_interval=monte_carlo['confidence_interval'],
            probability_of_profit=monte_carlo['probability_of_profit'],
            expected_shortfall=monte_carlo['expected_shortfall'],
            equity_curve=equity_curve,
            trades=trades,
            daily_returns=returns,
            monthly_returns=returns.resample('M').apply(lambda x: (1+x).prod()-1)
        )
    
    def _calculate_ulcer_index(self, equity_curve: pd.Series) -> float:
        """Calculate Ulcer Index (measure of downside volatility)."""
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return np.sqrt(np.mean(drawdown ** 2))
    
    def _calculate_max_dd_duration(self, equity_curve: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        rolling_max = equity_curve.expanding().max()
        drawdown_periods = (equity_curve < rolling_max).astype(int)
        
        # Find consecutive drawdown periods
        max_duration = 0
        current_duration = 0
        
        for is_drawdown in drawdown_periods:
            if is_drawdown:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def _generate_report(self, results: BacktestResult):
        """Generate comprehensive backtest report."""
        logger.info("=" * 50)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 50)
        logger.info(f"Total Return: {results.total_return:.2%}")
        logger.info(f"Annual Return: {results.annual_return:.2%}")
        logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        logger.info(f"Sortino Ratio: {results.sortino_ratio:.2f}")
        logger.info(f"Max Drawdown: {results.max_drawdown:.2%}")
        logger.info(f"Win Rate: {results.win_rate:.2%}")
        logger.info(f"Profit Factor: {results.profit_factor:.2f}")
        logger.info(f"95% Confidence Interval: "
                   f"[{results.confidence_interval[0]:.2%}, "
                   f"{results.confidence_interval[1]:.2%}]")
        logger.info(f"Probability of Profit: {results.probability_of_profit:.2%}")
        logger.info("=" * 50)


# Global backtesting engine instance
_backtest_engine: Optional[AdvancedBacktestingEngine] = None


async def get_backtest_engine() -> AdvancedBacktestingEngine:
    """Get or create backtesting engine."""
    global _backtest_engine
    if _backtest_engine is None:
        config = BacktestConfig()
        _backtest_engine = AdvancedBacktestingEngine(config)
    return _backtest_engine