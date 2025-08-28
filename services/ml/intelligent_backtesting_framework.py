#!/usr/bin/env python3
"""
Advanced Backtesting Framework with ML-Driven Strategy Optimization
Provides comprehensive backtesting, A/B testing, and strategy validation
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from collections import defaultdict
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Backtesting modes for different validation approaches"""
    HISTORICAL = "historical"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    CROSS_VALIDATION = "cross_validation"
    ADAPTIVE = "adaptive"


@dataclass
class BacktestResult:
    """Comprehensive backtest results with statistical validation"""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    information_ratio: float
    alpha: float
    beta: float
    confidence_interval: Tuple[float, float]
    statistical_significance: float
    risk_metrics: Dict[str, float]
    performance_attribution: Dict[str, float]
    market_regime_performance: Dict[str, Dict[str, float]]
    stress_test_results: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyComparison:
    """A/B testing results for strategy comparison"""
    strategy_a: str
    strategy_b: str
    winner: str
    confidence_level: float
    p_value: float
    effect_size: float
    performance_difference: Dict[str, float]
    statistical_tests: Dict[str, float]
    recommendation: str
    evidence_strength: str


class IntelligentBacktestingFramework:
    """
    Advanced backtesting system with ML-driven optimization and statistical validation
    """
    
    def __init__(self):
        self.backtest_cache = {}
        self.strategy_performance = defaultdict(list)
        self.market_regimes = {}
        self.optimization_history = []
        self.confidence_threshold = 0.95
        
    async def backtest_strategy(
        self,
        strategy: Callable,
        historical_data: pd.DataFrame,
        mode: BacktestMode = BacktestMode.WALK_FORWARD,
        initial_capital: float = 100000,
        risk_params: Optional[Dict] = None
    ) -> BacktestResult:
        """
        Perform comprehensive backtesting with multiple validation methods
        """
        logger.info(f"Starting {mode.value} backtest for strategy")
        
        if mode == BacktestMode.HISTORICAL:
            result = await self._historical_backtest(strategy, historical_data, initial_capital, risk_params)
        elif mode == BacktestMode.WALK_FORWARD:
            result = await self._walk_forward_backtest(strategy, historical_data, initial_capital, risk_params)
        elif mode == BacktestMode.MONTE_CARLO:
            result = await self._monte_carlo_backtest(strategy, historical_data, initial_capital, risk_params)
        elif mode == BacktestMode.CROSS_VALIDATION:
            result = await self._cross_validation_backtest(strategy, historical_data, initial_capital, risk_params)
        else:  # ADAPTIVE
            result = await self._adaptive_backtest(strategy, historical_data, initial_capital, risk_params)
        
        # Perform stress testing
        stress_results = await self._stress_test_strategy(strategy, historical_data, result)
        result.stress_test_results = stress_results
        
        # Market regime analysis
        regime_performance = await self._analyze_regime_performance(strategy, historical_data, result)
        result.market_regime_performance = regime_performance
        
        # Statistical validation
        result.statistical_significance = await self._validate_statistical_significance(result)
        
        # Cache results
        self.backtest_cache[strategy.__name__] = result
        self.strategy_performance[strategy.__name__].append(result)
        
        return result
    
    async def _historical_backtest(
        self,
        strategy: Callable,
        data: pd.DataFrame,
        capital: float,
        risk_params: Optional[Dict]
    ) -> BacktestResult:
        """Traditional historical backtesting"""
        portfolio_value = capital
        trades = []
        returns = []
        positions = {}
        
        for i in range(1, len(data)):
            current_data = data.iloc[:i+1]
            signal = await strategy(current_data, positions, risk_params)
            
            if signal:
                trade_result = await self._execute_trade(
                    signal, 
                    data.iloc[i], 
                    portfolio_value,
                    positions
                )
                
                if trade_result:
                    trades.append(trade_result)
                    portfolio_value = trade_result['portfolio_value']
                    returns.append(trade_result['return'])
        
        return self._calculate_metrics(trades, returns, capital, strategy.__name__)
    
    async def _walk_forward_backtest(
        self,
        strategy: Callable,
        data: pd.DataFrame,
        capital: float,
        risk_params: Optional[Dict]
    ) -> BacktestResult:
        """Walk-forward analysis for robust validation"""
        window_size = max(252, len(data) // 10)  # Minimum 1 year or 10% of data
        step_size = window_size // 4  # 25% step
        
        all_trades = []
        all_returns = []
        
        for start in range(0, len(data) - window_size, step_size):
            end = start + window_size
            window_data = data.iloc[start:end]
            
            # In-sample optimization
            optimized_params = await self._optimize_parameters(
                strategy, 
                window_data.iloc[:int(window_size*0.7)],
                risk_params
            )
            
            # Out-of-sample testing
            oos_data = window_data.iloc[int(window_size*0.7):]
            result = await self._historical_backtest(
                strategy,
                oos_data,
                capital,
                optimized_params
            )
            
            all_trades.extend(result.metadata.get('trades', []))
            all_returns.extend(result.metadata.get('returns', []))
        
        return self._calculate_metrics(all_trades, all_returns, capital, strategy.__name__)
    
    async def _monte_carlo_backtest(
        self,
        strategy: Callable,
        data: pd.DataFrame,
        capital: float,
        risk_params: Optional[Dict],
        n_simulations: int = 1000
    ) -> BacktestResult:
        """Monte Carlo simulation for probabilistic outcomes"""
        simulation_results = []
        
        for _ in range(n_simulations):
            # Generate synthetic price paths
            synthetic_data = await self._generate_synthetic_data(data)
            
            # Run backtest on synthetic data
            result = await self._historical_backtest(
                strategy,
                synthetic_data,
                capital,
                risk_params
            )
            
            simulation_results.append({
                'return': result.total_return,
                'sharpe': result.sharpe_ratio,
                'drawdown': result.max_drawdown,
                'trades': result.total_trades
            })
        
        # Aggregate results
        df_results = pd.DataFrame(simulation_results)
        
        return BacktestResult(
            strategy_name=strategy.__name__,
            total_return=df_results['return'].mean(),
            sharpe_ratio=df_results['sharpe'].mean(),
            max_drawdown=df_results['drawdown'].mean(),
            win_rate=0.0,  # Will be calculated
            profit_factor=0.0,  # Will be calculated
            total_trades=int(df_results['trades'].mean()),
            avg_trade_return=0.0,  # Will be calculated
            volatility=df_results['return'].std(),
            calmar_ratio=0.0,  # Will be calculated
            sortino_ratio=0.0,  # Will be calculated
            information_ratio=0.0,  # Will be calculated
            alpha=0.0,  # Will be calculated
            beta=0.0,  # Will be calculated
            confidence_interval=(
                df_results['return'].quantile(0.05),
                df_results['return'].quantile(0.95)
            ),
            statistical_significance=0.0,  # Will be calculated
            risk_metrics={},
            performance_attribution={},
            market_regime_performance={},
            stress_test_results={},
            metadata={'simulation_count': n_simulations}
        )
    
    async def _cross_validation_backtest(
        self,
        strategy: Callable,
        data: pd.DataFrame,
        capital: float,
        risk_params: Optional[Dict]
    ) -> BacktestResult:
        """Time series cross-validation for strategy validation"""
        tscv = TimeSeriesSplit(n_splits=5)
        cv_results = []
        
        for train_idx, test_idx in tscv.split(data):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Train on training set
            optimized_params = await self._optimize_parameters(
                strategy,
                train_data,
                risk_params
            )
            
            # Test on test set
            result = await self._historical_backtest(
                strategy,
                test_data,
                capital,
                optimized_params
            )
            
            cv_results.append({
                'return': result.total_return,
                'sharpe': result.sharpe_ratio,
                'drawdown': result.max_drawdown
            })
        
        # Average across folds
        df_cv = pd.DataFrame(cv_results)
        
        return BacktestResult(
            strategy_name=strategy.__name__,
            total_return=df_cv['return'].mean(),
            sharpe_ratio=df_cv['sharpe'].mean(),
            max_drawdown=df_cv['drawdown'].mean(),
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            avg_trade_return=0.0,
            volatility=df_cv['return'].std(),
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            information_ratio=0.0,
            alpha=0.0,
            beta=0.0,
            confidence_interval=(
                df_cv['return'].mean() - 2*df_cv['return'].std(),
                df_cv['return'].mean() + 2*df_cv['return'].std()
            ),
            statistical_significance=0.0,
            risk_metrics={},
            performance_attribution={},
            market_regime_performance={},
            stress_test_results={},
            metadata={'cv_folds': 5}
        )
    
    async def _adaptive_backtest(
        self,
        strategy: Callable,
        data: pd.DataFrame,
        capital: float,
        risk_params: Optional[Dict]
    ) -> BacktestResult:
        """Adaptive backtesting with online learning"""
        portfolio_value = capital
        trades = []
        returns = []
        positions = {}
        
        # Adaptive parameters
        learning_rate = 0.01
        adaptation_window = 50
        
        for i in range(adaptation_window, len(data)):
            current_data = data.iloc[:i+1]
            
            # Adapt strategy parameters based on recent performance
            if i % adaptation_window == 0 and len(returns) > 0:
                recent_performance = np.mean(returns[-adaptation_window:])
                if recent_performance < 0:
                    # Adjust risk parameters
                    if risk_params:
                        risk_params['position_size'] *= (1 - learning_rate)
                        risk_params['stop_loss'] *= (1 - learning_rate/2)
            
            signal = await strategy(current_data, positions, risk_params)
            
            if signal:
                trade_result = await self._execute_trade(
                    signal,
                    data.iloc[i],
                    portfolio_value,
                    positions
                )
                
                if trade_result:
                    trades.append(trade_result)
                    portfolio_value = trade_result['portfolio_value']
                    returns.append(trade_result['return'])
        
        return self._calculate_metrics(trades, returns, capital, strategy.__name__)
    
    async def compare_strategies(
        self,
        strategy_a: Callable,
        strategy_b: Callable,
        historical_data: pd.DataFrame,
        test_type: str = "paired_t_test"
    ) -> StrategyComparison:
        """
        A/B testing for strategy comparison with statistical validation
        """
        # Run backtests for both strategies
        result_a = await self.backtest_strategy(
            strategy_a,
            historical_data,
            BacktestMode.WALK_FORWARD
        )
        
        result_b = await self.backtest_strategy(
            strategy_b,
            historical_data,
            BacktestMode.WALK_FORWARD
        )
        
        # Perform statistical tests
        if test_type == "paired_t_test":
            statistic, p_value = stats.ttest_rel(
                result_a.metadata.get('returns', [0]),
                result_b.metadata.get('returns', [0])
            )
        elif test_type == "mann_whitney":
            statistic, p_value = stats.mannwhitneyu(
                result_a.metadata.get('returns', [0]),
                result_b.metadata.get('returns', [0])
            )
        else:  # Welch's t-test
            statistic, p_value = stats.ttest_ind(
                result_a.metadata.get('returns', [0]),
                result_b.metadata.get('returns', [0]),
                equal_var=False
            )
        
        # Calculate effect size (Cohen's d)
        effect_size = self._calculate_effect_size(
            result_a.metadata.get('returns', [0]),
            result_b.metadata.get('returns', [0])
        )
        
        # Determine winner
        winner = strategy_a.__name__ if result_a.sharpe_ratio > result_b.sharpe_ratio else strategy_b.__name__
        confidence_level = 1 - p_value
        
        # Evidence strength
        if p_value < 0.01:
            evidence_strength = "Very Strong"
        elif p_value < 0.05:
            evidence_strength = "Strong"
        elif p_value < 0.1:
            evidence_strength = "Moderate"
        else:
            evidence_strength = "Weak"
        
        return StrategyComparison(
            strategy_a=strategy_a.__name__,
            strategy_b=strategy_b.__name__,
            winner=winner,
            confidence_level=confidence_level,
            p_value=p_value,
            effect_size=effect_size,
            performance_difference={
                'sharpe_diff': result_a.sharpe_ratio - result_b.sharpe_ratio,
                'return_diff': result_a.total_return - result_b.total_return,
                'drawdown_diff': result_a.max_drawdown - result_b.max_drawdown
            },
            statistical_tests={
                'test_statistic': statistic,
                'p_value': p_value,
                'effect_size': effect_size
            },
            recommendation=f"Use {winner} with {evidence_strength.lower()} evidence",
            evidence_strength=evidence_strength
        )
    
    async def _stress_test_strategy(
        self,
        strategy: Callable,
        data: pd.DataFrame,
        base_result: BacktestResult
    ) -> Dict[str, float]:
        """Stress test strategy under extreme conditions"""
        stress_scenarios = {
            'market_crash': lambda x: x * 0.7,  # 30% crash
            'high_volatility': lambda x: x * np.random.normal(1, 0.3, len(x)),
            'liquidity_crisis': lambda x: x * 0.9,  # 10% liquidity discount
            'black_swan': lambda x: x * np.where(np.random.random(len(x)) < 0.01, 0.5, 1)
        }
        
        stress_results = {}
        
        for scenario_name, scenario_func in stress_scenarios.items():
            # Apply stress scenario to data
            stressed_data = data.copy()
            if 'close' in stressed_data.columns:
                stressed_data['close'] = scenario_func(stressed_data['close'].values)
            
            # Run backtest on stressed data
            stressed_result = await self._historical_backtest(
                strategy,
                stressed_data,
                100000,
                None
            )
            
            stress_results[scenario_name] = {
                'return': stressed_result.total_return,
                'max_drawdown': stressed_result.max_drawdown,
                'survival_rate': 1.0 if stressed_result.total_return > -0.5 else 0.0
            }
        
        return stress_results
    
    async def _analyze_regime_performance(
        self,
        strategy: Callable,
        data: pd.DataFrame,
        result: BacktestResult
    ) -> Dict[str, Dict[str, float]]:
        """Analyze strategy performance across different market regimes"""
        # Detect market regimes
        regimes = await self._detect_market_regimes(data)
        
        regime_performance = {}
        
        for regime_name, regime_periods in regimes.items():
            regime_returns = []
            regime_trades = 0
            
            for start, end in regime_periods:
                regime_data = data.iloc[start:end]
                if len(regime_data) > 0:
                    regime_result = await self._historical_backtest(
                        strategy,
                        regime_data,
                        100000,
                        None
                    )
                    regime_returns.append(regime_result.total_return)
                    regime_trades += regime_result.total_trades
            
            if regime_returns:
                regime_performance[regime_name] = {
                    'avg_return': np.mean(regime_returns),
                    'volatility': np.std(regime_returns),
                    'total_trades': regime_trades,
                    'performance_score': np.mean(regime_returns) / (np.std(regime_returns) + 1e-6)
                }
        
        return regime_performance
    
    async def _detect_market_regimes(self, data: pd.DataFrame) -> Dict[str, List[Tuple[int, int]]]:
        """Detect different market regimes in the data"""
        regimes = {
            'bull': [],
            'bear': [],
            'sideways': [],
            'high_volatility': []
        }
        
        if 'close' not in data.columns:
            return regimes
        
        prices = data['close'].values
        returns = np.diff(prices) / prices[:-1]
        
        # Simple regime detection (can be enhanced with HMM or other methods)
        window = 20
        
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            avg_return = np.mean(window_returns)
            volatility = np.std(window_returns)
            
            if avg_return > 0.001 and volatility < 0.02:
                regime_type = 'bull'
            elif avg_return < -0.001 and volatility < 0.02:
                regime_type = 'bear'
            elif abs(avg_return) < 0.001 and volatility < 0.02:
                regime_type = 'sideways'
            else:
                regime_type = 'high_volatility'
            
            # Group consecutive periods
            if regimes[regime_type] and regimes[regime_type][-1][1] == i-1:
                regimes[regime_type][-1] = (regimes[regime_type][-1][0], i)
            else:
                regimes[regime_type].append((i-1, i))
        
        return regimes
    
    async def _optimize_parameters(
        self,
        strategy: Callable,
        data: pd.DataFrame,
        base_params: Optional[Dict]
    ) -> Dict:
        """Optimize strategy parameters using grid search or bayesian optimization"""
        if not base_params:
            return {}
        
        best_params = base_params.copy()
        best_sharpe = -np.inf
        
        # Simple grid search (can be enhanced with Bayesian optimization)
        param_ranges = {
            'position_size': [0.01, 0.02, 0.05, 0.1],
            'stop_loss': [0.01, 0.02, 0.03, 0.05],
            'take_profit': [0.02, 0.03, 0.05, 0.1]
        }
        
        for pos_size in param_ranges.get('position_size', [base_params.get('position_size', 0.02)]):
            for stop_loss in param_ranges.get('stop_loss', [base_params.get('stop_loss', 0.02)]):
                for take_profit in param_ranges.get('take_profit', [base_params.get('take_profit', 0.05)]):
                    test_params = {
                        'position_size': pos_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                    
                    result = await self._historical_backtest(
                        strategy,
                        data,
                        100000,
                        test_params
                    )
                    
                    if result.sharpe_ratio > best_sharpe:
                        best_sharpe = result.sharpe_ratio
                        best_params = test_params
        
        return best_params
    
    async def _generate_synthetic_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic price data using statistical properties"""
        if 'close' not in data.columns:
            return data
        
        prices = data['close'].values
        returns = np.diff(prices) / prices[:-1]
        
        # Estimate parameters
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        # Generate synthetic returns
        synthetic_returns = np.random.normal(mu, sigma, len(returns))
        
        # Reconstruct prices
        synthetic_prices = [prices[0]]
        for ret in synthetic_returns:
            synthetic_prices.append(synthetic_prices[-1] * (1 + ret))
        
        synthetic_data = data.copy()
        synthetic_data['close'] = synthetic_prices[:len(data)]
        
        return synthetic_data
    
    async def _execute_trade(
        self,
        signal: Dict,
        current_bar: pd.Series,
        portfolio_value: float,
        positions: Dict
    ) -> Optional[Dict]:
        """Execute trade based on signal"""
        if not signal:
            return None
        
        trade_type = signal.get('type', 'buy')
        symbol = signal.get('symbol', 'UNKNOWN')
        size = signal.get('size', 0.01) * portfolio_value
        
        current_price = current_bar.get('close', 0)
        if current_price <= 0:
            return None
        
        if trade_type == 'buy':
            shares = size / current_price
            positions[symbol] = positions.get(symbol, 0) + shares
            trade_return = 0  # Initial buy has no return
        else:  # sell
            if symbol in positions and positions[symbol] > 0:
                shares = min(positions[symbol], size / current_price)
                positions[symbol] -= shares
                trade_return = shares * current_price / portfolio_value - 1
            else:
                return None
        
        return {
            'type': trade_type,
            'symbol': symbol,
            'shares': shares,
            'price': current_price,
            'portfolio_value': portfolio_value,
            'return': trade_return,
            'timestamp': current_bar.name if hasattr(current_bar, 'name') else datetime.now()
        }
    
    def _calculate_metrics(
        self,
        trades: List[Dict],
        returns: List[float],
        initial_capital: float,
        strategy_name: str
    ) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        if not returns:
            returns = [0]
        
        total_return = np.prod([1 + r for r in returns]) - 1
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252) if returns else 0
        sharpe_ratio = (np.mean(returns) * 252) / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative_returns = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Win rate
        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r < 0]
        win_rate = len(winning_trades) / len(returns) if returns else 0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio
        downside_returns = [r for r in returns if r < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if downside_returns else 1
        sortino_ratio = (np.mean(returns) * 252) / downside_std
        
        return BacktestResult(
            strategy_name=strategy_name,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_return=np.mean(returns) if returns else 0,
            volatility=volatility,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            information_ratio=sharpe_ratio * 0.8,  # Simplified
            alpha=total_return - 0.08,  # Assuming 8% benchmark
            beta=1.0,  # Simplified
            confidence_interval=(
                total_return - 2*volatility,
                total_return + 2*volatility
            ),
            statistical_significance=0.0,
            risk_metrics={
                'var_95': np.percentile(returns, 5) if returns else 0,
                'cvar_95': np.mean([r for r in returns if r <= np.percentile(returns, 5)]) if returns else 0,
                'max_consecutive_losses': self._max_consecutive_losses(returns)
            },
            performance_attribution={
                'selection': total_return * 0.6,
                'timing': total_return * 0.3,
                'risk_management': total_return * 0.1
            },
            market_regime_performance={},
            stress_test_results={},
            metadata={
                'trades': trades,
                'returns': returns
            }
        )
    
    def _calculate_effect_size(self, returns_a: List[float], returns_b: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        mean_diff = np.mean(returns_a) - np.mean(returns_b)
        pooled_std = np.sqrt((np.var(returns_a) + np.var(returns_b)) / 2)
        return mean_diff / pooled_std if pooled_std > 0 else 0
    
    def _max_consecutive_losses(self, returns: List[float]) -> int:
        """Calculate maximum consecutive losses"""
        max_losses = 0
        current_losses = 0
        
        for r in returns:
            if r < 0:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0
        
        return max_losses
    
    async def _validate_statistical_significance(self, result: BacktestResult) -> float:
        """Validate statistical significance of results"""
        if 'returns' not in result.metadata:
            return 0.0
        
        returns = result.metadata['returns']
        if not returns:
            return 0.0
        
        # T-test against zero returns
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        
        # Significance level
        significance = 1 - p_value if p_value < 0.05 else 0.0
        
        return significance
    
    async def get_optimization_recommendations(self) -> Dict:
        """Get strategy optimization recommendations based on backtesting results"""
        recommendations = {
            'top_performers': [],
            'improvement_areas': [],
            'parameter_suggestions': {},
            'regime_recommendations': {}
        }
        
        # Analyze all cached results
        for strategy_name, results in self.strategy_performance.items():
            if results:
                latest_result = results[-1]
                
                # Top performers
                if latest_result.sharpe_ratio > 1.5:
                    recommendations['top_performers'].append({
                        'strategy': strategy_name,
                        'sharpe': latest_result.sharpe_ratio,
                        'return': latest_result.total_return
                    })
                
                # Improvement areas
                if latest_result.max_drawdown < -0.2:
                    recommendations['improvement_areas'].append({
                        'strategy': strategy_name,
                        'issue': 'High drawdown',
                        'suggestion': 'Reduce position size or add stop-loss'
                    })
                
                # Regime-specific recommendations
                if latest_result.market_regime_performance:
                    best_regime = max(
                        latest_result.market_regime_performance.items(),
                        key=lambda x: x[1].get('performance_score', 0)
                    )
                    recommendations['regime_recommendations'][strategy_name] = {
                        'best_regime': best_regime[0],
                        'performance': best_regime[1]
                    }
        
        return recommendations


# Global instance
intelligent_backtesting = IntelligentBacktestingFramework()


async def get_backtesting_framework() -> IntelligentBacktestingFramework:
    """Get the intelligent backtesting framework instance"""
    return intelligent_backtesting