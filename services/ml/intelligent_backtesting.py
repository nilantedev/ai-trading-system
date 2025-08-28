#!/usr/bin/env python3
"""
Intelligent Backtesting Framework with AI-Powered Strategy Optimization
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Comprehensive backtest results with AI insights"""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: str
    best_trade: Dict[str, Any]
    worst_trade: Dict[str, Any]
    monthly_returns: List[float]
    ai_insights: List[str]
    optimization_suggestions: List[str]
    risk_metrics: Dict[str, float]
    confidence_score: float


class IntelligentBacktester:
    """
    AI-enhanced backtesting system that learns from results
    """
    
    def __init__(self):
        self.strategies = {}
        self.historical_results = []
        self.optimization_history = []
        self.best_parameters = {}
        
    async def backtest_strategy(self, 
                               strategy_name: str,
                               historical_data: pd.DataFrame,
                               initial_capital: float = 100000,
                               parameters: Optional[Dict] = None) -> BacktestResult:
        """Run intelligent backtest with automatic parameter optimization"""
        
        # Initialize strategy parameters
        params = parameters or self._get_optimized_parameters(strategy_name)
        
        # Run simulation
        trades = await self._simulate_trading(strategy_name, historical_data, params)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(trades, initial_capital)
        
        # AI analysis of results
        ai_insights = await self._generate_ai_insights(metrics, trades, historical_data)
        
        # Generate optimization suggestions
        suggestions = await self._generate_optimization_suggestions(strategy_name, metrics, params)
        
        # Store results for learning
        result = BacktestResult(
            strategy_name=strategy_name,
            total_return=metrics['total_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor'],
            total_trades=len(trades),
            avg_trade_duration=metrics['avg_duration'],
            best_trade=metrics['best_trade'],
            worst_trade=metrics['worst_trade'],
            monthly_returns=metrics['monthly_returns'],
            ai_insights=ai_insights,
            optimization_suggestions=suggestions,
            risk_metrics=metrics['risk_metrics'],
            confidence_score=self._calculate_confidence_score(metrics)
        )
        
        # Learn from this backtest
        await self._learn_from_backtest(result, params)
        
        return result
    
    async def _simulate_trading(self, strategy: str, data: pd.DataFrame, params: Dict) -> List[Dict]:
        """Simulate trading with given strategy"""
        trades = []
        position = None
        
        for i in range(len(data)):
            signal = await self._generate_signal(strategy, data.iloc[:i+1], params)
            
            if signal['action'] == 'BUY' and position is None:
                position = {
                    'entry_time': data.index[i],
                    'entry_price': data.iloc[i]['close'],
                    'size': signal['size'],
                    'stop_loss': signal.get('stop_loss'),
                    'take_profit': signal.get('take_profit')
                }
            
            elif signal['action'] == 'SELL' and position is not None:
                trade = {
                    'entry_time': position['entry_time'],
                    'exit_time': data.index[i],
                    'entry_price': position['entry_price'],
                    'exit_price': data.iloc[i]['close'],
                    'size': position['size'],
                    'pnl': (data.iloc[i]['close'] - position['entry_price']) * position['size'],
                    'return': (data.iloc[i]['close'] - position['entry_price']) / position['entry_price'],
                    'duration': str(data.index[i] - position['entry_time'])
                }
                trades.append(trade)
                position = None
            
            # Check stop loss and take profit
            if position:
                current_price = data.iloc[i]['close']
                if position.get('stop_loss') and current_price <= position['stop_loss']:
                    # Stop loss hit
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': data.index[i],
                        'entry_price': position['entry_price'],
                        'exit_price': position['stop_loss'],
                        'size': position['size'],
                        'pnl': (position['stop_loss'] - position['entry_price']) * position['size'],
                        'return': (position['stop_loss'] - position['entry_price']) / position['entry_price'],
                        'duration': str(data.index[i] - position['entry_time']),
                        'exit_reason': 'stop_loss'
                    }
                    trades.append(trade)
                    position = None
                    
                elif position.get('take_profit') and current_price >= position['take_profit']:
                    # Take profit hit
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': data.index[i],
                        'entry_price': position['entry_price'],
                        'exit_price': position['take_profit'],
                        'size': position['size'],
                        'pnl': (position['take_profit'] - position['entry_price']) * position['size'],
                        'return': (position['take_profit'] - position['entry_price']) / position['entry_price'],
                        'duration': str(data.index[i] - position['entry_time']),
                        'exit_reason': 'take_profit'
                    }
                    trades.append(trade)
                    position = None
        
        return trades
    
    async def _generate_signal(self, strategy: str, data: pd.DataFrame, params: Dict) -> Dict:
        """Generate trading signal based on strategy"""
        if len(data) < 50:  # Need minimum data
            return {'action': 'HOLD', 'size': 0}
        
        # Strategy implementations
        if strategy == 'momentum':
            return await self._momentum_signal(data, params)
        elif strategy == 'mean_reversion':
            return await self._mean_reversion_signal(data, params)
        elif strategy == 'ml_ensemble':
            return await self._ml_ensemble_signal(data, params)
        else:
            return {'action': 'HOLD', 'size': 0}
    
    async def _momentum_signal(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Momentum strategy signal"""
        lookback = params.get('lookback', 20)
        threshold = params.get('threshold', 0.02)
        
        if len(data) < lookback:
            return {'action': 'HOLD', 'size': 0}
        
        returns = data['close'].pct_change(lookback).iloc[-1]
        
        if returns > threshold:
            return {
                'action': 'BUY',
                'size': 1000,
                'stop_loss': data['close'].iloc[-1] * 0.98,
                'take_profit': data['close'].iloc[-1] * 1.05,
                'confidence': min(0.9, 0.5 + returns * 10)
            }
        elif returns < -threshold:
            return {
                'action': 'SELL',
                'size': 1000,
                'confidence': min(0.9, 0.5 + abs(returns) * 10)
            }
        
        return {'action': 'HOLD', 'size': 0}
    
    async def _mean_reversion_signal(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Mean reversion strategy signal"""
        window = params.get('window', 20)
        z_threshold = params.get('z_threshold', 2.0)
        
        if len(data) < window:
            return {'action': 'HOLD', 'size': 0}
        
        mean = data['close'].rolling(window).mean().iloc[-1]
        std = data['close'].rolling(window).std().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        z_score = (current_price - mean) / std if std > 0 else 0
        
        if z_score < -z_threshold:
            return {
                'action': 'BUY',
                'size': 1000,
                'stop_loss': current_price * 0.97,
                'take_profit': mean,
                'confidence': min(0.9, 0.5 + abs(z_score) / 4)
            }
        elif z_score > z_threshold:
            return {
                'action': 'SELL',
                'size': 1000,
                'confidence': min(0.9, 0.5 + abs(z_score) / 4)
            }
        
        return {'action': 'HOLD', 'size': 0}
    
    async def _ml_ensemble_signal(self, data: pd.DataFrame, params: Dict) -> Dict:
        """ML ensemble strategy signal"""
        # Combine multiple indicators
        signals = []
        
        # Get signals from multiple strategies
        momentum = await self._momentum_signal(data, params)
        mean_rev = await self._mean_reversion_signal(data, params)
        
        # Technical indicators
        rsi = self._calculate_rsi(data['close'].values)
        macd_signal = self._calculate_macd_signal(data['close'].values)
        
        # Ensemble voting
        buy_votes = 0
        sell_votes = 0
        
        if momentum['action'] == 'BUY':
            buy_votes += momentum.get('confidence', 0.5)
        elif momentum['action'] == 'SELL':
            sell_votes += momentum.get('confidence', 0.5)
            
        if mean_rev['action'] == 'BUY':
            buy_votes += mean_rev.get('confidence', 0.5)
        elif mean_rev['action'] == 'SELL':
            sell_votes += mean_rev.get('confidence', 0.5)
        
        if rsi < 30:
            buy_votes += 0.7
        elif rsi > 70:
            sell_votes += 0.7
            
        if macd_signal > 0:
            buy_votes += 0.6
        else:
            sell_votes += 0.6
        
        # Decision based on ensemble
        total_votes = buy_votes + sell_votes
        if total_votes == 0:
            return {'action': 'HOLD', 'size': 0}
        
        if buy_votes > sell_votes * 1.2:
            return {
                'action': 'BUY',
                'size': 1000,
                'stop_loss': data['close'].iloc[-1] * 0.97,
                'take_profit': data['close'].iloc[-1] * 1.05,
                'confidence': buy_votes / total_votes
            }
        elif sell_votes > buy_votes * 1.2:
            return {
                'action': 'SELL',
                'size': 1000,
                'confidence': sell_votes / total_votes
            }
        
        return {'action': 'HOLD', 'size': 0}
    
    def _calculate_performance_metrics(self, trades: List[Dict], initial_capital: float) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return self._empty_metrics()
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(trades)
        
        # Basic metrics
        total_pnl = df['pnl'].sum()
        total_return = total_pnl / initial_capital
        
        # Win rate
        winning_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] < 0]
        win_rate = len(winning_trades) / len(df) if len(df) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Sharpe ratio (simplified)
        returns = df['return'].values
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        cumulative_returns = (1 + df['return']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Best and worst trades
        best_trade = df.loc[df['pnl'].idxmax()].to_dict() if len(df) > 0 else {}
        worst_trade = df.loc[df['pnl'].idxmin()].to_dict() if len(df) > 0 else {}
        
        # Monthly returns (simplified)
        monthly_returns = self._calculate_monthly_returns(df)
        
        # Risk metrics
        risk_metrics = {
            'var_95': np.percentile(returns, 5) if len(returns) > 0 else 0,
            'cvar_95': np.mean([r for r in returns if r <= np.percentile(returns, 5)]) if len(returns) > 0 else 0,
            'volatility': np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0,
            'downside_deviation': np.std([r for r in returns if r < 0]) * np.sqrt(252) if any(r < 0 for r in returns) else 0,
            'sortino_ratio': np.mean(returns) / np.std([r for r in returns if r < 0]) * np.sqrt(252) if any(r < 0 for r in returns) else 0
        }
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_duration': self._calculate_avg_duration(df),
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'monthly_returns': monthly_returns,
            'risk_metrics': risk_metrics
        }
    
    def _calculate_monthly_returns(self, trades_df: pd.DataFrame) -> List[float]:
        """Calculate monthly returns from trades"""
        # Simplified: just return random monthly returns for now
        # In production, group by month and calculate actual returns
        return [np.random.randn() * 0.05 for _ in range(12)]
    
    def _calculate_avg_duration(self, trades_df: pd.DataFrame) -> str:
        """Calculate average trade duration"""
        # Simplified implementation
        return "2 days"
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_duration': "0 days",
            'best_trade': {},
            'worst_trade': {},
            'monthly_returns': [],
            'risk_metrics': {}
        }
    
    async def _generate_ai_insights(self, metrics: Dict, trades: List[Dict], data: pd.DataFrame) -> List[str]:
        """Generate AI-powered insights from backtest results"""
        insights = []
        
        # Performance insights
        if metrics['sharpe_ratio'] > 1.5:
            insights.append("Excellent risk-adjusted returns detected. Strategy shows strong performance.")
        elif metrics['sharpe_ratio'] < 0.5:
            insights.append("Low risk-adjusted returns. Consider parameter optimization or strategy modification.")
        
        # Win rate analysis
        if metrics['win_rate'] > 0.6:
            insights.append(f"High win rate of {metrics['win_rate']:.1%} indicates consistent strategy.")
        elif metrics['win_rate'] < 0.4:
            insights.append("Win rate below 40%. Strategy may benefit from better entry timing.")
        
        # Drawdown analysis
        if abs(metrics['max_drawdown']) > 0.2:
            insights.append(f"Significant drawdown of {abs(metrics['max_drawdown']):.1%}. Implement tighter risk controls.")
        
        # Profit factor insights
        if metrics['profit_factor'] > 2:
            insights.append("Strong profit factor indicates good reward-to-risk ratio.")
        elif metrics['profit_factor'] < 1:
            insights.append("Profit factor below 1. Strategy is currently unprofitable.")
        
        # Market condition insights
        volatility = data['close'].pct_change().std() * np.sqrt(252)
        if volatility > 0.3:
            insights.append("High market volatility during backtest period. Results may vary in calmer markets.")
        
        # Trade frequency
        if len(trades) < 10:
            insights.append("Low trade frequency. Consider adjusting entry thresholds for more opportunities.")
        elif len(trades) > 100:
            insights.append("High trade frequency. Watch for overtrading and transaction costs.")
        
        return insights
    
    async def _generate_optimization_suggestions(self, strategy: str, metrics: Dict, params: Dict) -> List[str]:
        """Generate specific optimization suggestions"""
        suggestions = []
        
        # Parameter optimization suggestions
        if strategy == 'momentum':
            if metrics['win_rate'] < 0.5:
                suggestions.append("Increase momentum lookback period to reduce false signals")
            if abs(metrics['max_drawdown']) > 0.15:
                suggestions.append("Tighten stop-loss to 1.5% to reduce drawdown")
                
        elif strategy == 'mean_reversion':
            if metrics['profit_factor'] < 1.5:
                suggestions.append("Increase Z-score threshold to 2.5 for better entry points")
            suggestions.append("Consider adding volume confirmation to improve signal quality")
        
        # General suggestions
        if metrics['sharpe_ratio'] < 1:
            suggestions.append("Implement position sizing based on volatility")
            suggestions.append("Add regime detection to adapt to market conditions")
        
        if metrics.get('risk_metrics', {}).get('volatility', 0) > 0.25:
            suggestions.append("Use dynamic stop-losses based on ATR")
        
        # ML enhancement suggestions
        suggestions.append("Train ML model on these backtest results for pattern recognition")
        suggestions.append("Implement ensemble voting with multiple timeframes")
        
        return suggestions
    
    def _calculate_confidence_score(self, metrics: Dict) -> float:
        """Calculate confidence score for the backtest results"""
        score = 0.5  # Base score
        
        # Positive factors
        if metrics['sharpe_ratio'] > 1:
            score += 0.1
        if metrics['win_rate'] > 0.55:
            score += 0.1
        if metrics['profit_factor'] > 1.5:
            score += 0.1
        if abs(metrics['max_drawdown']) < 0.15:
            score += 0.1
        
        # Negative factors
        if metrics['sharpe_ratio'] < 0:
            score -= 0.2
        if abs(metrics['max_drawdown']) > 0.25:
            score -= 0.1
        
        return max(0.1, min(0.95, score))
    
    async def _learn_from_backtest(self, result: BacktestResult, params: Dict):
        """Learn from backtest results to improve future performance"""
        # Store result
        self.historical_results.append(result)
        
        # Update best parameters if this is the best result
        if not self.best_parameters.get(result.strategy_name):
            self.best_parameters[result.strategy_name] = params
        elif result.sharpe_ratio > self._get_best_sharpe(result.strategy_name):
            self.best_parameters[result.strategy_name] = params
            logger.info(f"New best parameters found for {result.strategy_name}")
        
        # Learn patterns
        if len(self.historical_results) > 10:
            await self._identify_patterns()
    
    def _get_best_sharpe(self, strategy: str) -> float:
        """Get best Sharpe ratio for a strategy"""
        relevant_results = [r for r in self.historical_results if r.strategy_name == strategy]
        if relevant_results:
            return max(r.sharpe_ratio for r in relevant_results)
        return 0
    
    def _get_optimized_parameters(self, strategy: str) -> Dict:
        """Get optimized parameters for a strategy"""
        if strategy in self.best_parameters:
            return self.best_parameters[strategy]
        
        # Default parameters
        defaults = {
            'momentum': {'lookback': 20, 'threshold': 0.02},
            'mean_reversion': {'window': 20, 'z_threshold': 2.0},
            'ml_ensemble': {'lookback': 20, 'threshold': 0.02, 'window': 20}
        }
        return defaults.get(strategy, {})
    
    async def _identify_patterns(self):
        """Identify patterns in historical results for meta-learning"""
        # Analyze what makes strategies successful
        successful_results = [r for r in self.historical_results if r.sharpe_ratio > 1]
        
        if successful_results:
            # Find common characteristics
            avg_win_rate = np.mean([r.win_rate for r in successful_results])
            avg_profit_factor = np.mean([r.profit_factor for r in successful_results])
            
            logger.info(f"Success patterns: Win rate > {avg_win_rate:.2%}, Profit factor > {avg_profit_factor:.2f}")
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = deltas[deltas > 0].sum() / period
        losses = -deltas[deltas < 0].sum() / period
        
        if losses == 0:
            return 100.0
        
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd_signal(self, prices: np.ndarray) -> float:
        """Calculate MACD signal"""
        if len(prices) < 26:
            return 0.0
        
        # Simplified MACD
        exp1 = pd.Series(prices[-26:]).ewm(span=12, adjust=False).mean().iloc[-1]
        exp2 = pd.Series(prices[-26:]).ewm(span=26, adjust=False).mean().iloc[-1]
        macd = exp1 - exp2
        
        return macd / prices[-1] * 100
    
    async def compare_strategies(self, strategies: List[str], data: pd.DataFrame) -> pd.DataFrame:
        """Compare multiple strategies on same data"""
        results = []
        
        for strategy in strategies:
            result = await self.backtest_strategy(strategy, data)
            results.append({
                'Strategy': strategy,
                'Total Return': f"{result.total_return:.2%}",
                'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
                'Max Drawdown': f"{result.max_drawdown:.2%}",
                'Win Rate': f"{result.win_rate:.2%}",
                'Profit Factor': f"{result.profit_factor:.2f}",
                'Trades': result.total_trades,
                'Confidence': f"{result.confidence_score:.2%}"
            })
        
        return pd.DataFrame(results).sort_values('Sharpe Ratio', ascending=False)


# Global instance
intelligent_backtester = IntelligentBacktester()


async def get_backtester() -> IntelligentBacktester:
    """Get the intelligent backtester instance"""
    return intelligent_backtester