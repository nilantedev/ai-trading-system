#!/usr/bin/env python3
"""
Performance Analytics Service - Advanced risk-adjusted performance metrics and feedback loops
Comprehensive analysis of trading strategy performance with advanced risk metrics and feedback mechanisms.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "../../shared/python-common"))

from trading_common import get_settings, get_logger, MarketData
from trading_common.cache import get_trading_cache
from trading_common.database import get_database

logger = get_logger(__name__)
settings = get_settings()


class PerformanceMetricType(str, Enum):
    """Types of performance metrics."""
    RETURN_BASED = "return_based"           # Return-based metrics
    RISK_ADJUSTED = "risk_adjusted"         # Risk-adjusted metrics
    DRAWDOWN = "drawdown"                   # Drawdown-based metrics
    TAIL_RISK = "tail_risk"                 # Tail risk metrics
    BEHAVIORAL = "behavioral"               # Behavioral metrics


@dataclass
class AdvancedMetrics:
    """Advanced performance and risk metrics."""
    # Basic return metrics
    total_return: float
    annualized_return: float
    volatility: float
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Drawdown metrics
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: float  # Average days in drawdown
    recovery_factor: float   # Net profit / max drawdown
    
    # Tail risk metrics
    var_95: float           # Value at Risk 95%
    var_99: float           # Value at Risk 99% 
    expected_shortfall_95: float  # Conditional VaR 95%
    expected_shortfall_99: float  # Conditional VaR 99%
    tail_ratio: float       # 95% VaR / 5% VaR
    
    # Higher moment metrics
    skewness: float
    kurtosis: float
    
    # Trade-based metrics
    win_rate: float
    profit_factor: float
    payoff_ratio: float     # Avg win / Avg loss
    kelly_criterion: float  # Optimal position size
    
    # Advanced risk metrics
    ulcer_index: float      # Drawdown-based risk measure
    sterling_ratio: float   # Risk-adjusted return considering drawdowns
    burke_ratio: float      # Modified Sharpe with drawdown in denominator
    
    # All optional fields must come after required fields
    # Optional risk-adjusted metrics
    treynor_ratio: Optional[float] = None
    information_ratio: Optional[float] = None
    
    # Market-relative metrics
    beta: Optional[float] = None
    alpha: Optional[float] = None
    correlation_to_market: Optional[float] = None
    tracking_error: Optional[float] = None
    
    # Time-based metrics
    up_capture: Optional[float] = None     # Upside capture ratio
    down_capture: Optional[float] = None   # Downside capture ratio
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PerformanceReport:
    """Comprehensive performance analysis report."""
    strategy_name: str
    analysis_period: Tuple[datetime, datetime]
    benchmark_symbol: str
    
    # Core metrics
    metrics: AdvancedMetrics
    
    # Time series data
    returns: pd.Series
    cumulative_returns: pd.Series
    drawdown_series: pd.Series
    rolling_sharpe: pd.Series
    
    # Benchmark comparison
    benchmark_metrics: Optional[AdvancedMetrics] = None
    excess_returns: Optional[pd.Series] = None
    
    # Attribution analysis
    sector_attribution: Optional[Dict[str, float]] = None
    factor_attribution: Optional[Dict[str, float]] = None
    
    # Risk decomposition
    risk_contributions: Optional[Dict[str, float]] = None
    
    # Recommendations
    performance_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of key metrics."""
        return {
            "Total Return": self.metrics.total_return,
            "Annualized Return": self.metrics.annualized_return,
            "Volatility": self.metrics.volatility,
            "Sharpe Ratio": self.metrics.sharpe_ratio,
            "Sortino Ratio": self.metrics.sortino_ratio,
            "Max Drawdown": self.metrics.max_drawdown,
            "Win Rate": self.metrics.win_rate,
            "Profit Factor": self.metrics.profit_factor
        }


class PerformanceAnalyticsService:
    """Advanced performance analytics with feedback loops."""
    
    def __init__(self):
        self.cache = None
        self.db = None
        self.risk_free_rate = 0.02  # Annual risk-free rate
        self.benchmark_data = {}    # Cached benchmark data
        
    async def start(self):
        """Initialize the performance analytics service."""
        logger.info("Starting Performance Analytics Service")
        
        self.cache = await get_trading_cache()
        self.db = await get_database()
        
        # Create analytics tables
        await self._create_analytics_tables()
        
        logger.info("Performance Analytics Service started")
    
    async def _create_analytics_tables(self):
        """Create performance analytics database tables."""
        # Performance metrics table
        await self.db.execute("""
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            strategy_name VARCHAR(255) NOT NULL,
            analysis_date DATE NOT NULL,
            period_start DATE NOT NULL,
            period_end DATE NOT NULL,
            benchmark_symbol VARCHAR(10),
            metrics JSON NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(strategy_name, analysis_date)
        )
        """)
        
        # Daily performance tracking
        await self.db.execute("""
        CREATE TABLE IF NOT EXISTS daily_performance (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            strategy_name VARCHAR(255) NOT NULL,
            trade_date DATE NOT NULL,
            daily_return FLOAT NOT NULL,
            cumulative_return FLOAT NOT NULL,
            drawdown FLOAT NOT NULL,
            portfolio_value FLOAT NOT NULL,
            benchmark_return FLOAT,
            excess_return FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(strategy_name, trade_date)
        )
        """)
        
        # Performance alerts table
        await self.db.execute("""
        CREATE TABLE IF NOT EXISTS performance_alerts (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            strategy_name VARCHAR(255) NOT NULL,
            alert_type VARCHAR(50) NOT NULL,
            metric_name VARCHAR(100) NOT NULL,
            threshold_value FLOAT NOT NULL,
            actual_value FLOAT NOT NULL,
            severity VARCHAR(20) NOT NULL,
            message TEXT,
            triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
            is_resolved BOOLEAN DEFAULT FALSE
        )
        """)
        
        # Create indexes
        await self.db.execute("""
        CREATE INDEX IF NOT EXISTS idx_performance_metrics_strategy_date
        ON performance_metrics(strategy_name, analysis_date DESC)
        """)
        
        await self.db.execute("""
        CREATE INDEX IF NOT EXISTS idx_daily_performance_strategy_date
        ON daily_performance(strategy_name, trade_date DESC)
        """)
    
    async def analyze_strategy_performance(
        self, 
        strategy_name: str,
        returns: pd.Series,
        benchmark_symbol: str = "SPY",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> PerformanceReport:
        """Comprehensive performance analysis of a trading strategy."""
        
        logger.info(f"Analyzing performance for strategy: {strategy_name}")
        
        # Ensure returns are clean
        returns = returns.dropna().copy()
        
        if len(returns) < 30:  # Need minimum data
            raise ValueError("Insufficient return data for analysis")
        
        # Get analysis period
        if start_date is None:
            start_date = returns.index[0].to_pydatetime() if hasattr(returns.index[0], 'to_pydatetime') else returns.index[0]
        if end_date is None:
            end_date = returns.index[-1].to_pydatetime() if hasattr(returns.index[-1], 'to_pydatetime') else returns.index[-1]
        
        # Calculate basic metrics
        metrics = await self._calculate_advanced_metrics(returns, benchmark_symbol)
        
        # Calculate time series
        cumulative_returns = (1 + returns).cumprod() - 1
        drawdown_series = await self._calculate_drawdown_series(cumulative_returns)
        rolling_sharpe = await self._calculate_rolling_sharpe(returns, window=252)
        
        # Get benchmark data and comparison
        benchmark_returns = await self._get_benchmark_returns(benchmark_symbol, start_date, end_date)
        benchmark_metrics = None
        excess_returns = None
        
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # Align returns with benchmark
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
            if len(aligned_benchmark) > 30:
                benchmark_metrics = await self._calculate_advanced_metrics(aligned_benchmark, None)
                excess_returns = aligned_returns - aligned_benchmark
        
        # Performance diagnostics
        issues, recommendations = await self._diagnose_performance(metrics, returns)
        
        # Create report
        report = PerformanceReport(
            strategy_name=strategy_name,
            analysis_period=(start_date, end_date),
            benchmark_symbol=benchmark_symbol,
            metrics=metrics,
            returns=returns,
            cumulative_returns=cumulative_returns,
            drawdown_series=drawdown_series,
            rolling_sharpe=rolling_sharpe,
            benchmark_metrics=benchmark_metrics,
            excess_returns=excess_returns,
            performance_issues=issues,
            recommendations=recommendations
        )
        
        # Store results
        await self._store_performance_analysis(report)
        
        # Check for alerts
        await self._check_performance_alerts(strategy_name, metrics)
        
        logger.info(f"Performance analysis completed for {strategy_name}")
        return report
    
    async def _calculate_advanced_metrics(self, returns: pd.Series, benchmark_symbol: Optional[str]) -> AdvancedMetrics:
        """Calculate comprehensive set of performance metrics."""
        
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        days = len(returns)
        years = days / 252.0 if days > 0 else 1
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        volatility = returns.std() * np.sqrt(252)
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod() - 1
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max)
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
        
        # Drawdown duration analysis
        drawdown_duration = await self._calculate_avg_drawdown_duration(drawdown)
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Risk-adjusted metrics
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # VaR and Expected Shortfall
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01) 
        expected_shortfall_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        expected_shortfall_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99
        
        # Tail ratio
        var_5 = returns.quantile(0.95)
        tail_ratio = abs(var_95 / var_5) if var_5 != 0 else 1
        
        # Higher moments
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Trade-based metrics (simplified - would need actual trade data)
        win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
        profit_factor = abs(avg_win * len(returns[returns > 0]) / (avg_loss * len(returns[returns < 0]))) if avg_loss != 0 and len(returns[returns < 0]) > 0 else 1
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 1
        
        # Kelly Criterion (simplified)
        kelly_criterion = win_rate - ((1 - win_rate) / payoff_ratio) if payoff_ratio > 0 else 0
        
        # Ulcer Index (drawdown-based risk measure)
        ulcer_index = np.sqrt((drawdown ** 2).mean())
        
        # Sterling Ratio 
        sterling_ratio = annualized_return / abs(avg_drawdown) if avg_drawdown != 0 else 0
        
        # Burke Ratio (modified Sharpe with drawdown)
        burke_ratio = excess_return / ulcer_index if ulcer_index > 0 else 0
        
        # Market-relative metrics (if benchmark available)
        beta = None
        alpha = None
        correlation_to_market = None
        tracking_error = None
        treynor_ratio = None
        information_ratio = None
        up_capture = None
        down_capture = None
        
        if benchmark_symbol:
            try:
                benchmark_returns = await self._get_benchmark_returns(
                    benchmark_symbol, 
                    returns.index[0].to_pydatetime() if hasattr(returns.index[0], 'to_pydatetime') else returns.index[0],
                    returns.index[-1].to_pydatetime() if hasattr(returns.index[-1], 'to_pydatetime') else returns.index[-1]
                )
                
                if benchmark_returns is not None and len(benchmark_returns) > 0:
                    aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
                    
                    if len(aligned_benchmark) > 30:
                        # Beta calculation
                        covariance = aligned_returns.cov(aligned_benchmark)
                        benchmark_var = aligned_benchmark.var()
                        beta = covariance / benchmark_var if benchmark_var > 0 else 0
                        
                        # Alpha (Jensen's alpha)
                        benchmark_annual = (1 + aligned_benchmark).prod() ** (252/len(aligned_benchmark)) - 1
                        alpha = annualized_return - (self.risk_free_rate + beta * (benchmark_annual - self.risk_free_rate))
                        
                        # Correlation
                        correlation_to_market = aligned_returns.corr(aligned_benchmark)
                        
                        # Tracking error
                        excess_ret = aligned_returns - aligned_benchmark
                        tracking_error = excess_ret.std() * np.sqrt(252)
                        
                        # Treynor ratio
                        treynor_ratio = excess_return / beta if beta > 0 else 0
                        
                        # Information ratio
                        information_ratio = excess_ret.mean() * 252 / tracking_error if tracking_error > 0 else 0
                        
                        # Up/Down capture
                        up_market = aligned_benchmark > 0
                        down_market = aligned_benchmark < 0
                        
                        if len(aligned_benchmark[up_market]) > 0:
                            up_capture = (aligned_returns[up_market].mean() / aligned_benchmark[up_market].mean()) if aligned_benchmark[up_market].mean() != 0 else 0
                        
                        if len(aligned_benchmark[down_market]) > 0:
                            down_capture = (aligned_returns[down_market].mean() / aligned_benchmark[down_market].mean()) if aligned_benchmark[down_market].mean() != 0 else 0
                            
            except Exception as e:
                logger.warning(f"Could not calculate market-relative metrics: {e}")
        
        return AdvancedMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            treynor_ratio=treynor_ratio,
            information_ratio=information_ratio,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            drawdown_duration=drawdown_duration,
            recovery_factor=recovery_factor,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=expected_shortfall_95,
            expected_shortfall_99=expected_shortfall_99,
            tail_ratio=tail_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
            win_rate=win_rate,
            profit_factor=profit_factor,
            payoff_ratio=payoff_ratio,
            kelly_criterion=kelly_criterion,
            beta=beta,
            alpha=alpha,
            correlation_to_market=correlation_to_market,
            tracking_error=tracking_error,
            ulcer_index=ulcer_index,
            sterling_ratio=sterling_ratio,
            burke_ratio=burke_ratio,
            up_capture=up_capture,
            down_capture=down_capture
        )
    
    async def _calculate_drawdown_series(self, cumulative_returns: pd.Series) -> pd.Series:
        """Calculate drawdown time series."""
        running_max = cumulative_returns.expanding().max()
        return (cumulative_returns - running_max)
    
    async def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        rolling_return = returns.rolling(window=window).mean() * 252
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_return - self.risk_free_rate) / rolling_vol
        return rolling_sharpe.fillna(0)
    
    async def _calculate_avg_drawdown_duration(self, drawdown: pd.Series) -> float:
        """Calculate average drawdown duration in days."""
        in_drawdown = drawdown < -0.001  # Consider drawdowns > 0.1%
        
        if not in_drawdown.any():
            return 0.0
        
        # Find drawdown periods
        drawdown_periods = []
        start_dd = None
        
        for i, in_dd in enumerate(in_drawdown):
            if in_dd and start_dd is None:
                start_dd = i
            elif not in_dd and start_dd is not None:
                drawdown_periods.append(i - start_dd)
                start_dd = None
        
        # Handle case where we end in drawdown
        if start_dd is not None:
            drawdown_periods.append(len(in_drawdown) - start_dd)
        
        return np.mean(drawdown_periods) if drawdown_periods else 0.0
    
    async def _get_benchmark_returns(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.Series]:
        """Get benchmark returns for comparison."""
        # In practice, would fetch from market data service
        # For now, generate mock benchmark data
        try:
            # This would be replaced with actual data fetching
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            # Mock SPY-like returns (random walk with slight positive drift)
            np.random.seed(42)  # For reproducibility
            daily_returns = np.random.normal(0.0003, 0.015, len(date_range))  # ~8% annual return, 15% vol
            return pd.Series(daily_returns, index=date_range, name=symbol)
        except Exception as e:
            logger.warning(f"Could not fetch benchmark data for {symbol}: {e}")
            return None
    
    async def _diagnose_performance(self, metrics: AdvancedMetrics, returns: pd.Series) -> Tuple[List[str], List[str]]:
        """Diagnose performance issues and generate recommendations."""
        issues = []
        recommendations = []
        
        # Check Sharpe ratio
        if metrics.sharpe_ratio < 0.5:
            issues.append("Low risk-adjusted returns (Sharpe < 0.5)")
            recommendations.append("Review strategy logic and risk management")
        
        # Check max drawdown
        if metrics.max_drawdown < -0.15:  # 15% drawdown
            issues.append(f"High maximum drawdown: {metrics.max_drawdown:.1%}")
            recommendations.append("Implement stronger position sizing and stop-loss mechanisms")
        
        # Check volatility
        if metrics.volatility > 0.30:  # 30% annualized volatility
            issues.append(f"High volatility: {metrics.volatility:.1%}")
            recommendations.append("Consider reducing position sizes or adding diversification")
        
        # Check win rate
        if metrics.win_rate < 0.4:  # Less than 40% win rate
            issues.append(f"Low win rate: {metrics.win_rate:.1%}")
            recommendations.append("Review entry signals and consider longer holding periods")
        
        # Check skewness (prefer positive skew)
        if metrics.skewness < -0.5:
            issues.append("Negative return skewness (tail risk)")
            recommendations.append("Implement tail risk hedging or adjust position sizing")
        
        # Check recovery factor
        if metrics.recovery_factor < 2.0:
            issues.append("Low recovery factor (profits vs max drawdown)")
            recommendations.append("Improve risk management to reduce drawdown severity")
        
        # Check profit factor
        if metrics.profit_factor < 1.2:
            issues.append("Low profit factor (wins vs losses)")
            recommendations.append("Focus on improving trade selection or exit timing")
        
        return issues, recommendations
    
    async def _store_performance_analysis(self, report: PerformanceReport):
        """Store performance analysis results."""
        await self.db.execute("""
        INSERT INTO performance_metrics 
        (strategy_name, analysis_date, period_start, period_end, benchmark_symbol, metrics)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (strategy_name, analysis_date) DO UPDATE SET
            metrics = EXCLUDED.metrics
        """, [
            report.strategy_name,
            datetime.utcnow().date(),
            report.analysis_period[0].date(),
            report.analysis_period[1].date(),
            report.benchmark_symbol,
            json.dumps(report.metrics.to_dict())
        ])
        
        # Store daily performance data
        daily_data = []
        cumulative = 0.0
        running_max = 0.0
        
        for date, daily_return in report.returns.items():
            cumulative = (1 + cumulative) * (1 + daily_return) - 1
            running_max = max(running_max, cumulative)
            drawdown = cumulative - running_max
            
            daily_data.append([
                report.strategy_name,
                date.date() if hasattr(date, 'date') else date,
                float(daily_return),
                cumulative,
                drawdown,
                (1 + cumulative) * 100000  # Assuming $100k initial
            ])
        
        if daily_data:
            await self.db.executemany("""
            INSERT INTO daily_performance 
            (strategy_name, trade_date, daily_return, cumulative_return, drawdown, portfolio_value)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (strategy_name, trade_date) DO UPDATE SET
                daily_return = EXCLUDED.daily_return,
                cumulative_return = EXCLUDED.cumulative_return,
                drawdown = EXCLUDED.drawdown,
                portfolio_value = EXCLUDED.portfolio_value
            """, daily_data)
    
    async def _check_performance_alerts(self, strategy_name: str, metrics: AdvancedMetrics):
        """Check for performance alert conditions."""
        alerts = []
        
        # Alert thresholds
        thresholds = {
            'max_drawdown': -0.10,      # 10% drawdown
            'sharpe_ratio': 0.5,        # Minimum Sharpe
            'volatility': 0.25,         # Maximum volatility
            'win_rate': 0.35,           # Minimum win rate
        }
        
        # Check each threshold
        if metrics.max_drawdown < thresholds['max_drawdown']:
            alerts.append({
                'type': 'drawdown_breach',
                'metric': 'max_drawdown', 
                'threshold': thresholds['max_drawdown'],
                'actual': metrics.max_drawdown,
                'severity': 'HIGH' if metrics.max_drawdown < -0.20 else 'MEDIUM',
                'message': f"Maximum drawdown of {metrics.max_drawdown:.1%} exceeds threshold"
            })
        
        if metrics.sharpe_ratio < thresholds['sharpe_ratio']:
            alerts.append({
                'type': 'performance_degradation',
                'metric': 'sharpe_ratio',
                'threshold': thresholds['sharpe_ratio'],
                'actual': metrics.sharpe_ratio,
                'severity': 'MEDIUM',
                'message': f"Sharpe ratio of {metrics.sharpe_ratio:.2f} below acceptable threshold"
            })
        
        if metrics.volatility > thresholds['volatility']:
            alerts.append({
                'type': 'high_volatility',
                'metric': 'volatility',
                'threshold': thresholds['volatility'],
                'actual': metrics.volatility,
                'severity': 'LOW',
                'message': f"Volatility of {metrics.volatility:.1%} exceeds target"
            })
        
        # Store alerts
        for alert in alerts:
            await self.db.execute("""
            INSERT INTO performance_alerts 
            (strategy_name, alert_type, metric_name, threshold_value, actual_value, 
             severity, message)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, [
                strategy_name,
                alert['type'],
                alert['metric'],
                alert['threshold'],
                alert['actual'],
                alert['severity'],
                alert['message']
            ])
            
            logger.warning(f"Performance alert for {strategy_name}: {alert['message']}")
    
    async def get_strategy_performance_summary(self, strategy_name: str, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for a strategy."""
        # Get recent performance metrics
        query = """
        SELECT * FROM performance_metrics
        WHERE strategy_name = %s 
        AND analysis_date >= %s
        ORDER BY analysis_date DESC
        LIMIT 1
        """
        
        cutoff_date = datetime.utcnow().date() - timedelta(days=days)
        row = await self.db.fetch_one(query, [strategy_name, cutoff_date])
        
        if not row:
            return {"error": "No recent performance data found"}
        
        metrics = json.loads(row['metrics'])
        
        # Get recent alerts
        alerts_query = """
        SELECT alert_type, severity, message, triggered_at
        FROM performance_alerts
        WHERE strategy_name = %s 
        AND triggered_at >= %s
        AND is_resolved = FALSE
        ORDER BY triggered_at DESC
        """
        
        alert_rows = await self.db.fetch_all(
            alerts_query, 
            [strategy_name, datetime.utcnow() - timedelta(days=days)]
        )
        
        return {
            "strategy_name": strategy_name,
            "last_analysis": row['analysis_date'].isoformat(),
            "analysis_period": [row['period_start'].isoformat(), row['period_end'].isoformat()],
            "key_metrics": {
                "total_return": metrics.get('total_return'),
                "sharpe_ratio": metrics.get('sharpe_ratio'), 
                "max_drawdown": metrics.get('max_drawdown'),
                "win_rate": metrics.get('win_rate'),
                "volatility": metrics.get('volatility')
            },
            "active_alerts": [
                {
                    "type": alert['alert_type'],
                    "severity": alert['severity'],
                    "message": alert['message'],
                    "triggered_at": alert['triggered_at'].isoformat()
                }
                for alert in alert_rows
            ]
        }
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get service health status."""
        # Count recent analyses
        recent_analyses = await self.db.fetch_one("""
        SELECT COUNT(*) as count FROM performance_metrics
        WHERE analysis_date >= %s
        """, [datetime.utcnow().date() - timedelta(days=7)])
        
        # Count active alerts
        active_alerts = await self.db.fetch_one("""
        SELECT COUNT(*) as count FROM performance_alerts
        WHERE is_resolved = FALSE
        """)
        
        return {
            "service": "performance_analytics",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "recent_analyses": recent_analyses['count'] if recent_analyses else 0,
                "active_alerts": active_alerts['count'] if active_alerts else 0
            }
        }


# Global performance analytics service instance
_performance_analytics_service: Optional[PerformanceAnalyticsService] = None


async def get_performance_analytics_service() -> PerformanceAnalyticsService:
    """Get global performance analytics service instance."""
    global _performance_analytics_service
    if _performance_analytics_service is None:
        _performance_analytics_service = PerformanceAnalyticsService()
        await _performance_analytics_service.start()
    return _performance_analytics_service