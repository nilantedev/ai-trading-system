#!/usr/bin/env python3
"""
Kelly Criterion Position Sizing - Optimal position sizing for maximum geometric growth
Implements the Kelly formula with ML enhancements and safety constraints.
"""

import numpy as np
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics
import math

from trading_common import get_logger
from trading_common.cache import get_trading_cache

logger = get_logger(__name__)


@dataclass
class TradeHistory:
    """Historical trade data for Kelly calculation."""
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    win: bool
    duration_hours: float
    timestamp: datetime
    strategy: str


@dataclass
class KellyParameters:
    """Kelly Criterion parameters for a symbol/strategy."""
    symbol: str
    strategy: str
    win_rate: float           # Probability of winning (p)
    avg_win_percent: float    # Average winning percentage
    avg_loss_percent: float   # Average losing percentage  
    kelly_fraction: float     # Calculated Kelly fraction
    adjusted_fraction: float  # Safety-adjusted Kelly fraction
    confidence_interval: Tuple[float, float]  # 95% confidence interval
    sample_size: int         # Number of trades in sample
    last_updated: datetime


@dataclass 
class PositionSizeRecommendation:
    """Position sizing recommendation with rationale."""
    symbol: str
    recommended_size: float      # Dollar amount
    position_percent: float      # % of portfolio
    kelly_fraction: float        # Raw Kelly fraction
    safety_factor: float         # Applied safety multiplier
    max_risk_percent: float      # Maximum risk per trade
    confidence_level: str        # LOW/MEDIUM/HIGH confidence
    reasoning: str              # Explanation of sizing decision


class KellyPositionSizer:
    """
    Kelly Criterion position sizer with machine learning enhancements.
    
    Calculates optimal position sizes to maximize geometric growth rate
    while incorporating safety constraints and regime adjustments.
    """
    
    def __init__(self, 
                 max_kelly_fraction: float = 0.25,
                 safety_factor: float = 0.5,
                 min_sample_size: int = 20,
                 max_single_position: float = 0.10):
        """
        Initialize Kelly position sizer.
        
        Args:
            max_kelly_fraction: Maximum Kelly fraction allowed (cap at 25%)
            safety_factor: Safety multiplier applied to Kelly fraction (0.5 = half Kelly)
            min_sample_size: Minimum trades required for Kelly calculation
            max_single_position: Maximum single position as % of portfolio (10%)
        """
        self.max_kelly_fraction = max_kelly_fraction
        self.safety_factor = safety_factor
        self.min_sample_size = min_sample_size
        self.max_single_position = max_single_position
        
        # Historical trade data
        self.trade_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.kelly_params: Dict[str, KellyParameters] = {}
        
        # Performance tracking
        self.cache = None
        self.calculations_performed = 0
        self.recommendations_given = 0
        
        # Market regime adjustments
        self.regime_multipliers = {
            "bull_trending": 1.0,      # Full Kelly in trending bull market
            "bear_trending": 0.3,      # Very conservative in bear market  
            "high_volatility": 0.4,    # Reduced size in high vol
            "low_volatility": 0.8,     # Slightly larger in low vol
            "sideways": 0.6,           # Moderate sizing in range-bound
            "unknown": 0.5             # Conservative default
        }
    
    async def initialize(self):
        """Initialize the Kelly position sizer."""
        self.cache = get_trading_cache()
        
        # Load historical trade data from cache if available
        await self._load_trade_history()
        
        logger.info("Kelly Position Sizer initialized")
    
    async def calculate_position_size(self,
                                    symbol: str,
                                    signal_confidence: float,
                                    portfolio_value: float,
                                    current_price: float,
                                    stop_loss_price: Optional[float] = None,
                                    strategy: str = "default",
                                    market_regime: str = "unknown") -> PositionSizeRecommendation:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            symbol: Trading symbol
            signal_confidence: Confidence in signal (0-1)
            portfolio_value: Total portfolio value
            current_price: Current stock price
            stop_loss_price: Stop loss price (for risk calculation)
            strategy: Trading strategy name
            market_regime: Current market regime
            
        Returns:
            PositionSizeRecommendation with sizing details
        """
        self.calculations_performed += 1
        
        # Get or calculate Kelly parameters for this symbol/strategy
        key = f"{symbol}:{strategy}"
        
        if key not in self.kelly_params:
            kelly_params = await self._calculate_kelly_parameters(symbol, strategy)
            if kelly_params:
                self.kelly_params[key] = kelly_params
        else:
            kelly_params = self.kelly_params[key]
            
            # Recalculate if parameters are old (>7 days) or sample size increased significantly
            age_days = (datetime.utcnow() - kelly_params.last_updated).days
            current_sample_size = len(self.trade_history[key])
            
            if age_days > 7 or current_sample_size > kelly_params.sample_size * 1.5:
                updated_params = await self._calculate_kelly_parameters(symbol, strategy)
                if updated_params:
                    self.kelly_params[key] = updated_params
                    kelly_params = updated_params
        
        # Calculate position size
        if kelly_params and kelly_params.sample_size >= self.min_sample_size:
            recommendation = await self._calculate_with_kelly(
                symbol, signal_confidence, portfolio_value, current_price,
                stop_loss_price, strategy, market_regime, kelly_params
            )
        else:
            # Fallback to conservative sizing when insufficient data
            recommendation = await self._calculate_conservative_fallback(
                symbol, signal_confidence, portfolio_value, current_price,
                stop_loss_price, strategy, market_regime
            )
        
        self.recommendations_given += 1
        
        # Cache the recommendation
        await self._cache_recommendation(recommendation)
        
        return recommendation
    
    async def _calculate_kelly_parameters(self, symbol: str, strategy: str) -> Optional[KellyParameters]:
        """Calculate Kelly parameters from historical trade data."""
        key = f"{symbol}:{strategy}"
        
        if key not in self.trade_history or len(self.trade_history[key]) < self.min_sample_size:
            return None
        
        trades = list(self.trade_history[key])
        
        # Separate wins and losses
        winning_trades = [t for t in trades if t.win]
        losing_trades = [t for t in trades if not t.win]
        
        if not winning_trades or not losing_trades:
            return None  # Need both wins and losses for Kelly
        
        # Calculate probabilities and averages
        win_rate = len(winning_trades) / len(trades)
        avg_win_percent = statistics.mean([t.pnl_percent for t in winning_trades])
        avg_loss_percent = abs(statistics.mean([t.pnl_percent for t in losing_trades]))
        
        # Kelly fraction calculation: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        if avg_loss_percent > 0:
            b = avg_win_percent / avg_loss_percent  # Payoff ratio
            kelly_fraction = (b * win_rate - (1 - win_rate)) / b
        else:
            kelly_fraction = 0.0
        
        # Apply safety constraints
        kelly_fraction = max(0.0, min(kelly_fraction, self.max_kelly_fraction))
        adjusted_fraction = kelly_fraction * self.safety_factor
        
        # Calculate confidence interval using bootstrap method
        confidence_interval = self._calculate_confidence_interval(trades, win_rate, avg_win_percent, avg_loss_percent)
        
        return KellyParameters(
            symbol=symbol,
            strategy=strategy,
            win_rate=win_rate,
            avg_win_percent=avg_win_percent,
            avg_loss_percent=avg_loss_percent,
            kelly_fraction=kelly_fraction,
            adjusted_fraction=adjusted_fraction,
            confidence_interval=confidence_interval,
            sample_size=len(trades),
            last_updated=datetime.utcnow()
        )
    
    def _calculate_confidence_interval(self, trades: List[TradeHistory], 
                                     win_rate: float, avg_win: float, avg_loss: float) -> Tuple[float, float]:
        """Calculate 95% confidence interval for Kelly fraction using bootstrap."""
        if len(trades) < 10:
            return (0.0, 0.1)  # Wide interval for low sample size
        
        # Bootstrap resampling
        bootstrap_kellys = []
        n_bootstrap = 1000
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(trades, size=len(trades), replace=True)
            
            wins = [t for t in bootstrap_sample if t.win]
            losses = [t for t in bootstrap_sample if not t.win]
            
            if wins and losses:
                bs_win_rate = len(wins) / len(bootstrap_sample)
                bs_avg_win = statistics.mean([t.pnl_percent for t in wins])
                bs_avg_loss = abs(statistics.mean([t.pnl_percent for t in losses]))
                
                if bs_avg_loss > 0:
                    bs_b = bs_avg_win / bs_avg_loss
                    bs_kelly = (bs_b * bs_win_rate - (1 - bs_win_rate)) / bs_b
                    bs_kelly = max(0.0, min(bs_kelly, self.max_kelly_fraction))
                    bootstrap_kellys.append(bs_kelly)
        
        if bootstrap_kellys:
            lower = np.percentile(bootstrap_kellys, 2.5)
            upper = np.percentile(bootstrap_kellys, 97.5)
            return (float(lower), float(upper))
        else:
            return (0.0, 0.1)
    
    async def _calculate_with_kelly(self,
                                  symbol: str,
                                  signal_confidence: float,
                                  portfolio_value: float,
                                  current_price: float,
                                  stop_loss_price: Optional[float],
                                  strategy: str,
                                  market_regime: str,
                                  kelly_params: KellyParameters) -> PositionSizeRecommendation:
        """Calculate position size using Kelly parameters."""
        
        # Base Kelly fraction
        base_kelly = kelly_params.adjusted_fraction
        
        # Adjust for signal confidence
        confidence_multiplier = signal_confidence ** 1.5  # Square root for safety
        
        # Adjust for market regime
        regime_multiplier = self.regime_multipliers.get(market_regime, 0.5)
        
        # Adjust for sample size confidence
        sample_confidence = min(kelly_params.sample_size / 100, 1.0)  # Full confidence at 100+ trades
        
        # Final Kelly fraction
        final_kelly = base_kelly * confidence_multiplier * regime_multiplier * sample_confidence
        
        # Convert to dollar amount
        target_position_value = final_kelly * portfolio_value
        
        # Apply maximum position size constraint
        max_position_value = self.max_single_position * portfolio_value
        target_position_value = min(target_position_value, max_position_value)
        
        # Calculate risk per trade
        if stop_loss_price and current_price > 0:
            risk_per_share = abs(current_price - stop_loss_price)
            max_risk_percent = risk_per_share / current_price
        else:
            max_risk_percent = 0.03  # Default 3% risk
        
        # Ensure risk doesn't exceed portfolio risk limits
        max_portfolio_risk = 0.02  # 2% of portfolio at risk
        if target_position_value * max_risk_percent > portfolio_value * max_portfolio_risk:
            # Reduce position size to maintain risk limit
            target_position_value = (portfolio_value * max_portfolio_risk) / max_risk_percent
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(kelly_params, sample_confidence, signal_confidence)
        
        # Generate reasoning
        reasoning = self._generate_kelly_reasoning(
            kelly_params, confidence_multiplier, regime_multiplier, 
            sample_confidence, final_kelly, confidence_level
        )
        
        return PositionSizeRecommendation(
            symbol=symbol,
            recommended_size=target_position_value,
            position_percent=target_position_value / portfolio_value,
            kelly_fraction=final_kelly,
            safety_factor=self.safety_factor,
            max_risk_percent=max_risk_percent,
            confidence_level=confidence_level,
            reasoning=reasoning
        )
    
    async def _calculate_conservative_fallback(self,
                                             symbol: str,
                                             signal_confidence: float,
                                             portfolio_value: float,
                                             current_price: float,
                                             stop_loss_price: Optional[float],
                                             strategy: str,
                                             market_regime: str) -> PositionSizeRecommendation:
        """Conservative position sizing when insufficient historical data."""
        
        # Very conservative base size
        base_fraction = 0.02  # 2% of portfolio
        
        # Adjust for signal confidence (but cap low)
        confidence_adjustment = min(signal_confidence * 0.5, 0.5)
        
        # Market regime adjustment
        regime_multiplier = self.regime_multipliers.get(market_regime, 0.5)
        
        final_fraction = base_fraction * (0.5 + confidence_adjustment) * regime_multiplier
        target_position_value = final_fraction * portfolio_value
        
        # Calculate risk
        if stop_loss_price and current_price > 0:
            max_risk_percent = abs(current_price - stop_loss_price) / current_price
        else:
            max_risk_percent = 0.03
        
        reasoning = (f"Conservative sizing due to insufficient trade history "
                    f"(need {self.min_sample_size}+ trades for Kelly calculation). "
                    f"Using {final_fraction:.1%} allocation with {regime_multiplier:.1f}x "
                    f"regime adjustment for {market_regime} market.")
        
        return PositionSizeRecommendation(
            symbol=symbol,
            recommended_size=target_position_value,
            position_percent=final_fraction,
            kelly_fraction=0.0,  # No Kelly calculation available
            safety_factor=1.0,
            max_risk_percent=max_risk_percent,
            confidence_level="LOW",
            reasoning=reasoning
        )
    
    def _determine_confidence_level(self, kelly_params: KellyParameters, 
                                   sample_confidence: float, signal_confidence: float) -> str:
        """Determine confidence level for position sizing."""
        
        # Factors for confidence
        sample_factor = sample_confidence
        kelly_stability = 1.0 - (kelly_params.confidence_interval[1] - kelly_params.confidence_interval[0])
        win_rate_factor = 1.0 if kelly_params.win_rate > 0.55 else 0.5
        signal_factor = signal_confidence
        
        overall_confidence = (sample_factor + kelly_stability + win_rate_factor + signal_factor) / 4
        
        if overall_confidence > 0.8:
            return "HIGH"
        elif overall_confidence > 0.6:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_kelly_reasoning(self, kelly_params: KellyParameters,
                                 confidence_mult: float, regime_mult: float,
                                 sample_conf: float, final_kelly: float,
                                 confidence_level: str) -> str:
        """Generate human-readable reasoning for Kelly sizing decision."""
        
        return (
            f"Kelly Criterion sizing: {kelly_params.win_rate:.1%} win rate, "
            f"{kelly_params.avg_win_percent:.1%} avg win vs {kelly_params.avg_loss_percent:.1%} avg loss "
            f"over {kelly_params.sample_size} trades. "
            f"Base Kelly: {kelly_params.kelly_fraction:.1%}, "
            f"adjusted for confidence ({confidence_mult:.2f}x), "
            f"regime ({regime_mult:.2f}x), and sample size ({sample_conf:.2f}x). "
            f"Final allocation: {final_kelly:.1%} ({confidence_level} confidence)."
        )
    
    async def record_trade_outcome(self, trade: TradeHistory):
        """Record a completed trade for Kelly parameter updates."""
        key = f"{trade.symbol}:{trade.strategy}"
        self.trade_history[key].append(trade)
        
        # Clear cached Kelly parameters to force recalculation
        if key in self.kelly_params:
            del self.kelly_params[key]
        
        logger.debug(f"Recorded trade outcome for {trade.symbol}: {trade.pnl_percent:.2%} ({'WIN' if trade.win else 'LOSS'})")
    
    async def _load_trade_history(self):
        """Load historical trade data from cache."""
        if not self.cache:
            return
        
        try:
            cached_history = await self.cache.get_json("kelly_trade_history")
            if cached_history:
                # Reconstruct trade history from cached data
                for key, trades in cached_history.items():
                    for trade_data in trades:
                        trade = TradeHistory(**trade_data)
                        self.trade_history[key].append(trade)
                
                logger.info(f"Loaded {sum(len(trades) for trades in self.trade_history.values())} historical trades")
        except Exception as e:
            logger.warning(f"Failed to load trade history: {e}")
    
    async def _cache_recommendation(self, recommendation: PositionSizeRecommendation):
        """Cache position sizing recommendation."""
        if not self.cache:
            return
        
        try:
            cache_key = f"kelly_recommendation:{recommendation.symbol}:latest"
            cache_data = {
                "symbol": recommendation.symbol,
                "recommended_size": recommendation.recommended_size,
                "position_percent": recommendation.position_percent,
                "kelly_fraction": recommendation.kelly_fraction,
                "confidence_level": recommendation.confidence_level,
                "reasoning": recommendation.reasoning,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.cache.set_json(cache_key, cache_data, ttl=300)  # 5 minutes
        except Exception as e:
            logger.warning(f"Failed to cache recommendation: {e}")
    
    async def get_kelly_statistics(self) -> Dict:
        """Get Kelly position sizer performance statistics."""
        total_symbols = len(self.kelly_params)
        total_trades = sum(len(trades) for trades in self.trade_history.values())
        
        # Calculate average performance metrics
        if self.kelly_params:
            avg_win_rate = statistics.mean([p.win_rate for p in self.kelly_params.values()])
            avg_kelly_fraction = statistics.mean([p.kelly_fraction for p in self.kelly_params.values()])
        else:
            avg_win_rate = 0.0
            avg_kelly_fraction = 0.0
        
        return {
            "total_symbols_tracked": total_symbols,
            "total_trade_history": total_trades,
            "calculations_performed": self.calculations_performed,
            "recommendations_given": self.recommendations_given,
            "avg_win_rate": avg_win_rate,
            "avg_kelly_fraction": avg_kelly_fraction,
            "min_sample_size": self.min_sample_size,
            "safety_factor": self.safety_factor,
            "max_kelly_fraction": self.max_kelly_fraction
        }


# Global Kelly sizer instance
_kelly_sizer: Optional[KellyPositionSizer] = None


async def get_kelly_position_sizer() -> KellyPositionSizer:
    """Get or create global Kelly position sizer instance."""
    global _kelly_sizer
    if _kelly_sizer is None:
        _kelly_sizer = KellyPositionSizer(
            max_kelly_fraction=0.25,  # Cap at 25%
            safety_factor=0.5,        # Half-Kelly for safety
            min_sample_size=20,       # Need 20+ trades for Kelly
            max_single_position=0.10  # Max 10% per position
        )
        await _kelly_sizer.initialize()
    return _kelly_sizer


async def calculate_optimal_position_size(symbol: str,
                                        signal_confidence: float,
                                        portfolio_value: float,
                                        current_price: float,
                                        stop_loss_price: Optional[float] = None,
                                        strategy: str = "default",
                                        market_regime: str = "unknown") -> PositionSizeRecommendation:
    """
    Calculate optimal position size using Kelly Criterion.
    
    This is the main entry point for position sizing calculations.
    """
    kelly_sizer = await get_kelly_position_sizer()
    return await kelly_sizer.calculate_position_size(
        symbol, signal_confidence, portfolio_value, current_price,
        stop_loss_price, strategy, market_regime
    )