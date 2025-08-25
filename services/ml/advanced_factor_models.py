#!/usr/bin/env python3
"""
Advanced Factor Models - Fama-French-Carhart Five-Factor Model Implementation
PhD-level implementation with dynamic factor loadings and risk-adjusted alpha generation.
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML imports
from sklearn.linear_model import LinearRegression, RidgeRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from trading_common import MarketData, get_settings, get_logger
from trading_common.cache import get_trading_cache

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class FactorLoadings:
    """Factor loadings for a stock."""
    symbol: str
    timestamp: datetime
    
    # Fama-French Five-Factor loadings
    market_beta: float          # Market factor (RMRF)
    size_loading: float         # Small Minus Big (SMB)
    value_loading: float        # High Minus Low (HML)
    profitability_loading: float # Robust Minus Weak (RMW)  
    investment_loading: float   # Conservative Minus Aggressive (CMA)
    
    # Carhart momentum factor
    momentum_loading: float     # Winners Minus Losers (WML)
    
    # Model statistics
    r_squared: float
    alpha: float               # Risk-adjusted alpha
    alpha_t_stat: float        # Statistical significance of alpha
    residual_volatility: float # Idiosyncratic risk
    
    # Factor loadings stability
    loading_stability: float   # How stable the loadings are over time
    model_confidence: float    # Confidence in the factor model fit
    
    def get_expected_return(self, factor_returns: 'FactorReturns') -> float:
        """Calculate expected return based on factor exposures."""
        expected_return = (
            self.market_beta * factor_returns.market_premium +
            self.size_loading * factor_returns.smb +
            self.value_loading * factor_returns.hml +
            self.profitability_loading * factor_returns.rmw +
            self.investment_loading * factor_returns.cma +
            self.momentum_loading * factor_returns.wml
        )
        return expected_return + self.alpha
    
    def get_factor_attribution(self, factor_returns: 'FactorReturns') -> Dict[str, float]:
        """Break down return attribution by factor."""
        return {
            'alpha': self.alpha,
            'market': self.market_beta * factor_returns.market_premium,
            'size': self.size_loading * factor_returns.smb,
            'value': self.value_loading * factor_returns.hml,
            'profitability': self.profitability_loading * factor_returns.rmw,
            'investment': self.investment_loading * factor_returns.cma,
            'momentum': self.momentum_loading * factor_returns.wml
        }


@dataclass
class FactorReturns:
    """Current factor returns for the market."""
    timestamp: datetime
    risk_free_rate: float      # Risk-free rate
    market_return: float       # Market return
    market_premium: float      # Market - Risk-free (RMRF)
    smb: float                 # Small Minus Big
    hml: float                 # High Minus Low  
    rmw: float                 # Robust Minus Weak
    cma: float                 # Conservative Minus Aggressive
    wml: float                 # Winners Minus Losers (Momentum)
    
    # Factor regime indicators
    bull_market_indicator: float    # 0-1, higher = more bullish
    volatility_regime: str         # 'low', 'medium', 'high'
    factor_momentum: Dict[str, float]  # Momentum in each factor


@dataclass
class StockFactorProfile:
    """Comprehensive factor profile for a stock."""
    symbol: str
    current_loadings: FactorLoadings
    historical_loadings: List[FactorLoadings] = field(default_factory=list)
    
    # Factor timing signals
    factor_timing_signals: Dict[str, float] = field(default_factory=dict)
    
    # Risk metrics
    systematic_risk: float = 0.0      # Risk from factor exposures
    idiosyncratic_risk: float = 0.0   # Stock-specific risk
    total_risk: float = 0.0           # Total volatility
    
    # Performance attribution
    performance_attribution: Dict[str, float] = field(default_factory=dict)
    alpha_persistence: float = 0.0    # How persistent the alpha is
    
    def get_risk_adjusted_signal(self) -> float:
        """Get risk-adjusted trading signal based on factor analysis."""
        
        # Base signal from alpha
        alpha_signal = self.current_loadings.alpha * 100  # Scale alpha
        
        # Adjust for alpha significance
        if self.current_loadings.alpha_t_stat > 2.0:
            alpha_signal *= 1.5  # Boost significant alpha
        elif self.current_loadings.alpha_t_stat < 1.0:
            alpha_signal *= 0.5  # Reduce insignificant alpha
        
        # Factor timing adjustments
        timing_adjustment = 0.0
        for factor_name, timing_signal in self.factor_timing_signals.items():
            factor_loading = getattr(self.current_loadings, f"{factor_name}_loading", 0.0)
            timing_adjustment += factor_loading * timing_signal * 0.1
        
        # Risk adjustment
        risk_penalty = self.idiosyncratic_risk * 10  # Penalize high idiosyncratic risk
        
        final_signal = alpha_signal + timing_adjustment - risk_penalty
        
        # Normalize to [-1, 1] range
        return np.tanh(final_signal / 100.0)


class FactorDataConstructor:
    """Constructs factor returns and portfolios from market data."""
    
    def __init__(self):
        self.cache = None
        
        # Factor construction parameters
        self.lookback_window = 252  # 1 year for factor construction
        self.rebalance_frequency = 21  # Rebalance factors monthly
        
        # Size breakpoints (market cap percentiles)
        self.size_breakpoints = [0.5]  # Median split
        
        # Value breakpoints (book-to-market percentiles)  
        self.value_breakpoints = [0.3, 0.7]  # 30th and 70th percentiles
        
        # Profitability breakpoints (ROE percentiles)
        self.profitability_breakpoints = [0.3, 0.7]
        
        # Investment breakpoints (asset growth percentiles)
        self.investment_breakpoints = [0.3, 0.7]
        
        # Momentum lookback period
        self.momentum_window = 252  # 12 months
        self.momentum_skip = 21     # Skip last month
        
    async def initialize(self):
        """Initialize factor data constructor."""
        self.cache = get_trading_cache()
        logger.info("Factor Data Constructor initialized")
    
    async def construct_factor_returns(self, symbols: List[str],
                                     market_data: Dict[str, List[MarketData]],
                                     fundamental_data: Dict[str, Dict]) -> FactorReturns:
        """Construct factor returns using market and fundamental data."""
        
        current_date = datetime.utcnow().date()
        
        # Calculate individual stock metrics
        stock_metrics = {}
        for symbol in symbols:
            if symbol not in market_data or len(market_data[symbol]) < self.lookback_window:
                continue
                
            metrics = await self._calculate_stock_metrics(
                symbol, market_data[symbol], fundamental_data.get(symbol, {})
            )
            if metrics:
                stock_metrics[symbol] = metrics
        
        if len(stock_metrics) < 10:
            logger.warning("Insufficient data for factor construction")
            return self._get_default_factor_returns()
        
        # Construct factor portfolios
        factor_portfolios = await self._construct_factor_portfolios(stock_metrics)
        
        # Calculate factor returns
        factor_returns = await self._calculate_factor_returns(factor_portfolios, stock_metrics)
        
        # Add regime indicators
        factor_returns.bull_market_indicator = await self._calculate_bull_market_indicator(factor_returns)
        factor_returns.volatility_regime = await self._determine_volatility_regime(stock_metrics)
        factor_returns.factor_momentum = await self._calculate_factor_momentum(factor_returns)
        
        logger.info(f"Constructed factor returns: Market={factor_returns.market_premium:.4f}, "
                   f"SMB={factor_returns.smb:.4f}, HML={factor_returns.hml:.4f}")
        
        return factor_returns
    
    async def _calculate_stock_metrics(self, symbol: str, 
                                     data: List[MarketData],
                                     fundamentals: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Calculate metrics needed for factor construction."""
        
        if len(data) < 30:
            return None
            
        df = self._market_data_to_df(data)
        
        try:
            # Market cap (proxy using volume * price)
            market_cap = df['close'].iloc[-1] * df['volume'].iloc[-1]
            
            # Calculate returns
            returns = df['close'].pct_change().dropna()
            if len(returns) < 20:
                return None
                
            current_return = returns.iloc[-1]
            
            # Momentum calculation (12-1 months)
            if len(returns) >= self.momentum_window + self.momentum_skip:
                momentum_start = -(self.momentum_window + self.momentum_skip)
                momentum_end = -self.momentum_skip
                momentum_return = (df['close'].iloc[momentum_end] / df['close'].iloc[momentum_start]) - 1
            else:
                momentum_return = 0.0
            
            # Volatility
            volatility = returns.std() * np.sqrt(252)
            
            # Fundamental metrics (from company data)
            book_to_market = fundamentals.get('pb_ratio', 1.0)
            if book_to_market > 0:
                book_to_market = 1.0 / book_to_market  # Convert P/B to B/M
            else:
                book_to_market = 0.5  # Default
            
            profitability = fundamentals.get('roe', 0.1)  # Return on Equity
            investment = fundamentals.get('revenue_growth', 0.05)  # Asset growth proxy
            
            return {
                'symbol': symbol,
                'market_cap': market_cap,
                'book_to_market': book_to_market,
                'profitability': profitability,
                'investment': investment,
                'momentum': momentum_return,
                'current_return': current_return,
                'volatility': volatility
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate metrics for {symbol}: {e}")
            return None
    
    async def _construct_factor_portfolios(self, stock_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, List[str]]]:
        """Construct factor portfolios based on stock characteristics."""
        
        metrics_df = pd.DataFrame.from_dict(stock_metrics, orient='index')
        
        # Size portfolios (Small vs Big)
        size_breakpoint = metrics_df['market_cap'].quantile(self.size_breakpoints[0])
        small_stocks = metrics_df[metrics_df['market_cap'] <= size_breakpoint].index.tolist()
        big_stocks = metrics_df[metrics_df['market_cap'] > size_breakpoint].index.tolist()
        
        # Value portfolios (High vs Low book-to-market)
        value_low = metrics_df['book_to_market'].quantile(self.value_breakpoints[0])
        value_high = metrics_df['book_to_market'].quantile(self.value_breakpoints[1])
        
        low_value = metrics_df[metrics_df['book_to_market'] <= value_low].index.tolist()
        mid_value = metrics_df[(metrics_df['book_to_market'] > value_low) & 
                              (metrics_df['book_to_market'] < value_high)].index.tolist()
        high_value = metrics_df[metrics_df['book_to_market'] >= value_high].index.tolist()
        
        # Profitability portfolios (Robust vs Weak)
        prof_low = metrics_df['profitability'].quantile(self.profitability_breakpoints[0])
        prof_high = metrics_df['profitability'].quantile(self.profitability_breakpoints[1])
        
        weak_prof = metrics_df[metrics_df['profitability'] <= prof_low].index.tolist()
        robust_prof = metrics_df[metrics_df['profitability'] >= prof_high].index.tolist()
        
        # Investment portfolios (Conservative vs Aggressive)
        inv_low = metrics_df['investment'].quantile(self.investment_breakpoints[0])
        inv_high = metrics_df['investment'].quantile(self.investment_breakpoints[1])
        
        conservative = metrics_df[metrics_df['investment'] <= inv_low].index.tolist()
        aggressive = metrics_df[metrics_df['investment'] >= inv_high].index.tolist()
        
        # Momentum portfolios (Winners vs Losers)
        momentum_low = metrics_df['momentum'].quantile(0.3)
        momentum_high = metrics_df['momentum'].quantile(0.7)
        
        losers = metrics_df[metrics_df['momentum'] <= momentum_low].index.tolist()
        winners = metrics_df[metrics_df['momentum'] >= momentum_high].index.tolist()
        
        return {
            'size': {'small': small_stocks, 'big': big_stocks},
            'value': {'low': low_value, 'mid': mid_value, 'high': high_value},
            'profitability': {'weak': weak_prof, 'robust': robust_prof},
            'investment': {'conservative': conservative, 'aggressive': aggressive},
            'momentum': {'losers': losers, 'winners': winners}
        }
    
    async def _calculate_factor_returns(self, portfolios: Dict[str, Dict[str, List[str]]],
                                      stock_metrics: Dict[str, Dict[str, float]]) -> FactorReturns:
        """Calculate factor returns from portfolios."""
        
        def portfolio_return(stocks: List[str]) -> float:
            """Calculate value-weighted portfolio return."""
            if not stocks:
                return 0.0
                
            total_weight = 0.0
            weighted_return = 0.0
            
            for symbol in stocks:
                if symbol in stock_metrics:
                    weight = stock_metrics[symbol]['market_cap']
                    ret = stock_metrics[symbol]['current_return']
                    
                    weighted_return += weight * ret
                    total_weight += weight
            
            return weighted_return / total_weight if total_weight > 0 else 0.0
        
        # Calculate portfolio returns
        small_return = portfolio_return(portfolios['size']['small'])
        big_return = portfolio_return(portfolios['size']['big'])
        
        high_value_return = portfolio_return(portfolios['value']['high'])
        low_value_return = portfolio_return(portfolios['value']['low'])
        
        robust_return = portfolio_return(portfolios['profitability']['robust'])
        weak_return = portfolio_return(portfolios['profitability']['weak'])
        
        conservative_return = portfolio_return(portfolios['investment']['conservative'])
        aggressive_return = portfolio_return(portfolios['investment']['aggressive'])
        
        winners_return = portfolio_return(portfolios['momentum']['winners'])
        losers_return = portfolio_return(portfolios['momentum']['losers'])
        
        # Market return (value-weighted return of all stocks)
        market_return = portfolio_return(list(stock_metrics.keys()))
        
        # Risk-free rate (placeholder - would get from treasury data)
        risk_free_rate = 0.02 / 252  # 2% annualized, daily
        
        # Calculate factor returns
        market_premium = market_return - risk_free_rate
        smb = small_return - big_return
        hml = high_value_return - low_value_return
        rmw = robust_return - weak_return
        cma = conservative_return - aggressive_return
        wml = winners_return - losers_return
        
        return FactorReturns(
            timestamp=datetime.utcnow(),
            risk_free_rate=risk_free_rate,
            market_return=market_return,
            market_premium=market_premium,
            smb=smb,
            hml=hml,
            rmw=rmw,
            cma=cma,
            wml=wml,
            bull_market_indicator=0.5,  # Will be calculated
            volatility_regime='medium',  # Will be calculated
            factor_momentum={}  # Will be calculated
        )
    
    async def _calculate_bull_market_indicator(self, factor_returns: FactorReturns) -> float:
        """Calculate bull market indicator."""
        # Simple implementation - would use more sophisticated methods
        if factor_returns.market_premium > 0.001:  # Positive market premium
            return 0.8
        elif factor_returns.market_premium > 0:
            return 0.6
        else:
            return 0.3
    
    async def _determine_volatility_regime(self, stock_metrics: Dict[str, Dict[str, float]]) -> str:
        """Determine current volatility regime."""
        volatilities = [metrics['volatility'] for metrics in stock_metrics.values()]
        avg_vol = np.mean(volatilities)
        
        if avg_vol < 0.15:
            return 'low'
        elif avg_vol < 0.25:
            return 'medium'
        else:
            return 'high'
    
    async def _calculate_factor_momentum(self, factor_returns: FactorReturns) -> Dict[str, float]:
        """Calculate momentum in factor returns."""
        # Would use historical factor returns for this calculation
        # For now, return neutral values
        return {
            'market': 0.0,
            'size': 0.0,
            'value': 0.0,
            'profitability': 0.0,
            'investment': 0.0,
            'momentum': 0.0
        }
    
    def _get_default_factor_returns(self) -> FactorReturns:
        """Return default factor returns when construction fails."""
        return FactorReturns(
            timestamp=datetime.utcnow(),
            risk_free_rate=0.00008,  # ~2% annual
            market_return=0.0003,    # ~8% annual
            market_premium=0.00022,  # ~6% annual
            smb=0.0,
            hml=0.0,
            rmw=0.0,
            cma=0.0,
            wml=0.0,
            bull_market_indicator=0.5,
            volatility_regime='medium',
            factor_momentum={}
        )
    
    def _market_data_to_df(self, data: List[MarketData]) -> pd.DataFrame:
        """Convert market data to DataFrame."""
        rows = []
        for md in data:
            rows.append({
                'timestamp': md.timestamp,
                'open': md.open,
                'high': md.high,
                'low': md.low,
                'close': md.close,
                'volume': md.volume
            })
        
        df = pd.DataFrame(rows)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df


class FactorModelEstimator:
    """Estimates factor loadings and alpha for individual stocks."""
    
    def __init__(self):
        self.cache = None
        
        # Estimation parameters
        self.estimation_window = 252  # 1 year of data
        self.min_observations = 60    # Minimum observations required
        self.rolling_window = 126     # 6 months for rolling estimates
        
        # Model selection criteria
        self.use_robust_regression = True
        self.alpha_confidence_level = 0.05  # 95% confidence
        
    async def initialize(self):
        """Initialize the factor model estimator."""
        self.cache = get_trading_cache()
        logger.info("Factor Model Estimator initialized")
    
    async def estimate_factor_loadings(self, symbol: str,
                                     stock_returns: pd.Series,
                                     factor_returns_history: List[FactorReturns]) -> Optional[FactorLoadings]:
        """Estimate factor loadings for a stock."""
        
        if len(stock_returns) < self.min_observations or not factor_returns_history:
            logger.warning(f"Insufficient data for factor loading estimation: {symbol}")
            return None
        
        try:
            # Prepare factor return matrix
            factor_df = self._prepare_factor_dataframe(factor_returns_history)
            
            # Align stock returns with factor returns
            aligned_data = self._align_returns(stock_returns, factor_df)
            
            if len(aligned_data) < self.min_observations:
                logger.warning(f"Insufficient aligned data for {symbol}: {len(aligned_data)} observations")
                return None
            
            # Estimate factor model
            loadings = await self._estimate_loadings_ols(symbol, aligned_data)
            
            # Calculate loading stability
            stability = await self._calculate_loading_stability(symbol, aligned_data)
            loadings.loading_stability = stability
            
            return loadings
            
        except Exception as e:
            logger.error(f"Failed to estimate factor loadings for {symbol}: {e}")
            return None
    
    def _prepare_factor_dataframe(self, factor_returns_history: List[FactorReturns]) -> pd.DataFrame:
        """Prepare factor returns as DataFrame."""
        data = []
        for fr in factor_returns_history:
            data.append({
                'timestamp': fr.timestamp,
                'market_premium': fr.market_premium,
                'smb': fr.smb,
                'hml': fr.hml,
                'rmw': fr.rmw,
                'cma': fr.cma,
                'wml': fr.wml
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df
    
    def _align_returns(self, stock_returns: pd.Series, factor_df: pd.DataFrame) -> pd.DataFrame:
        """Align stock returns with factor returns."""
        
        # Create combined DataFrame
        combined = pd.DataFrame({
            'stock_return': stock_returns,
            'market_premium': factor_df['market_premium'],
            'smb': factor_df['smb'],
            'hml': factor_df['hml'],
            'rmw': factor_df['rmw'],
            'cma': factor_df['cma'],
            'wml': factor_df['wml']
        })
        
        # Remove missing values
        aligned = combined.dropna()
        
        # Use most recent observations if we have too many
        if len(aligned) > self.estimation_window:
            aligned = aligned.iloc[-self.estimation_window:]
        
        return aligned
    
    async def _estimate_loadings_ols(self, symbol: str, data: pd.DataFrame) -> FactorLoadings:
        """Estimate factor loadings using OLS regression."""
        
        # Prepare regression data
        y = data['stock_return'].values
        X = data[['market_premium', 'smb', 'hml', 'rmw', 'cma', 'wml']].values
        
        # Add constant for alpha
        X = sm.add_constant(X)
        
        # Fit regression model
        if self.use_robust_regression:
            model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
            results = model.fit()
        else:
            model = sm.OLS(y, X)
            results = model.fit()
        
        # Extract results
        params = results.params
        t_stats = results.tvalues
        
        alpha = params[0]
        alpha_t_stat = t_stats[0]
        market_beta = params[1]
        size_loading = params[2]
        value_loading = params[3]
        profitability_loading = params[4]
        investment_loading = params[5]
        momentum_loading = params[6]
        
        # Calculate model statistics
        r_squared = results.rsquared
        residual_volatility = np.sqrt(results.mse_resid * 252)  # Annualized
        
        # Model confidence based on R-squared and sample size
        model_confidence = min(r_squared * (len(data) / 252), 0.95)
        
        return FactorLoadings(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            market_beta=market_beta,
            size_loading=size_loading,
            value_loading=value_loading,
            profitability_loading=profitability_loading,
            investment_loading=investment_loading,
            momentum_loading=momentum_loading,
            r_squared=r_squared,
            alpha=alpha,
            alpha_t_stat=alpha_t_stat,
            residual_volatility=residual_volatility,
            loading_stability=0.0,  # Will be calculated
            model_confidence=model_confidence
        )
    
    async def _calculate_loading_stability(self, symbol: str, data: pd.DataFrame) -> float:
        """Calculate how stable the factor loadings are over time."""
        
        if len(data) < self.rolling_window * 2:
            return 0.5  # Default stability
        
        try:
            # Fit rolling regression to see how loadings change
            y = data['stock_return']
            X = data[['market_premium', 'smb', 'hml', 'rmw', 'cma', 'wml']]
            X = sm.add_constant(X)
            
            rolling_model = RollingOLS(y, X, window=self.rolling_window)
            rolling_results = rolling_model.fit()
            
            # Calculate coefficient of variation for each loading
            loading_stabilities = []
            for i, param_name in enumerate(['const', 'market_premium', 'smb', 'hml', 'rmw', 'cma', 'wml']):
                param_series = rolling_results.params.iloc[:, i]
                param_series = param_series.dropna()
                
                if len(param_series) > 5:
                    cv = param_series.std() / (abs(param_series.mean()) + 1e-6)
                    stability = max(0, 1 - cv)  # Lower CV = higher stability
                    loading_stabilities.append(stability)
            
            # Average stability across all loadings
            avg_stability = np.mean(loading_stabilities) if loading_stabilities else 0.5
            return min(max(avg_stability, 0.0), 1.0)
            
        except Exception as e:
            logger.debug(f"Failed to calculate loading stability for {symbol}: {e}")
            return 0.5


class AdvancedFactorService:
    """Main service for advanced factor model analysis."""
    
    def __init__(self):
        self.factor_constructor = FactorDataConstructor()
        self.model_estimator = FactorModelEstimator()
        
        self.cache = None
        
        # Current factor state
        self.current_factor_returns = None
        self.factor_returns_history = []
        
        # Stock factor profiles
        self.stock_profiles = {}  # symbol -> StockFactorProfile
        
        # Update scheduling
        self.last_factor_update = None
        self.factor_update_frequency = timedelta(hours=1)  # Update factors hourly
        
        # Performance tracking
        self.factor_models_fitted = 0
        self.average_r_squared = 0.0
        self.significant_alphas = 0
        
    async def initialize(self):
        """Initialize the advanced factor service."""
        await self.factor_constructor.initialize()
        await self.model_estimator.initialize()
        
        self.cache = get_trading_cache()
        
        logger.info("Advanced Factor Service initialized")
    
    async def update_factor_models(self, symbols: List[str],
                                 market_data: Dict[str, List[MarketData]],
                                 fundamental_data: Dict[str, Dict] = None) -> Dict[str, Any]:
        """Update factor models for all symbols."""
        
        if fundamental_data is None:
            fundamental_data = {}
        
        logger.info(f"Updating factor models for {len(symbols)} symbols")
        
        # Construct current factor returns
        factor_returns = await self.factor_constructor.construct_factor_returns(
            symbols, market_data, fundamental_data
        )
        
        self.current_factor_returns = factor_returns
        self.factor_returns_history.append(factor_returns)
        
        # Keep only recent factor history
        if len(self.factor_returns_history) > 252:
            self.factor_returns_history = self.factor_returns_history[-252:]
        
        # Update individual stock factor loadings
        updated_profiles = {}
        r_squared_values = []
        significant_alphas = 0
        
        for symbol in symbols:
            if symbol not in market_data or len(market_data[symbol]) < 60:
                continue
            
            try:
                # Calculate stock returns
                stock_data = market_data[symbol]
                df = self._market_data_to_df(stock_data)
                stock_returns = df['close'].pct_change().dropna()
                
                # Estimate factor loadings
                loadings = await self.model_estimator.estimate_factor_loadings(
                    symbol, stock_returns, self.factor_returns_history
                )
                
                if loadings:
                    # Create or update stock profile
                    if symbol in self.stock_profiles:
                        profile = self.stock_profiles[symbol]
                        profile.historical_loadings.append(profile.current_loadings)
                        profile.current_loadings = loadings
                    else:
                        profile = StockFactorProfile(
                            symbol=symbol,
                            current_loadings=loadings
                        )
                    
                    # Update risk metrics
                    await self._update_risk_metrics(profile)
                    
                    # Calculate factor timing signals
                    profile.factor_timing_signals = await self._calculate_factor_timing_signals(loadings)
                    
                    # Performance attribution
                    profile.performance_attribution = loadings.get_factor_attribution(factor_returns)
                    
                    self.stock_profiles[symbol] = profile
                    updated_profiles[symbol] = profile
                    
                    # Track statistics
                    r_squared_values.append(loadings.r_squared)
                    if abs(loadings.alpha_t_stat) > 2.0:  # Significant at 5% level
                        significant_alphas += 1
                    
                    self.factor_models_fitted += 1
                    
            except Exception as e:
                logger.warning(f"Failed to update factor model for {symbol}: {e}")
        
        # Update service statistics
        if r_squared_values:
            self.average_r_squared = np.mean(r_squared_values)
        self.significant_alphas = significant_alphas
        
        # Cache factor returns and profiles
        await self._cache_factor_data()
        
        update_summary = {
            'symbols_updated': len(updated_profiles),
            'average_r_squared': self.average_r_squared,
            'significant_alphas': significant_alphas,
            'current_factor_returns': {
                'market_premium': factor_returns.market_premium,
                'smb': factor_returns.smb,
                'hml': factor_returns.hml,
                'rmw': factor_returns.rmw,
                'cma': factor_returns.cma,
                'wml': factor_returns.wml
            }
        }
        
        logger.info(f"Updated {len(updated_profiles)} factor models. "
                   f"Avg R²={self.average_r_squared:.3f}, "
                   f"Significant alphas={significant_alphas}")
        
        return update_summary
    
    async def get_factor_signals(self, symbols: List[str]) -> Dict[str, float]:
        """Get factor-based trading signals for symbols."""
        
        signals = {}
        
        for symbol in symbols:
            if symbol in self.stock_profiles:
                profile = self.stock_profiles[symbol]
                signal = profile.get_risk_adjusted_signal()
                signals[symbol] = signal
            else:
                signals[symbol] = 0.0  # Neutral signal for unknown symbols
        
        return signals
    
    async def get_factor_attribution(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed factor attribution for a symbol."""
        
        if symbol not in self.stock_profiles:
            return None
        
        profile = self.stock_profiles[symbol]
        loadings = profile.current_loadings
        
        attribution = {
            'symbol': symbol,
            'timestamp': loadings.timestamp.isoformat(),
            'alpha': {
                'value': loadings.alpha,
                't_statistic': loadings.alpha_t_stat,
                'significant': abs(loadings.alpha_t_stat) > 2.0
            },
            'factor_loadings': {
                'market_beta': loadings.market_beta,
                'size_loading': loadings.size_loading,
                'value_loading': loadings.value_loading,
                'profitability_loading': loadings.profitability_loading,
                'investment_loading': loadings.investment_loading,
                'momentum_loading': loadings.momentum_loading
            },
            'model_statistics': {
                'r_squared': loadings.r_squared,
                'residual_volatility': loadings.residual_volatility,
                'loading_stability': loadings.loading_stability,
                'model_confidence': loadings.model_confidence
            },
            'risk_decomposition': {
                'systematic_risk': profile.systematic_risk,
                'idiosyncratic_risk': profile.idiosyncratic_risk,
                'total_risk': profile.total_risk
            },
            'performance_attribution': profile.performance_attribution,
            'factor_timing_signals': profile.factor_timing_signals
        }
        
        return attribution
    
    async def get_portfolio_factor_exposure(self, portfolio: Dict[str, float]) -> Dict[str, Any]:
        """Calculate factor exposures for a portfolio."""
        
        total_beta = 0.0
        total_size = 0.0
        total_value = 0.0
        total_profitability = 0.0
        total_investment = 0.0
        total_momentum = 0.0
        total_alpha = 0.0
        
        total_weight = sum(abs(weight) for weight in portfolio.values())
        
        if total_weight == 0:
            return {}
        
        for symbol, weight in portfolio.items():
            if symbol in self.stock_profiles:
                loadings = self.stock_profiles[symbol].current_loadings
                normalized_weight = weight / total_weight
                
                total_beta += normalized_weight * loadings.market_beta
                total_size += normalized_weight * loadings.size_loading
                total_value += normalized_weight * loadings.value_loading
                total_profitability += normalized_weight * loadings.profitability_loading
                total_investment += normalized_weight * loadings.investment_loading
                total_momentum += normalized_weight * loadings.momentum_loading
                total_alpha += normalized_weight * loadings.alpha
        
        return {
            'portfolio_factor_exposures': {
                'market_beta': total_beta,
                'size_loading': total_size,
                'value_loading': total_value,
                'profitability_loading': total_profitability,
                'investment_loading': total_investment,
                'momentum_loading': total_momentum
            },
            'portfolio_alpha': total_alpha,
            'expected_return': total_alpha + total_beta * self.current_factor_returns.market_premium if self.current_factor_returns else total_alpha
        }
    
    async def _update_risk_metrics(self, profile: StockFactorProfile):
        """Update risk metrics for a stock profile."""
        
        loadings = profile.current_loadings
        
        if self.current_factor_returns:
            # Calculate systematic risk from factor exposures
            # (simplified - would use factor covariance matrix)
            factor_variances = {
                'market': 0.04,  # 20% annual vol
                'size': 0.01,    # 10% annual vol
                'value': 0.01,
                'profitability': 0.005,
                'investment': 0.005,
                'momentum': 0.015
            }
            
            systematic_variance = (
                (loadings.market_beta ** 2) * factor_variances['market'] +
                (loadings.size_loading ** 2) * factor_variances['size'] +
                (loadings.value_loading ** 2) * factor_variances['value'] +
                (loadings.profitability_loading ** 2) * factor_variances['profitability'] +
                (loadings.investment_loading ** 2) * factor_variances['investment'] +
                (loadings.momentum_loading ** 2) * factor_variances['momentum']
            )
            
            profile.systematic_risk = np.sqrt(systematic_variance)
            profile.idiosyncratic_risk = loadings.residual_volatility
            profile.total_risk = np.sqrt(systematic_variance + loadings.residual_volatility ** 2)
    
    async def _calculate_factor_timing_signals(self, loadings: FactorLoadings) -> Dict[str, float]:
        """Calculate factor timing signals."""
        
        # Placeholder implementation - would use factor momentum, valuation, etc.
        return {
            'market': 0.0,
            'size': 0.1,      # Slight positive for small-cap
            'value': 0.05,    # Slight positive for value
            'profitability': 0.1,  # Positive for high profitability
            'investment': -0.05,   # Slight negative for aggressive investment
            'momentum': 0.0
        }
    
    def _market_data_to_df(self, data: List[MarketData]) -> pd.DataFrame:
        """Convert market data to DataFrame."""
        rows = []
        for md in data:
            rows.append({
                'timestamp': md.timestamp,
                'close': md.close
            })
        
        df = pd.DataFrame(rows)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df
    
    async def _cache_factor_data(self):
        """Cache factor returns and model results."""
        
        if self.cache and self.current_factor_returns:
            # Cache current factor returns
            factor_data = {
                'timestamp': self.current_factor_returns.timestamp.isoformat(),
                'market_premium': self.current_factor_returns.market_premium,
                'smb': self.current_factor_returns.smb,
                'hml': self.current_factor_returns.hml,
                'rmw': self.current_factor_returns.rmw,
                'cma': self.current_factor_returns.cma,
                'wml': self.current_factor_returns.wml,
                'bull_market_indicator': self.current_factor_returns.bull_market_indicator,
                'volatility_regime': self.current_factor_returns.volatility_regime
            }
            
            await self.cache.set_json("current_factor_returns", factor_data, ttl=3600)
            
            # Cache service statistics
            service_stats = {
                'factor_models_fitted': self.factor_models_fitted,
                'average_r_squared': self.average_r_squared,
                'significant_alphas': self.significant_alphas,
                'symbols_tracked': len(self.stock_profiles)
            }
            
            await self.cache.set_json("factor_service_stats", service_stats, ttl=3600)
    
    async def get_service_performance(self) -> Dict[str, Any]:
        """Get service performance metrics."""
        
        return {
            'factor_models_fitted': self.factor_models_fitted,
            'symbols_tracked': len(self.stock_profiles),
            'average_r_squared': self.average_r_squared,
            'significant_alphas': self.significant_alphas,
            'factor_update_frequency': self.factor_update_frequency.total_seconds() / 3600,
            'current_factor_regime': {
                'bull_market_indicator': self.current_factor_returns.bull_market_indicator if self.current_factor_returns else 0.5,
                'volatility_regime': self.current_factor_returns.volatility_regime if self.current_factor_returns else 'unknown'
            }
        }


# Global factor service instance
factor_service: Optional[AdvancedFactorService] = None


async def get_factor_service() -> AdvancedFactorService:
    """Get or create factor service instance."""
    global factor_service
    if factor_service is None:
        factor_service = AdvancedFactorService()
        await factor_service.initialize()
    return factor_service