#!/usr/bin/env python3
"""
Company Intelligence Service - Comprehensive company data aggregation and tracking
Maintains up-to-date company files with all relevant financial, operational, and market data.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import os
from decimal import Decimal
import hashlib

from trading_common import get_settings, get_logger
from trading_common.cache import get_trading_cache

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class FinancialMetrics:
    """Key financial metrics for a company."""
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None  # Return on Equity
    roa: Optional[float] = None  # Return on Assets
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    free_cash_flow: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    beta: Optional[float] = None
    shares_outstanding: Optional[int] = None
    float_shares: Optional[int] = None
    insider_ownership: Optional[float] = None
    institutional_ownership: Optional[float] = None
    short_interest: Optional[float] = None
    days_to_cover: Optional[float] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EarningsData:
    """Earnings and guidance information."""
    next_earnings_date: Optional[datetime] = None
    last_earnings_date: Optional[datetime] = None
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    earnings_surprise: Optional[float] = None
    revenue_surprise: Optional[float] = None
    guidance_raised: Optional[bool] = None
    guidance_lowered: Optional[bool] = None
    analyst_upgrades: int = 0
    analyst_downgrades: int = 0
    analyst_rating_avg: Optional[float] = None
    price_target_avg: Optional[float] = None
    price_target_high: Optional[float] = None
    price_target_low: Optional[float] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BusinessIntelligence:
    """Business and operational intelligence."""
    industry: Optional[str] = None
    sector: Optional[str] = None
    business_description: Optional[str] = None
    key_executives: List[Dict[str, str]] = field(default_factory=list)
    headquarters: Optional[str] = None
    employees: Optional[int] = None
    website: Optional[str] = None
    competitors: List[str] = field(default_factory=list)
    key_products: List[str] = field(default_factory=list)
    revenue_streams: List[Dict[str, float]] = field(default_factory=list)  # segment -> % of revenue
    geographic_exposure: List[Dict[str, float]] = field(default_factory=list)  # region -> % of revenue
    major_customers: List[str] = field(default_factory=list)
    key_suppliers: List[str] = field(default_factory=list)
    regulatory_issues: List[Dict[str, str]] = field(default_factory=list)
    esg_score: Optional[float] = None
    recent_news_sentiment: Optional[float] = None
    social_sentiment: Optional[float] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators."""
    price_current: Optional[float] = None
    price_52w_high: Optional[float] = None
    price_52w_low: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    volume_avg: Optional[int] = None
    volume_current: Optional[int] = None
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    trend_direction: Optional[str] = None  # 'bullish', 'bearish', 'sideways'
    volatility: Optional[float] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RiskFactors:
    """Risk assessment and factors."""
    financial_risk_score: Optional[float] = None  # 0-10, higher = more risk
    operational_risk_score: Optional[float] = None
    market_risk_score: Optional[float] = None
    liquidity_risk_score: Optional[float] = None
    credit_risk_score: Optional[float] = None
    regulatory_risk_score: Optional[float] = None
    overall_risk_score: Optional[float] = None
    key_risk_factors: List[str] = field(default_factory=list)
    risk_mitigation_factors: List[str] = field(default_factory=list)
    debt_maturity_profile: List[Dict[str, float]] = field(default_factory=list)  # year -> amount
    litigation_exposure: Optional[float] = None
    currency_exposure: List[str] = field(default_factory=list)
    commodity_exposure: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class InsiderActivity:
    """Insider trading and ownership activity."""
    recent_insider_buys: List[Dict[str, Any]] = field(default_factory=list)
    recent_insider_sells: List[Dict[str, Any]] = field(default_factory=list)
    insider_sentiment_score: Optional[float] = None  # -1 to 1
    insider_ownership_changes: List[Dict[str, Any]] = field(default_factory=list)
    institutional_flows: List[Dict[str, Any]] = field(default_factory=list)
    major_holders: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CompanyIntelligenceProfile:
    """Comprehensive company intelligence profile."""
    symbol: str
    company_name: str
    exchange: str
    
    # Core data components
    financial_metrics: FinancialMetrics = field(default_factory=FinancialMetrics)
    earnings_data: EarningsData = field(default_factory=EarningsData)
    business_intelligence: BusinessIntelligence = field(default_factory=BusinessIntelligence)
    technical_indicators: TechnicalIndicators = field(default_factory=TechnicalIndicators)
    risk_factors: RiskFactors = field(default_factory=RiskFactors)
    insider_activity: InsiderActivity = field(default_factory=InsiderActivity)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    data_quality_score: float = 0.0  # 0-1, completeness of data
    next_update_due: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=1))
    
    def calculate_data_quality_score(self) -> float:
        """Calculate how complete the data is."""
        total_fields = 0
        populated_fields = 0
        
        # Check each dataclass for populated fields
        for component in [self.financial_metrics, self.earnings_data, self.business_intelligence, 
                         self.technical_indicators, self.risk_factors, self.insider_activity]:
            for field_name, field_value in asdict(component).items():
                if field_name != 'last_updated':  # Skip metadata fields
                    total_fields += 1
                    if field_value is not None and field_value != [] and field_value != {}:
                        populated_fields += 1
        
        self.data_quality_score = populated_fields / total_fields if total_fields > 0 else 0.0
        return self.data_quality_score
    
    def needs_update(self) -> bool:
        """Check if profile needs updating."""
        return datetime.utcnow() >= self.next_update_due
    
    def get_investment_thesis(self) -> Dict[str, Any]:
        """Generate investment thesis based on available data."""
        thesis = {
            'bullish_factors': [],
            'bearish_factors': [],
            'overall_sentiment': 'neutral',
            'confidence': 0.5
        }
        
        # Analyze financial metrics
        if self.financial_metrics.pe_ratio and self.financial_metrics.pe_ratio < 15:
            thesis['bullish_factors'].append(f"Low P/E ratio ({self.financial_metrics.pe_ratio:.1f})")
        elif self.financial_metrics.pe_ratio and self.financial_metrics.pe_ratio > 30:
            thesis['bearish_factors'].append(f"High P/E ratio ({self.financial_metrics.pe_ratio:.1f})")
        
        if self.financial_metrics.revenue_growth and self.financial_metrics.revenue_growth > 0.15:
            thesis['bullish_factors'].append(f"Strong revenue growth ({self.financial_metrics.revenue_growth*100:.1f}%)")
        elif self.financial_metrics.revenue_growth and self.financial_metrics.revenue_growth < -0.05:
            thesis['bearish_factors'].append(f"Declining revenue ({self.financial_metrics.revenue_growth*100:.1f}%)")
        
        # Analyze earnings data
        if self.earnings_data.earnings_surprise and self.earnings_data.earnings_surprise > 0.05:
            thesis['bullish_factors'].append(f"Recent earnings beat ({self.earnings_data.earnings_surprise*100:.1f}%)")
        elif self.earnings_data.earnings_surprise and self.earnings_data.earnings_surprise < -0.05:
            thesis['bearish_factors'].append(f"Recent earnings miss ({self.earnings_data.earnings_surprise*100:.1f}%)")
        
        # Analyze technical indicators
        if (self.technical_indicators.price_current and self.technical_indicators.sma_50 and 
            self.technical_indicators.price_current > self.technical_indicators.sma_50):
            thesis['bullish_factors'].append("Price above 50-day moving average")
        
        # Analyze insider activity
        if self.insider_activity.insider_sentiment_score and self.insider_activity.insider_sentiment_score > 0.3:
            thesis['bullish_factors'].append("Positive insider sentiment")
        elif self.insider_activity.insider_sentiment_score and self.insider_activity.insider_sentiment_score < -0.3:
            thesis['bearish_factors'].append("Negative insider sentiment")
        
        # Determine overall sentiment
        bullish_count = len(thesis['bullish_factors'])
        bearish_count = len(thesis['bearish_factors'])
        
        if bullish_count > bearish_count * 1.5:
            thesis['overall_sentiment'] = 'bullish'
            thesis['confidence'] = min(0.9, 0.5 + (bullish_count - bearish_count) * 0.1)
        elif bearish_count > bullish_count * 1.5:
            thesis['overall_sentiment'] = 'bearish'
            thesis['confidence'] = min(0.9, 0.5 + (bearish_count - bullish_count) * 0.1)
        
        return thesis


class CompanyDataCollector:
    """Collects comprehensive company data from multiple sources."""
    
    def __init__(self):
        self.session = None
        self.data_sources = {
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
            'finnhub': os.getenv('FINNHUB_API_KEY'),
            'polygon': os.getenv('POLYGON_API_KEY'),
            'fmp': os.getenv('FMP_API_KEY')  # Financial Modeling Prep
        }
        
    async def initialize(self):
        """Initialize data collector."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        logger.info("Company data collector initialized")
    
    async def collect_financial_metrics(self, symbol: str) -> FinancialMetrics:
        """Collect financial metrics from various sources."""
        metrics = FinancialMetrics()
        
        # Try multiple sources for robustness
        tasks = [
            self._collect_alpha_vantage_fundamentals(symbol),
            self._collect_finnhub_metrics(symbol),
            self._collect_polygon_fundamentals(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results (later sources override earlier ones if both have data)
        for result in results:
            if isinstance(result, dict):
                for key, value in result.items():
                    if value is not None and hasattr(metrics, key):
                        setattr(metrics, key, value)
        
        return metrics
    
    async def _collect_alpha_vantage_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Collect fundamentals from Alpha Vantage."""
        if not self.data_sources['alpha_vantage']:
            return {}
            
        try:
            # Company Overview
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.data_sources['alpha_vantage']
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    return {}
                    
                data = await response.json()
                
                return {
                    'market_cap': self._safe_float(data.get('MarketCapitalization')),
                    'pe_ratio': self._safe_float(data.get('PERatio')),
                    'pb_ratio': self._safe_float(data.get('PriceToBookRatio')),
                    'ps_ratio': self._safe_float(data.get('PriceToSalesRatioTTM')),
                    'ev_ebitda': self._safe_float(data.get('EVToEBITDA')),
                    'debt_to_equity': self._safe_float(data.get('DebtToEquityRatio')),
                    'roe': self._safe_float(data.get('ReturnOnEquityTTM')),
                    'roa': self._safe_float(data.get('ReturnOnAssetsTTM')),
                    'gross_margin': self._safe_float(data.get('GrossProfitTTM')),
                    'operating_margin': self._safe_float(data.get('OperatingMarginTTM')),
                    'net_margin': self._safe_float(data.get('ProfitMargin')),
                    'revenue_growth': self._safe_float(data.get('RevenueGrowthTTM')),
                    'earnings_growth': self._safe_float(data.get('QuarterlyEarningsGrowthYOY')),
                    'dividend_yield': self._safe_float(data.get('DividendYield')),
                    'beta': self._safe_float(data.get('Beta')),
                    'shares_outstanding': self._safe_int(data.get('SharesOutstanding'))
                }
                
        except Exception as e:
            logger.warning(f"Alpha Vantage fundamentals error for {symbol}: {e}")
            return {}
    
    async def _collect_finnhub_metrics(self, symbol: str) -> Dict[str, Any]:
        """Collect metrics from Finnhub."""
        if not self.data_sources['finnhub']:
            return {}
            
        try:
            # Basic financials
            url = 'https://finnhub.io/api/v1/stock/metric'
            params = {
                'symbol': symbol,
                'metric': 'all',
                'token': self.data_sources['finnhub']
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    return {}
                    
                data = await response.json()
                metric_data = data.get('metric', {})
                
                return {
                    'market_cap': metric_data.get('marketCapitalization'),
                    'enterprise_value': metric_data.get('enterpriseValue'),
                    'pe_ratio': metric_data.get('peBasicExclExtraTTM'),
                    'pb_ratio': metric_data.get('pbAnnual'),
                    'ev_ebitda': metric_data.get('evEbitdaTTM'),
                    'roe': metric_data.get('roeTTM'),
                    'roa': metric_data.get('roaTTM'),
                    'free_cash_flow': metric_data.get('fcfTTM'),
                    'beta': metric_data.get('beta'),
                    'float_shares': metric_data.get('floatShares'),
                    'insider_ownership': metric_data.get('insiderOwnership'),
                    'institutional_ownership': metric_data.get('institutionalOwnership')
                }
                
        except Exception as e:
            logger.warning(f"Finnhub metrics error for {symbol}: {e}")
            return {}
    
    async def _collect_polygon_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Collect fundamentals from Polygon."""
        if not self.data_sources['polygon']:
            return {}
            
        try:
            # Company details
            url = f'https://api.polygon.io/v3/reference/tickers/{symbol}'
            params = {'apikey': self.data_sources['polygon']}
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    return {}
                    
                data = await response.json()
                results = data.get('results', {})
                
                return {
                    'market_cap': results.get('market_cap'),
                    'shares_outstanding': results.get('share_class_shares_outstanding'),
                    'dividend_yield': results.get('dividend_yield')
                }
                
        except Exception as e:
            logger.warning(f"Polygon fundamentals error for {symbol}: {e}")
            return {}
    
    async def collect_earnings_data(self, symbol: str) -> EarningsData:
        """Collect earnings and analyst data."""
        earnings = EarningsData()
        
        # Collect from multiple sources
        tasks = [
            self._collect_alpha_vantage_earnings(symbol),
            self._collect_finnhub_earnings(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results
        for result in results:
            if isinstance(result, dict):
                for key, value in result.items():
                    if value is not None and hasattr(earnings, key):
                        setattr(earnings, key, value)
        
        return earnings
    
    async def _collect_alpha_vantage_earnings(self, symbol: str) -> Dict[str, Any]:
        """Collect earnings data from Alpha Vantage."""
        if not self.data_sources['alpha_vantage']:
            return {}
            
        try:
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'EARNINGS',
                'symbol': symbol,
                'apikey': self.data_sources['alpha_vantage']
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    return {}
                    
                data = await response.json()
                quarterly_earnings = data.get('quarterlyEarnings', [])
                
                if quarterly_earnings:
                    latest = quarterly_earnings[0]
                    return {
                        'last_earnings_date': self._safe_date(latest.get('reportedDate')),
                        'eps_actual': self._safe_float(latest.get('reportedEPS')),
                        'eps_estimate': self._safe_float(latest.get('estimatedEPS')),
                        'earnings_surprise': self._safe_float(latest.get('surprise'))
                    }
                
        except Exception as e:
            logger.warning(f"Alpha Vantage earnings error for {symbol}: {e}")
            
        return {}
    
    async def _collect_finnhub_earnings(self, symbol: str) -> Dict[str, Any]:
        """Collect earnings data from Finnhub."""
        if not self.data_sources['finnhub']:
            return {}
            
        try:
            # Earnings calendar
            url = 'https://finnhub.io/api/v1/calendar/earnings'
            params = {
                'symbol': symbol,
                'token': self.data_sources['finnhub']
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    return {}
                    
                data = await response.json()
                earnings_calendar = data.get('earningsCalendar', [])
                
                if earnings_calendar:
                    next_earnings = earnings_calendar[0]
                    return {
                        'next_earnings_date': self._safe_date(next_earnings.get('date')),
                        'eps_estimate': self._safe_float(next_earnings.get('epsEstimate'))
                    }
                
        except Exception as e:
            logger.warning(f"Finnhub earnings error for {symbol}: {e}")
            
        return {}
    
    async def collect_business_intelligence(self, symbol: str) -> BusinessIntelligence:
        """Collect business and operational intelligence."""
        intel = BusinessIntelligence()
        
        # This would integrate with multiple data sources
        # For now, implementing basic structure
        
        try:
            # Get company profile from Alpha Vantage
            if self.data_sources['alpha_vantage']:
                url = 'https://www.alphavantage.co/query'
                params = {
                    'function': 'OVERVIEW',
                    'symbol': symbol,
                    'apikey': self.data_sources['alpha_vantage']
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        intel.industry = data.get('Industry')
                        intel.sector = data.get('Sector')
                        intel.business_description = data.get('Description')
                        intel.employees = self._safe_int(data.get('FullTimeEmployees'))
                        intel.headquarters = f"{data.get('Address', '')}, {data.get('Country', '')}"
                        
        except Exception as e:
            logger.warning(f"Business intelligence collection error for {symbol}: {e}")
        
        return intel
    
    async def collect_technical_indicators(self, symbol: str) -> TechnicalIndicators:
        """Collect technical analysis indicators."""
        tech = TechnicalIndicators()
        
        try:
            if self.data_sources['alpha_vantage']:
                # Get daily price data
                url = 'https://www.alphavantage.co/query'
                params = {
                    'function': 'DAILY',
                    'symbol': symbol,
                    'apikey': self.data_sources['alpha_vantage']
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        time_series = data.get('Time Series (Daily)', {})
                        
                        if time_series:
                            # Get latest price
                            latest_date = max(time_series.keys())
                            latest_data = time_series[latest_date]
                            tech.price_current = self._safe_float(latest_data.get('4. close'))
                            tech.volume_current = self._safe_int(latest_data.get('5. volume'))
                            
                            # Calculate 52-week high/low
                            prices = [self._safe_float(day_data.get('4. close')) for day_data in time_series.values()]
                            prices = [p for p in prices if p is not None]
                            
                            if prices:
                                tech.price_52w_high = max(prices[:252])  # 252 trading days
                                tech.price_52w_low = min(prices[:252])
                
        except Exception as e:
            logger.warning(f"Technical indicators collection error for {symbol}: {e}")
        
        return tech
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float."""
        if value is None or value == 'None' or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _safe_int(self, value) -> Optional[int]:
        """Safely convert value to int."""
        if value is None or value == 'None' or value == '':
            return None
        try:
            return int(float(value))  # Convert via float first to handle strings like "1.5E+9"
        except (ValueError, TypeError):
            return None
    
    def _safe_date(self, value) -> Optional[datetime]:
        """Safely convert value to datetime."""
        if value is None or value == 'None' or value == '':
            return None
        try:
            return datetime.strptime(value, '%Y-%m-%d')
        except (ValueError, TypeError):
            return None
    
    async def close(self):
        """Close data collector."""
        if self.session:
            await self.session.close()


class CompanyIntelligenceService:
    """Main company intelligence service."""
    
    def __init__(self):
        self.cache = None
        self.collector = CompanyDataCollector()
        
        # In-memory cache for frequently accessed profiles
        self.profile_cache = {}
        self.cache_expiry = {}
        
        # Tracking
        self.profiles_created = 0
        self.profiles_updated = 0
        self.update_queue = asyncio.Queue(maxsize=1000)
        
        # Update scheduling
        self.is_running = False
        self.update_intervals = {
            'financial_metrics': timedelta(hours=6),
            'earnings_data': timedelta(hours=2),
            'business_intelligence': timedelta(days=7),
            'technical_indicators': timedelta(minutes=15),
            'risk_factors': timedelta(hours=12),
            'insider_activity': timedelta(hours=4)
        }
    
    async def initialize(self):
        """Initialize the intelligence service."""
        self.cache = get_trading_cache()
        await self.collector.initialize()
        
        # Start background update tasks
        self.is_running = True
        asyncio.create_task(self._process_update_queue())
        asyncio.create_task(self._schedule_periodic_updates())
        
        logger.info("Company Intelligence Service initialized")
    
    async def get_company_profile(self, symbol: str, force_update: bool = False) -> CompanyIntelligenceProfile:
        """Get comprehensive company profile."""
        
        # Check in-memory cache first
        if not force_update and symbol in self.profile_cache:
            if symbol in self.cache_expiry and datetime.utcnow() < self.cache_expiry[symbol]:
                return self.profile_cache[symbol]
        
        # Check persistent cache
        if not force_update:
            cached_profile = await self._get_cached_profile(symbol)
            if cached_profile and not cached_profile.needs_update():
                self.profile_cache[symbol] = cached_profile
                self.cache_expiry[symbol] = datetime.utcnow() + timedelta(minutes=30)
                return cached_profile
        
        # Create or update profile
        profile = await self._create_or_update_profile(symbol)
        
        # Cache the profile
        await self._cache_profile(profile)
        self.profile_cache[symbol] = profile
        self.cache_expiry[symbol] = datetime.utcnow() + timedelta(minutes=30)
        
        return profile
    
    async def _create_or_update_profile(self, symbol: str) -> CompanyIntelligenceProfile:
        """Create or update a company profile."""
        
        # Try to get existing profile
        existing_profile = await self._get_cached_profile(symbol)
        
        if existing_profile:
            profile = existing_profile
            self.profiles_updated += 1
        else:
            # Create new profile
            profile = CompanyIntelligenceProfile(
                symbol=symbol,
                company_name=f"{symbol} Corp",  # Would get actual name from API
                exchange="NASDAQ"  # Would determine actual exchange
            )
            self.profiles_created += 1
        
        # Collect data from all sources concurrently
        logger.info(f"Updating company profile for {symbol}")
        
        tasks = [
            self.collector.collect_financial_metrics(symbol),
            self.collector.collect_earnings_data(symbol),
            self.collector.collect_business_intelligence(symbol),
            self.collector.collect_technical_indicators(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update profile components
        if isinstance(results[0], FinancialMetrics):
            profile.financial_metrics = results[0]
        if isinstance(results[1], EarningsData):
            profile.earnings_data = results[1]
        if isinstance(results[2], BusinessIntelligence):
            profile.business_intelligence = results[2]
        if isinstance(results[3], TechnicalIndicators):
            profile.technical_indicators = results[3]
        
        # Update metadata
        profile.last_updated = datetime.utcnow()
        profile.next_update_due = datetime.utcnow() + timedelta(hours=1)  # Default update interval
        profile.calculate_data_quality_score()
        
        logger.info(f"Updated profile for {symbol} - data quality: {profile.data_quality_score:.2f}")
        
        return profile
    
    async def _get_cached_profile(self, symbol: str) -> Optional[CompanyIntelligenceProfile]:
        """Get profile from cache."""
        if not self.cache:
            return None
            
        try:
            cache_key = f"company_profile:{symbol}"
            cached_data = await self.cache.get_json(cache_key)
            
            if cached_data:
                # Reconstruct profile from cached data
                profile = CompanyIntelligenceProfile(**cached_data)
                return profile
                
        except Exception as e:
            logger.warning(f"Failed to get cached profile for {symbol}: {e}")
        
        return None
    
    async def _cache_profile(self, profile: CompanyIntelligenceProfile):
        """Cache company profile."""
        if not self.cache:
            return
            
        try:
            cache_key = f"company_profile:{profile.symbol}"
            profile_data = asdict(profile)
            
            # Convert datetime objects to strings for JSON serialization
            self._serialize_datetimes(profile_data)
            
            await self.cache.set_json(cache_key, profile_data, ttl=3600)  # 1 hour TTL
            
        except Exception as e:
            logger.warning(f"Failed to cache profile for {profile.symbol}: {e}")
    
    def _serialize_datetimes(self, data: Any):
        """Recursively convert datetime objects to strings."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, datetime):
                    data[key] = value.isoformat()
                elif isinstance(value, (dict, list)):
                    self._serialize_datetimes(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, datetime):
                    data[i] = item.isoformat()
                elif isinstance(item, (dict, list)):
                    self._serialize_datetimes(item)
    
    async def queue_update(self, symbol: str, priority: bool = False):
        """Queue a profile update."""
        try:
            if priority:
                # For priority updates, put at front of queue
                await self.update_queue.put((0, symbol))  # 0 = high priority
            else:
                await self.update_queue.put((1, symbol))  # 1 = normal priority
        except asyncio.QueueFull:
            logger.warning(f"Update queue full, skipping {symbol}")
    
    async def _process_update_queue(self):
        """Process profile updates from queue."""
        while self.is_running:
            try:
                priority, symbol = await asyncio.wait_for(self.update_queue.get(), timeout=1.0)
                
                # Update the profile
                await self.get_company_profile(symbol, force_update=True)
                
                # Mark task as done
                self.update_queue.task_done()
                
                # Small delay to prevent overwhelming APIs
                await asyncio.sleep(1.0)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing profile update: {e}")
    
    async def _schedule_periodic_updates(self):
        """Schedule periodic updates for all tracked symbols."""
        while self.is_running:
            try:
                # Wait 5 minutes between scheduling cycles
                await asyncio.sleep(300)
                
                # Get all symbols that need updating
                symbols_to_update = await self._get_symbols_needing_update()
                
                for symbol in symbols_to_update:
                    await self.queue_update(symbol)
                    
                logger.debug(f"Scheduled updates for {len(symbols_to_update)} symbols")
                
            except Exception as e:
                logger.error(f"Error in periodic update scheduling: {e}")
    
    async def _get_symbols_needing_update(self) -> List[str]:
        """Get symbols that need profile updates."""
        # This would query the cache or database for symbols that need updating
        # For now, return a sample list
        return []  # Would return actual symbols from watchlist/portfolio
    
    async def search_companies(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """Search for companies by name or symbol."""
        # This would implement company search functionality
        # Return format: [{"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"}]
        return []
    
    async def get_sector_analysis(self, sector: str) -> Dict[str, Any]:
        """Get analysis for a specific sector."""
        # This would aggregate data for all companies in a sector
        return {
            'sector': sector,
            'total_companies': 0,
            'avg_pe_ratio': 0.0,
            'avg_revenue_growth': 0.0,
            'top_performers': [],
            'underperformers': [],
            'sector_sentiment': 'neutral'
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics."""
        return {
            'profiles_created': self.profiles_created,
            'profiles_updated': self.profiles_updated,
            'cached_profiles': len(self.profile_cache),
            'update_queue_size': self.update_queue.qsize(),
            'is_running': self.is_running,
            'data_sources_available': len([k for k, v in self.collector.data_sources.items() if v])
        }
    
    async def stop(self):
        """Stop the intelligence service."""
        self.is_running = False
        await self.collector.close()
        logger.info("Company Intelligence Service stopped")


# Global intelligence service instance
intelligence_service: Optional[CompanyIntelligenceService] = None


async def get_company_intelligence_service() -> CompanyIntelligenceService:
    """Get or create company intelligence service instance."""
    global intelligence_service
    if intelligence_service is None:
        intelligence_service = CompanyIntelligenceService()
        await intelligence_service.initialize()
    return intelligence_service