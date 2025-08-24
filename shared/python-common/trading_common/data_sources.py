"""Dynamic data source management with hot-swappable API keys."""

import os
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from functools import lru_cache

from pydantic import BaseSettings, Field
from .logging import get_logger
from .exceptions import ConfigError

logger = get_logger(__name__)


class DataSourceStatus(Enum):
    """Data source availability status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    DISABLED = "disabled"


class DataSourceType(Enum):
    """Types of data sources."""
    MARKET_DATA = "market_data"
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    BROKER = "broker"
    AI_MODEL = "ai_model"


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    name: str
    type: DataSourceType
    required_keys: List[str]
    optional_keys: List[str] = field(default_factory=list)
    enabled: bool = True
    priority: int = 1  # 1 = highest priority
    rate_limit: Optional[int] = None  # requests per minute
    cost_per_month: float = 0.0
    description: str = ""
    
    def is_configured(self) -> bool:
        """Check if all required API keys are available."""
        for key in self.required_keys:
            if not os.getenv(key):
                return False
        return True
    
    def get_missing_keys(self) -> List[str]:
        """Get list of missing required API keys."""
        return [key for key in self.required_keys if not os.getenv(key)]


class DataSourceRegistry:
    """Registry of all available data sources with dynamic management."""
    
    def __init__(self):
        self.sources: Dict[str, DataSourceConfig] = {}
        self.status: Dict[str, DataSourceStatus] = {}
        self._initialize_sources()
    
    def _initialize_sources(self):
        """Initialize all known data sources."""
        
        # Market Data Sources
        self.sources["polygon"] = DataSourceConfig(
            name="polygon",
            type=DataSourceType.MARKET_DATA,
            required_keys=["POLYGON_API_KEY"],
            priority=1,
            cost_per_month=99.0,
            description="Polygon.io - Real-time market data + basic news"
        )
        
        self.sources["alpha_vantage"] = DataSourceConfig(
            name="alpha_vantage", 
            type=DataSourceType.MARKET_DATA,
            required_keys=["ALPHA_VANTAGE_API_KEY"],
            priority=2,
            cost_per_month=49.99,
            description="Alpha Vantage - Market data + news with sentiment"
        )
        
        self.sources["iex_cloud"] = DataSourceConfig(
            name="iex_cloud",
            type=DataSourceType.MARKET_DATA,
            required_keys=["IEX_CLOUD_API_KEY"],
            priority=3,
            cost_per_month=9.0,
            description="IEX Cloud - Budget-friendly market data"
        )
        
        # News Sources
        self.sources["benzinga"] = DataSourceConfig(
            name="benzinga",
            type=DataSourceType.NEWS,
            required_keys=["BENZINGA_API_KEY"],
            priority=1,
            cost_per_month=199.0,
            description="Benzinga - Premium financial news & alerts"
        )
        
        self.sources["newsapi"] = DataSourceConfig(
            name="newsapi",
            type=DataSourceType.NEWS,
            required_keys=["NEWS_API_KEY"],
            priority=2,
            cost_per_month=449.0,
            description="NewsAPI - 80,000+ global news sources"
        )
        
        # Social Media Sources
        self.sources["twitter"] = DataSourceConfig(
            name="twitter",
            type=DataSourceType.SOCIAL_MEDIA,
            required_keys=["TWITTER_API_KEY", "TWITTER_API_SECRET"],
            optional_keys=["TWITTER_BEARER_TOKEN"],
            priority=1,
            cost_per_month=100.0,
            description="Twitter/X - Real-time social sentiment"
        )
        
        self.sources["reddit"] = DataSourceConfig(
            name="reddit",
            type=DataSourceType.SOCIAL_MEDIA,
            required_keys=["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"],
            optional_keys=["REDDIT_USER_AGENT"],
            priority=1,
            cost_per_month=100.0,
            description="Reddit - WSB, stocks, investing sentiment"
        )
        
        self.sources["discord"] = DataSourceConfig(
            name="discord",
            type=DataSourceType.SOCIAL_MEDIA,
            required_keys=["DISCORD_BOT_TOKEN"],
            priority=2,
            cost_per_month=0.0,
            description="Discord - Trading servers & options flow"
        )
        
        self.sources["stocktwits"] = DataSourceConfig(
            name="stocktwits",
            type=DataSourceType.SOCIAL_MEDIA,
            required_keys=["STOCKTWITS_ACCESS_TOKEN"],
            priority=3,
            cost_per_month=0.0,
            description="StockTwits - Pure financial social sentiment"
        )
        
        # Broker Sources
        self.sources["alpaca"] = DataSourceConfig(
            name="alpaca",
            type=DataSourceType.BROKER,
            required_keys=["ALPACA_API_KEY", "ALPACA_SECRET_KEY"],
            optional_keys=["ALPACA_BASE_URL"],
            priority=1,
            cost_per_month=0.0,
            description="Alpaca - Commission-free trading"
        )
        
        # AI Model Sources
        self.sources["openai"] = DataSourceConfig(
            name="openai",
            type=DataSourceType.AI_MODEL,
            required_keys=["OPENAI_API_KEY"],
            priority=2,  # Local models are priority 1
            cost_per_month=200.0,  # Estimated usage
            description="OpenAI GPT - Advanced language models"
        )
        
        self.sources["anthropic"] = DataSourceConfig(
            name="anthropic",
            type=DataSourceType.AI_MODEL,
            required_keys=["ANTHROPIC_API_KEY"],
            priority=2,
            cost_per_month=150.0,
            description="Anthropic Claude - Advanced AI analysis"
        )
    
    def get_available_sources(self, source_type: Optional[DataSourceType] = None) -> List[DataSourceConfig]:
        """Get all available (configured) data sources."""
        sources = []
        for source in self.sources.values():
            if source_type and source.type != source_type:
                continue
            if source.is_configured() and source.enabled:
                sources.append(source)
        
        # Sort by priority (lower number = higher priority)
        sources.sort(key=lambda x: x.priority)
        return sources
    
    def get_unavailable_sources(self, source_type: Optional[DataSourceType] = None) -> List[DataSourceConfig]:
        """Get all unavailable (not configured) data sources."""
        sources = []
        for source in self.sources.values():
            if source_type and source.type != source_type:
                continue
            if not source.is_configured():
                sources.append(source)
        
        sources.sort(key=lambda x: x.priority)
        return sources
    
    def get_source_status(self, source_name: str) -> DataSourceStatus:
        """Get current status of a data source."""
        if source_name not in self.sources:
            return DataSourceStatus.UNAVAILABLE
        
        source = self.sources[source_name]
        if not source.enabled:
            return DataSourceStatus.DISABLED
        
        if not source.is_configured():
            return DataSourceStatus.UNAVAILABLE
        
        # Check cached status
        return self.status.get(source_name, DataSourceStatus.AVAILABLE)
    
    def update_source_status(self, source_name: str, status: DataSourceStatus):
        """Update the status of a data source."""
        if source_name in self.sources:
            self.status[source_name] = status
            logger.info(f"Data source {source_name} status updated to {status.value}")
    
    def get_configuration_report(self) -> Dict[str, Any]:
        """Get comprehensive report of data source configuration."""
        report = {
            "available_sources": {},
            "unavailable_sources": {},
            "total_monthly_cost": 0.0,
            "coverage": {
                "market_data": [],
                "news": [],
                "social_media": [],
                "brokers": [],
                "ai_models": []
            }
        }
        
        for source_name, source in self.sources.items():
            source_info = {
                "type": source.type.value,
                "priority": source.priority,
                "cost": source.cost_per_month,
                "description": source.description,
                "status": self.get_source_status(source_name).value
            }
            
            if source.is_configured() and source.enabled:
                report["available_sources"][source_name] = source_info
                report["total_monthly_cost"] += source.cost_per_month
                
                # Add to coverage
                if source.type == DataSourceType.MARKET_DATA:
                    report["coverage"]["market_data"].append(source_name)
                elif source.type == DataSourceType.NEWS:
                    report["coverage"]["news"].append(source_name)
                elif source.type == DataSourceType.SOCIAL_MEDIA:
                    report["coverage"]["social_media"].append(source_name)
                elif source.type == DataSourceType.BROKER:
                    report["coverage"]["brokers"].append(source_name)
                elif source.type == DataSourceType.AI_MODEL:
                    report["coverage"]["ai_models"].append(source_name)
            else:
                source_info["missing_keys"] = source.get_missing_keys()
                report["unavailable_sources"][source_name] = source_info
        
        return report
    
    def get_fallback_chain(self, source_type: DataSourceType) -> List[str]:
        """Get fallback chain for a specific data source type."""
        available = self.get_available_sources(source_type)
        return [source.name for source in available]


@lru_cache()
def get_data_source_registry() -> DataSourceRegistry:
    """Get cached data source registry instance."""
    return DataSourceRegistry()


def is_data_source_available(source_name: str) -> bool:
    """Check if a specific data source is available."""
    registry = get_data_source_registry()
    return registry.get_source_status(source_name) == DataSourceStatus.AVAILABLE


def get_primary_data_source(source_type: DataSourceType) -> Optional[str]:
    """Get the primary (highest priority available) data source for a type."""
    registry = get_data_source_registry()
    available = registry.get_available_sources(source_type)
    return available[0].name if available else None


def log_data_source_status():
    """Log current data source configuration status."""
    registry = get_data_source_registry()
    report = registry.get_configuration_report()
    
    logger.info(f"Data Source Configuration Report:")
    logger.info(f"Available sources: {len(report['available_sources'])}")
    logger.info(f"Unavailable sources: {len(report['unavailable_sources'])}")
    logger.info(f"Total monthly cost: ${report['total_monthly_cost']}")
    
    for source_type, sources in report['coverage'].items():
        if sources:
            logger.info(f"{source_type.replace('_', ' ').title()}: {', '.join(sources)}")
    
    if report['unavailable_sources']:
        logger.warning("Unavailable sources (missing API keys):")
        for source_name, info in report['unavailable_sources'].items():
            logger.warning(f"  {source_name}: {', '.join(info['missing_keys'])}")