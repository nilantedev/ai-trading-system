"""Data source management service with hot-swapping capabilities."""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json

from ..data_sources import (
    DataSourceRegistry, DataSourceType, DataSourceStatus,
    get_data_source_registry, is_data_source_available
)
from ..logging import get_logger
from ..exceptions import TradingError

logger = get_logger(__name__)


class DataSourceManager:
    """Manages data sources with hot-swapping and fallback capabilities."""
    
    def __init__(self):
        self.registry = get_data_source_registry()
        self.active_connections: Dict[str, Any] = {}
        self.fallback_chains: Dict[DataSourceType, List[str]] = {}
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}
        self._initialize_fallback_chains()
    
    def _initialize_fallback_chains(self):
        """Initialize fallback chains for each data source type."""
        for source_type in DataSourceType:
            self.fallback_chains[source_type] = self.registry.get_fallback_chain(source_type)
    
    async def initialize_available_sources(self):
        """Initialize connections to all available data sources."""
        logger.info("Initializing available data sources...")
        
        for source_type in DataSourceType:
            available = self.registry.get_available_sources(source_type)
            logger.info(f"{source_type.value}: {len(available)} sources available")
            
            for source in available:
                try:
                    await self._initialize_source_connection(source.name, source.type)
                except Exception as e:
                    logger.error(f"Failed to initialize {source.name}: {e}")
                    self.registry.update_source_status(source.name, DataSourceStatus.ERROR)
    
    async def _initialize_source_connection(self, source_name: str, source_type: DataSourceType):
        """Initialize connection to a specific data source."""
        if source_type == DataSourceType.MARKET_DATA:
            await self._init_market_data_source(source_name)
        elif source_type == DataSourceType.NEWS:
            await self._init_news_source(source_name)
        elif source_type == DataSourceType.SOCIAL_MEDIA:
            await self._init_social_media_source(source_name)
        elif source_type == DataSourceType.BROKER:
            await self._init_broker_source(source_name)
        elif source_type == DataSourceType.AI_MODEL:
            await self._init_ai_model_source(source_name)
    
    async def _init_market_data_source(self, source_name: str):
        """Initialize market data source connection."""
        if source_name == "polygon":
            # Initialize Polygon.io connection
            # This would be implemented with actual Polygon client
            logger.info("Initializing Polygon.io market data connection")
            self.active_connections[source_name] = {"type": "polygon", "status": "connected"}
            self.registry.update_source_status(source_name, DataSourceStatus.AVAILABLE)
        
        elif source_name == "alpha_vantage":
            logger.info("Initializing Alpha Vantage market data connection")
            self.active_connections[source_name] = {"type": "alpha_vantage", "status": "connected"}
            self.registry.update_source_status(source_name, DataSourceStatus.AVAILABLE)
        
        # Add other market data sources as needed
    
    async def _init_news_source(self, source_name: str):
        """Initialize news source connection."""
        if source_name == "benzinga":
            logger.info("Initializing Benzinga news connection")
            self.active_connections[source_name] = {"type": "benzinga", "status": "connected"}
            self.registry.update_source_status(source_name, DataSourceStatus.AVAILABLE)
        
        elif source_name == "newsapi":
            logger.info("Initializing NewsAPI connection")
            self.active_connections[source_name] = {"type": "newsapi", "status": "connected"}
            self.registry.update_source_status(source_name, DataSourceStatus.AVAILABLE)
    
    async def _init_social_media_source(self, source_name: str):
        """Initialize social media source connection."""
        if source_name == "twitter":
            logger.info("Initializing Twitter/X connection")
            self.active_connections[source_name] = {"type": "twitter", "status": "connected"}
            self.registry.update_source_status(source_name, DataSourceStatus.AVAILABLE)
        
        elif source_name == "reddit":
            logger.info("Initializing Reddit connection") 
            self.active_connections[source_name] = {"type": "reddit", "status": "connected"}
            self.registry.update_source_status(source_name, DataSourceStatus.AVAILABLE)
        
        # Add other social media sources
    
    async def _init_broker_source(self, source_name: str):
        """Initialize broker source connection."""
        if source_name == "alpaca":
            logger.info("Initializing Alpaca broker connection")
            self.active_connections[source_name] = {"type": "alpaca", "status": "connected"}
            self.registry.update_source_status(source_name, DataSourceStatus.AVAILABLE)
    
    async def _init_ai_model_source(self, source_name: str):
        """Initialize AI model source connection."""
        if source_name == "openai":
            logger.info("Initializing OpenAI connection")
            self.active_connections[source_name] = {"type": "openai", "status": "connected"}
            self.registry.update_source_status(source_name, DataSourceStatus.AVAILABLE)
    
    async def get_market_data(self, symbol: str, data_type: str = "quote") -> Optional[Dict[str, Any]]:
        """Get market data with automatic fallback."""
        fallback_chain = self.fallback_chains[DataSourceType.MARKET_DATA]
        
        for source_name in fallback_chain:
            if self.registry.get_source_status(source_name) == DataSourceStatus.AVAILABLE:
                try:
                    # This would call the actual API
                    logger.debug(f"Getting market data for {symbol} from {source_name}")
                    
                    # Mock response for now
                    return {
                        "symbol": symbol,
                        "source": source_name,
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"price": 100.0, "volume": 1000}  # Mock data
                    }
                
                except Exception as e:
                    logger.warning(f"Market data failed from {source_name}: {e}")
                    self.registry.update_source_status(source_name, DataSourceStatus.ERROR)
                    continue
        
        logger.error(f"No available market data sources for {symbol}")
        return None
    
    async def get_news(self, query: str = "", limit: int = 10) -> List[Dict[str, Any]]:
        """Get news with automatic fallback."""
        fallback_chain = self.fallback_chains[DataSourceType.NEWS]
        
        for source_name in fallback_chain:
            if self.registry.get_source_status(source_name) == DataSourceStatus.AVAILABLE:
                try:
                    logger.debug(f"Getting news from {source_name}")
                    
                    # Mock response for now
                    return [{
                        "title": f"Sample news from {source_name}",
                        "source": source_name,
                        "timestamp": datetime.utcnow().isoformat(),
                        "url": "https://example.com/news/1",
                        "sentiment": 0.5
                    }] * min(limit, 5)  # Mock data
                
                except Exception as e:
                    logger.warning(f"News failed from {source_name}: {e}")
                    self.registry.update_source_status(source_name, DataSourceStatus.ERROR)
                    continue
        
        logger.error("No available news sources")
        return []
    
    async def get_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get social media sentiment with automatic fallback."""
        fallback_chain = self.fallback_chains[DataSourceType.SOCIAL_MEDIA]
        sentiment_data = {"symbol": symbol, "sources": {}, "overall_sentiment": 0.0}
        
        for source_name in fallback_chain:
            if self.registry.get_source_status(source_name) == DataSourceStatus.AVAILABLE:
                try:
                    logger.debug(f"Getting social sentiment for {symbol} from {source_name}")
                    
                    # Mock response for now
                    sentiment_data["sources"][source_name] = {
                        "sentiment": 0.6,  # Mock sentiment
                        "volume": 100,     # Mock post count
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                except Exception as e:
                    logger.warning(f"Social sentiment failed from {source_name}: {e}")
                    self.registry.update_source_status(source_name, DataSourceStatus.ERROR)
        
        # Calculate overall sentiment from available sources
        if sentiment_data["sources"]:
            total_sentiment = sum(data["sentiment"] for data in sentiment_data["sources"].values())
            sentiment_data["overall_sentiment"] = total_sentiment / len(sentiment_data["sources"])
        
        return sentiment_data
    
    async def add_new_api_key(self, source_name: str, force_reconnect: bool = True):
        """Hot-swap: Add new API key and initialize connection."""
        if source_name not in self.registry.sources:
            raise TradingError(f"Unknown data source: {source_name}")
        
        source = self.registry.sources[source_name]
        
        if not source.is_configured():
            missing = source.get_missing_keys()
            raise TradingError(f"Missing API keys for {source_name}: {missing}")
        
        logger.info(f"Adding new API key for {source_name}")
        
        # Disconnect existing connection if any
        if source_name in self.active_connections:
            await self._disconnect_source(source_name)
        
        # Initialize new connection
        await self._initialize_source_connection(source_name, source.type)
        
        # Update fallback chains
        self._initialize_fallback_chains()
        
        logger.info(f"Successfully added and connected {source_name}")
    
    async def _disconnect_source(self, source_name: str):
        """Disconnect from a data source."""
        if source_name in self.active_connections:
            logger.info(f"Disconnecting from {source_name}")
            # This would close the actual connection
            del self.active_connections[source_name]
            self.registry.update_source_status(source_name, DataSourceStatus.DISABLED)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        report = self.registry.get_configuration_report()
        report["active_connections"] = len(self.active_connections)
        report["fallback_chains"] = {
            source_type.value: chain 
            for source_type, chain in self.fallback_chains.items()
        }
        
        return report
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all active sources."""
        health_status = {
            "overall_health": "healthy",
            "sources": {},
            "issues": []
        }
        
        for source_name in self.active_connections:
            try:
                # This would ping the actual API
                health_status["sources"][source_name] = "healthy"
            except Exception as e:
                health_status["sources"][source_name] = f"unhealthy: {e}"
                health_status["issues"].append(f"{source_name}: {e}")
                self.registry.update_source_status(source_name, DataSourceStatus.ERROR)
        
        if health_status["issues"]:
            health_status["overall_health"] = "degraded"
        
        return health_status


# Global instance
_data_source_manager: Optional[DataSourceManager] = None


def get_data_source_manager() -> DataSourceManager:
    """Get global data source manager instance."""
    global _data_source_manager
    if _data_source_manager is None:
        _data_source_manager = DataSourceManager()
    return _data_source_manager


async def initialize_data_sources():
    """Initialize all available data sources."""
    manager = get_data_source_manager()
    await manager.initialize_available_sources()
    
    # Log status
    status = manager.get_system_status()
    logger.info(f"Data source system initialized:")
    logger.info(f"  Active connections: {status['active_connections']}")
    logger.info(f"  Monthly cost: ${status['total_monthly_cost']}")
    
    return manager