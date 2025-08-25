"""Data ingestion pipeline interfaces for trading system."""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

from .models import MarketData, NewsEvent, TechnicalIndicator, OptionsData
from .database import get_database_manager, QuestDBOperations
from .cache import get_trading_cache
from .config import get_settings
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class IngestionMetrics:
    """Metrics for data ingestion monitoring."""
    records_processed: int = 0
    records_success: int = 0
    records_failed: int = 0
    processing_time_ms: float = 0
    last_update: datetime = field(default_factory=datetime.utcnow)
    error_messages: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.records_processed == 0:
            return 0.0
        return (self.records_success / self.records_processed) * 100
    
    @property
    def throughput_per_second(self) -> float:
        """Calculate throughput in records per second."""
        if self.processing_time_ms == 0:
            return 0.0
        return (self.records_processed / self.processing_time_ms) * 1000


class DataIngestionBase(ABC):
    """Abstract base class for data ingestion pipelines."""
    
    def __init__(self, name: str):
        self.name = name
        self.db_manager = None
        self.cache = None
        self.metrics = IngestionMetrics()
        self._running = False
        self._error_callbacks: List[Callable] = []
        
    async def initialize(self):
        """Initialize database connections and cache."""
        self.db_manager = await get_database_manager()
        self.cache = await get_trading_cache()
        logger.info(f"Initialized {self.name} ingestion pipeline")
    
    async def start(self):
        """Start the ingestion pipeline."""
        if self._running:
            logger.warning(f"{self.name} pipeline already running")
            return
        
        self._running = True
        logger.info(f"Starting {self.name} ingestion pipeline")
        
        try:
            await self.run_pipeline()
        except Exception as e:
            logger.error(f"Pipeline {self.name} failed: {e}")
            await self._handle_error(e)
        finally:
            self._running = False
    
    async def stop(self):
        """Stop the ingestion pipeline."""
        self._running = False
        logger.info(f"Stopping {self.name} ingestion pipeline")
    
    def add_error_callback(self, callback: Callable[[Exception, "DataIngestionBase"], None]):
        """Add error callback function."""
        self._error_callbacks.append(callback)
    
    async def _handle_error(self, error: Exception):
        """Handle pipeline errors."""
        self.metrics.error_messages.append(str(error))
        for callback in self._error_callbacks:
            try:
                await callback(error, self)
            except Exception as cb_error:
                logger.error(f"Error callback failed: {cb_error}")
    
    @abstractmethod
    async def run_pipeline(self):
        """Main pipeline logic - to be implemented by subclasses."""
        pass
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        return {
            'name': self.name,
            'running': self._running,
            'records_processed': self.metrics.records_processed,
            'records_success': self.metrics.records_success,
            'records_failed': self.metrics.records_failed,
            'success_rate': self.metrics.success_rate,
            'throughput_per_second': self.metrics.throughput_per_second,
            'processing_time_ms': self.metrics.processing_time_ms,
            'last_update': self.metrics.last_update.isoformat(),
            'recent_errors': self.metrics.error_messages[-5:]  # Last 5 errors
        }


class MarketDataIngestion(DataIngestionBase):
    """Market data ingestion pipeline."""
    
    def __init__(self, symbols: List[str], timeframe: str = "1m", 
                 batch_size: int = 100, poll_interval: int = 60):
        super().__init__(f"MarketData-{timeframe}")
        self.symbols = symbols
        self.timeframe = timeframe
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.questdb_ops = None
    
    async def initialize(self):
        """Initialize with QuestDB operations."""
        await super().initialize()
        self.questdb_ops = QuestDBOperations(self.db_manager)
    
    async def run_pipeline(self):
        """Main market data ingestion loop."""
        logger.info(f"Starting market data ingestion for {len(self.symbols)} symbols")
        
        while self._running:
            start_time = datetime.utcnow()
            batch_records = []
            
            for symbol in self.symbols:
                try:
                    # Simulate fetching market data (replace with actual API calls)
                    market_data = await self._fetch_market_data(symbol)
                    if market_data:
                        batch_records.append(market_data)
                        
                        # Cache latest data
                        await self.cache.cache_market_data(market_data, self.timeframe)
                    
                except Exception as e:
                    logger.error(f"Failed to fetch data for {symbol}: {e}")
                    self.metrics.records_failed += 1
                    continue
            
            # Batch insert to database
            if batch_records:
                try:
                    success_count = await self.questdb_ops.insert_market_data(batch_records)
                    self.metrics.records_success += success_count
                    logger.debug(f"Inserted {success_count} market data records")
                except Exception as e:
                    logger.error(f"Batch insert failed: {e}")
                    self.metrics.records_failed += len(batch_records)
            
            # Update metrics
            self.metrics.records_processed += len(batch_records)
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.metrics.processing_time_ms += processing_time
            self.metrics.last_update = datetime.utcnow()
            
            # Wait for next poll
            if self._running:
                await asyncio.sleep(self.poll_interval)
    
    async def _fetch_market_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch market data for a symbol (mock implementation)."""
        # This would be replaced with actual API calls to Polygon, Alpaca, etc.
        # For now, return a mock data point
        import random
        
        base_price = 100.0 + random.uniform(-50, 50)
        return MarketData(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            open=base_price + random.uniform(-1, 1),
            high=base_price + random.uniform(0, 2),
            low=base_price - random.uniform(0, 2),
            close=base_price + random.uniform(-1, 1),
            volume=random.randint(1000, 100000),
            vwap=base_price + random.uniform(-0.5, 0.5),
            trade_count=random.randint(10, 1000),
            data_source="mock_provider"
        )


class NewsIngestion(DataIngestionBase):
    """News data ingestion pipeline."""
    
    def __init__(self, sources: List[str], keywords: List[str], 
                 poll_interval: int = 300):
        super().__init__("NewsIngestion")
        self.sources = sources
        self.keywords = keywords
        self.poll_interval = poll_interval
        self.questdb_ops = None
    
    async def initialize(self):
        """Initialize with QuestDB operations."""
        await super().initialize()
        self.questdb_ops = QuestDBOperations(self.db_manager)
    
    async def run_pipeline(self):
        """Main news ingestion loop."""
        logger.info(f"Starting news ingestion for {len(self.sources)} sources")
        
        while self._running:
            start_time = datetime.utcnow()
            news_records = []
            
            for source in self.sources:
                try:
                    # Simulate fetching news (replace with actual API calls)
                    news_items = await self._fetch_news_from_source(source)
                    news_records.extend(news_items)
                    
                except Exception as e:
                    logger.error(f"Failed to fetch news from {source}: {e}")
                    self.metrics.records_failed += 1
                    continue
            
            # Process and store news
            for news_item in news_records:
                try:
                    # Process sentiment analysis (placeholder)
                    news_item = await self._analyze_sentiment(news_item)
                    
                    # Store in database (would need to implement news storage)
                    # await self.questdb_ops.insert_news_event(news_item)
                    
                    self.metrics.records_success += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process news item: {e}")
                    self.metrics.records_failed += 1
            
            # Update metrics
            self.metrics.records_processed += len(news_records)
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.metrics.processing_time_ms += processing_time
            self.metrics.last_update = datetime.utcnow()
            
            # Wait for next poll
            if self._running:
                await asyncio.sleep(self.poll_interval)
    
    async def _fetch_news_from_source(self, source: str) -> List[NewsEvent]:
        """Fetch news from a specific source (mock implementation)."""
        # Mock implementation - replace with actual news API calls
        import random
        
        news_items = []
        for i in range(random.randint(1, 5)):
            news_items.append(NewsEvent(
                id=f"news_{datetime.utcnow().timestamp()}_{i}",
                timestamp=datetime.utcnow(),
                headline=f"Mock headline from {source}",
                content=f"Mock news content from {source} about market conditions",
                source=source,
                symbols=["AAPL", "GOOGL", "MSFT"],  # Mock symbols
                news_type="market_update",
                language="en",
                url=f"https://{source}.com/article/{i}"
            ))
        
        return news_items
    
    async def _analyze_sentiment(self, news_item: NewsEvent) -> NewsEvent:
        """Analyze sentiment of news item (placeholder)."""
        # Mock sentiment analysis - replace with actual NLP
        import random
        
        news_item.sentiment_score = random.uniform(-1, 1)
        news_item.relevance_score = random.uniform(0.3, 1.0)
        
        return news_item


class TechnicalIndicatorIngestion(DataIngestionBase):
    """Technical indicator calculation and ingestion pipeline."""
    
    def __init__(self, symbols: List[str], indicators: List[str],
                 timeframes: List[str], calculation_interval: int = 300):
        super().__init__("TechnicalIndicators")
        self.symbols = symbols
        self.indicators = indicators
        self.timeframes = timeframes
        self.calculation_interval = calculation_interval
        self.questdb_ops = None
    
    async def initialize(self):
        """Initialize with QuestDB operations."""
        await super().initialize()
        self.questdb_ops = QuestDBOperations(self.db_manager)
    
    async def run_pipeline(self):
        """Main technical indicator calculation loop."""
        logger.info(f"Starting technical indicator calculations")
        
        while self._running:
            start_time = datetime.utcnow()
            
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    for indicator in self.indicators:
                        try:
                            # Calculate indicator
                            indicator_value = await self._calculate_indicator(
                                symbol, indicator, timeframe
                            )
                            
                            if indicator_value is not None:
                                # Cache indicator value
                                await self.cache.cache_technical_indicator(
                                    symbol, indicator, timeframe, indicator_value
                                )
                                
                                self.metrics.records_success += 1
                            
                        except Exception as e:
                            logger.error(f"Indicator calculation failed for {symbol} {indicator}: {e}")
                            self.metrics.records_failed += 1
            
            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.metrics.processing_time_ms += processing_time
            self.metrics.last_update = datetime.utcnow()
            
            # Wait for next calculation cycle
            if self._running:
                await asyncio.sleep(self.calculation_interval)
    
    async def _calculate_indicator(self, symbol: str, indicator: str, timeframe: str) -> Optional[float]:
        """Calculate technical indicator (placeholder implementation)."""
        # Get recent market data
        market_data = await self.questdb_ops.get_latest_market_data(symbol, limit=50)
        
        if not market_data:
            return None
        
        # Mock indicator calculation - replace with actual TA-Lib or custom calculations
        import random
        
        if indicator == "sma_20":
            return random.uniform(90, 110)
        elif indicator == "rsi":
            return random.uniform(20, 80)
        elif indicator == "macd":
            return random.uniform(-2, 2)
        else:
            return random.uniform(0, 100)


class IngestionManager:
    """Manage multiple ingestion pipelines."""
    
    def __init__(self):
        self.pipelines: Dict[str, DataIngestionBase] = {}
        self.running = False
    
    def add_pipeline(self, pipeline: DataIngestionBase):
        """Add a pipeline to the manager."""
        self.pipelines[pipeline.name] = pipeline
        logger.info(f"Added pipeline: {pipeline.name}")
    
    async def start_all(self):
        """Start all registered pipelines."""
        if self.running:
            logger.warning("Ingestion manager already running")
            return
        
        self.running = True
        logger.info(f"Starting {len(self.pipelines)} ingestion pipelines")
        
        # Initialize all pipelines
        for pipeline in self.pipelines.values():
            await pipeline.initialize()
        
        # Start all pipelines concurrently
        tasks = [pipeline.start() for pipeline in self.pipelines.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop_all(self):
        """Stop all pipelines."""
        self.running = False
        logger.info("Stopping all ingestion pipelines")
        
        for pipeline in self.pipelines.values():
            await pipeline.stop()
    
    async def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all pipelines."""
        metrics = {}
        for name, pipeline in self.pipelines.items():
            metrics[name] = await pipeline.get_metrics()
        return metrics
    
    def get_pipeline(self, name: str) -> Optional[DataIngestionBase]:
        """Get pipeline by name."""
        return self.pipelines.get(name)


# Global ingestion manager instance
_ingestion_manager: Optional[IngestionManager] = None

def get_ingestion_manager() -> IngestionManager:
    """Get or create global ingestion manager."""
    global _ingestion_manager
    if _ingestion_manager is None:
        _ingestion_manager = IngestionManager()
    return _ingestion_manager