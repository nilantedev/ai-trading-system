#!/usr/bin/env python3
"""Stream Processing Service - Real-time data stream processing and routing."""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import time

from trading_common import MarketData, NewsItem, get_settings, get_logger
from trading_common.cache import get_trading_cache
from trading_common.messaging import get_message_consumer, get_message_producer
from trading_common.ai_models import generate_response, ModelType

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ProcessedMarketData:
    """Enhanced market data with processing metadata."""
    market_data: MarketData
    processing_time: datetime
    latency_ms: float
    indicators: Dict[str, float]
    quality_score: float
    processor_id: str


@dataclass
class StreamMetrics:
    """Stream processing performance metrics."""
    messages_processed: int = 0
    messages_failed: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    throughput_per_second: float = 0.0
    last_reset: datetime = datetime.utcnow()
    
    def update(self, latency_ms: float, success: bool = True):
        """Update metrics with new processing result."""
        if success:
            self.messages_processed += 1
            self.total_latency_ms += latency_ms
            self.avg_latency_ms = self.total_latency_ms / self.messages_processed
        else:
            self.messages_failed += 1
        
        # Calculate throughput
        elapsed_seconds = (datetime.utcnow() - self.last_reset).total_seconds()
        if elapsed_seconds > 0:
            self.throughput_per_second = (self.messages_processed + self.messages_failed) / elapsed_seconds


class MarketDataProcessor:
    """Processes market data streams with basic indicators."""
    
    def __init__(self):
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
    async def process_market_data(self, market_data: MarketData) -> ProcessedMarketData:
        """Process market data and calculate basic indicators."""
        start_time = time.time()
        
        symbol = market_data.symbol
        
        # Update price and volume history
        self.price_history[symbol].append(market_data.close)
        self.volume_history[symbol].append(market_data.volume)
        
        # Calculate basic indicators
        indicators = {}
        
        # Simple Moving Average (20 periods)
        if len(self.price_history[symbol]) >= 20:
            sma_20 = statistics.mean(list(self.price_history[symbol])[-20:])
            indicators['sma_20'] = sma_20
        
        # Simple Moving Average (50 periods)
        if len(self.price_history[symbol]) >= 50:
            sma_50 = statistics.mean(list(self.price_history[symbol])[-50:])
            indicators['sma_50'] = sma_50
            
            # Golden Cross / Death Cross signals
            if 'sma_20' in indicators:
                if indicators['sma_20'] > sma_50:
                    indicators['ma_signal'] = 1.0  # Bullish
                else:
                    indicators['ma_signal'] = -1.0  # Bearish
        
        # Price change percentage
        if len(self.price_history[symbol]) >= 2:
            prev_price = list(self.price_history[symbol])[-2]
            price_change = (market_data.close - prev_price) / prev_price
            indicators['price_change'] = price_change
        
        # Volume analysis
        if len(self.volume_history[symbol]) >= 10:
            avg_volume = statistics.mean(list(self.volume_history[symbol])[-10:])
            volume_ratio = market_data.volume / avg_volume if avg_volume > 0 else 1.0
            indicators['volume_ratio'] = volume_ratio
            
            # High volume breakout
            if volume_ratio > 2.0 and indicators.get('price_change', 0) > 0.02:
                indicators['volume_breakout'] = 1.0
            else:
                indicators['volume_breakout'] = 0.0
        
        # Calculate quality score based on data completeness and validity
        quality_score = self._calculate_data_quality(market_data, indicators)
        
        # Calculate processing latency
        processing_latency = (time.time() - start_time) * 1000
        
        return ProcessedMarketData(
            market_data=market_data,
            processing_time=datetime.utcnow(),
            latency_ms=processing_latency,
            indicators=indicators,
            quality_score=quality_score,
            processor_id="market_processor_v1"
        )
    
    def _calculate_data_quality(self, data: MarketData, indicators: Dict[str, float]) -> float:
        """Calculate data quality score (0-1)."""
        quality_factors = []
        
        # Price validity
        if data.close > 0 and data.high >= data.close >= data.low and data.open > 0:
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.0)
        
        # Volume validity
        if data.volume >= 0:
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.5)
        
        # Timestamp recency (within last hour)
        if data.timestamp and (datetime.utcnow() - data.timestamp).total_seconds() < 3600:
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.7)
        
        # Indicator availability
        indicator_ratio = len(indicators) / 6.0  # Expected max 6 indicators
        quality_factors.append(min(indicator_ratio, 1.0))
        
        return statistics.mean(quality_factors)


class NewsProcessor:
    """Processes news items for sentiment and relevance."""
    
    def __init__(self):
        self.sentiment_cache: Dict[str, float] = {}
    
    async def process_news(self, news_item: NewsItem) -> Dict[str, Any]:
        """Process news item and enhance with analysis."""
        start_time = time.time()
        
        # Use cached sentiment if available
        content_hash = hash(news_item.title + news_item.content)
        
        if content_hash in self.sentiment_cache:
            sentiment_score = self.sentiment_cache[content_hash]
        else:
            # Analyze sentiment if not cached
            if news_item.sentiment_score is None:
                sentiment_score = await self._analyze_sentiment(news_item.title + " " + news_item.content)
                self.sentiment_cache[content_hash] = sentiment_score
            else:
                sentiment_score = news_item.sentiment_score
        
        # Calculate market impact score
        impact_score = self._calculate_market_impact(news_item, sentiment_score)
        
        # Processing latency
        processing_latency = (time.time() - start_time) * 1000
        
        return {
            'news_item': news_item,
            'enhanced_sentiment': sentiment_score,
            'market_impact_score': impact_score,
            'processing_time': datetime.utcnow(),
            'latency_ms': processing_latency,
            'processor_id': 'news_processor_v1'
        }
    
    async def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment using AI models."""
        try:
            prompt = f"""
            Analyze the financial sentiment of this text on a scale from -1 (very negative) to +1 (very positive).
            Return only a number between -1 and 1.
            
            Text: {text[:500]}
            """
            
            response = await generate_response(
                prompt,
                model_preference=[ModelType.LOCAL_OLLAMA]  # Only local models
            )
            
            # Extract numeric sentiment
            sentiment_text = response.content.strip()
            try:
                sentiment = float(sentiment_text)
                return max(-1.0, min(1.0, sentiment))
            except ValueError:
                return 0.0
                
        except Exception as e:
            logger.warning(f"AI sentiment analysis failed: {e}")
            return 0.0
    
    def _calculate_market_impact(self, news_item: NewsItem, sentiment: float) -> float:
        """Calculate potential market impact score."""
        impact_factors = []
        
        # Sentiment strength
        impact_factors.append(abs(sentiment))
        
        # Source credibility (simplified)
        credible_sources = ['reuters', 'bloomberg', 'cnbc', 'financial times', 'wsj']
        source_credibility = 1.0 if any(source in news_item.source.lower() for source in credible_sources) else 0.7
        impact_factors.append(source_credibility)
        
        # Recency factor
        if news_item.published_at:
            hours_old = (datetime.utcnow() - news_item.published_at).total_seconds() / 3600
            recency_factor = max(0.1, 1.0 - (hours_old / 24))  # Decay over 24 hours
            impact_factors.append(recency_factor)
        else:
            impact_factors.append(0.5)
        
        # Symbol relevance
        relevance_factor = news_item.relevance_score if news_item.relevance_score else 0.5
        impact_factors.append(relevance_factor)
        
        return statistics.mean(impact_factors)


class StreamProcessingService:
    """Main stream processing service coordinator."""
    
    def __init__(self):
        self.consumer = None
        self.producer = None
        self.cache = None
        
        self.market_processor = MarketDataProcessor()
        self.news_processor = NewsProcessor()
        
        self.metrics = StreamMetrics()
        self.is_running = False
        
        # Processing queues for different data types
        self.market_data_queue = asyncio.Queue(maxsize=1000)
        self.news_queue = asyncio.Queue(maxsize=500)
        
    async def start(self):
        """Initialize and start stream processing."""
        logger.info("Starting Stream Processing Service")
        
        try:
            # Initialize connections
            self.consumer = await get_message_consumer()
            self.producer = await get_message_producer()
            self.cache = get_trading_cache()
            
            # Subscribe to message topics
            await self._setup_subscriptions()
            
            # Start processing tasks
            self.is_running = True
            
            # Start concurrent processing tasks
            tasks = [
                asyncio.create_task(self._process_market_data_queue()),
                asyncio.create_task(self._process_news_queue()),
                asyncio.create_task(self._metrics_reporter()),
                asyncio.create_task(self._consume_messages())
            ]
            
            logger.info("Stream processing started with 4 concurrent tasks")
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Failed to start stream processing: {e}")
            raise
    
    async def stop(self):
        """Stop stream processing gracefully."""
        logger.info("Stopping Stream Processing Service")
        self.is_running = False
        
        if self.consumer:
            await self.consumer.close()
        if self.producer:
            await self.producer.close()
        
        logger.info("Stream Processing Service stopped")
    
    async def _setup_subscriptions(self):
        """Subscribe to relevant message topics."""
        try:
            # Subscribe to market data
            await self.consumer.subscribe_market_data(
                self._handle_market_data_message,
                subscription_name="stream-processor-market"
            )
            
            # Subscribe to trading signals for processing
            await self.consumer.subscribe_trading_signals(
                self._handle_signal_message,
                subscription_name="stream-processor-signals"
            )
            
            logger.info("Subscribed to message topics")
        except Exception as e:
            logger.warning(f"Message subscription setup failed: {e}")
    
    async def _consume_messages(self):
        """Start consuming messages from subscribed topics."""
        try:
            await self.consumer.start_consuming()
        except Exception as e:
            logger.error(f"Message consumption failed: {e}")
    
    async def _handle_market_data_message(self, message):
        """Handle incoming market data messages."""
        try:
            # Convert message to MarketData object
            if hasattr(message, 'symbol'):
                # Direct MarketDataMessage object
                market_data = MarketData(
                    symbol=message.symbol,
                    timestamp=datetime.fromisoformat(message.timestamp),
                    open=message.open,
                    high=message.high,
                    low=message.low,
                    close=message.close,
                    volume=message.volume,
                    timeframe="1min",  # Default
                    data_source=message.data_source if hasattr(message, 'data_source') else "stream"
                )
            else:
                # JSON message
                data = json.loads(message) if isinstance(message, str) else message
                market_data = MarketData(**data)
            
            # Add to processing queue
            await self.market_data_queue.put(market_data)
            
        except Exception as e:
            logger.error(f"Failed to handle market data message: {e}")
            self.metrics.update(0, success=False)
    
    async def _handle_signal_message(self, message):
        """Handle trading signal messages for processing."""
        try:
            # Process signals for routing and enhancement
            logger.debug(f"Received trading signal: {message}")
            # Could add signal processing logic here
        except Exception as e:
            logger.error(f"Failed to handle signal message: {e}")
    
    async def _process_market_data_queue(self):
        """Process market data from the queue."""
        while self.is_running:
            try:
                # Wait for market data with timeout
                market_data = await asyncio.wait_for(
                    self.market_data_queue.get(), 
                    timeout=1.0
                )
                
                # Process the market data
                processed_data = await self.market_processor.process_market_data(market_data)
                
                # Cache processed data
                if self.cache:
                    await self._cache_processed_data(processed_data)
                
                # Publish enhanced data
                await self._publish_processed_data(processed_data)
                
                # Update metrics
                self.metrics.update(processed_data.latency_ms, success=True)
                
                # Mark queue task as done
                self.market_data_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Market data processing error: {e}")
                self.metrics.update(0, success=False)
    
    async def _process_news_queue(self):
        """Process news data from the queue."""
        while self.is_running:
            try:
                # Wait for news data with timeout
                news_item = await asyncio.wait_for(
                    self.news_queue.get(),
                    timeout=1.0
                )
                
                # Process the news item
                processed_news = await self.news_processor.process_news(news_item)
                
                # Cache processed news
                if self.cache:
                    await self._cache_processed_news(processed_news)
                
                # Update metrics
                self.metrics.update(processed_news['latency_ms'], success=True)
                
                # Mark queue task as done
                self.news_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"News processing error: {e}")
                self.metrics.update(0, success=False)
    
    async def _cache_processed_data(self, processed_data: ProcessedMarketData):
        """Cache processed market data."""
        try:
            cache_key = f"processed_market:{processed_data.market_data.symbol}:latest"
            cache_data = asdict(processed_data)
            
            # Convert datetime objects to ISO strings for JSON serialization
            cache_data['processing_time'] = processed_data.processing_time.isoformat()
            cache_data['market_data']['timestamp'] = processed_data.market_data.timestamp.isoformat()
            
            await self.cache.set_json(cache_key, cache_data, ttl=300)  # 5 minutes
            
        except Exception as e:
            logger.warning(f"Failed to cache processed data: {e}")
    
    async def _cache_processed_news(self, processed_news: Dict[str, Any]):
        """Cache processed news data."""
        try:
            news_item = processed_news['news_item']
            cache_key = f"processed_news:{hash(news_item.title)}:latest"
            
            # Serialize for caching
            cache_data = {
                'title': news_item.title,
                'source': news_item.source,
                'published_at': news_item.published_at.isoformat() if news_item.published_at else None,
                'enhanced_sentiment': processed_news['enhanced_sentiment'],
                'market_impact_score': processed_news['market_impact_score'],
                'symbols': news_item.symbols,
                'processing_time': processed_news['processing_time'].isoformat()
            }
            
            await self.cache.set_json(cache_key, cache_data, ttl=3600)  # 1 hour
            
        except Exception as e:
            logger.warning(f"Failed to cache processed news: {e}")
    
    async def _publish_processed_data(self, processed_data: ProcessedMarketData):
        """Publish processed data to downstream topics."""
        try:
            if self.producer:
                # Create enhanced message
                message_data = {
                    'symbol': processed_data.market_data.symbol,
                    'timestamp': processed_data.processing_time.isoformat(),
                    'price': processed_data.market_data.close,
                    'indicators': processed_data.indicators,
                    'quality_score': processed_data.quality_score,
                    'latency_ms': processed_data.latency_ms
                }
                
                # Would publish to processed data topic
                logger.debug(f"Publishing processed data for {processed_data.market_data.symbol}")
                
        except Exception as e:
            logger.warning(f"Failed to publish processed data: {e}")
    
    async def _metrics_reporter(self):
        """Periodic metrics reporting."""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Report every 30 seconds
                
                logger.info(
                    f"Stream Metrics - Processed: {self.metrics.messages_processed}, "
                    f"Failed: {self.metrics.messages_failed}, "
                    f"Avg Latency: {self.metrics.avg_latency_ms:.2f}ms, "
                    f"Throughput: {self.metrics.throughput_per_second:.2f}/sec"
                )
                
                # Cache metrics for monitoring
                if self.cache:
                    metrics_data = {
                        'messages_processed': self.metrics.messages_processed,
                        'messages_failed': self.metrics.messages_failed,
                        'avg_latency_ms': self.metrics.avg_latency_ms,
                        'throughput_per_second': self.metrics.throughput_per_second,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    await self.cache.set_json('stream_processor_metrics', metrics_data, ttl=300)
                
            except Exception as e:
                logger.warning(f"Metrics reporting error: {e}")
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get service health status."""
        queue_status = {
            'market_data_queue_size': self.market_data_queue.qsize(),
            'news_queue_size': self.news_queue.qsize(),
            'is_running': self.is_running
        }
        
        return {
            'service': 'stream_processor',
            'status': 'healthy' if self.is_running else 'stopped',
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {
                'messages_processed': self.metrics.messages_processed,
                'messages_failed': self.metrics.messages_failed,
                'avg_latency_ms': round(self.metrics.avg_latency_ms, 2),
                'throughput_per_second': round(self.metrics.throughput_per_second, 2)
            },
            'queue_status': queue_status,
            'connections': {
                'consumer': self.consumer is not None,
                'producer': self.producer is not None,
                'cache': self.cache is not None
            }
        }


# Global service instance
stream_processing_service: Optional[StreamProcessingService] = None


async def get_stream_processing_service() -> StreamProcessingService:
    """Get or create stream processing service instance."""
    global stream_processing_service
    if stream_processing_service is None:
        stream_processing_service = StreamProcessingService()
    return stream_processing_service