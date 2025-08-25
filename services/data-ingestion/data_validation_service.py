#!/usr/bin/env python3
"""Data Validation Service - Quality checks and data validation."""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

from trading_common import MarketData, NewsItem, get_settings, get_logger
from trading_common.cache import get_trading_cache
from trading_common.messaging import get_pulsar_client

logger = get_logger(__name__)
settings = get_settings()


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    suggested_fix: Optional[str] = None
    timestamp: datetime = datetime.utcnow()


@dataclass
class DataQualityMetrics:
    """Data quality metrics."""
    total_records: int
    valid_records: int
    invalid_records: int
    completeness_score: float  # 0-1
    accuracy_score: float  # 0-1
    timeliness_score: float  # 0-1
    consistency_score: float  # 0-1
    overall_score: float  # 0-1
    issues: List[ValidationResult]
    timestamp: datetime = datetime.utcnow()


class DataValidationService:
    """Handles data quality checks and validation."""
    
    def __init__(self):
        self.cache = None
        self.pulsar_client = None
        self.producer = None
        
        # Validation rules configuration
        self.market_data_rules = {
            'price_range': (0.01, 100000.0),  # Min/max prices
            'volume_range': (0, 1000000000),  # Min/max volume
            'price_change_limit': 0.50,  # Max 50% price change per minute
            'stale_data_minutes': 15,  # Data older than 15 minutes is stale
        }
        
        self.news_rules = {
            'min_title_length': 10,
            'max_title_length': 200,
            'min_content_length': 50,
            'max_content_length': 10000,
            'sentiment_range': (-1.0, 1.0),
            'relevance_range': (0.0, 1.0),
            'stale_news_hours': 24,  # News older than 24 hours is stale
        }
    
    async def start(self):
        """Initialize service connections."""
        logger.info("Starting Data Validation Service")
        
        # Initialize cache
        self.cache = get_trading_cache()
        
        # Initialize message producer
        try:
            self.pulsar_client = get_pulsar_client()
            self.producer = self.pulsar_client.create_producer(
                topic='persistent://trading/development/data-quality',
                producer_name='validation-service'
            )
            logger.info("Connected to message system")
        except Exception as e:
            logger.warning(f"Failed to connect to message system: {e}")
    
    async def stop(self):
        """Cleanup service connections."""
        if self.producer:
            self.producer.close()
        if self.pulsar_client:
            self.pulsar_client.close()
        logger.info("Data Validation Service stopped")
    
    async def validate_market_data(self, data: MarketData) -> List[ValidationResult]:
        """Validate market data quality."""
        results = []
        
        # Basic field validation
        results.extend(self._validate_market_data_fields(data))
        
        # Price validation
        results.extend(self._validate_price_data(data))
        
        # Volume validation
        results.extend(self._validate_volume_data(data))
        
        # Temporal validation
        results.extend(self._validate_timestamp(data))
        
        # Historical consistency check
        if self.cache:
            historical_results = await self._validate_historical_consistency(data)
            results.extend(historical_results)
        
        return results
    
    async def validate_news_data(self, news: NewsItem) -> List[ValidationResult]:
        """Validate news data quality."""
        results = []
        
        # Basic field validation
        results.extend(self._validate_news_fields(news))
        
        # Content validation
        results.extend(self._validate_news_content(news))
        
        # Sentiment validation
        results.extend(self._validate_sentiment_scores(news))
        
        # Temporal validation
        results.extend(self._validate_news_timestamp(news))
        
        return results
    
    async def calculate_data_quality_metrics(
        self, 
        symbol: str, 
        hours_back: int = 24
    ) -> DataQualityMetrics:
        """Calculate comprehensive data quality metrics."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        
        # Get market data for the period
        market_data_records = await self._get_market_data_for_period(symbol, start_time, end_time)
        
        # Get news data for the period
        news_records = await self._get_news_data_for_period(symbol, start_time, end_time)
        
        total_records = len(market_data_records) + len(news_records)
        all_issues = []
        valid_count = 0
        
        # Validate market data
        for data in market_data_records:
            issues = await self.validate_market_data(data)
            all_issues.extend(issues)
            if not any(issue.severity == ValidationSeverity.ERROR for issue in issues):
                valid_count += 1
        
        # Validate news data
        for news in news_records:
            issues = await self.validate_news_data(news)
            all_issues.extend(issues)
            if not any(issue.severity == ValidationSeverity.ERROR for issue in issues):
                valid_count += 1
        
        # Calculate scores
        completeness_score = self._calculate_completeness_score(market_data_records, hours_back)
        accuracy_score = valid_count / max(total_records, 1)
        timeliness_score = self._calculate_timeliness_score(market_data_records, news_records)
        consistency_score = self._calculate_consistency_score(market_data_records)
        
        overall_score = (completeness_score + accuracy_score + timeliness_score + consistency_score) / 4
        
        metrics = DataQualityMetrics(
            total_records=total_records,
            valid_records=valid_count,
            invalid_records=total_records - valid_count,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            timeliness_score=timeliness_score,
            consistency_score=consistency_score,
            overall_score=overall_score,
            issues=all_issues
        )
        
        # Publish metrics to message system
        if self.producer:
            try:
                await self._publish_quality_metrics(metrics)
            except Exception as e:
                logger.warning(f"Failed to publish quality metrics: {e}")
        
        return metrics
    
    def _validate_market_data_fields(self, data: MarketData) -> List[ValidationResult]:
        """Validate required fields in market data."""
        results = []
        
        # Check required fields
        if not data.symbol:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Missing symbol",
                field="symbol",
                suggested_fix="Provide valid symbol identifier"
            ))
        
        if not data.timestamp:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Missing timestamp",
                field="timestamp",
                suggested_fix="Provide valid timestamp"
            ))
        
        if data.open is None or data.high is None or data.low is None or data.close is None:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Missing OHLC price data",
                field="prices",
                suggested_fix="Ensure all OHLC prices are provided"
            ))
        
        return results
    
    def _validate_price_data(self, data: MarketData) -> List[ValidationResult]:
        """Validate price data reasonableness."""
        results = []
        
        prices = [data.open, data.high, data.low, data.close]
        
        # Check price range
        min_price, max_price = self.market_data_rules['price_range']
        for price in prices:
            if price is not None:
                if price < min_price:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Price {price} below minimum threshold {min_price}",
                        field="price",
                        suggested_fix="Check data source for potential errors"
                    ))
                elif price > max_price:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        message=f"Price {price} above maximum threshold {max_price}",
                        field="price",
                        suggested_fix="Verify high-value stock data"
                    ))
        
        # Check OHLC relationships
        if all(p is not None for p in prices):
            if data.high < max(data.open, data.close):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="High price is lower than open or close",
                    field="high",
                    suggested_fix="Verify OHLC data integrity"
                ))
            
            if data.low > min(data.open, data.close):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="Low price is higher than open or close",
                    field="low",
                    suggested_fix="Verify OHLC data integrity"
                ))
        
        return results
    
    def _validate_volume_data(self, data: MarketData) -> List[ValidationResult]:
        """Validate volume data."""
        results = []
        
        if data.volume is not None:
            min_vol, max_vol = self.market_data_rules['volume_range']
            
            if data.volume < min_vol:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message="Volume is negative or zero",
                    field="volume",
                    suggested_fix="Check if zero volume is expected"
                ))
            elif data.volume > max_vol:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Unusually high volume: {data.volume}",
                    field="volume",
                    suggested_fix="Verify high volume event"
                ))
        
        return results
    
    def _validate_timestamp(self, data: MarketData) -> List[ValidationResult]:
        """Validate timestamp data."""
        results = []
        
        if data.timestamp:
            now = datetime.utcnow()
            age_minutes = (now - data.timestamp).total_seconds() / 60
            
            # Check for stale data
            if age_minutes > self.market_data_rules['stale_data_minutes']:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Stale data: {age_minutes:.1f} minutes old",
                    field="timestamp",
                    suggested_fix="Check data feed latency"
                ))
            
            # Check for future timestamp
            if data.timestamp > now:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="Future timestamp detected",
                    field="timestamp",
                    suggested_fix="Verify system time synchronization"
                ))
        
        return results
    
    async def _validate_historical_consistency(self, data: MarketData) -> List[ValidationResult]:
        """Validate data against historical patterns."""
        results = []
        
        try:
            # Get recent data for comparison
            if self.cache:
                recent_data = await self.cache.get_recent_market_data(data.symbol, hours=1)
                
                if recent_data and len(recent_data) > 0:
                    last_price = recent_data[-1].close
                    current_price = data.close
                    
                    if last_price and current_price:
                        price_change = abs(current_price - last_price) / last_price
                        
                        if price_change > self.market_data_rules['price_change_limit']:
                            results.append(ValidationResult(
                                is_valid=False,
                                severity=ValidationSeverity.WARNING,
                                message=f"Large price change detected: {price_change:.2%}",
                                field="price_consistency",
                                suggested_fix="Verify market event or data source"
                            ))
        
        except Exception as e:
            logger.warning(f"Historical consistency check failed: {e}")
        
        return results
    
    def _validate_news_fields(self, news: NewsItem) -> List[ValidationResult]:
        """Validate news data fields."""
        results = []
        
        # Check title
        if not news.title:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Missing news title",
                field="title"
            ))
        elif len(news.title) < self.news_rules['min_title_length']:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="Title too short",
                field="title"
            ))
        elif len(news.title) > self.news_rules['max_title_length']:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="Title too long",
                field="title"
            ))
        
        # Check content
        if news.content and len(news.content) > self.news_rules['max_content_length']:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="Content too long",
                field="content",
                suggested_fix="Truncate content to reasonable length"
            ))
        
        return results
    
    def _validate_news_content(self, news: NewsItem) -> List[ValidationResult]:
        """Validate news content quality."""
        results = []
        
        # Check for duplicate content patterns
        if news.title and news.content:
            title_words = set(news.title.lower().split())
            content_words = set(news.content.lower().split())
            
            # If title and content are too similar, it might be duplicate
            if len(title_words.intersection(content_words)) / len(title_words.union(content_words)) > 0.9:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.INFO,
                    message="Title and content are very similar",
                    field="content_quality"
                ))
        
        return results
    
    def _validate_sentiment_scores(self, news: NewsItem) -> List[ValidationResult]:
        """Validate sentiment scores."""
        results = []
        
        # Check sentiment score range
        if news.sentiment_score is not None:
            min_sent, max_sent = self.news_rules['sentiment_range']
            if news.sentiment_score < min_sent or news.sentiment_score > max_sent:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Sentiment score {news.sentiment_score} out of range [{min_sent}, {max_sent}]",
                    field="sentiment_score"
                ))
        
        # Check relevance score range
        if news.relevance_score is not None:
            min_rel, max_rel = self.news_rules['relevance_range']
            if news.relevance_score < min_rel or news.relevance_score > max_rel:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Relevance score {news.relevance_score} out of range [{min_rel}, {max_rel}]",
                    field="relevance_score"
                ))
        
        return results
    
    def _validate_news_timestamp(self, news: NewsItem) -> List[ValidationResult]:
        """Validate news timestamp."""
        results = []
        
        if news.published_at:
            now = datetime.utcnow()
            age_hours = (now - news.published_at).total_seconds() / 3600
            
            # Check for very old news
            if age_hours > self.news_rules['stale_news_hours']:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.INFO,
                    message=f"Old news: {age_hours:.1f} hours old",
                    field="published_at"
                ))
            
            # Check for future timestamp
            if news.published_at > now:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="Future publication date",
                    field="published_at"
                ))
        
        return results
    
    def _calculate_completeness_score(self, market_data: List[MarketData], hours_back: int) -> float:
        """Calculate data completeness score."""
        if not market_data:
            return 0.0
        
        # Expected data points (assuming 1-minute bars during trading hours)
        # Simplified: assume 6.5 hours per trading day
        expected_points = hours_back * 60 if hours_back <= 24 else int(hours_back / 24) * 390
        actual_points = len(market_data)
        
        return min(actual_points / expected_points, 1.0)
    
    def _calculate_timeliness_score(
        self, 
        market_data: List[MarketData], 
        news_data: List[NewsItem]
    ) -> float:
        """Calculate data timeliness score."""
        now = datetime.utcnow()
        timeliness_scores = []
        
        # Score market data timeliness
        for data in market_data[-10:]:  # Check last 10 records
            if data.timestamp:
                age_minutes = (now - data.timestamp).total_seconds() / 60
                score = max(0, 1 - (age_minutes / 60))  # Full score for data < 1 hour old
                timeliness_scores.append(score)
        
        # Score news data timeliness
        for news in news_data[-5:]:  # Check last 5 news items
            if news.published_at:
                age_hours = (now - news.published_at).total_seconds() / 3600
                score = max(0, 1 - (age_hours / 24))  # Full score for news < 24 hours old
                timeliness_scores.append(score)
        
        return statistics.mean(timeliness_scores) if timeliness_scores else 0.0
    
    def _calculate_consistency_score(self, market_data: List[MarketData]) -> float:
        """Calculate data consistency score."""
        if len(market_data) < 2:
            return 1.0
        
        consistency_scores = []
        
        # Check price continuity
        for i in range(1, len(market_data)):
            prev_data = market_data[i-1]
            curr_data = market_data[i]
            
            if prev_data.close and curr_data.open:
                # In continuous trading, current open should be close to previous close
                price_gap = abs(curr_data.open - prev_data.close) / prev_data.close
                score = max(0, 1 - (price_gap * 10))  # Penalize gaps > 10%
                consistency_scores.append(score)
        
        return statistics.mean(consistency_scores) if consistency_scores else 1.0
    
    async def _get_market_data_for_period(
        self, 
        symbol: str, 
        start: datetime, 
        end: datetime
    ) -> List[MarketData]:
        """Get market data for validation period."""
        if self.cache:
            try:
                hours_back = int((end - start).total_seconds() / 3600)
                return await self.cache.get_recent_market_data(symbol, hours_back)
            except Exception as e:
                logger.warning(f"Failed to get market data for validation: {e}")
        
        return []
    
    async def _get_news_data_for_period(
        self, 
        symbol: str, 
        start: datetime, 
        end: datetime
    ) -> List[NewsItem]:
        """Get news data for validation period."""
        if self.cache:
            try:
                hours_back = int((end - start).total_seconds() / 3600)
                return await self.cache.get_recent_news(symbol, hours_back)
            except Exception as e:
                logger.warning(f"Failed to get news data for validation: {e}")
        
        return []
    
    async def _publish_quality_metrics(self, metrics: DataQualityMetrics):
        """Publish quality metrics to message stream."""
        if self.producer:
            message = json.dumps(asdict(metrics), default=str)
            self.producer.send(message.encode('utf-8'))
    
    async def get_service_health(self) -> Dict:
        """Get service health status."""
        return {
            'service': 'data_validation',
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'validation_rules': {
                'market_data_rules': len(self.market_data_rules),
                'news_rules': len(self.news_rules)
            },
            'connections': {
                'cache': self.cache is not None,
                'message_producer': self.producer is not None
            }
        }


# Global service instance
data_validation_service: Optional[DataValidationService] = None


async def get_data_validation_service() -> DataValidationService:
    """Get or create data validation service instance."""
    global data_validation_service
    if data_validation_service is None:
        data_validation_service = DataValidationService()
        await data_validation_service.start()
    return data_validation_service