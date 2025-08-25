#!/usr/bin/env python3
"""News Integration Service - Multi-source financial news and sentiment analysis."""

import asyncio
import json
import logging
import aiohttp
import time
import re
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod

from trading_common import get_settings, get_logger
from trading_common.cache import get_trading_cache
from trading_common.messaging import get_message_consumer, get_message_producer

logger = get_logger(__name__)
settings = get_settings()


class NewsProvider(Enum):
    NEWSAPI = "newsapi"
    FINNHUB = "finnhub"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    REDDIT = "reddit"


class SentimentScore(Enum):
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


class NewsCategory(Enum):
    GENERAL = "general"
    EARNINGS = "earnings"
    MERGERS = "mergers"
    IPO = "ipo"
    ANALYST = "analyst"
    FDA = "fda"
    LEGAL = "legal"
    FINANCIAL = "financial"


@dataclass
class NewsArticle:
    """News article representation."""
    article_id: str
    title: str
    description: str
    content: Optional[str]
    url: str
    source: str
    author: Optional[str]
    published_at: datetime
    symbols: List[str]  # Related stock symbols
    category: NewsCategory
    sentiment_score: float  # -1.0 to 1.0
    sentiment_label: SentimentScore
    relevance_score: float  # 0.0 to 1.0
    impact_score: float  # 0.0 to 1.0 (predicted market impact)
    provider: NewsProvider
    language: str = "en"
    keywords: List[str] = None


@dataclass
class NewsRequest:
    """News request configuration."""
    request_id: str
    provider: NewsProvider
    symbols: List[str]
    category: Optional[NewsCategory] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = 100
    language: str = "en"


@dataclass
class SentimentAnalysis:
    """Sentiment analysis result."""
    symbol: str
    timestamp: datetime
    overall_sentiment: float  # -1.0 to 1.0
    sentiment_label: SentimentScore
    confidence: float  # 0.0 to 1.0
    article_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    trend: str  # "improving", "declining", "stable"
    key_themes: List[str]


class NewsProviderAPI(ABC):
    """Abstract base class for news provider APIs."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url')
        
        # Rate limiting
        self.rate_limit = asyncio.Semaphore(config.get('rate_limit', 10))
        self.min_request_interval = config.get('min_request_interval', 1.0)
        self.last_request_time = 0
        
        # Status tracking
        self.request_count = 0
        self.error_count = 0
        self.last_success = None
        self.last_error = None
    
    async def initialize(self):
        """Initialize the news provider connection."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'AI-Trading-System/1.0'}
        )
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Apply rate limiting."""
        async with self.rate_limit:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - time_since_last)
            
            self.last_request_time = time.time()
    
    @abstractmethod
    async def get_news(self, request: NewsRequest) -> List[NewsArticle]:
        """Get news articles."""
        pass
    
    @abstractmethod
    async def get_company_news(self, symbol: str, days: int = 7) -> List[NewsArticle]:
        """Get company-specific news."""
        pass


class NewsAPIProvider(NewsProviderAPI):
    """NewsAPI.org provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        config['base_url'] = 'https://newsapi.org/v2'
        config['rate_limit'] = 1000  # 1000 requests per day
        config['min_request_interval'] = 0.1
        super().__init__(config)
    
    async def get_news(self, request: NewsRequest) -> List[NewsArticle]:
        """Get news from NewsAPI."""
        await self._rate_limit()
        
        try:
            # Build query string for symbols
            query_parts = []
            for symbol in request.symbols:
                query_parts.extend([
                    symbol,
                    f"${symbol}",
                    f"NYSE:{symbol}",
                    f"NASDAQ:{symbol}"
                ])
            
            query = " OR ".join(query_parts)
            
            params = {
                'q': query,
                'apiKey': self.api_key,
                'pageSize': min(request.limit, 100),
                'language': request.language,
                'sortBy': 'publishedAt'
            }
            
            if request.start_date:
                params['from'] = request.start_date.strftime('%Y-%m-%d')
            if request.end_date:
                params['to'] = request.end_date.strftime('%Y-%m-%d')
            
            # Use everything endpoint for general news
            url = f"{self.base_url}/everything"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('status') == 'ok':
                        articles = []
                        
                        for article_data in data.get('articles', []):
                            if article_data.get('title') and article_data.get('publishedAt'):
                                article = await self._parse_article(article_data, request.symbols)
                                if article:
                                    articles.append(article)
                        
                        self.last_success = datetime.utcnow()
                        self.request_count += 1
                        
                        return articles
                    else:
                        logger.warning(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                else:
                    self.error_count += 1
                    self.last_error = datetime.utcnow()
                    logger.error(f"NewsAPI HTTP error: {response.status}")
            
        except Exception as e:
            self.error_count += 1
            self.last_error = datetime.utcnow()
            logger.error(f"Failed to get NewsAPI articles: {e}")
        
        return []
    
    async def get_company_news(self, symbol: str, days: int = 7) -> List[NewsArticle]:
        """Get company-specific news from NewsAPI."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        request = NewsRequest(
            request_id=f"company_news_{symbol}",
            provider=NewsProvider.NEWSAPI,
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date,
            limit=50
        )
        
        return await self.get_news(request)
    
    async def _parse_article(self, article_data: Dict[str, Any], symbols: List[str]) -> Optional[NewsArticle]:
        """Parse article data from NewsAPI."""
        try:
            published_at_str = article_data.get('publishedAt', '')
            published_at = datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
            
            title = article_data.get('title', '')
            description = article_data.get('description', '') or ''
            content = article_data.get('content', '') or description
            
            # Extract mentioned symbols
            mentioned_symbols = self._extract_symbols(f"{title} {description}", symbols)
            
            # Basic sentiment analysis
            sentiment_score, sentiment_label = self._analyze_sentiment(f"{title} {description}")
            
            # Determine category
            category = self._categorize_news(title, description)
            
            article = NewsArticle(
                article_id=f"newsapi_{hash(article_data.get('url', ''))}",
                title=title,
                description=description,
                content=content,
                url=article_data.get('url', ''),
                source=article_data.get('source', {}).get('name', 'Unknown'),
                author=article_data.get('author'),
                published_at=published_at,
                symbols=mentioned_symbols,
                category=category,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                relevance_score=self._calculate_relevance(title, description, symbols),
                impact_score=self._estimate_impact(title, description, category),
                provider=NewsProvider.NEWSAPI,
                keywords=self._extract_keywords(f"{title} {description}")
            )
            
            return article
            
        except Exception as e:
            logger.warning(f"Failed to parse NewsAPI article: {e}")
            return None
    
    def _extract_symbols(self, text: str, candidate_symbols: List[str]) -> List[str]:
        """Extract stock symbols mentioned in text."""
        mentioned = []
        text_upper = text.upper()
        
        for symbol in candidate_symbols:
            symbol_upper = symbol.upper()
            # Look for symbol mentions with various patterns
            patterns = [
                rf'\b{symbol_upper}\b',
                rf'\${symbol_upper}\b',
                rf'\b{symbol_upper}:\w+',
                rf'NYSE:{symbol_upper}',
                rf'NASDAQ:{symbol_upper}'
            ]
            
            for pattern in patterns:
                if re.search(pattern, text_upper):
                    mentioned.append(symbol_upper)
                    break
        
        return mentioned
    
    def _analyze_sentiment(self, text: str) -> tuple[float, SentimentScore]:
        """Basic sentiment analysis using keyword matching."""
        positive_keywords = [
            'bullish', 'buy', 'strong', 'growth', 'profit', 'gain', 'rise', 'up',
            'positive', 'good', 'excellent', 'outstanding', 'beat', 'exceed',
            'upgrade', 'outperform', 'rally', 'surge', 'soar', 'jump'
        ]
        
        negative_keywords = [
            'bearish', 'sell', 'weak', 'loss', 'decline', 'fall', 'down',
            'negative', 'bad', 'poor', 'terrible', 'miss', 'below',
            'downgrade', 'underperform', 'crash', 'plunge', 'drop', 'tumble'
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        # Calculate sentiment score
        total_words = len(text.split())
        if total_words == 0:
            return 0.0, SentimentScore.NEUTRAL
        
        sentiment_score = (positive_count - negative_count) / max(total_words / 20, 1)
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        # Determine sentiment label
        if sentiment_score >= 0.3:
            sentiment_label = SentimentScore.POSITIVE
        elif sentiment_score >= 0.6:
            sentiment_label = SentimentScore.VERY_POSITIVE
        elif sentiment_score <= -0.3:
            sentiment_label = SentimentScore.NEGATIVE
        elif sentiment_score <= -0.6:
            sentiment_label = SentimentScore.VERY_NEGATIVE
        else:
            sentiment_label = SentimentScore.NEUTRAL
        
        return sentiment_score, sentiment_label
    
    def _categorize_news(self, title: str, description: str) -> NewsCategory:
        """Categorize news article based on content."""
        text = f"{title} {description}".lower()
        
        if any(word in text for word in ['earnings', 'quarterly', 'revenue', 'eps']):
            return NewsCategory.EARNINGS
        elif any(word in text for word in ['merger', 'acquisition', 'acquire', 'takeover']):
            return NewsCategory.MERGERS
        elif any(word in text for word in ['ipo', 'public offering', 'debut']):
            return NewsCategory.IPO
        elif any(word in text for word in ['analyst', 'rating', 'price target', 'upgrade', 'downgrade']):
            return NewsCategory.ANALYST
        elif any(word in text for word in ['fda', 'approval', 'clinical', 'drug']):
            return NewsCategory.FDA
        elif any(word in text for word in ['lawsuit', 'legal', 'court', 'settlement']):
            return NewsCategory.LEGAL
        elif any(word in text for word in ['financial', 'balance sheet', 'debt', 'cash']):
            return NewsCategory.FINANCIAL
        else:
            return NewsCategory.GENERAL
    
    def _calculate_relevance(self, title: str, description: str, symbols: List[str]) -> float:
        """Calculate relevance score for the symbols."""
        text = f"{title} {description}".lower()
        relevance = 0.0
        
        # Check for symbol mentions
        for symbol in symbols:
            symbol_lower = symbol.lower()
            if symbol_lower in text:
                relevance += 0.3
            if f"${symbol_lower}" in text:
                relevance += 0.2
        
        # Check for financial keywords
        financial_keywords = ['stock', 'shares', 'market', 'trading', 'price', 'volume']
        relevance += sum(0.1 for keyword in financial_keywords if keyword in text)
        
        return min(relevance, 1.0)
    
    def _estimate_impact(self, title: str, description: str, category: NewsCategory) -> float:
        """Estimate potential market impact."""
        text = f"{title} {description}".lower()
        
        # Base impact by category
        category_impact = {
            NewsCategory.EARNINGS: 0.8,
            NewsCategory.MERGERS: 0.9,
            NewsCategory.IPO: 0.6,
            NewsCategory.ANALYST: 0.5,
            NewsCategory.FDA: 0.7,
            NewsCategory.LEGAL: 0.6,
            NewsCategory.FINANCIAL: 0.7,
            NewsCategory.GENERAL: 0.3
        }
        
        impact = category_impact.get(category, 0.3)
        
        # Adjust for high-impact keywords
        high_impact_words = ['breaking', 'major', 'significant', 'huge', 'massive', 'unprecedented']
        if any(word in text for word in high_impact_words):
            impact *= 1.3
        
        return min(impact, 1.0)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        # Simple keyword extraction (would use NLP library in production)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [word for word in words if word not in stop_words]
        
        # Return most common keywords
        from collections import Counter
        return [word for word, count in Counter(keywords).most_common(10)]


class FinnhubProvider(NewsProviderAPI):
    """Finnhub news provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        config['base_url'] = 'https://finnhub.io/api/v1'
        config['rate_limit'] = 60  # 60 calls per minute
        config['min_request_interval'] = 1.0
        super().__init__(config)
    
    async def get_news(self, request: NewsRequest) -> List[NewsArticle]:
        """Get news from Finnhub."""
        articles = []
        
        for symbol in request.symbols:
            symbol_articles = await self.get_company_news(symbol, 7)
            articles.extend(symbol_articles)
        
        return articles[:request.limit]
    
    async def get_company_news(self, symbol: str, days: int = 7) -> List[NewsArticle]:
        """Get company news from Finnhub."""
        await self._rate_limit()
        
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            params = {
                'symbol': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': self.api_key
            }
            
            url = f"{self.base_url}/company-news"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = []
                    
                    for article_data in data:
                        article = await self._parse_finnhub_article(article_data, symbol)
                        if article:
                            articles.append(article)
                    
                    self.last_success = datetime.utcnow()
                    self.request_count += 1
                    
                    return articles
                else:
                    self.error_count += 1
                    self.last_error = datetime.utcnow()
                    logger.error(f"Finnhub API error: {response.status}")
            
        except Exception as e:
            self.error_count += 1
            self.last_error = datetime.utcnow()
            logger.error(f"Failed to get Finnhub news for {symbol}: {e}")
        
        return []
    
    async def _parse_finnhub_article(self, article_data: Dict[str, Any], symbol: str) -> Optional[NewsArticle]:
        """Parse Finnhub article data."""
        try:
            published_at = datetime.fromtimestamp(article_data.get('datetime', 0))
            
            title = article_data.get('headline', '')
            description = article_data.get('summary', '')
            
            # Finnhub provides sentiment
            sentiment_score = float(article_data.get('sentiment', 0))
            
            if sentiment_score >= 0.3:
                sentiment_label = SentimentScore.POSITIVE
            elif sentiment_score >= 0.6:
                sentiment_label = SentimentScore.VERY_POSITIVE
            elif sentiment_score <= -0.3:
                sentiment_label = SentimentScore.NEGATIVE
            elif sentiment_score <= -0.6:
                sentiment_label = SentimentScore.VERY_NEGATIVE
            else:
                sentiment_label = SentimentScore.NEUTRAL
            
            article = NewsArticle(
                article_id=f"finnhub_{article_data.get('id', hash(article_data.get('url', '')))}",
                title=title,
                description=description,
                content=description,
                url=article_data.get('url', ''),
                source=article_data.get('source', 'Finnhub'),
                author=None,
                published_at=published_at,
                symbols=[symbol],
                category=self._categorize_news_finnhub(article_data.get('category', '')),
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                relevance_score=1.0,  # Finnhub pre-filters for relevance
                impact_score=0.5,  # Default impact
                provider=NewsProvider.FINNHUB
            )
            
            return article
            
        except Exception as e:
            logger.warning(f"Failed to parse Finnhub article: {e}")
            return None
    
    def _categorize_news_finnhub(self, category: str) -> NewsCategory:
        """Map Finnhub category to our category."""
        category_map = {
            'earnings': NewsCategory.EARNINGS,
            'merger': NewsCategory.MERGERS,
            'ipo': NewsCategory.IPO,
            'general': NewsCategory.GENERAL
        }
        
        return category_map.get(category.lower(), NewsCategory.GENERAL)


class NewsIntegrationService:
    """Service for aggregating news from multiple sources."""
    
    def __init__(self):
        self.consumer = None
        self.producer = None
        self.cache = None
        
        self.providers: Dict[str, NewsProviderAPI] = {}
        self.is_running = False
        
        # News processing
        self.news_queue = asyncio.Queue(maxsize=10000)
        self.processed_articles = {}  # URL -> ArticleID to prevent duplicates
        
        # Sentiment tracking
        self.symbol_sentiment = {}  # Symbol -> SentimentAnalysis
        
        # Performance metrics
        self.articles_processed = 0
        self.sentiment_analyses = 0
        self.api_errors = 0
        
        # Tracked symbols for news monitoring
        self.tracked_symbols: Set[str] = set()
        
    async def start(self):
        """Initialize and start news integration service."""
        logger.info("Starting News Integration Service")
        
        try:
            # Initialize connections
            self.consumer = await get_message_consumer()
            self.producer = await get_message_producer()
            self.cache = get_trading_cache()
            
            # Initialize news providers
            await self._initialize_providers()
            
            # Subscribe to news requests
            await self._setup_subscriptions()
            
            # Start processing tasks
            self.is_running = True
            
            tasks = [
                asyncio.create_task(self._process_news_queue()),
                asyncio.create_task(self._consume_messages()),
                asyncio.create_task(self._periodic_news_collection()),
                asyncio.create_task(self._sentiment_analysis_task()),
                asyncio.create_task(self._cleanup_old_data())
            ]
            
            logger.info("News integration service started with 5 concurrent tasks")
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Failed to start news integration service: {e}")
            raise
    
    async def stop(self):
        """Stop news integration service gracefully."""
        logger.info("Stopping News Integration Service")
        self.is_running = False
        
        # Clean up provider connections
        for provider in self.providers.values():
            await provider.cleanup()
        
        if self.consumer:
            await self.consumer.close()
        if self.producer:
            await self.producer.close()
        
        logger.info("News Integration Service stopped")
    
    async def _initialize_providers(self):
        """Initialize configured news providers."""
        try:
            provider_configs = settings.get('news_providers', {})
            
            for provider_name, config in provider_configs.items():
                provider_type = NewsProvider(config.get('type'))
                
                if provider_type == NewsProvider.NEWSAPI:
                    provider = NewsAPIProvider(config)
                elif provider_type == NewsProvider.FINNHUB:
                    provider = FinnhubProvider(config)
                else:
                    logger.warning(f"Unsupported news provider: {provider_type}")
                    continue
                
                await provider.initialize()
                self.providers[provider_name] = provider
                
                logger.info(f"Initialized {provider_type.value} provider: {provider_name}")
            
            if not self.providers:
                # Initialize default NewsAPI provider (requires API key)
                logger.info("No news providers configured - news service will run in limited mode")
                
        except Exception as e:
            logger.error(f"Failed to initialize news providers: {e}")
            # Don't raise - service can run without news providers for testing
    
    async def _setup_subscriptions(self):
        """Subscribe to news requests and symbol tracking."""
        try:
            await self.consumer.subscribe_news_requests(
                self._handle_news_request,
                subscription_name="news-service-requests"
            )
            
            await self.consumer.subscribe_symbol_updates(
                self._handle_symbol_update,
                subscription_name="news-service-symbols"
            )
            
            logger.info("Subscribed to news requests and symbol updates")
        except Exception as e:
            logger.warning(f"News service subscription setup failed: {e}")
    
    async def _consume_messages(self):
        """Consume messages from subscribed topics."""
        try:
            await self.consumer.start_consuming()
        except Exception as e:
            logger.error(f"Message consumption failed: {e}")
    
    async def _handle_news_request(self, message):
        """Handle incoming news request."""
        try:
            request_data = json.loads(message) if isinstance(message, str) else message
            
            news_request = NewsRequest(
                request_id=request_data.get('request_id', f"news_req_{time.time()}"),
                provider=NewsProvider(request_data.get('provider', 'newsapi')),
                symbols=request_data.get('symbols', []),
                category=NewsCategory(request_data['category']) if request_data.get('category') else None,
                start_date=datetime.fromisoformat(request_data['start_date']) if request_data.get('start_date') else None,
                end_date=datetime.fromisoformat(request_data['end_date']) if request_data.get('end_date') else None,
                limit=request_data.get('limit', 100)
            )
            
            await self.news_queue.put(news_request)
            
        except Exception as e:
            logger.error(f"Failed to handle news request: {e}")
    
    async def _handle_symbol_update(self, message):
        """Handle symbol tracking updates."""
        try:
            symbol_data = json.loads(message) if isinstance(message, str) else message
            
            action = symbol_data.get('action', 'add')
            symbol = symbol_data.get('symbol', '').upper()
            
            if action == 'add' and symbol:
                self.tracked_symbols.add(symbol)
                logger.info(f"Added {symbol} to news tracking")
            elif action == 'remove' and symbol:
                self.tracked_symbols.discard(symbol)
                logger.info(f"Removed {symbol} from news tracking")
            
        except Exception as e:
            logger.error(f"Failed to handle symbol update: {e}")
    
    async def _process_news_queue(self):
        """Process news requests."""
        while self.is_running:
            try:
                # Wait for news request
                news_request = await asyncio.wait_for(
                    self.news_queue.get(),
                    timeout=1.0
                )
                
                # Execute request
                articles = await self._execute_news_request(news_request)
                
                # Process and publish articles
                for article in articles:
                    if article.url not in self.processed_articles:
                        self.processed_articles[article.url] = article.article_id
                        
                        # Cache article
                        await self._cache_article(article)
                        
                        # Publish article
                        await self._publish_article(article)
                        
                        self.articles_processed += 1
                
                self.news_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"News processing error: {e}")
                self.api_errors += 1
    
    async def _execute_news_request(self, request: NewsRequest) -> List[NewsArticle]:
        """Execute news request with provider fallback."""
        articles = []
        
        for provider_name, provider in self.providers.items():
            try:
                provider_articles = await provider.get_news(request)
                articles.extend(provider_articles)
                
                if len(articles) >= request.limit:
                    break
                    
            except Exception as e:
                logger.warning(f"News provider {provider_name} failed: {e}")
                continue
        
        return articles[:request.limit]
    
    async def _periodic_news_collection(self):
        """Collect news periodically for tracked symbols."""
        while self.is_running:
            try:
                await asyncio.sleep(1800)  # Collect every 30 minutes
                
                if self.tracked_symbols and self.providers:
                    # Create news request for tracked symbols
                    symbols_list = list(self.tracked_symbols)
                    
                    # Process in batches of 10 symbols
                    for i in range(0, len(symbols_list), 10):
                        batch = symbols_list[i:i+10]
                        
                        request = NewsRequest(
                            request_id=f"periodic_{datetime.utcnow().timestamp()}",
                            provider=NewsProvider.NEWSAPI,
                            symbols=batch,
                            limit=50
                        )
                        
                        await self.news_queue.put(request)
                
            except Exception as e:
                logger.warning(f"Periodic news collection error: {e}")
    
    async def _sentiment_analysis_task(self):
        """Perform periodic sentiment analysis for tracked symbols."""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Analyze every hour
                
                for symbol in self.tracked_symbols:
                    sentiment = await self._calculate_symbol_sentiment(symbol)
                    if sentiment:
                        self.symbol_sentiment[symbol] = sentiment
                        
                        # Cache and publish sentiment
                        await self._cache_sentiment(sentiment)
                        await self._publish_sentiment(sentiment)
                        
                        self.sentiment_analyses += 1
                
            except Exception as e:
                logger.warning(f"Sentiment analysis error: {e}")
    
    async def _calculate_symbol_sentiment(self, symbol: str) -> Optional[SentimentAnalysis]:
        """Calculate sentiment analysis for a symbol based on recent news."""
        try:
            if not self.cache:
                return None
            
            # Get recent articles for symbol from cache
            cache_pattern = f"news_article:*{symbol}*"
            article_keys = []  # Would get from cache scan
            
            articles = []
            for key in article_keys[:50]:  # Limit to recent 50 articles
                cached_article = await self.cache.get_json(key)
                if cached_article and symbol.upper() in cached_article.get('symbols', []):
                    articles.append(cached_article)
            
            if not articles:
                return None
            
            # Calculate aggregated sentiment
            sentiment_scores = [article['sentiment_score'] for article in articles]
            overall_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            # Count sentiment categories
            positive_count = sum(1 for score in sentiment_scores if score > 0.1)
            negative_count = sum(1 for score in sentiment_scores if score < -0.1)
            neutral_count = len(sentiment_scores) - positive_count - negative_count
            
            # Determine overall sentiment label
            if overall_sentiment >= 0.3:
                sentiment_label = SentimentScore.POSITIVE
            elif overall_sentiment >= 0.6:
                sentiment_label = SentimentScore.VERY_POSITIVE
            elif overall_sentiment <= -0.3:
                sentiment_label = SentimentScore.NEGATIVE
            elif overall_sentiment <= -0.6:
                sentiment_label = SentimentScore.VERY_NEGATIVE
            else:
                sentiment_label = SentimentScore.NEUTRAL
            
            # Calculate confidence based on article count and consistency
            confidence = min(len(articles) / 20.0, 1.0)  # More articles = higher confidence
            score_std = (sum((score - overall_sentiment) ** 2 for score in sentiment_scores) / len(sentiment_scores)) ** 0.5
            confidence *= max(0.5, 1.0 - score_std)  # Lower variance = higher confidence
            
            # Determine trend (simplified)
            recent_articles = sorted(articles, key=lambda x: x['published_at'])[-10:]
            if len(recent_articles) >= 5:
                recent_sentiment = sum(article['sentiment_score'] for article in recent_articles[-5:]) / 5
                older_sentiment = sum(article['sentiment_score'] for article in recent_articles[:5]) / 5
                
                if recent_sentiment > older_sentiment + 0.1:
                    trend = "improving"
                elif recent_sentiment < older_sentiment - 0.1:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "stable"
            
            # Extract key themes (simplified)
            all_keywords = []
            for article in articles:
                all_keywords.extend(article.get('keywords', []))
            
            from collections import Counter
            key_themes = [word for word, count in Counter(all_keywords).most_common(5)]
            
            return SentimentAnalysis(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                overall_sentiment=overall_sentiment,
                sentiment_label=sentiment_label,
                confidence=confidence,
                article_count=len(articles),
                positive_count=positive_count,
                negative_count=negative_count,
                neutral_count=neutral_count,
                trend=trend,
                key_themes=key_themes
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate sentiment for {symbol}: {e}")
            return None
    
    async def _cleanup_old_data(self):
        """Clean up old news data and processed article tracking."""
        while self.is_running:
            try:
                await asyncio.sleep(86400)  # Clean up daily
                
                # Remove old processed article URLs (keep 7 days)
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                
                # Would implement cleanup logic for processed_articles
                # based on article timestamps
                
                logger.debug("Performed news data cleanup")
                
            except Exception as e:
                logger.warning(f"News data cleanup error: {e}")
    
    async def _cache_article(self, article: NewsArticle):
        """Cache news article."""
        try:
            if self.cache:
                cache_key = f"news_article:{article.article_id}"
                article_data = asdict(article)
                
                # Convert datetime and enum fields
                article_data['published_at'] = article.published_at.isoformat()
                article_data['category'] = article.category.value
                article_data['sentiment_label'] = article.sentiment_label.value
                article_data['provider'] = article.provider.value
                
                await self.cache.set_json(cache_key, article_data, ttl=604800)  # 7 days
        except Exception as e:
            logger.warning(f"Failed to cache article: {e}")
    
    async def _cache_sentiment(self, sentiment: SentimentAnalysis):
        """Cache sentiment analysis."""
        try:
            if self.cache:
                cache_key = f"news_sentiment:{sentiment.symbol}:latest"
                sentiment_data = asdict(sentiment)
                sentiment_data['timestamp'] = sentiment.timestamp.isoformat()
                sentiment_data['sentiment_label'] = sentiment.sentiment_label.value
                
                await self.cache.set_json(cache_key, sentiment_data, ttl=3600)  # 1 hour
        except Exception as e:
            logger.warning(f"Failed to cache sentiment: {e}")
    
    async def _publish_article(self, article: NewsArticle):
        """Publish news article."""
        try:
            if self.producer:
                article_message = {
                    'article_id': article.article_id,
                    'title': article.title,
                    'symbols': article.symbols,
                    'sentiment_score': article.sentiment_score,
                    'sentiment_label': article.sentiment_label.value,
                    'category': article.category.value,
                    'impact_score': article.impact_score,
                    'published_at': article.published_at.isoformat(),
                    'url': article.url
                }
                
                # Would publish to news topic
                logger.debug(f"Publishing news article: {article.title[:50]}...")
                
        except Exception as e:
            logger.warning(f"Failed to publish article: {e}")
    
    async def _publish_sentiment(self, sentiment: SentimentAnalysis):
        """Publish sentiment analysis."""
        try:
            if self.producer:
                sentiment_message = {
                    'symbol': sentiment.symbol,
                    'overall_sentiment': sentiment.overall_sentiment,
                    'sentiment_label': sentiment.sentiment_label.value,
                    'confidence': sentiment.confidence,
                    'trend': sentiment.trend,
                    'article_count': sentiment.article_count,
                    'timestamp': sentiment.timestamp.isoformat()
                }
                
                # Would publish to sentiment topic
                logger.debug(f"Publishing sentiment analysis for {sentiment.symbol}")
                
        except Exception as e:
            logger.warning(f"Failed to publish sentiment: {e}")
    
    async def add_tracked_symbol(self, symbol: str):
        """Add symbol to news tracking."""
        self.tracked_symbols.add(symbol.upper())
        logger.info(f"Added {symbol} to news tracking")
    
    async def remove_tracked_symbol(self, symbol: str):
        """Remove symbol from news tracking."""
        self.tracked_symbols.discard(symbol.upper())
        logger.info(f"Removed {symbol} from news tracking")
    
    async def get_symbol_sentiment(self, symbol: str) -> Optional[SentimentAnalysis]:
        """Get latest sentiment analysis for a symbol."""
        return self.symbol_sentiment.get(symbol.upper())
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get service health status."""
        provider_health = {}
        for provider_name, provider in self.providers.items():
            provider_health[provider_name] = {
                'request_count': provider.request_count,
                'error_count': provider.error_count,
                'last_success': provider.last_success.isoformat() if provider.last_success else None,
                'last_error': provider.last_error.isoformat() if provider.last_error else None
            }
        
        return {
            'service': 'news_integration_service',
            'status': 'healthy' if self.is_running else 'stopped',
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {
                'articles_processed': self.articles_processed,
                'sentiment_analyses': self.sentiment_analyses,
                'api_errors': self.api_errors,
                'tracked_symbols': len(self.tracked_symbols),
                'active_providers': len(self.providers)
            },
            'providers': provider_health,
            'tracked_symbols': list(self.tracked_symbols),
            'connections': {
                'consumer': self.consumer is not None,
                'producer': self.producer is not None,
                'cache': self.cache is not None
            }
        }


# Global service instance
news_service: Optional[NewsIntegrationService] = None


async def get_news_service() -> NewsIntegrationService:
    """Get or create news integration service instance."""
    global news_service
    if news_service is None:
        news_service = NewsIntegrationService()
    return news_service