#!/usr/bin/env python3
"""
Social Media Data Collector - Real-time social sentiment and trend analysis
Collects data from Twitter, Reddit, Discord, Telegram, and news sources for trading signals.
"""

import asyncio
import aiohttp
import json
import logging
import re
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import hashlib
import os
from urllib.parse import urlencode

from trading_common import get_settings, get_logger
from trading_common.cache import get_trading_cache
from trading_common.messaging import get_pulsar_client
try:
    from shared.vector.weaviate_schema import get_weaviate_client  # type: ignore
except Exception:  # noqa: BLE001
    get_weaviate_client = None  # type: ignore

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class SocialSignal:
    """Social media signal data."""
    source: str  # twitter, reddit, discord, telegram, news
    symbol: str
    sentiment_score: float  # -1 to 1
    engagement_score: float  # 0-1 based on likes, shares, comments
    influence_score: float  # 0-1 based on author influence
    content: str
    author: str
    timestamp: datetime
    url: Optional[str] = None
    # Use a default factory for lists to avoid mutable default pitfalls and None checks
    tags: List[str] = field(default_factory=list)
    

@dataclass
class SocialTrendData:
    """Aggregated social trend data for a symbol."""
    symbol: str
    timestamp: datetime
    overall_sentiment: float  # -1 to 1
    mention_volume: int
    trending_score: float  # 0-1
    top_keywords: List[str]
    source_breakdown: Dict[str, int]  # mentions per source
    influential_mentions: List[SocialSignal]


class TwitterCollector:
    """Twitter data collector using Twitter API v2."""
    
    def __init__(self):
        self.api_key = os.getenv('TWITTER_API_KEY')
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.session = None
        
    async def initialize(self):
        """Initialize Twitter collector."""
        if not self.bearer_token:
            logger.warning("Twitter Bearer Token not found - Twitter data unavailable")
            return False
            
        self.session = aiohttp.ClientSession(
            headers={
                'Authorization': f'Bearer {self.bearer_token}',
                'Content-Type': 'application/json'
            },
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return True
    
    async def collect_symbol_mentions(self, symbol: str, hours_back: int = 1) -> List[SocialSignal]:
        """Collect Twitter mentions for a symbol."""
        if not self.session:
            return []
            
        try:
            # Search for symbol mentions
            query_params = {
                'query': f'${symbol} OR #{symbol} -is:retweet lang:en',
                'tweet.fields': 'created_at,public_metrics,author_id,context_annotations',
                'user.fields': 'public_metrics,verified',
                'expansions': 'author_id',
                'max_results': 100
            }
            
            url = f"https://api.twitter.com/2/tweets/search/recent?{urlencode(query_params)}"
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Twitter API error {response.status} for {symbol}")
                    return []
                    
                data = await response.json()
                return self._process_twitter_data(data, symbol)
                
        except Exception as e:
            logger.error(f"Twitter collection error for {symbol}: {e}")
            return []
    
    def _process_twitter_data(self, data: Dict, symbol: str) -> List[SocialSignal]:
        """Process Twitter API response."""
        signals = []
        
        tweets = data.get('data', [])
        users = {user['id']: user for user in data.get('includes', {}).get('users', [])}
        
        for tweet in tweets:
            try:
                # Calculate sentiment (simplified - would use proper sentiment analysis)
                content = tweet.get('text', '')
                sentiment = self._calculate_sentiment(content)
                
                # Calculate engagement score
                metrics = tweet.get('public_metrics', {})
                engagement = (
                    metrics.get('retweet_count', 0) * 3 +
                    metrics.get('like_count', 0) * 1 +
                    metrics.get('reply_count', 0) * 2
                )
                engagement_score = min(engagement / 1000, 1.0)  # Normalize
                
                # Calculate influence score
                author_id = tweet.get('author_id')
                author = users.get(author_id, {})
                follower_count = author.get('public_metrics', {}).get('followers_count', 0)
                is_verified = author.get('verified', False)
                
                influence_score = min(follower_count / 100000, 1.0)  # Normalize to 100k followers
                if is_verified:
                    influence_score *= 1.5
                influence_score = min(influence_score, 1.0)
                
                signal = SocialSignal(
                    source='twitter',
                    symbol=symbol,
                    sentiment_score=sentiment,
                    engagement_score=engagement_score,
                    influence_score=influence_score,
                    content=content[:500],  # Truncate for storage
                    author=author.get('username', 'unknown'),
                    timestamp=datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00')),
                    url=f"https://twitter.com/{author.get('username', 'unknown')}/status/{tweet['id']}"
                )
                signals.append(signal)
                
            except Exception as e:
                logger.warning(f"Error processing tweet: {e}")
                continue
                
        return signals
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score (simplified version)."""
        positive_words = ['bullish', 'buy', 'moon', 'rocket', 'up', 'rise', 'pump', 'long', 'calls']
        negative_words = ['bearish', 'sell', 'crash', 'dump', 'down', 'fall', 'short', 'puts']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
            
        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        return sentiment
    
    async def close(self):
        """Close Twitter collector."""
        if self.session:
            await self.session.close()


class RedditCollector:
    """Reddit data collector using Reddit API."""
    
    def __init__(self):
        self.client_id = os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.session = None
        self.access_token = None
        
    async def initialize(self):
        """Initialize Reddit collector."""
        if not self.client_id or not self.client_secret:
            logger.warning("Reddit credentials not found - Reddit data unavailable")
            return False
            
        # Get access token
        await self._get_access_token()
        return self.access_token is not None
    
    async def _get_access_token(self):
        """Get Reddit API access token."""
        try:
            auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
            data = {
                'grant_type': 'client_credentials'
            }
            headers = {
                'User-Agent': 'TradingBot/1.0'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'https://www.reddit.com/api/v1/access_token',
                    auth=auth,
                    data=data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        self.access_token = token_data['access_token']
                        
                        # Create session with token
                        self.session = aiohttp.ClientSession(
                            headers={
                                'Authorization': f'bearer {self.access_token}',
                                'User-Agent': 'TradingBot/1.0'
                            },
                            timeout=aiohttp.ClientTimeout(total=30)
                        )
                        
        except Exception as e:
            logger.error(f"Failed to get Reddit access token: {e}")
    
    async def collect_symbol_mentions(self, symbol: str, hours_back: int = 1) -> List[SocialSignal]:
        """Collect Reddit mentions for a symbol."""
        if not self.session:
            return []
            
        signals = []
        
        # Search in multiple subreddits
        subreddits = ['wallstreetbets', 'stocks', 'investing', 'SecurityAnalysis', 'ValueInvesting']
        
        for subreddit in subreddits:
            try:
                url = f'https://oauth.reddit.com/r/{subreddit}/search.json'
                # Map hours_back to Reddit time filter buckets
                if hours_back <= 1:
                    t_filter = 'hour'
                elif hours_back <= 24:
                    t_filter = 'day'
                elif hours_back <= 7 * 24:
                    t_filter = 'week'
                elif hours_back <= 30 * 24:
                    t_filter = 'month'
                elif hours_back <= 365 * 24:
                    t_filter = 'year'
                else:
                    t_filter = 'all'
                params = {
                    'q': f'${symbol}',
                    'sort': 'new',
                    'limit': 25,
                    't': t_filter
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status != 200:
                        continue
                        
                    data = await response.json()
                    subreddit_signals = self._process_reddit_data(data, symbol, subreddit, hours_back)
                    signals.extend(subreddit_signals)
                    
            except Exception as e:
                logger.warning(f"Reddit collection error for {subreddit}/{symbol}: {e}")
                continue
                
        return signals
    
    def _process_reddit_data(self, data: Dict, symbol: str, subreddit: str, hours_back: int) -> List[SocialSignal]:
        """Process Reddit API response."""
        signals = []
        
        posts = data.get('data', {}).get('children', [])
        
        for post_data in posts:
            try:
                post = post_data.get('data', {})
                
                # Skip if too old
                created_utc = post.get('created_utc', 0)
                post_time = datetime.fromtimestamp(created_utc)
                # Honor requested hours_back instead of a fixed 1 hour window
                if (datetime.utcnow() - post_time).total_seconds() > max(1, hours_back) * 3600:
                    continue
                
                title = post.get('title', '')
                selftext = post.get('selftext', '')
                content = f"{title} {selftext}"
                
                # Calculate sentiment
                sentiment = self._calculate_sentiment(content)
                
                # Calculate engagement
                upvote_ratio = post.get('upvote_ratio', 0.5)
                num_comments = post.get('num_comments', 0)
                score = post.get('score', 0)
                
                engagement_score = min((score + num_comments * 2) / 100, 1.0)
                
                # Influence based on upvote ratio and subreddit
                influence_multiplier = {'wallstreetbets': 1.5, 'stocks': 1.2}.get(subreddit, 1.0)
                influence_score = min(upvote_ratio * influence_multiplier, 1.0)
                
                signal = SocialSignal(
                    source='reddit',
                    symbol=symbol,
                    sentiment_score=sentiment,
                    engagement_score=engagement_score,
                    influence_score=influence_score,
                    content=content[:500],
                    author=post.get('author', 'unknown'),
                    timestamp=post_time,
                    url=f"https://reddit.com{post.get('permalink', '')}"
                )
                signals.append(signal)
                
            except Exception as e:
                logger.warning(f"Error processing Reddit post: {e}")
                continue
                
        return signals
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score."""
        positive_words = ['bullish', 'buy', 'calls', 'moon', 'rocket', 'diamond hands', 'hold', 'pump']
        negative_words = ['bearish', 'sell', 'puts', 'crash', 'dump', 'paper hands', 'short']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
            
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    async def close(self):
        """Close Reddit collector."""
        if self.session:
            await self.session.close()


class NewsCollector:
    """Financial news collector."""
    
    def __init__(self):
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.session = None
        
    async def initialize(self):
        """Initialize news collector."""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return True
    
    async def collect_symbol_news(self, symbol: str, hours_back: int = 2) -> List[SocialSignal]:
        """Collect news mentions for a symbol."""
        if not self.session:
            return []
            
        signals = []
        
        # Multiple news sources
        sources = [
            ('newsapi', self._collect_newsapi),
            ('finnhub', self._collect_finnhub_news),
            ('alpha_vantage', self._collect_alpha_vantage_news)
        ]
        
        for source_name, collector_func in sources:
            try:
                source_signals = await collector_func(symbol, hours_back)
                signals.extend(source_signals)
            except Exception as e:
                logger.warning(f"News collection error for {source_name}/{symbol}: {e}")
                
        return signals
    
    async def _collect_newsapi(self, symbol: str, hours_back: int) -> List[SocialSignal]:
        """Collect from NewsAPI."""
        if not self.news_api_key:
            return []
            
        try:
            from_date = (datetime.utcnow() - timedelta(hours=hours_back)).strftime('%Y-%m-%d')
            
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': f'"{symbol}"',
                'apiKey': self.news_api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'from': from_date,
                'pageSize': 50
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    return []
                    
                data = await response.json()
                return self._process_news_data(data, symbol, 'newsapi')
                
        except Exception as e:
            logger.warning(f"NewsAPI error: {e}")
            return []
    
    async def _collect_finnhub_news(self, symbol: str, hours_back: int) -> List[SocialSignal]:
        """Collect from Finnhub."""
        finnhub_key = os.getenv('FINNHUB_API_KEY')
        if not finnhub_key:
            return []
            
        try:
            from_date = datetime.utcnow() - timedelta(hours=hours_back)
            to_date = datetime.utcnow()
            
            url = 'https://finnhub.io/api/v1/company-news'
            params = {
                'symbol': symbol,
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'token': finnhub_key
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    return []
                    
                data = await response.json()
                return self._process_finnhub_news(data, symbol)
                
        except Exception as e:
            logger.warning(f"Finnhub news error: {e}")
            return []
    
    async def _collect_alpha_vantage_news(self, symbol: str, hours_back: int) -> List[SocialSignal]:
        """Collect from Alpha Vantage."""
        av_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not av_key:
            return []
            
        try:
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': av_key,
                'limit': 50
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    return []
                    
                data = await response.json()
                return self._process_alpha_vantage_news(data, symbol)
                
        except Exception as e:
            logger.warning(f"Alpha Vantage news error: {e}")
            return []
    
    def _process_news_data(self, data: Dict, symbol: str, source: str) -> List[SocialSignal]:
        """Process news API data."""
        signals = []
        
        articles = data.get('articles', [])
        
        for article in articles:
            try:
                title = article.get('title', '')
                description = article.get('description', '')
                content = f"{title} {description}"
                
                # Calculate sentiment
                sentiment = self._calculate_news_sentiment(content)
                
                # News engagement is based on source reputation
                source_name = article.get('source', {}).get('name', '')
                engagement_score = self._get_source_credibility(source_name)
                
                signal = SocialSignal(
                    source=f'news_{source}',
                    symbol=symbol,
                    sentiment_score=sentiment,
                    engagement_score=engagement_score,
                    influence_score=engagement_score,  # News influence = credibility
                    content=content[:500],
                    author=source_name,
                    timestamp=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                    url=article.get('url')
                )
                signals.append(signal)
                
            except Exception as e:
                logger.warning(f"Error processing news article: {e}")
                continue
                
        return signals
    
    def _process_finnhub_news(self, data: List, symbol: str) -> List[SocialSignal]:
        """Process Finnhub news data."""
        signals = []
        
        for article in data:
            try:
                headline = article.get('headline', '')
                summary = article.get('summary', '')
                content = f"{headline} {summary}"
                
                sentiment = self._calculate_news_sentiment(content)
                
                signal = SocialSignal(
                    source='news_finnhub',
                    symbol=symbol,
                    sentiment_score=sentiment,
                    engagement_score=0.8,  # Finnhub is credible
                    influence_score=0.8,
                    content=content[:500],
                    author='Finnhub',
                    timestamp=datetime.fromtimestamp(article.get('datetime', 0)),
                    url=article.get('url')
                )
                signals.append(signal)
                
            except Exception as e:
                logger.warning(f"Error processing Finnhub article: {e}")
                continue
                
        return signals
    
    def _process_alpha_vantage_news(self, data: Dict, symbol: str) -> List[SocialSignal]:
        """Process Alpha Vantage news sentiment data."""
        signals = []
        
        feed = data.get('feed', [])
        
        for article in feed:
            try:
                title = article.get('title', '')
                summary = article.get('summary', '')
                content = f"{title} {summary}"
                
                # Use provided sentiment if available
                ticker_sentiment = article.get('ticker_sentiment', [])
                sentiment_score = 0.0
                
                for ticker_data in ticker_sentiment:
                    if ticker_data.get('ticker') == symbol:
                        sentiment_score = float(ticker_data.get('ticker_sentiment_score', 0))
                        break
                
                # Fallback to our sentiment calculation
                if sentiment_score == 0.0:
                    sentiment_score = self._calculate_news_sentiment(content)
                
                signal = SocialSignal(
                    source='news_alpha_vantage',
                    symbol=symbol,
                    sentiment_score=sentiment_score,
                    engagement_score=0.75,
                    influence_score=0.75,
                    content=content[:500],
                    author='Alpha Vantage',
                    timestamp=datetime.strptime(article.get('time_published', '20240101T000000'), '%Y%m%dT%H%M%S'),
                    url=article.get('url')
                )
                signals.append(signal)
                
            except Exception as e:
                logger.warning(f"Error processing Alpha Vantage article: {e}")
                continue
                
        return signals
    
    def _calculate_news_sentiment(self, text: str) -> float:
        """Calculate sentiment for news content."""
        positive_words = ['profit', 'growth', 'increase', 'positive', 'strong', 'beat', 'exceed', 'bullish']
        negative_words = ['loss', 'decline', 'decrease', 'negative', 'weak', 'miss', 'below', 'bearish']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
            
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _get_source_credibility(self, source_name: str) -> float:
        """Get credibility score for news source."""
        high_credibility = ['reuters', 'bloomberg', 'wall street journal', 'financial times']
        medium_credibility = ['cnbc', 'marketwatch', 'yahoo finance', 'seeking alpha']
        
        source_lower = source_name.lower()
        
        if any(source in source_lower for source in high_credibility):
            return 0.9
        elif any(source in source_lower for source in medium_credibility):
            return 0.7
        else:
            return 0.5
    
    async def close(self):
        """Close news collector."""
        if self.session:
            await self.session.close()


class SocialMediaCollector:
    """Main social media data collection service."""
    
    def __init__(self):
        self.twitter = TwitterCollector()
        self.reddit = RedditCollector()
        self.news = NewsCollector()
        self.enable_structured_logs = os.getenv('ENABLE_STRUCTURED_INGEST_LOGS','false').lower() in ('1','true','yes')
        # Messaging & persistence
        self.pulsar_client = None
        self.producer = None
        self._pulsar_topic = 'persistent://trading/production/social-data'
        self.enable_qdb_persist = os.getenv('ENABLE_QUESTDB_SOCIAL_PERSIST', 'false').lower() in ('1','true','yes')
        try:
            from questdb.ingress import Sender as _QSender, TimestampNanos as _QTs  # type: ignore
            self._qdb_sender_cls = _QSender
            self._qdb_ts = _QTs
        except Exception:  # noqa: BLE001
            self._qdb_sender_cls = None
            self._qdb_ts = None
        self._qdb_conf: Optional[str] = None
        # Optional vector store persistence
        self.enable_weaviate_persist = os.getenv('ENABLE_WEAVIATE_SOCIAL_PERSIST', 'false').lower() in ('1','true','yes')

        self.cache = None
        self.is_initialized = False
        
        # Tracking
        self.signals_collected = 0
        self.collection_errors = 0
        
    async def initialize(self):
        """Initialize all collectors."""
        logger.info("Initializing Social Media Collector")
        
        # Acquire cache client (async) – previously missing await caused a coroutine to leak
        try:
            self.cache = await get_trading_cache()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Cache init failed for social collector: {e}")
            self.cache = None
        
        # Initialize collectors
        twitter_ok = await self.twitter.initialize()
        reddit_ok = await self.reddit.initialize()
        news_ok = await self.news.initialize()
        
        # Init Pulsar producer (best-effort)
        try:
            self.pulsar_client = get_pulsar_client()
            self.producer = self.pulsar_client.create_producer(
                topic=self._pulsar_topic,
                producer_name='social-media-collector'
            )
            logger.info("Social producer connected")
        except Exception as e:
            logger.warning(f"Social producer init failed: {e}")

        # Configure QuestDB sender conf string (reusing env vars)
        try:
            if self.enable_qdb_persist and self._qdb_sender_cls:
                host = os.getenv('QUESTDB_HOST', 'trading-questdb')
                proto = os.getenv('QUESTDB_INGEST_PROTOCOL', 'tcp').strip().lower()
                if proto == 'http':
                    port = int(os.getenv('QUESTDB_HTTP_PORT', '9000'))
                    self._qdb_conf = f"http::addr={host}:{port};"
                else:
                    port = int(os.getenv('QUESTDB_LINE_TCP_PORT', '9009'))
                    self._qdb_conf = f"tcp::addr={host}:{port};"
        except Exception:
            self._qdb_conf = None

        self.is_initialized = True
        
        logger.info(f"Social Media Collector initialized - "
                   f"Twitter: {'✓' if twitter_ok else '✗'}, "
                   f"Reddit: {'✓' if reddit_ok else '✗'}, "
                   f"News: {'✓' if news_ok else '✗'}")
    
    async def collect_social_data(self, symbols: List[str], hours_back: int = 1) -> Dict[str, List[SocialSignal]]:
        """Collect social data for multiple symbols."""
        if not self.is_initialized:
            await self.initialize()
            
        results = {}
        
        for symbol in symbols:
            try:
                symbol_signals = []
                if self.enable_structured_logs:
                    try:
                        logger.info("social_collect_symbol_start", extra={"event":"social_collect_symbol_start","symbol":symbol,"hours_back":hours_back})
                    except Exception:
                        pass
                
                # Collect from all sources concurrently
                tasks = [
                    self.twitter.collect_symbol_mentions(symbol, hours_back),
                    self.reddit.collect_symbol_mentions(symbol, hours_back),
                    self.news.collect_symbol_news(symbol, hours_back)
                ]
                
                source_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Combine results
                for result in source_results:
                    if isinstance(result, list):
                        symbol_signals.extend(result)
                    elif isinstance(result, Exception):
                        logger.warning(f"Collection error for {symbol}: {result}")
                        self.collection_errors += 1
                
                results[symbol] = symbol_signals
                self.signals_collected += len(symbol_signals)
                
                # Cache results (ensure datetime is JSON-serializable)
                if self.cache and symbol_signals:
                    cache_key = f"social_signals:{symbol}:latest"
                    cache_data = []
                    for signal in symbol_signals:
                        payload = asdict(signal)
                        ts = payload.get("timestamp")
                        if isinstance(ts, datetime):
                            payload["timestamp"] = ts.isoformat()
                        cache_data.append(payload)
                    await self.cache.set_json(cache_key, cache_data, ttl=300)
                # Persist and publish if enabled
                try:
                    await self._persist_and_publish(symbol_signals)
                except Exception as e:
                    logger.debug(f"Persist/publish social signals failed for {symbol}: {e}")
                
                logger.debug(f"Collected {len(symbol_signals)} social signals for {symbol}")
                if self.enable_structured_logs:
                    try:
                        logger.info(
                            "social_collect_symbol_complete",
                            extra={"event":"social_collect_symbol_complete","symbol":symbol,"count":len(symbol_signals)}
                        )
                    except Exception:
                        pass
                
            except Exception as e:
                logger.error(f"Failed to collect social data for {symbol}: {e}")
                results[symbol] = []
                
        return results
    
    async def get_trending_analysis(self, symbols: List[str]) -> Dict[str, SocialTrendData]:
        """Get trending analysis for symbols."""
        social_data = await self.collect_social_data(symbols, hours_back=2)
        
        trending_data = {}
        
        for symbol, signals in social_data.items():
            if not signals:
                continue
                
            # Calculate aggregate metrics
            mention_volume = len(signals)
            overall_sentiment = sum(s.sentiment_score for s in signals) / len(signals) if signals else 0.0
            
            # Calculate trending score
            recent_signals = [s for s in signals if (datetime.utcnow() - s.timestamp).total_seconds() < 3600]
            trending_score = min(len(recent_signals) / 50, 1.0)  # Normalize to 50 mentions
            
            # Extract top keywords
            all_content = ' '.join(s.content for s in signals)
            keywords = self._extract_keywords(all_content)
            
            # Source breakdown
            source_breakdown = defaultdict(int)
            for signal in signals:
                source_breakdown[signal.source] += 1
            
            # Get most influential mentions
            influential_mentions = sorted(signals, key=lambda x: x.influence_score, reverse=True)[:5]
            
            trending_data[symbol] = SocialTrendData(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                overall_sentiment=overall_sentiment,
                mention_volume=mention_volume,
                trending_score=trending_score,
                top_keywords=keywords[:10],
                source_breakdown=dict(source_breakdown),
                influential_mentions=influential_mentions
            )
            
        return trending_data
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract trending keywords from text."""
        # Simple keyword extraction (would use proper NLP in production)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Count frequency
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
            
        # Return top keywords
        return [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)]
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get collector performance metrics."""
        return {
            'signals_collected': self.signals_collected,
            'collection_errors': self.collection_errors,
            'error_rate': self.collection_errors / max(self.signals_collected, 1),
            'collectors_initialized': self.is_initialized,
            'twitter_available': self.twitter.session is not None,
            'reddit_available': self.reddit.session is not None,
            'news_available': self.news.session is not None
        }
    
    async def close(self):
        """Close all collectors."""
        await self.twitter.close()
        await self.reddit.close()
        await self.news.close()
        try:
            if self.pulsar_client:
                if self.producer:
                    self.producer.close()
                self.pulsar_client.close()
        except Exception:
            pass

    async def _persist_and_publish(self, signals: List[SocialSignal]) -> None:
        """Persist signals to QuestDB and publish to Pulsar if configured.

        Table: social_signals (created implicitly)
          symbols: symbol, source
          columns: sentiment (float), engagement (float), influence (float), author (str), url (str), content (str)
          ts: signal.timestamp
        """
        if not signals:
            return
        # Publish
        if self.producer:
            try:
                for s in signals:
                    payload = asdict(s)
                    # Convert datetime to iso
                    if isinstance(payload.get('timestamp'), datetime):
                        payload['timestamp'] = payload['timestamp'].isoformat()
                    self.producer.send(json.dumps(payload, default=str).encode('utf-8'))
            except Exception as e:
                logger.debug(f"Social publish failed: {e}")
        # Persist to QuestDB (ILP preferred)
        ilp_ok = False
        if self.enable_qdb_persist and self._qdb_sender_cls and self._qdb_ts and self._qdb_conf:
            try:
                with self._qdb_sender_cls.from_conf(self._qdb_conf) as s:  # type: ignore[arg-type]
                    for sig in signals:
                        try:
                            at_ts = self._qdb_ts.from_datetime(sig.timestamp)
                            s.row(
                                'social_signals',
                                symbols={'symbol': sig.symbol.upper(), 'source': sig.source},
                                columns={
                                    'sentiment': float(sig.sentiment_score or 0.0),
                                    'engagement': float(sig.engagement_score or 0.0),
                                    'influence': float(sig.influence_score or 0.0),
                                    'author': str(sig.author or '')[:120],
                                    'url': str(sig.url or '')[:300],
                                    'content': str(sig.content or '')[:500],
                                },
                                at=at_ts,
                            )
                        except Exception:
                            continue
                    s.flush()
                ilp_ok = True
            except Exception as e:
                logger.debug(f"Social QuestDB ILP persist failed: {e}")

        # HTTP /exec fallback with CREATE TABLE IF NOT EXISTS
        if not ilp_ok:
            try:
                host = os.getenv('QUESTDB_HOST', 'trading-questdb')
                http_port = int(os.getenv('QUESTDB_HTTP_PORT', '9000'))
                qdb_url = os.getenv('QUESTDB_HTTP_URL', f"http://{host}:{http_port}/exec")
                import aiohttp as _aio
                async with _aio.ClientSession(timeout=_aio.ClientTimeout(total=20)) as sess:
                    # Create table if missing
                    create_sql = (
                        "create table if not exists social_signals ("
                        "symbol symbol, source symbol, sentiment double, engagement double, influence double, "
                        "author string, url string, content string, ts timestamp) timestamp(ts) PARTITION BY DAY"
                    )
                    async with sess.get(qdb_url, params={"query": create_sql}) as r:
                        await r.text()
                    # Insert rows in small batches
                    vals = []
                    for sig in signals:
                        try:
                            ts_iso = (sig.timestamp.isoformat() if isinstance(sig.timestamp, datetime) else str(sig.timestamp)).replace('Z','')
                            ts_iso = ts_iso.split('.')[0] + ".000000Z"
                            vals.append(
                                "('" + sig.symbol.upper().replace("'","") + "','" + sig.source.replace("'","") + "',"
                                + f"{float(sig.sentiment_score or 0.0):.6f},{float(sig.engagement_score or 0.0):.6f},{float(sig.influence_score or 0.0):.6f},"
                                + "'" + (sig.author or '').replace("'","")[:120] + "','" + (sig.url or '').replace("'","")[:300] + "','"
                                + (sig.content or '').replace("'","")[:500] + "',"
                                + f"to_timestamp('{ts_iso}', 'yyyy-MM-ddTHH:mm:ss.SSSSSSZ'))"
                            )
                        except Exception:
                            continue
                    # Chunk inserts to avoid overly large query strings
                    for i in range(0, len(vals), 500):
                        chunk = vals[i:i+500]
                        if not chunk:
                            continue
                        ins = "insert into social_signals(symbol,source,sentiment,engagement,influence,author,url,content,ts) values " + ",".join(chunk)
                        async with sess.get(qdb_url, params={"query": ins}) as r2:
                            await r2.text()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Social QuestDB HTTP persist failed: {e}")
        # Best-effort Weaviate indexing (optional)
        if self.enable_weaviate_persist and signals and get_weaviate_client is not None:
            try:
                client = await asyncio.to_thread(get_weaviate_client)
                coll = client.collections.get('SocialSentiment')
                batch_size = int(os.getenv('WEAVIATE_INDEX_BATCH_SIZE', '64'))
                # Map signals to properties matching desired schema
                payload = []
                for s in signals:
                    try:
                        props = {
                            'symbol': s.symbol,
                            'platform': s.source,
                            'sentiment': f"{float(s.sentiment_score or 0.0):.4f}",
                            'momentum': f"{float(s.engagement_score or 0.0):.4f}",
                            'topics': [],
                            # Weaviate expects RFC3339 (with timezone). Normalize to UTC with Z suffix.
                            'timestamp': (
                                s.timestamp.replace(tzinfo=timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')
                                if isinstance(s.timestamp, datetime) else str(s.timestamp)
                            ),
                            'influential_mentions': f"{float(s.influence_score or 0.0):.4f}",
                            'confidence': '',
                        }
                        payload.append(props)
                    except Exception:
                        continue
                for i in range(0, len(payload), batch_size):
                    chunk = payload[i:i+batch_size]
                    with coll.batch.dynamic() as batch:
                        for props in chunk:
                            batch.add_object(properties=props)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Social Weaviate indexing skipped/failed: {e}")


# Global social media collector instance
social_collector: Optional[SocialMediaCollector] = None


async def get_social_media_collector() -> SocialMediaCollector:
    """Get or create social media collector instance."""
    global social_collector
    if social_collector is None:
        social_collector = SocialMediaCollector()
        await social_collector.initialize()
    return social_collector