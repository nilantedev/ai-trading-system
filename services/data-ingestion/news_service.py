#!/usr/bin/env python3
"""News Service - Financial news and sentiment data collection."""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import asdict
import os
import re
from urllib.parse import quote

from trading_common import NewsItem, SocialSentiment, get_settings, get_logger
from trading_common.cache import get_trading_cache
from trading_common.messaging import get_pulsar_client
from trading_common.ai_models import generate_response, ModelType

logger = get_logger(__name__)
settings = get_settings()


class NewsService:
    """Handles financial news ingestion and sentiment analysis."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache = None
        self.pulsar_client = None
        self.producer = None
        
        # API configurations
        self.news_api_config = {
            'api_key': os.getenv('NEWS_API_KEY'),
            'base_url': 'https://newsapi.org/v2'
        }
        
        self.finnhub_config = {
            'api_key': os.getenv('FINNHUB_API_KEY'),
            'base_url': 'https://finnhub.io/api/v1'
        }
        
        self.reddit_config = {
            'client_id': os.getenv('REDDIT_CLIENT_ID'),
            'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
            'user_agent': 'AI-Trading-System/1.0'
        }
        
        # Financial keywords for filtering
        self.financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'dividend', 'merger', 'acquisition',
            'IPO', 'stock', 'shares', 'market', 'trading', 'investor', 'analyst',
            'upgrade', 'downgrade', 'rating', 'SEC', 'FDA', 'federal reserve',
            'inflation', 'interest rate', 'GDP', 'unemployment', 'CPI'
        ]
    
    async def start(self):
        """Initialize service connections."""
        logger.info("Starting News Service")
        
        # Initialize HTTP session
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'AI-Trading-System/1.0'}
        )
        
        # Initialize cache
        self.cache = get_trading_cache()
        
        # Initialize message producer
        try:
            self.pulsar_client = get_pulsar_client()
            self.producer = self.pulsar_client.create_producer(
                topic='persistent://trading/development/news-data',
                producer_name='news-service'
            )
            logger.info("Connected to message system")
        except Exception as e:
            logger.warning(f"Failed to connect to message system: {e}")
    
    async def stop(self):
        """Cleanup service connections."""
        if self.session:
            await self.session.close()
        if self.producer:
            self.producer.close()
        if self.pulsar_client:
            self.pulsar_client.close()
        logger.info("News Service stopped")
    
    async def collect_financial_news(
        self, 
        symbols: Optional[List[str]] = None,
        hours_back: int = 1,
        max_articles: int = 50
    ) -> List[NewsItem]:
        """Collect financial news from multiple sources."""
        logger.info(f"Collecting financial news for symbols: {symbols}")
        
        all_news = []
        
        # Collect from NewsAPI
        if self.news_api_config['api_key']:
            news_api_articles = await self._collect_from_newsapi(symbols, hours_back, max_articles // 2)
            all_news.extend(news_api_articles)
        
        # Collect from Finnhub
        if self.finnhub_config['api_key'] and symbols:
            finnhub_articles = await self._collect_from_finnhub(symbols, max_articles // 2)
            all_news.extend(finnhub_articles)
        
        # Remove duplicates based on URL or title similarity
        unique_news = self._deduplicate_news(all_news)
        
        # Analyze sentiment for each article
        for article in unique_news:
            if not article.sentiment_score:
                sentiment = await self._analyze_sentiment(article.title + " " + article.content)
                article.sentiment_score = sentiment
        
        # Cache and publish news
        for article in unique_news:
            if self.cache:
                await self.cache.set_news_item(article)
            
            if self.producer:
                try:
                    await self._publish_news(article)
                except Exception as e:
                    logger.warning(f"Failed to publish news: {e}")
        
        logger.info(f"Collected {len(unique_news)} unique news articles")
        return unique_news
    
    async def collect_social_sentiment(
        self, 
        symbols: List[str],
        hours_back: int = 1
    ) -> List[SocialSentiment]:
        """Collect social media sentiment for symbols."""
        logger.info(f"Collecting social sentiment for: {symbols}")
        
        sentiment_data = []
        
        # Collect from Reddit (if configured)
        if self.reddit_config['client_id']:
            reddit_sentiment = await self._collect_reddit_sentiment(symbols, hours_back)
            sentiment_data.extend(reddit_sentiment)
        
        return sentiment_data
    
    async def _collect_from_newsapi(
        self, 
        symbols: Optional[List[str]], 
        hours_back: int,
        max_articles: int
    ) -> List[NewsItem]:
        """Collect news from NewsAPI."""
        try:
            # Build query
            if symbols:
                query = f"({' OR '.join(symbols)}) AND (stock OR shares OR trading OR market)"
            else:
                query = "stock market OR trading OR finance OR earnings"
            
            from_date = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()
            
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': min(max_articles, 100),
                'apiKey': self.news_api_config['api_key']
            }
            
            url = f"{self.news_api_config['base_url']}/everything"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get('articles', [])
                    
                    news_items = []
                    for article in articles:
                        if self._is_financial_news(article.get('title', '') + ' ' + article.get('description', '')):
                            news_item = NewsItem(
                                title=article.get('title', ''),
                                content=article.get('description', ''),
                                source=article.get('source', {}).get('name', 'NewsAPI'),
                                published_at=datetime.fromisoformat(
                                    article.get('publishedAt', '').replace('Z', '+00:00')
                                ),
                                url=article.get('url', ''),
                                sentiment_score=None,  # Will be analyzed later
                                relevance_score=self._calculate_relevance(article, symbols),
                                symbols=self._extract_symbols(article.get('title', '') + ' ' + article.get('description', ''), symbols)
                            )
                            news_items.append(news_item)
                    
                    return news_items
                else:
                    logger.warning(f"NewsAPI error: {response.status}")
                    
        except Exception as e:
            logger.error(f"NewsAPI collection error: {e}")
        
        return []
    
    async def _collect_from_finnhub(self, symbols: List[str], max_articles: int) -> List[NewsItem]:
        """Collect company-specific news from Finnhub."""
        try:
            all_news = []
            
            for symbol in symbols[:5]:  # Limit to avoid rate limits
                params = {
                    'symbol': symbol,
                    'from': (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d'),
                    'to': datetime.utcnow().strftime('%Y-%m-%d'),
                    'token': self.finnhub_config['api_key']
                }
                
                url = f"{self.finnhub_config['base_url']}/company-news"
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        articles = await response.json()
                        
                        for article in articles[:max_articles // len(symbols)]:
                            news_item = NewsItem(
                                title=article.get('headline', ''),
                                content=article.get('summary', ''),
                                source='Finnhub',
                                published_at=datetime.fromtimestamp(article.get('datetime', 0)),
                                url=article.get('url', ''),
                                sentiment_score=None,
                                relevance_score=0.9,  # Company-specific news is highly relevant
                                symbols=[symbol]
                            )
                            all_news.append(news_item)
                    else:
                        logger.warning(f"Finnhub error for {symbol}: {response.status}")
                
                # Rate limiting
                await asyncio.sleep(0.2)
            
            return all_news
            
        except Exception as e:
            logger.error(f"Finnhub collection error: {e}")
        
        return []
    
    async def _collect_reddit_sentiment(
        self, 
        symbols: List[str], 
        hours_back: int
    ) -> List[SocialSentiment]:
        """Collect sentiment from Reddit financial subreddits."""
        try:
            # This would implement Reddit API calls to collect posts/comments
            # from subreddits like r/stocks, r/investing, r/SecurityAnalysis
            # For now, return mock data structure
            
            sentiment_data = []
            subreddits = ['stocks', 'investing', 'SecurityAnalysis', 'wallstreetbets']
            
            for symbol in symbols:
                # Mock sentiment calculation
                sentiment_data.append(SocialSentiment(
                    symbol=symbol,
                    source='reddit',
                    timestamp=datetime.utcnow(),
                    sentiment_score=0.5,  # Would calculate from actual data
                    volume=100,  # Number of mentions
                    trending_score=0.3,
                    metadata={'subreddits': subreddits}
                ))
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Reddit sentiment collection error: {e}")
        
        return []
    
    def _is_financial_news(self, text: str) -> bool:
        """Check if text contains financial keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.financial_keywords)
    
    def _calculate_relevance(
        self, 
        article: Dict, 
        symbols: Optional[List[str]] = None
    ) -> float:
        """Calculate relevance score for an article."""
        if not symbols:
            return 0.5
        
        title = article.get('title', '').upper()
        description = article.get('description', '').upper()
        text = title + ' ' + description
        
        # Count symbol mentions
        symbol_mentions = sum(1 for symbol in symbols if symbol in text)
        
        # Base relevance
        relevance = min(symbol_mentions / len(symbols), 1.0)
        
        # Boost for financial keywords
        financial_mentions = sum(1 for keyword in self.financial_keywords 
                               if keyword.upper() in text)
        relevance += min(financial_mentions * 0.1, 0.3)
        
        return min(relevance, 1.0)
    
    def _extract_symbols(
        self, 
        text: str, 
        possible_symbols: Optional[List[str]] = None
    ) -> List[str]:
        """Extract stock symbols mentioned in text."""
        if not possible_symbols:
            return []
        
        text_upper = text.upper()
        mentioned_symbols = []
        
        for symbol in possible_symbols:
            if symbol in text_upper:
                mentioned_symbols.append(symbol)
        
        return mentioned_symbols
    
    def _deduplicate_news(self, news_list: List[NewsItem]) -> List[NewsItem]:
        """Remove duplicate news articles."""
        seen_urls = set()
        seen_titles = set()
        unique_news = []
        
        for article in news_list:
            # Skip if URL already seen
            if article.url and article.url in seen_urls:
                continue
            
            # Skip if title is very similar (simple check)
            title_words = set(article.title.lower().split())
            is_duplicate = False
            
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                if len(title_words.intersection(seen_words)) / max(len(title_words), len(seen_words)) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_news.append(article)
                if article.url:
                    seen_urls.add(article.url)
                seen_titles.add(article.title.lower())
        
        return unique_news
    
    async def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using AI models."""
        try:
            prompt = f"""
            Analyze the sentiment of this financial news text and provide a sentiment score from -1 (very negative) to +1 (very positive):
            
            Text: {text[:1000]}  # Limit text length
            
            Consider:
            - Positive indicators: growth, profit, bullish, upgrade, beat expectations
            - Negative indicators: loss, decline, bearish, downgrade, miss expectations
            - Neutral indicators: mixed signals, uncertain outlook
            
            Provide only a decimal number between -1 and 1.
            """
            
            response = await generate_response(
                prompt, 
                model_preference=[ModelType.LOCAL_OLLAMA, ModelType.OPENAI]
            )
            
            # Extract numeric score from response
            score_text = response.content.strip()
            try:
                score = float(score_text)
                return max(-1.0, min(1.0, score))  # Clamp to [-1, 1]
            except ValueError:
                # Fallback: simple keyword-based sentiment
                return self._simple_sentiment_analysis(text)
                
        except Exception as e:
            logger.warning(f"AI sentiment analysis failed: {e}, using fallback")
            return self._simple_sentiment_analysis(text)
    
    def _simple_sentiment_analysis(self, text: str) -> float:
        """Simple keyword-based sentiment analysis fallback."""
        text_lower = text.lower()
        
        positive_words = [
            'profit', 'growth', 'increase', 'bullish', 'upgrade', 'beat', 'strong',
            'positive', 'gain', 'rally', 'boost', 'surge', 'soar', 'outperform'
        ]
        
        negative_words = [
            'loss', 'decline', 'bearish', 'downgrade', 'miss', 'weak', 'negative',
            'drop', 'fall', 'crash', 'plunge', 'underperform', 'concern', 'risk'
        ]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count == 0 and negative_count == 0:
            return 0.0
        
        total_words = positive_count + negative_count
        return (positive_count - negative_count) / total_words
    
    async def _publish_news(self, news_item: NewsItem):
        """Publish news item to message stream."""
        if self.producer:
            message = json.dumps(asdict(news_item), default=str)
            self.producer.send(message.encode('utf-8'))
    
    async def get_service_health(self) -> Dict:
        """Get service health status."""
        return {
            'service': 'news',
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'data_sources': {
                'news_api': bool(self.news_api_config['api_key']),
                'finnhub': bool(self.finnhub_config['api_key']),
                'reddit': bool(self.reddit_config['client_id'])
            },
            'connections': {
                'http_session': self.session is not None,
                'cache': self.cache is not None,
                'message_producer': self.producer is not None
            }
        }


# Global service instance
news_service: Optional[NewsService] = None


async def get_news_service() -> NewsService:
    """Get or create news service instance."""
    global news_service
    if news_service is None:
        news_service = NewsService()
        await news_service.start()
    return news_service