#!/usr/bin/env python3
"""
FinBERT Sentiment Analysis Service
Financial sentiment analysis using the ProsusAI/finbert model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Result from sentiment analysis."""
    text: str
    sentiment: str  # positive, negative, neutral
    confidence: float
    scores: Dict[str, float]
    
    
class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial text.
    Uses ProsusAI/finbert model pre-trained on financial data.
    """
    
    def __init__(self):
        """Initialize FinBERT model and tokenizer."""
        self.model_name = "ProsusAI/finbert"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Load pre-trained FinBERT
            logger.info(f"Loading FinBERT model from {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Label mapping for FinBERT
            self.labels = ['positive', 'negative', 'neutral']
            
            logger.info(f"FinBERT model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Financial text to analyze
            
        Returns:
            SentimentResult with sentiment and confidence scores
        """
        if not text or not text.strip():
            return SentimentResult(
                text=text,
                sentiment='neutral',
                confidence=0.0,
                scores={'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
            )
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions = predictions.cpu().numpy()[0]
            
            # Get sentiment scores
            scores = {
                label: float(score) 
                for label, score in zip(self.labels, predictions)
            }
            
            # Determine primary sentiment
            sentiment_idx = np.argmax(predictions)
            sentiment = self.labels[sentiment_idx]
            confidence = float(predictions[sentiment_idx])
            
            return SentimentResult(
                text=text[:100] + "..." if len(text) > 100 else text,
                sentiment=sentiment,
                confidence=confidence,
                scores=scores
            )
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return SentimentResult(
                text=text[:100] + "..." if len(text) > 100 else text,
                sentiment='neutral',
                confidence=0.0,
                scores={'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
            )
    
    def batch_analyze(self, texts: List[str], batch_size: int = 32) -> List[SentimentResult]:
        """
        Analyze sentiment for multiple texts in batches.
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process at once
            
        Returns:
            List of SentimentResult objects
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Filter empty texts
            valid_texts = [t for t in batch if t and t.strip()]
            if not valid_texts:
                results.extend([
                    SentimentResult(
                        text=t,
                        sentiment='neutral',
                        confidence=0.0,
                        scores={'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
                    ) for t in batch
                ])
                continue
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    valid_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get predictions for batch
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predictions = predictions.cpu().numpy()
                
                # Process each result
                for text, pred in zip(valid_texts, predictions):
                    scores = {
                        label: float(score) 
                        for label, score in zip(self.labels, pred)
                    }
                    
                    sentiment_idx = np.argmax(pred)
                    sentiment = self.labels[sentiment_idx]
                    confidence = float(pred[sentiment_idx])
                    
                    results.append(SentimentResult(
                        text=text[:100] + "..." if len(text) > 100 else text,
                        sentiment=sentiment,
                        confidence=confidence,
                        scores=scores
                    ))
                    
            except Exception as e:
                logger.error(f"Error in batch analysis: {e}")
                # Fallback to neutral for failed batch
                results.extend([
                    SentimentResult(
                        text=t[:100] + "..." if len(t) > 100 else t,
                        sentiment='neutral',
                        confidence=0.0,
                        scores={'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
                    ) for t in valid_texts
                ])
        
        return results
    
    def analyze_news_impact(self, headlines: List[str]) -> Dict[str, any]:
        """
        Analyze overall sentiment impact from news headlines.
        
        Args:
            headlines: List of news headlines
            
        Returns:
            Dictionary with sentiment statistics and market impact assessment
        """
        if not headlines:
            return {
                'total_headlines': 0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'average_confidence': 0.0,
                'market_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'high_impact_headlines': []
            }
        
        # Analyze all headlines
        results = self.batch_analyze(headlines)
        
        # Calculate statistics
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_confidence = 0.0
        high_impact = []
        
        for result in results:
            sentiment_counts[result.sentiment] += 1
            total_confidence += result.confidence
            
            # High impact: strong sentiment with high confidence
            if result.confidence > 0.8 and result.sentiment != 'neutral':
                high_impact.append({
                    'text': result.text,
                    'sentiment': result.sentiment,
                    'confidence': result.confidence
                })
        
        # Calculate sentiment score (-1 to 1)
        total = len(results)
        sentiment_score = (
            (sentiment_counts['positive'] - sentiment_counts['negative']) / total
            if total > 0 else 0.0
        )
        
        # Determine overall market sentiment
        if sentiment_score > 0.3:
            market_sentiment = 'bullish'
        elif sentiment_score < -0.3:
            market_sentiment = 'bearish'
        else:
            market_sentiment = 'neutral'
        
        return {
            'total_headlines': total,
            'sentiment_distribution': sentiment_counts,
            'average_confidence': total_confidence / total if total > 0 else 0.0,
            'market_sentiment': market_sentiment,
            'sentiment_score': sentiment_score,
            'high_impact_headlines': sorted(
                high_impact, 
                key=lambda x: x['confidence'], 
                reverse=True
            )[:5]  # Top 5 high impact headlines
        }
    
    async def analyze_sentiment_async(self, text: str) -> SentimentResult:
        """Async wrapper for sentiment analysis."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze_sentiment, text)
    
    async def batch_analyze_async(
        self, 
        texts: List[str], 
        batch_size: int = 32
    ) -> List[SentimentResult]:
        """Async wrapper for batch sentiment analysis."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.batch_analyze, texts, batch_size)


# Singleton instance
_finbert_instance: Optional[FinBERTSentimentAnalyzer] = None


def get_finbert_analyzer() -> FinBERTSentimentAnalyzer:
    """Get or create FinBERT analyzer instance."""
    global _finbert_instance
    if _finbert_instance is None:
        _finbert_instance = FinBERTSentimentAnalyzer()
    return _finbert_instance


# Example usage and testing
if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = get_finbert_analyzer()
    
    # Test texts
    test_texts = [
        "Apple's Q3 earnings exceeded expectations with record revenue growth.",
        "The Federal Reserve raised interest rates by 75 basis points amid inflation concerns.",
        "Tesla stock plummeted after disappointing delivery numbers.",
        "Gold prices remained stable as investors await economic data.",
        "Banking sector faces significant losses due to credit defaults.",
        "Microsoft announces major AI partnership, stock surges 5%."
    ]
    
    print("FinBERT Sentiment Analysis Test")
    print("=" * 50)
    
    # Analyze each text
    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"\nText: {text[:80]}...")
        print(f"Sentiment: {result.sentiment}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Scores: {result.scores}")
    
    # Test batch analysis
    print("\n" + "=" * 50)
    print("Batch Analysis - Market Impact Assessment")
    print("=" * 50)
    
    impact = analyzer.analyze_news_impact(test_texts)
    print(f"\nMarket Sentiment: {impact['market_sentiment']}")
    print(f"Sentiment Score: {impact['sentiment_score']:.3f}")
    print(f"Distribution: {impact['sentiment_distribution']}")
    print(f"Average Confidence: {impact['average_confidence']:.3f}")
    
    if impact['high_impact_headlines']:
        print("\nHigh Impact Headlines:")
        for item in impact['high_impact_headlines']:
            print(f"  - [{item['sentiment']}] {item['text']} (conf: {item['confidence']:.3f})")