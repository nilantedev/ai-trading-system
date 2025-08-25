#!/usr/bin/env python3
"""
Unit tests for Signal Generation Service
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sys
sys.path.append('services/signal-generator')
from signal_generation_service import SignalGenerationService


class TestSignalGenerationService:
    """Test cases for SignalGenerationService class."""

    @pytest_asyncio.fixture
    async def service(self, mock_redis, mock_database, test_settings):
        """Create SignalGenerationService instance for testing."""
        service = SignalGenerationService()
        service.redis_client = mock_redis
        service.database = mock_database
        service.settings = test_settings
        return service

    @pytest.mark.asyncio
    async def test_generate_signal_buy(self, service, sample_market_data):
        """Test BUY signal generation."""
        # Mock favorable conditions for BUY signal
        with patch.object(service, '_calculate_technical_indicators') as mock_indicators:
            mock_indicators.return_value = {
                "rsi": 35.0,  # Oversold
                "macd": 2.5,  # Bullish
                "bollinger_position": 0.2,  # Near lower band
                "volume_ratio": 1.8  # High volume
            }
            
            signal = await service.generate_signal("AAPL", sample_market_data)
            
            assert signal is not None
            assert signal["signal_type"] == "BUY"
            assert signal["symbol"] == "AAPL"
            assert signal["confidence"] > 0.7
            assert "price_target" in signal
            assert "stop_loss" in signal

    @pytest.mark.asyncio
    async def test_generate_signal_sell(self, service, sample_market_data):
        """Test SELL signal generation."""
        # Mock conditions for SELL signal
        with patch.object(service, '_calculate_technical_indicators') as mock_indicators:
            mock_indicators.return_value = {
                "rsi": 75.0,  # Overbought
                "macd": -1.5,  # Bearish
                "bollinger_position": 0.9,  # Near upper band
                "volume_ratio": 2.0  # High volume
            }
            
            signal = await service.generate_signal("AAPL", sample_market_data)
            
            assert signal is not None
            assert signal["signal_type"] == "SELL"
            assert signal["confidence"] > 0.7

    @pytest.mark.asyncio
    async def test_generate_signal_hold(self, service, sample_market_data):
        """Test HOLD signal generation."""
        # Mock neutral conditions
        with patch.object(service, '_calculate_technical_indicators') as mock_indicators:
            mock_indicators.return_value = {
                "rsi": 50.0,  # Neutral
                "macd": 0.1,  # Neutral
                "bollinger_position": 0.5,  # Middle
                "volume_ratio": 1.0  # Normal volume
            }
            
            signal = await service.generate_signal("AAPL", sample_market_data)
            
            assert signal is not None
            assert signal["signal_type"] == "HOLD"
            assert signal["confidence"] < 0.6  # Lower confidence for HOLD

    @pytest.mark.asyncio
    async def test_get_current_signals(self, service, sample_trading_signal):
        """Test retrieval of current active signals."""
        # Mock database returning signals
        service.database.fetch.return_value = [
            {
                "id": "signal_001",
                "symbol": "AAPL",
                "signal_type": "BUY",
                "confidence": 0.85,
                "created_at": datetime.utcnow()
            }
        ]
        
        signals = await service.get_current_signals()
        
        assert len(signals) == 1
        assert signals[0]["symbol"] == "AAPL"
        assert signals[0]["signal_type"] == "BUY"

    @pytest.mark.asyncio
    async def test_get_signals_by_symbol(self, service):
        """Test retrieval of signals for specific symbol."""
        mock_signals = [
            {
                "id": "signal_001",
                "symbol": "AAPL",
                "signal_type": "BUY",
                "confidence": 0.85
            },
            {
                "id": "signal_002", 
                "symbol": "AAPL",
                "signal_type": "SELL",
                "confidence": 0.75
            }
        ]
        
        service.database.fetch.return_value = mock_signals
        
        signals = await service.get_signals_by_symbol("AAPL")
        
        assert len(signals) == 2
        assert all(s["symbol"] == "AAPL" for s in signals)

    @pytest.mark.asyncio
    async def test_get_signals_by_strategy(self, service):
        """Test retrieval of signals by strategy."""
        mock_signals = [
            {
                "id": "signal_001",
                "strategy": "momentum_breakout",
                "signal_type": "BUY",
                "confidence": 0.88
            }
        ]
        
        service.database.fetch.return_value = mock_signals
        
        signals = await service.get_signals_by_strategy("momentum_breakout")
        
        assert len(signals) == 1
        assert signals[0]["strategy"] == "momentum_breakout"

    @pytest.mark.asyncio
    async def test_signal_validation(self, service, sample_trading_signal):
        """Test signal validation logic."""
        # Valid signal
        is_valid = service._validate_signal(sample_trading_signal)
        assert is_valid is True
        
        # Invalid signal - missing required fields
        invalid_signal = {
            "symbol": "AAPL",
            # Missing signal_type
            "confidence": 0.85
        }
        
        is_valid = service._validate_signal(invalid_signal)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_signal_strength_calculation(self, service):
        """Test signal strength calculation."""
        indicators = {
            "rsi": 30.0,  # Strong oversold
            "macd": 3.0,  # Strong bullish
            "bollinger_position": 0.1,  # Very oversold
            "volume_ratio": 2.5  # Very high volume
        }
        
        strength = service._calculate_signal_strength(indicators, "BUY")
        
        assert 0.8 <= strength <= 1.0  # Should be high strength

    @pytest.mark.asyncio
    async def test_price_target_calculation(self, service, sample_market_data):
        """Test price target calculation for signals."""
        # BUY signal price target
        price_target = service._calculate_price_target(
            sample_market_data["price"], "BUY", 0.85
        )
        
        assert price_target > sample_market_data["price"]  # Target should be higher for BUY
        
        # SELL signal price target
        price_target = service._calculate_price_target(
            sample_market_data["price"], "SELL", 0.85
        )
        
        assert price_target < sample_market_data["price"]  # Target should be lower for SELL

    @pytest.mark.asyncio
    async def test_stop_loss_calculation(self, service, sample_market_data):
        """Test stop loss calculation."""
        stop_loss = service._calculate_stop_loss(
            sample_market_data["price"], "BUY", 0.85
        )
        
        assert stop_loss < sample_market_data["price"]  # Stop loss below entry for BUY

    @pytest.mark.asyncio
    async def test_signal_expiry(self, service):
        """Test signal expiry logic."""
        # Old signal (expired)
        old_signal = {
            "timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
            "signal_type": "BUY"
        }
        
        is_expired = service._is_signal_expired(old_signal)
        assert is_expired is True
        
        # Recent signal (not expired)
        recent_signal = {
            "timestamp": datetime.utcnow().isoformat(),
            "signal_type": "BUY"
        }
        
        is_expired = service._is_signal_expired(recent_signal)
        assert is_expired is False

    @pytest.mark.asyncio
    async def test_multiple_strategy_signals(self, service, sample_market_data):
        """Test generating signals from multiple strategies."""
        strategies = ["momentum_breakout", "mean_reversion", "volume_spike"]
        
        with patch.object(service, '_run_strategy') as mock_strategy:
            mock_strategy.return_value = {
                "signal_type": "BUY",
                "confidence": 0.8
            }
            
            signals = await service.generate_multi_strategy_signals(
                "AAPL", sample_market_data, strategies
            )
            
            assert len(signals) == 3
            assert mock_strategy.call_count == 3

    @pytest.mark.asyncio
    async def test_signal_conflict_resolution(self, service):
        """Test resolution of conflicting signals."""
        conflicting_signals = [
            {"signal_type": "BUY", "confidence": 0.8, "strategy": "momentum"},
            {"signal_type": "SELL", "confidence": 0.75, "strategy": "mean_reversion"},
            {"signal_type": "BUY", "confidence": 0.85, "strategy": "volume_spike"}
        ]
        
        resolved_signal = service._resolve_signal_conflicts(conflicting_signals)
        
        # Should choose BUY based on majority and higher confidence
        assert resolved_signal["signal_type"] == "BUY"
        assert resolved_signal["confidence"] > 0.8

    @pytest.mark.asyncio
    async def test_signal_performance_tracking(self, service):
        """Test signal performance tracking."""
        signal_id = "signal_001"
        outcome = {
            "realized_return": 0.05,
            "max_drawdown": -0.02,
            "hold_period": 24  # hours
        }
        
        await service.track_signal_performance(signal_id, outcome)
        
        # Should update signal performance in database
        service.database.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_strategy_backtesting_data(self, service):
        """Test generation of backtesting data."""
        strategy_name = "momentum_breakout"
        
        backtest_data = await service.get_strategy_backtest_data(strategy_name, days=30)
        
        assert "total_signals" in backtest_data
        assert "win_rate" in backtest_data
        assert "average_return" in backtest_data

    @pytest.mark.asyncio
    async def test_signal_filtering(self, service):
        """Test signal filtering by various criteria."""
        filters = {
            "min_confidence": 0.8,
            "signal_types": ["BUY", "SELL"],
            "symbols": ["AAPL", "GOOGL"]
        }
        
        filtered_signals = await service.get_filtered_signals(filters)
        
        # Should apply all filters
        service.database.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_real_time_signal_generation(self, service, sample_market_data):
        """Test real-time signal generation with streaming data."""
        # Mock streaming data
        with patch.object(service, '_process_streaming_data') as mock_process:
            mock_process.return_value = True
            
            await service.process_real_time_data("AAPL", sample_market_data)
            
            mock_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_service_health_check(self, service):
        """Test service health check."""
        health = await service.get_service_health()
        
        assert "status" in health
        assert "active_strategies" in health
        assert "signals_generated_today" in health

    @pytest.mark.asyncio
    async def test_error_handling_invalid_data(self, service):
        """Test error handling for invalid market data."""
        invalid_data = {
            "symbol": "AAPL",
            "price": None,  # Invalid price
            "volume": -1000  # Invalid volume
        }
        
        with pytest.raises(ValueError):
            await service.generate_signal("AAPL", invalid_data)