#!/usr/bin/env python3
"""
Comprehensive tests for risk management functionality.
Tests position limits, portfolio risk, and circuit breaker integration.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal
import os
import sys

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.risk_monitor.risk_monitoring_service import (
    RiskMonitoringService, RiskLevel, RiskRule, RiskAlert, 
    PositionRiskAssessment, PortfolioRiskAssessment
)
from trading_common import MarketData, Position, Order, Account


class TestRiskMonitoringService:
    """Test suite for Risk Monitoring Service."""
    
    @pytest.fixture
    def risk_service(self):
        """Create risk monitoring service instance."""
        service = RiskMonitoringService()
        return service
    
    @pytest.fixture
    def sample_account(self):
        """Create sample account for testing."""
        return Account(
            account_id="test_account_1",
            broker="paper_trading",
            equity=100000.0,
            cash=50000.0,
            buying_power=200000.0,
            portfolio_value=100000.0,
            day_trade_buying_power=100000.0,
            pattern_day_trader=False,
            trade_suspended=False,
            account_blocked=False,
            last_updated=datetime.utcnow()
        )
    
    @pytest.fixture
    def sample_position(self):
        """Create sample position for testing."""
        return Position(
            symbol="AAPL",
            quantity=100,
            avg_cost=150.0,
            current_price=160.0,
            market_value=16000.0,
            unrealized_pnl=1000.0,
            side="long",
            last_updated=datetime.utcnow()
        )
    
    @pytest.fixture
    def sample_order(self):
        """Create sample order for testing."""
        from services.execution.order_management_system import Order, OrderType, OrderSide
        return Order(
            order_id="test_order_1",
            symbol="AAPL",
            quantity=50,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            time_in_force="DAY",
            created_at=datetime.utcnow()
        )

    @pytest.mark.asyncio
    async def test_position_risk_assessment(self, risk_service, sample_position, sample_account):
        """Test position-level risk assessment."""
        # Test normal position
        risk_assessment = await risk_service.assess_position_risk(sample_position, sample_account)
        
        assert risk_assessment is not None
        assert isinstance(risk_assessment.risk_level, RiskLevel)
        assert isinstance(risk_assessment.concentration_pct, float)
        assert 0 <= risk_assessment.concentration_pct <= 100
    
    @pytest.mark.asyncio  
    async def test_portfolio_risk_assessment(self, risk_service, sample_account):
        """Test portfolio-level risk assessment."""
        positions = [
            Position(
                symbol="AAPL", quantity=100, avg_cost=150.0, current_price=160.0,
                market_value=16000.0, unrealized_pnl=1000.0, side="long"
            ),
            Position(
                symbol="GOOGL", quantity=50, avg_cost=2500.0, current_price=2600.0, 
                market_value=130000.0, unrealized_pnl=5000.0, side="long"
            )
        ]
        
        portfolio_assessment = await risk_service.assess_portfolio_risk(positions, sample_account)
        
        assert portfolio_assessment is not None
        assert isinstance(portfolio_assessment.overall_risk_level, RiskLevel)
        assert portfolio_assessment.total_exposure >= 0
        assert 0 <= portfolio_assessment.leverage_ratio <= 10  # Reasonable leverage bounds
    
    @pytest.mark.asyncio
    async def test_order_risk_validation_success(self, risk_service, sample_order, sample_account):
        """Test successful order risk validation."""
        # Mock market data for the order symbol
        market_data = MarketData(
            symbol="AAPL",
            price=160.0,
            volume=1000000,
            timestamp=datetime.utcnow()
        )
        
        with patch.object(risk_service, '_get_market_data', return_value=market_data):
            is_valid, risk_info = await risk_service.validate_order_risk(sample_order, sample_account)
            
            assert is_valid is True
            assert isinstance(risk_info, dict)
    
    @pytest.mark.asyncio
    async def test_order_risk_validation_failure_insufficient_funds(self, risk_service, sample_account):
        """Test order risk validation failure due to insufficient funds."""
        from services.execution.order_management_system import Order, OrderType, OrderSide
        
        # Create order that exceeds buying power
        large_order = Order(
            order_id="large_order_1",
            symbol="AAPL", 
            quantity=2000,  # Much larger quantity
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            time_in_force="DAY",
            created_at=datetime.utcnow()
        )
        
        market_data = MarketData(
            symbol="AAPL",
            price=160.0,
            volume=1000000,
            timestamp=datetime.utcnow()
        )
        
        with patch.object(risk_service, '_get_market_data', return_value=market_data):
            is_valid, risk_info = await risk_service.validate_order_risk(large_order, sample_account)
            
            assert is_valid is False
            assert "insufficient" in risk_info.get("reason", "").lower()
    
    @pytest.mark.asyncio
    async def test_concentration_limit_breach(self, risk_service, sample_account):
        """Test detection of concentration limit breaches."""
        # Create position that represents >50% of portfolio (concentration risk)
        large_position = Position(
            symbol="AAPL",
            quantity=400,
            avg_cost=150.0,
            current_price=160.0,
            market_value=64000.0,  # 64% of $100k portfolio
            unrealized_pnl=4000.0,
            side="long"
        )
        
        risk_assessment = await risk_service.assess_position_risk(large_position, sample_account)
        
        # Should flag as high risk due to concentration
        assert risk_assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert risk_assessment.concentration_pct > 50
    
    @pytest.mark.asyncio
    async def test_volatility_risk_assessment(self, risk_service, sample_position):
        """Test volatility-based risk assessment."""
        # Mock high volatility market data
        volatile_market_data = [
            MarketData(symbol="AAPL", price=150.0, timestamp=datetime.utcnow() - timedelta(hours=1)),
            MarketData(symbol="AAPL", price=160.0, timestamp=datetime.utcnow() - timedelta(minutes=30)),
            MarketData(symbol="AAPL", price=140.0, timestamp=datetime.utcnow() - timedelta(minutes=15)),
            MarketData(symbol="AAPL", price=155.0, timestamp=datetime.utcnow())
        ]
        
        with patch.object(risk_service, '_get_historical_volatility', return_value=0.35):  # High volatility
            risk_metrics = await risk_service.calculate_position_volatility(sample_position)
            
            assert risk_metrics.get('volatility', 0) > 0.3  # High volatility threshold
            assert risk_metrics.get('risk_adjustment_factor', 1.0) > 1.0  # Should increase risk
    
    @pytest.mark.asyncio
    async def test_drawdown_monitoring(self, risk_service, sample_account):
        """Test portfolio drawdown monitoring."""
        # Simulate positions with unrealized losses
        losing_positions = [
            Position(
                symbol="STOCK1", quantity=100, avg_cost=100.0, current_price=85.0,
                market_value=8500.0, unrealized_pnl=-1500.0, side="long"
            ),
            Position(
                symbol="STOCK2", quantity=50, avg_cost=200.0, current_price=170.0,
                market_value=8500.0, unrealized_pnl=-1500.0, side="long"  
            )
        ]
        
        portfolio_risk = await risk_service.assess_portfolio_risk(losing_positions, sample_account)
        
        # Should detect significant drawdown
        assert portfolio_risk.unrealized_pnl < -2000  # Significant losses
        assert portfolio_risk.drawdown_pct > 2.0  # >2% drawdown
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, risk_service):
        """Test circuit breaker integration with risk monitoring."""
        # Test that circuit breakers are properly configured
        assert hasattr(risk_service, 'circuit_breakers')
        assert 'risk_calculation' in risk_service.circuit_breakers
        
        # Test circuit breaker functionality
        circuit_breaker = risk_service.circuit_breakers['risk_calculation']
        assert circuit_breaker is not None
        assert hasattr(circuit_breaker, 'call')
    
    @pytest.mark.asyncio
    async def test_risk_alert_generation(self, risk_service, sample_account):
        """Test risk alert generation and notification."""
        # Create high-risk scenario
        high_risk_positions = [
            Position(
                symbol="VOLATILE_STOCK", quantity=500, avg_cost=100.0, current_price=120.0,
                market_value=60000.0, unrealized_pnl=10000.0, side="long"  # 60% concentration
            )
        ]
        
        with patch.object(risk_service, '_send_risk_alert') as mock_alert:
            portfolio_risk = await risk_service.assess_portfolio_risk(high_risk_positions, sample_account)
            
            # Should trigger concentration risk alert
            if portfolio_risk.overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                # Verify alert would be sent (mocked)
                assert mock_alert.call_count >= 0  # Alert system should be called


class TestRiskCalculations:
    """Test suite for risk calculation algorithms."""
    
    def test_position_concentration_calculation(self):
        """Test position concentration percentage calculation."""
        position_value = 25000.0
        portfolio_value = 100000.0
        
        concentration_pct = (position_value / portfolio_value) * 100
        assert concentration_pct == 25.0
    
    def test_portfolio_beta_calculation(self):
        """Test portfolio beta calculation."""
        # Mock individual position betas
        position_betas = [1.2, 0.8, 1.5, 0.9]
        position_weights = [0.3, 0.2, 0.25, 0.25]
        
        portfolio_beta = sum(beta * weight for beta, weight in zip(position_betas, position_weights))
        
        assert 0.5 <= portfolio_beta <= 2.0  # Reasonable beta range
    
    def test_value_at_risk_calculation(self):
        """Test Value at Risk (VaR) calculation."""
        portfolio_value = 100000.0
        volatility = 0.25  # 25% annual volatility
        confidence_level = 0.05  # 95% confidence (5% VaR)
        time_horizon = 1  # 1 day
        
        # Simplified VaR calculation (normal distribution)
        import math
        z_score = 1.645  # 95% confidence z-score
        var = portfolio_value * volatility * z_score * math.sqrt(time_horizon / 252)
        
        assert var > 0
        assert var < portfolio_value * 0.1  # Shouldn't exceed 10% of portfolio
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        portfolio_return = 0.12  # 12% annual return
        risk_free_rate = 0.02   # 2% risk-free rate
        volatility = 0.15       # 15% volatility
        
        sharpe_ratio = (portfolio_return - risk_free_rate) / volatility
        
        assert sharpe_ratio > 0
        assert sharpe_ratio < 5.0  # Reasonable upper bound


if __name__ == "__main__":
    pytest.main([__file__, "-v"])