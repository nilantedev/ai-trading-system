#!/usr/bin/env python3
"""Risk Monitoring Service - Real-time risk analysis and alerting."""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from trading_common import MarketData, get_settings, get_logger
from trading_common.cache import get_trading_cache
from trading_common.messaging import get_message_consumer, get_message_producer

logger = get_logger(__name__)
settings = get_settings()


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    VOLUME_ANOMALY = "VOLUME_ANOMALY"
    PRICE_MOVEMENT = "PRICE_MOVEMENT"
    CORRELATION_BREAK = "CORRELATION_BREAK"
    LIQUIDITY_RISK = "LIQUIDITY_RISK"
    POSITION_LIMIT = "POSITION_LIMIT"
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"
    VAR_BREACH = "VAR_BREACH"


@dataclass
class RiskMetrics:
    """Risk metrics for a symbol or portfolio."""
    symbol: str
    timestamp: datetime
    volatility: float  # Historical volatility
    volume_ratio: float  # Current vs average volume
    price_change_1h: float  # 1-hour price change %
    price_change_24h: float  # 24-hour price change %
    liquidity_score: float  # 0-1, higher is more liquid
    correlation_breaks: int  # Number of correlation anomalies
    risk_level: RiskLevel
    risk_score: float  # 0-100, higher is riskier


@dataclass
class RiskAlert:
    """Risk alert notification."""
    alert_id: str
    symbol: str
    alert_type: AlertType
    risk_level: RiskLevel
    timestamp: datetime
    title: str
    description: str
    metrics: Dict[str, float]
    recommended_action: str
    severity_score: float  # 0-100


@dataclass
class PortfolioRisk:
    """Overall portfolio risk assessment."""
    timestamp: datetime
    total_risk_score: float  # 0-100
    risk_level: RiskLevel
    var_1d: float  # 1-day Value at Risk
    max_drawdown: float  # Maximum drawdown %
    concentration_risk: float  # Portfolio concentration risk
    correlation_risk: float  # Inter-asset correlation risk
    active_alerts: int
    high_risk_positions: List[str]  # Symbols with high risk


class RiskMonitoringService:
    """Service for monitoring and alerting on trading risks."""
    
    def __init__(self):
        self.consumer = None
        self.producer = None
        self.cache = None
        
        self.is_running = False
        
        # Risk monitoring queues
        self.market_data_queue = asyncio.Queue(maxsize=1000)
        self.signal_queue = asyncio.Queue(maxsize=500)
        self.position_queue = asyncio.Queue(maxsize=200)
        
        # Risk tracking data
        self.symbol_metrics = {}  # symbol -> RiskMetrics
        self.price_history = {}  # symbol -> List[price_data]
        self.volume_history = {}  # symbol -> List[volume_data]
        self.active_alerts = {}  # alert_id -> RiskAlert
        
        # Risk parameters
        self.risk_params = {
            'max_volatility': 0.05,  # 5% daily volatility threshold
            'volume_spike_threshold': 3.0,  # 3x normal volume
            'price_movement_threshold': 0.03,  # 3% price movement
            'correlation_threshold': 0.7,  # Correlation break threshold
            'max_position_size': 0.1,  # 10% max position size
            'max_portfolio_drawdown': 0.05,  # 5% max drawdown
            'var_confidence': 0.95  # 95% VaR confidence level
        }
        
        # Performance metrics
        self.alerts_generated = 0
        self.risk_checks_performed = 0
        self.critical_alerts = 0
        
    async def start(self):
        """Initialize and start risk monitoring service."""
        logger.info("Starting Risk Monitoring Service")
        
        try:
            # Initialize connections
            self.consumer = await get_message_consumer()
            self.producer = await get_message_producer()
            self.cache = get_trading_cache()
            
            # Subscribe to data streams
            await self._setup_subscriptions()
            
            # Start monitoring tasks
            self.is_running = True
            
            tasks = [
                asyncio.create_task(self._process_market_data_queue()),
                asyncio.create_task(self._process_signal_queue()),
                asyncio.create_task(self._process_position_queue()),
                asyncio.create_task(self._consume_messages()),
                asyncio.create_task(self._periodic_risk_assessment()),
                asyncio.create_task(self._alert_cleanup())
            ]
            
            logger.info("Risk monitoring service started with 6 concurrent tasks")
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Failed to start risk monitoring service: {e}")
            raise
    
    async def stop(self):
        """Stop risk monitoring service gracefully."""
        logger.info("Stopping Risk Monitoring Service")
        self.is_running = False
        
        if self.consumer:
            await self.consumer.close()
        if self.producer:
            await self.producer.close()
        
        logger.info("Risk Monitoring Service stopped")
    
    async def _setup_subscriptions(self):
        """Subscribe to market data, signals, and positions."""
        try:
            await self.consumer.subscribe_market_data(
                self._handle_market_data_message,
                subscription_name="risk-monitor-market"
            )
            
            await self.consumer.subscribe_trading_signals(
                self._handle_signal_message,
                subscription_name="risk-monitor-signals"
            )
            
            await self.consumer.subscribe_position_updates(
                self._handle_position_message,
                subscription_name="risk-monitor-positions"
            )
            
            logger.info("Subscribed to market data, signals, and positions")
        except Exception as e:
            logger.warning(f"Subscription setup failed: {e}")
    
    async def _consume_messages(self):
        """Consume messages from subscribed topics."""
        try:
            await self.consumer.start_consuming()
        except Exception as e:
            logger.error(f"Message consumption failed: {e}")
    
    async def _handle_market_data_message(self, message):
        """Handle incoming market data for risk monitoring."""
        try:
            # Parse market data
            if hasattr(message, 'symbol'):
                market_data = MarketData(
                    symbol=message.symbol,
                    timestamp=datetime.fromisoformat(message.timestamp),
                    open=message.open,
                    high=message.high,
                    low=message.low,
                    close=message.close,
                    volume=message.volume,
                    timeframe="1min",
                    data_source=message.data_source if hasattr(message, 'data_source') else "stream"
                )
            else:
                data = json.loads(message) if isinstance(message, str) else message
                market_data = MarketData(**data)
            
            # Add to processing queue
            await self.market_data_queue.put(market_data)
            
        except Exception as e:
            logger.error(f"Failed to handle market data message: {e}")
    
    async def _handle_signal_message(self, message):
        """Handle trading signal for risk assessment."""
        try:
            signal_data = json.loads(message) if isinstance(message, str) else message
            await self.signal_queue.put(signal_data)
        except Exception as e:
            logger.error(f"Failed to handle signal message: {e}")
    
    async def _handle_position_message(self, message):
        """Handle position update for risk monitoring."""
        try:
            position_data = json.loads(message) if isinstance(message, str) else message
            await self.position_queue.put(position_data)
        except Exception as e:
            logger.error(f"Failed to handle position message: {e}")
    
    async def _process_market_data_queue(self):
        """Process market data for risk analysis."""
        while self.is_running:
            try:
                # Wait for market data
                market_data = await asyncio.wait_for(
                    self.market_data_queue.get(),
                    timeout=1.0
                )
                
                # Update price and volume history
                await self._update_price_history(market_data)
                await self._update_volume_history(market_data)
                
                # Calculate risk metrics
                risk_metrics = await self._calculate_risk_metrics(market_data.symbol)
                
                if risk_metrics:
                    self.symbol_metrics[market_data.symbol] = risk_metrics
                    
                    # Check for risk alerts
                    alerts = await self._check_risk_alerts(risk_metrics, market_data)
                    
                    for alert in alerts:
                        await self._process_risk_alert(alert)
                
                self.risk_checks_performed += 1
                self.market_data_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Market data risk processing error: {e}")
    
    async def _process_signal_queue(self):
        """Process trading signals for risk validation."""
        while self.is_running:
            try:
                signal_data = await asyncio.wait_for(
                    self.signal_queue.get(),
                    timeout=1.0
                )
                
                # Validate signal against risk parameters
                risk_check = await self._validate_signal_risk(signal_data)
                
                if not risk_check['approved']:
                    # Generate risk alert for rejected signal
                    alert = RiskAlert(
                        alert_id=f"signal_risk_{signal_data['symbol']}_{datetime.utcnow().timestamp()}",
                        symbol=signal_data['symbol'],
                        alert_type=AlertType.POSITION_LIMIT,
                        risk_level=RiskLevel.HIGH,
                        timestamp=datetime.utcnow(),
                        title="Signal Rejected - Risk Limit",
                        description=f"Trading signal rejected: {risk_check['reason']}",
                        metrics=risk_check['metrics'],
                        recommended_action="Review position limits and risk parameters",
                        severity_score=75.0
                    )
                    
                    await self._process_risk_alert(alert)
                
                self.signal_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Signal risk processing error: {e}")
    
    async def _process_position_queue(self):
        """Process position updates for portfolio risk monitoring."""
        while self.is_running:
            try:
                position_data = await asyncio.wait_for(
                    self.position_queue.get(),
                    timeout=1.0
                )
                
                # Calculate portfolio risk metrics
                portfolio_risk = await self._calculate_portfolio_risk(position_data)
                
                if portfolio_risk.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    # Generate portfolio risk alert
                    alert = RiskAlert(
                        alert_id=f"portfolio_risk_{datetime.utcnow().timestamp()}",
                        symbol="PORTFOLIO",
                        alert_type=AlertType.DRAWDOWN_LIMIT,
                        risk_level=portfolio_risk.risk_level,
                        timestamp=datetime.utcnow(),
                        title="Portfolio Risk Alert",
                        description=f"Portfolio risk level: {portfolio_risk.risk_level.value}",
                        metrics={
                            'total_risk_score': portfolio_risk.total_risk_score,
                            'max_drawdown': portfolio_risk.max_drawdown,
                            'var_1d': portfolio_risk.var_1d
                        },
                        recommended_action="Review positions and consider risk reduction",
                        severity_score=portfolio_risk.total_risk_score
                    )
                    
                    await self._process_risk_alert(alert)
                
                self.position_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Position risk processing error: {e}")
    
    async def _update_price_history(self, market_data: MarketData):
        """Update price history for volatility calculations."""
        symbol = market_data.symbol
        
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        price_point = {
            'timestamp': market_data.timestamp,
            'price': market_data.close,
            'high': market_data.high,
            'low': market_data.low
        }
        
        self.price_history[symbol].append(price_point)
        
        # Keep only last 24 hours of data
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.price_history[symbol] = [
            p for p in self.price_history[symbol] 
            if p['timestamp'] > cutoff_time
        ]
    
    async def _update_volume_history(self, market_data: MarketData):
        """Update volume history for anomaly detection."""
        symbol = market_data.symbol
        
        if symbol not in self.volume_history:
            self.volume_history[symbol] = []
        
        volume_point = {
            'timestamp': market_data.timestamp,
            'volume': market_data.volume
        }
        
        self.volume_history[symbol].append(volume_point)
        
        # Keep only last 24 hours of data
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.volume_history[symbol] = [
            v for v in self.volume_history[symbol] 
            if v['timestamp'] > cutoff_time
        ]
    
    async def _calculate_risk_metrics(self, symbol: str) -> Optional[RiskMetrics]:
        """Calculate comprehensive risk metrics for a symbol."""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
                return None
            
            prices = self.price_history[symbol]
            volumes = self.volume_history.get(symbol, [])
            
            # Calculate volatility
            price_returns = []
            for i in range(1, len(prices)):
                prev_price = prices[i-1]['price']
                curr_price = prices[i]['price']
                if prev_price > 0:
                    return_pct = (curr_price - prev_price) / prev_price
                    price_returns.append(return_pct)
            
            volatility = 0.0
            if price_returns:
                mean_return = sum(price_returns) / len(price_returns)
                variance = sum((r - mean_return) ** 2 for r in price_returns) / len(price_returns)
                volatility = variance ** 0.5
            
            # Calculate volume ratio
            volume_ratio = 1.0
            if volumes and len(volumes) >= 2:
                current_volume = volumes[-1]['volume']
                avg_volume = sum(v['volume'] for v in volumes[:-1]) / (len(volumes) - 1)
                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
            
            # Calculate price changes
            price_change_1h = 0.0
            price_change_24h = 0.0
            
            current_price = prices[-1]['price']
            
            # 1-hour change
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            for price_point in reversed(prices):
                if price_point['timestamp'] <= one_hour_ago:
                    if price_point['price'] > 0:
                        price_change_1h = (current_price - price_point['price']) / price_point['price']
                    break
            
            # 24-hour change
            if len(prices) > 1:
                first_price = prices[0]['price']
                if first_price > 0:
                    price_change_24h = (current_price - first_price) / first_price
            
            # Calculate liquidity score (simplified)
            liquidity_score = 1.0
            if volumes:
                avg_volume = sum(v['volume'] for v in volumes) / len(volumes)
                if avg_volume < 1000:  # Low volume threshold
                    liquidity_score = 0.3
                elif avg_volume < 10000:
                    liquidity_score = 0.7
            
            # Determine risk level
            risk_score = 0.0
            risk_factors = []
            
            if volatility > self.risk_params['max_volatility']:
                risk_factors.append(('volatility', volatility * 100))
                risk_score += 30
            
            if volume_ratio > self.risk_params['volume_spike_threshold']:
                risk_factors.append(('volume_spike', volume_ratio))
                risk_score += 20
            
            if abs(price_change_1h) > self.risk_params['price_movement_threshold']:
                risk_factors.append(('price_movement', abs(price_change_1h) * 100))
                risk_score += 25
            
            if liquidity_score < 0.5:
                risk_factors.append(('liquidity', liquidity_score))
                risk_score += 15
            
            # Determine risk level
            if risk_score >= 70:
                risk_level = RiskLevel.CRITICAL
            elif risk_score >= 50:
                risk_level = RiskLevel.HIGH
            elif risk_score >= 25:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            return RiskMetrics(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                volatility=volatility,
                volume_ratio=volume_ratio,
                price_change_1h=price_change_1h,
                price_change_24h=price_change_24h,
                liquidity_score=liquidity_score,
                correlation_breaks=0,  # Would need more complex calculation
                risk_level=risk_level,
                risk_score=risk_score
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate risk metrics for {symbol}: {e}")
            return None
    
    async def _check_risk_alerts(self, risk_metrics: RiskMetrics, market_data: MarketData) -> List[RiskAlert]:
        """Check for various risk conditions and generate alerts."""
        alerts = []
        
        try:
            # Volatility spike alert
            if risk_metrics.volatility > self.risk_params['max_volatility']:
                alerts.append(RiskAlert(
                    alert_id=f"volatility_{risk_metrics.symbol}_{datetime.utcnow().timestamp()}",
                    symbol=risk_metrics.symbol,
                    alert_type=AlertType.VOLATILITY_SPIKE,
                    risk_level=RiskLevel.HIGH if risk_metrics.volatility > 0.1 else RiskLevel.MEDIUM,
                    timestamp=datetime.utcnow(),
                    title="High Volatility Alert",
                    description=f"Volatility spike detected: {risk_metrics.volatility:.2%}",
                    metrics={'volatility': risk_metrics.volatility, 'threshold': self.risk_params['max_volatility']},
                    recommended_action="Monitor position size and consider stop-loss adjustments",
                    severity_score=min(risk_metrics.volatility * 1000, 100)
                ))
            
            # Volume anomaly alert
            if risk_metrics.volume_ratio > self.risk_params['volume_spike_threshold']:
                alerts.append(RiskAlert(
                    alert_id=f"volume_{risk_metrics.symbol}_{datetime.utcnow().timestamp()}",
                    symbol=risk_metrics.symbol,
                    alert_type=AlertType.VOLUME_ANOMALY,
                    risk_level=RiskLevel.MEDIUM,
                    timestamp=datetime.utcnow(),
                    title="Volume Spike Alert",
                    description=f"Unusual volume activity: {risk_metrics.volume_ratio:.1f}x normal",
                    metrics={'volume_ratio': risk_metrics.volume_ratio, 'threshold': self.risk_params['volume_spike_threshold']},
                    recommended_action="Investigate news or events causing volume spike",
                    severity_score=min(risk_metrics.volume_ratio * 20, 100)
                ))
            
            # Price movement alert
            if abs(risk_metrics.price_change_1h) > self.risk_params['price_movement_threshold']:
                alerts.append(RiskAlert(
                    alert_id=f"price_{risk_metrics.symbol}_{datetime.utcnow().timestamp()}",
                    symbol=risk_metrics.symbol,
                    alert_type=AlertType.PRICE_MOVEMENT,
                    risk_level=RiskLevel.HIGH if abs(risk_metrics.price_change_1h) > 0.05 else RiskLevel.MEDIUM,
                    timestamp=datetime.utcnow(),
                    title="Significant Price Movement",
                    description=f"Large price move: {risk_metrics.price_change_1h:.2%} in 1 hour",
                    metrics={'price_change_1h': risk_metrics.price_change_1h, 'threshold': self.risk_params['price_movement_threshold']},
                    recommended_action="Review positions and market conditions",
                    severity_score=abs(risk_metrics.price_change_1h) * 1000
                ))
            
            # Liquidity risk alert
            if risk_metrics.liquidity_score < 0.5:
                alerts.append(RiskAlert(
                    alert_id=f"liquidity_{risk_metrics.symbol}_{datetime.utcnow().timestamp()}",
                    symbol=risk_metrics.symbol,
                    alert_type=AlertType.LIQUIDITY_RISK,
                    risk_level=RiskLevel.HIGH,
                    timestamp=datetime.utcnow(),
                    title="Low Liquidity Warning",
                    description=f"Low liquidity conditions detected (score: {risk_metrics.liquidity_score:.2f})",
                    metrics={'liquidity_score': risk_metrics.liquidity_score},
                    recommended_action="Consider position size limits and execution strategies",
                    severity_score=(1 - risk_metrics.liquidity_score) * 100
                ))
            
        except Exception as e:
            logger.error(f"Failed to check risk alerts for {risk_metrics.symbol}: {e}")
        
        return alerts
    
    async def _validate_signal_risk(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trading signal against risk parameters."""
        try:
            symbol = signal_data.get('symbol')
            position_size = signal_data.get('position_size', 0)
            
            # Check position size limit
            if position_size > self.risk_params['max_position_size']:
                return {
                    'approved': False,
                    'reason': f"Position size {position_size:.2%} exceeds limit {self.risk_params['max_position_size']:.2%}",
                    'metrics': {'position_size': position_size, 'limit': self.risk_params['max_position_size']}
                }
            
            # Check symbol risk level
            if symbol in self.symbol_metrics:
                risk_metrics = self.symbol_metrics[symbol]
                if risk_metrics.risk_level == RiskLevel.CRITICAL:
                    return {
                        'approved': False,
                        'reason': f"Symbol {symbol} has critical risk level",
                        'metrics': {'risk_score': risk_metrics.risk_score, 'risk_level': risk_metrics.risk_level.value}
                    }
            
            return {
                'approved': True,
                'reason': 'Signal approved',
                'metrics': {}
            }
            
        except Exception as e:
            logger.error(f"Signal risk validation error: {e}")
            return {
                'approved': False,
                'reason': f"Risk validation failed: {e}",
                'metrics': {}
            }
    
    async def _calculate_portfolio_risk(self, position_data: Dict[str, Any]) -> PortfolioRisk:
        """Calculate overall portfolio risk metrics."""
        try:
            # Simplified portfolio risk calculation
            total_risk_score = 0.0
            high_risk_positions = []
            active_alerts = len([a for a in self.active_alerts.values() 
                               if (datetime.utcnow() - a.timestamp).total_seconds() < 3600])
            
            # Calculate weighted risk score
            total_value = position_data.get('total_value', 1.0)
            positions = position_data.get('positions', {})
            
            for symbol, position in positions.items():
                if symbol in self.symbol_metrics:
                    risk_metrics = self.symbol_metrics[symbol]
                    weight = position.get('value', 0) / total_value
                    total_risk_score += risk_metrics.risk_score * weight
                    
                    if risk_metrics.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                        high_risk_positions.append(symbol)
            
            # Determine overall risk level
            if total_risk_score >= 70:
                risk_level = RiskLevel.CRITICAL
            elif total_risk_score >= 50:
                risk_level = RiskLevel.HIGH
            elif total_risk_score >= 25:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            return PortfolioRisk(
                timestamp=datetime.utcnow(),
                total_risk_score=total_risk_score,
                risk_level=risk_level,
                var_1d=total_risk_score * 0.01,  # Simplified VaR calculation
                max_drawdown=0.02,  # Would calculate from historical data
                concentration_risk=len(positions) / 10.0 if positions else 0.0,
                correlation_risk=0.5,  # Would calculate from correlations
                active_alerts=active_alerts,
                high_risk_positions=high_risk_positions
            )
            
        except Exception as e:
            logger.error(f"Portfolio risk calculation error: {e}")
            return PortfolioRisk(
                timestamp=datetime.utcnow(),
                total_risk_score=0.0,
                risk_level=RiskLevel.LOW,
                var_1d=0.0,
                max_drawdown=0.0,
                concentration_risk=0.0,
                correlation_risk=0.0,
                active_alerts=0,
                high_risk_positions=[]
            )
    
    async def _process_risk_alert(self, alert: RiskAlert):
        """Process and handle a risk alert."""
        try:
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            
            # Cache alert
            await self._cache_risk_alert(alert)
            
            # Publish alert
            await self._publish_risk_alert(alert)
            
            self.alerts_generated += 1
            
            if alert.risk_level == RiskLevel.CRITICAL:
                self.critical_alerts += 1
                logger.warning(f"CRITICAL RISK ALERT: {alert.title} for {alert.symbol}")
            else:
                logger.info(f"Risk alert generated: {alert.title} for {alert.symbol}")
            
        except Exception as e:
            logger.error(f"Failed to process risk alert: {e}")
    
    async def _cache_risk_alert(self, alert: RiskAlert):
        """Cache risk alert for retrieval."""
        try:
            if self.cache:
                cache_key = f"risk_alert:{alert.alert_id}"
                alert_data = asdict(alert)
                alert_data['timestamp'] = alert.timestamp.isoformat()
                alert_data['alert_type'] = alert.alert_type.value
                alert_data['risk_level'] = alert.risk_level.value
                
                await self.cache.set_json(cache_key, alert_data, ttl=3600)  # 1 hour
        except Exception as e:
            logger.warning(f"Failed to cache risk alert: {e}")
    
    async def _publish_risk_alert(self, alert: RiskAlert):
        """Publish risk alert to downstream services."""
        try:
            if self.producer:
                alert_message = {
                    'alert_id': alert.alert_id,
                    'symbol': alert.symbol,
                    'alert_type': alert.alert_type.value,
                    'risk_level': alert.risk_level.value,
                    'title': alert.title,
                    'severity_score': alert.severity_score,
                    'timestamp': alert.timestamp.isoformat()
                }
                
                # Would publish to risk alerts topic
                logger.debug(f"Publishing risk alert: {alert.title}")
                
        except Exception as e:
            logger.warning(f"Failed to publish risk alert: {e}")
    
    async def _periodic_risk_assessment(self):
        """Perform periodic comprehensive risk assessment."""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Review all active symbols for risk
                for symbol in list(self.symbol_metrics.keys()):
                    risk_metrics = self.symbol_metrics[symbol]
                    
                    # Check if risk level has changed
                    if risk_metrics.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                        logger.info(f"High risk symbol detected: {symbol} (score: {risk_metrics.risk_score})")
                
                # Clean up old price/volume history
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                for symbol in list(self.price_history.keys()):
                    self.price_history[symbol] = [
                        p for p in self.price_history[symbol]
                        if p['timestamp'] > cutoff_time
                    ]
                    
                    if not self.price_history[symbol]:
                        del self.price_history[symbol]
                
            except Exception as e:
                logger.warning(f"Periodic risk assessment error: {e}")
    
    async def _alert_cleanup(self):
        """Clean up expired alerts."""
        while self.is_running:
            try:
                await asyncio.sleep(600)  # Every 10 minutes
                
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                expired_alerts = []
                
                for alert_id, alert in self.active_alerts.items():
                    if alert.timestamp < cutoff_time:
                        expired_alerts.append(alert_id)
                
                for alert_id in expired_alerts:
                    del self.active_alerts[alert_id]
                
                if expired_alerts:
                    logger.debug(f"Cleaned up {len(expired_alerts)} expired risk alerts")
                
            except Exception as e:
                logger.warning(f"Alert cleanup error: {e}")
    
    async def get_risk_metrics(self, symbol: str) -> Optional[RiskMetrics]:
        """Get current risk metrics for a symbol."""
        return self.symbol_metrics.get(symbol)
    
    async def get_active_alerts(self, symbol: Optional[str] = None) -> List[RiskAlert]:
        """Get active risk alerts, optionally filtered by symbol."""
        if symbol:
            return [alert for alert in self.active_alerts.values() if alert.symbol == symbol]
        return list(self.active_alerts.values())
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get service health status."""
        return {
            'service': 'risk_monitoring_service',
            'status': 'healthy' if self.is_running else 'stopped',
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {
                'alerts_generated': self.alerts_generated,
                'critical_alerts': self.critical_alerts,
                'risk_checks_performed': self.risk_checks_performed,
                'active_alerts': len(self.active_alerts),
                'monitored_symbols': len(self.symbol_metrics)
            },
            'risk_parameters': self.risk_params,
            'connections': {
                'consumer': self.consumer is not None,
                'producer': self.producer is not None,
                'cache': self.cache is not None
            }
        }


# Global service instance
risk_service: Optional[RiskMonitoringService] = None


async def get_risk_service() -> RiskMonitoringService:
    """Get or create risk monitoring service instance."""
    global risk_service
    if risk_service is None:
        risk_service = RiskMonitoringService()
    return risk_service