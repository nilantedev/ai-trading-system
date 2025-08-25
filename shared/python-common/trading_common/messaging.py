"""Message infrastructure for AI trading system using Apache Pulsar."""

import json
import logging
import asyncio
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pulsar
from pulsar.schema import AvroSchema, JsonSchema, StringSchema

from .config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class MarketDataMessage:
    """Market data message schema."""
    symbol: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    data_source: str


@dataclass  
class TradingSignalMessage:
    """Trading signal message schema."""
    id: str
    timestamp: str
    symbol: str
    signal_type: str
    confidence: float
    target_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    strategy_name: str
    agent_id: str
    reasoning: str


@dataclass
class PortfolioUpdateMessage:
    """Portfolio update message schema."""
    portfolio_id: str
    timestamp: str
    total_value: float
    cash_balance: float
    positions_value: float
    unrealized_pnl: float
    action: str  # 'update', 'position_change', 'trade_executed'


class MessageProducer:
    """Pulsar message producer with retry logic and error handling."""
    
    def __init__(self, pulsar_url: str = "pulsar://localhost:6650"):
        self.pulsar_url = pulsar_url
        self.client: Optional[pulsar.Client] = None
        self.producers: Dict[str, pulsar.Producer] = {}
        self.settings = get_settings()
        
    async def connect(self):
        """Initialize Pulsar client and producers."""
        try:
            self.client = pulsar.Client(
                self.pulsar_url,
                connection_timeout_ms=10000,
                operation_timeout_seconds=30,
                log_conf_file_path=None,  # Disable Pulsar's internal logging
            )
            logger.info(f"Connected to Pulsar at {self.pulsar_url}")
            
            # Create producers for key topics
            await self._create_producer(
                'market_data',
                f'persistent://trading/{self.settings.environment}/market-data',
                AvroSchema(MarketDataMessage)
            )
            
            await self._create_producer(
                'trading_signals', 
                f'persistent://trading/{self.settings.environment}/trading-signals',
                AvroSchema(TradingSignalMessage)
            )
            
            await self._create_producer(
                'portfolio_updates',
                f'persistent://trading/{self.settings.environment}/portfolio-updates', 
                AvroSchema(PortfolioUpdateMessage)
            )
            
        except Exception as e:
            logger.error(f"Failed to connect to Pulsar: {e}")
            raise
            
    async def _create_producer(self, name: str, topic: str, schema):
        """Create a producer for a specific topic."""
        try:
            producer = self.client.create_producer(
                topic,
                schema=schema,
                batching_enabled=True,
                batching_max_messages=100,
                batching_max_publish_delay_ms=10,
                send_timeout_ms=30000,
                block_if_queue_full=False,
                max_pending_messages=1000,
            )
            self.producers[name] = producer
            logger.info(f"Created producer for {name} on topic {topic}")
        except Exception as e:
            logger.error(f"Failed to create producer for {name}: {e}")
            raise
            
    async def send_market_data(self, message: MarketDataMessage) -> str:
        """Send market data message."""
        return await self._send_message('market_data', message)
        
    async def send_trading_signal(self, message: TradingSignalMessage) -> str:
        """Send trading signal message."""
        return await self._send_message('trading_signals', message)
        
    async def send_portfolio_update(self, message: PortfolioUpdateMessage) -> str:
        """Send portfolio update message."""
        return await self._send_message('portfolio_updates', message)
        
    async def _send_message(self, producer_name: str, message: Any, retries: int = 3) -> str:
        """Send message with retry logic."""
        producer = self.producers.get(producer_name)
        if not producer:
            raise RuntimeError(f"Producer {producer_name} not initialized")
            
        last_exception = None
        for attempt in range(retries):
            try:
                # Send message asynchronously
                message_id = producer.send(message)
                logger.debug(f"Sent message to {producer_name}: {message_id}")
                return str(message_id)
            except Exception as e:
                last_exception = e
                logger.warning(f"Send attempt {attempt + 1} failed for {producer_name}: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
        logger.error(f"Failed to send message to {producer_name} after {retries} attempts")
        raise last_exception
        
    async def close(self):
        """Close all producers and client."""
        for name, producer in self.producers.items():
            try:
                producer.close()
                logger.info(f"Closed producer {name}")
            except Exception as e:
                logger.warning(f"Error closing producer {name}: {e}")
                
        if self.client:
            self.client.close()
            logger.info("Closed Pulsar client")


class MessageConsumer:
    """Pulsar message consumer with automatic acknowledgment and error handling."""
    
    def __init__(self, pulsar_url: str = "pulsar://localhost:6650"):
        self.pulsar_url = pulsar_url
        self.client: Optional[pulsar.Client] = None
        self.consumers: Dict[str, pulsar.Consumer] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.settings = get_settings()
        self._running = False
        
    async def connect(self):
        """Initialize Pulsar client."""
        try:
            self.client = pulsar.Client(
                self.pulsar_url,
                connection_timeout_ms=10000,
                operation_timeout_seconds=30,
                log_conf_file_path=None,
            )
            logger.info(f"Connected to Pulsar consumer at {self.pulsar_url}")
        except Exception as e:
            logger.error(f"Failed to connect consumer to Pulsar: {e}")
            raise
            
    async def subscribe_market_data(self, handler: Callable[[MarketDataMessage], None], subscription_name: str = "market-data-consumer"):
        """Subscribe to market data messages."""
        topic = f'persistent://trading/{self.settings.environment}/market-data'
        await self._subscribe(
            'market_data',
            topic, 
            subscription_name,
            AvroSchema(MarketDataMessage),
            handler
        )
        
    async def subscribe_trading_signals(self, handler: Callable[[TradingSignalMessage], None], subscription_name: str = "signal-consumer"):
        """Subscribe to trading signal messages."""
        topic = f'persistent://trading/{self.settings.environment}/trading-signals'
        await self._subscribe(
            'trading_signals',
            topic,
            subscription_name, 
            AvroSchema(TradingSignalMessage),
            handler
        )
        
    async def subscribe_portfolio_updates(self, handler: Callable[[PortfolioUpdateMessage], None], subscription_name: str = "portfolio-consumer"):
        """Subscribe to portfolio update messages."""
        topic = f'persistent://trading/{self.settings.environment}/portfolio-updates'
        await self._subscribe(
            'portfolio_updates', 
            topic,
            subscription_name,
            AvroSchema(PortfolioUpdateMessage), 
            handler
        )
        
    async def _subscribe(self, name: str, topic: str, subscription_name: str, schema, handler: Callable):
        """Subscribe to a topic with message handler."""
        try:
            consumer = self.client.subscribe(
                topic,
                subscription_name=subscription_name,
                schema=schema,
                consumer_type=pulsar.ConsumerType.Shared,
                receiver_queue_size=1000,
                max_total_receiver_queue_size_across_partitions=50000,
            )
            self.consumers[name] = consumer
            self.message_handlers[name] = handler
            logger.info(f"Subscribed to {name} on topic {topic}")
        except Exception as e:
            logger.error(f"Failed to subscribe to {name}: {e}")
            raise
            
    async def start_consuming(self):
        """Start consuming messages from all subscribed topics."""
        if not self.consumers:
            logger.warning("No consumers configured")
            return
            
        self._running = True
        logger.info("Started message consumption")
        
        # Start consumer tasks for each subscription
        tasks = []
        for name, consumer in self.consumers.items():
            task = asyncio.create_task(self._consume_messages(name, consumer))
            tasks.append(task)
            
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in message consumption: {e}")
            self._running = False
            
    async def stop_consuming(self):
        """Stop consuming messages."""
        self._running = False
        logger.info("Stopped message consumption")
        
    async def _consume_messages(self, name: str, consumer: pulsar.Consumer):
        """Consume messages for a specific topic."""
        handler = self.message_handlers.get(name)
        if not handler:
            logger.error(f"No handler configured for {name}")
            return
            
        while self._running:
            try:
                # Receive message with timeout
                msg = consumer.receive(timeout_millis=1000)
                
                try:
                    # Process message
                    await self._process_message(msg, handler)
                    # Acknowledge successful processing
                    consumer.acknowledge(msg)
                    logger.debug(f"Processed message from {name}: {msg.message_id()}")
                except Exception as e:
                    logger.error(f"Failed to process message from {name}: {e}")
                    # Negative acknowledge to trigger redelivery
                    consumer.negative_acknowledge(msg)
                    
            except Exception as e:
                if "Timeout" not in str(e):
                    logger.warning(f"Consumer {name} error: {e}")
                # Short sleep to prevent tight loop on persistent errors
                await asyncio.sleep(0.1)
                
    async def _process_message(self, msg, handler):
        """Process a single message."""
        try:
            # Extract message data
            message_data = msg.value()
            
            # Call handler
            if asyncio.iscoroutinefunction(handler):
                await handler(message_data)
            else:
                handler(message_data)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise
            
    async def close(self):
        """Close all consumers and client."""
        await self.stop_consuming()
        
        for name, consumer in self.consumers.items():
            try:
                consumer.unsubscribe()
                consumer.close()
                logger.info(f"Closed consumer {name}")
            except Exception as e:
                logger.warning(f"Error closing consumer {name}: {e}")
                
        if self.client:
            self.client.close()
            logger.info("Closed Pulsar consumer client")


# Global instances
_message_producer: Optional[MessageProducer] = None
_message_consumer: Optional[MessageConsumer] = None


async def get_message_producer() -> MessageProducer:
    """Get or create global message producer."""
    global _message_producer
    if _message_producer is None:
        settings = get_settings()
        _message_producer = MessageProducer(settings.messaging.pulsar_url)
        await _message_producer.connect()
    return _message_producer


async def get_message_consumer() -> MessageConsumer:
    """Get or create global message consumer.""" 
    global _message_consumer
    if _message_consumer is None:
        settings = get_settings()
        _message_consumer = MessageConsumer(settings.messaging.pulsar_url)
        await _message_consumer.connect()
    return _message_consumer


def get_pulsar_client():
    """Get Pulsar client for direct usage."""
    settings = get_settings()
    try:
        client = pulsar.Client(
            settings.messaging.pulsar_url,
            connection_timeout_ms=10000,
            operation_timeout_seconds=30,
            log_conf_file_path=None,
        )
        return client
    except Exception as e:
        logger.error(f"Failed to create Pulsar client: {e}")
        raise


async def close_messaging():
    """Close global messaging clients."""
    global _message_producer, _message_consumer
    
    if _message_producer:
        await _message_producer.close()
        _message_producer = None
        
    if _message_consumer:
        await _message_consumer.close()  
        _message_consumer = None
        
    logger.info("Closed all messaging connections")