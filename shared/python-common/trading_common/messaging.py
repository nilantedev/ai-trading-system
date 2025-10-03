"""Message infrastructure for AI trading system using Apache Pulsar."""

import json
import logging
import asyncio
import os
import random
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import pulsar
from pulsar.schema import AvroSchema, JsonSchema, StringSchema
from pulsar.schema import Record, String, Float, Integer, Long, Double

from .config import get_settings

logger = logging.getLogger(__name__)


class MarketDataMessage(Record):
    """Market data message schema for Pulsar."""
    _avro_namespace = 'com.trading.messages'
    symbol = String()
    timestamp = String()
    open = Double()
    high = Double()
    low = Double()
    close = Double()
    volume = Long()
    data_source = String()


class TradingSignalMessage(Record):
    """Trading signal message schema for Pulsar."""
    _avro_namespace = 'com.trading.messages'
    id = String()
    timestamp = String()
    symbol = String()
    signal_type = String()
    confidence = Double()
    target_price = Double()
    stop_loss = Double(default=0.0)  # Using default instead of Optional
    take_profit = Double(default=0.0)  # Using default instead of Optional
    strategy_name = String()
    agent_id = String()
    reasoning = String()


class PortfolioUpdateMessage(Record):
    """Portfolio update message schema for Pulsar."""
    _avro_namespace = 'com.trading.messages'
    portfolio_id = String()
    timestamp = String()
    total_value = Double()
    cash_balance = Double()
    positions_value = Double()
    unrealized_pnl = Double()
    action = String()  # 'update', 'position_change', 'trade_executed'


class OrderRequestMessage(Record):
    """Order request message schema for Pulsar."""
    _avro_namespace = 'com.trading.messages'
    order_id = String()
    timestamp = String()
    symbol = String()
    side = String()  # 'buy' or 'sell'
    order_type = String()  # 'market', 'limit', 'stop', etc.
    quantity = Double()
    price = Double(default=0.0)  # For limit/stop orders
    stop_price = Double(default=0.0)  # For stop orders
    time_in_force = String(default='DAY')  # 'DAY', 'GTC', 'IOC', 'FOK'
    strategy_name = String()
    agent_id = String()


class MessageProducer:
    """Pulsar message producer with retry logic and error handling."""
    
    def __init__(self, pulsar_url: str = "pulsar://trading-pulsar:6650"):
        self.pulsar_url = pulsar_url
        self.client: Optional[pulsar.Client] = None
        self.producers: Dict[str, pulsar.Producer] = {}
        self.settings = get_settings()
        
    async def connect(self):
        """Initialize Pulsar client and producers with retries and jittered backoff."""
        max_attempts = int(os.getenv("MSG_CONNECT_MAX_ATTEMPTS", "6"))
        base_delay = float(os.getenv("MSG_CONNECT_BASE_DELAY", "1.5"))
        for attempt in range(1, max_attempts + 1):
            try:
                self.client = pulsar.Client(
                    self.pulsar_url,
                    connection_timeout_ms=30000,  # 30 seconds
                    operation_timeout_seconds=60,  # 60 seconds
                    log_conf_file_path=None,
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

                await self._create_producer(
                    'order_requests',
                    f'persistent://trading/{self.settings.environment}/order-requests',
                    AvroSchema(OrderRequestMessage)
                )

                # Optional: indicator analysis channel (JSON string payload)
                await self._create_producer(
                    'indicator_analysis',
                    f'persistent://trading/{self.settings.environment}/indicator-analysis',
                    StringSchema()
                )

                # Order updates channel (JSON string payload)
                await self._create_producer(
                    'order_updates',
                    f'persistent://trading/{self.settings.environment}/order-updates',
                    StringSchema()
                )

                # Fills channel (JSON string payload)
                await self._create_producer(
                    'fills',
                    f'persistent://trading/{self.settings.environment}/fills',
                    StringSchema()
                )

                # Risk alerts channel (JSON string payload)
                await self._create_producer(
                    'risk_alerts',
                    f'persistent://trading/{self.settings.environment}/risk-alerts',
                    StringSchema()
                )

                return  # success
            except Exception as e:
                logger.error(f"Failed to connect to Pulsar (attempt {attempt}/{max_attempts}): {e}")
                if attempt >= max_attempts:
                    raise
                # Jittered exponential backoff
                delay = base_delay * (2 ** (attempt - 1))
                delay = min(delay, 30.0)  # cap delay
                delay += random.uniform(0, 0.5)
                await asyncio.sleep(delay)
            
    async def _create_producer(self, name: str, topic: str, schema):
        """Create a producer for a specific topic with retries."""
        max_attempts = int(os.getenv("MSG_PRODUCER_MAX_ATTEMPTS", "5"))
        base_delay = float(os.getenv("MSG_PRODUCER_BASE_DELAY", "0.8"))
        last_err: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                producer = self.client.create_producer(
                    topic,
                    schema=schema,
                    batching_enabled=True,
                    batching_max_messages=100,
                    batching_max_publish_delay_ms=10,
                    send_timeout_millis=30000,
                    block_if_queue_full=False,
                    max_pending_messages=1000,
                )
                self.producers[name] = producer
                logger.info(f"Created producer for {name} on topic {topic}")
                return
            except Exception as e:
                last_err = e
                logger.error(f"Failed to create producer for {name} (attempt {attempt}/{max_attempts}): {e}")
                if attempt >= max_attempts:
                    break
                delay = base_delay * (2 ** (attempt - 1))
                delay = min(delay, 15.0)
                delay += random.uniform(0, 0.3)
                await asyncio.sleep(delay)
        raise last_err if last_err else RuntimeError(f"Unknown error creating producer {name}")
            
    async def send_market_data(self, message: MarketDataMessage) -> str:
        """Send market data message."""
        return await self._send_message('market_data', message)
        
    async def send_trading_signal(self, message: TradingSignalMessage) -> str:
        """Send trading signal message."""
        return await self._send_message('trading_signals', message)
        
    async def send_portfolio_update(self, message: PortfolioUpdateMessage) -> str:
        """Send portfolio update message."""
        return await self._send_message('portfolio_updates', message)
        
    async def send_order_request(self, message: OrderRequestMessage) -> str:
        """Send order request message."""
        return await self._send_message('order_requests', message)

    async def send_indicator_analysis(self, payload: Any) -> str:
        """Send indicator analysis as JSON string payload.

        Accepts either a pre-serialized JSON string or a Python dict-like object
        which will be serialized via json.dumps.
        """
        if not isinstance(payload, str):
            try:
                payload = json.dumps(payload)
            except Exception as e:
                raise ValueError(f"Indicator analysis payload must be JSON-serializable: {e}")
        return await self._send_message('indicator_analysis', payload)

    async def send_order_update(self, payload: Any) -> str:
        """Send order update as JSON string payload to order-updates topic."""
        if not isinstance(payload, str):
            try:
                payload = json.dumps(payload)
            except Exception as e:
                raise ValueError(f"Order update payload must be JSON-serializable: {e}")
        return await self._send_message('order_updates', payload)

    async def send_fill(self, payload: Any) -> str:
        """Send fill notification as JSON string payload to fills topic."""
        if not isinstance(payload, str):
            try:
                payload = json.dumps(payload)
            except Exception as e:
                raise ValueError(f"Fill payload must be JSON-serializable: {e}")
        return await self._send_message('fills', payload)

    async def send_risk_alert(self, payload: Any) -> str:
        """Send risk alert as JSON string payload to risk-alerts topic."""
        if not isinstance(payload, str):
            try:
                payload = json.dumps(payload)
            except Exception as e:
                raise ValueError(f"Risk alert payload must be JSON-serializable: {e}")
        return await self._send_message('risk_alerts', payload)
        
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
    
    def __init__(self, pulsar_url: str = "pulsar://trading-pulsar:6650"):
        self.pulsar_url = pulsar_url
        self.client: Optional[pulsar.Client] = None
        self.consumers: Dict[str, pulsar.Consumer] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.settings = get_settings()
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=4)  # For running sync operations
        
    async def connect(self):
        """Initialize Pulsar client with retries."""
        max_attempts = int(os.getenv("MSG_CONNECT_MAX_ATTEMPTS", "6"))
        base_delay = float(os.getenv("MSG_CONNECT_BASE_DELAY", "1.5"))
        for attempt in range(1, max_attempts + 1):
            try:
                self.client = pulsar.Client(
                    self.pulsar_url,
                    connection_timeout_ms=30000,
                    operation_timeout_seconds=60,
                    log_conf_file_path=None,
                )
                logger.info(f"Connected to Pulsar consumer at {self.pulsar_url}")
                return
            except Exception as e:
                logger.error(f"Failed to connect consumer to Pulsar (attempt {attempt}/{max_attempts}): {e}")
                if attempt >= max_attempts:
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                delay = min(delay, 30.0)
                delay += random.uniform(0, 0.5)
                await asyncio.sleep(delay)
            
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
        
    async def subscribe_position_updates(self, handler: Callable[[PortfolioUpdateMessage], None], subscription_name: str = "position-consumer"):
        """Alias for subscribing to position/portfolio update messages.

        Some services refer to portfolio updates as position updates. This method
        delegates to subscribe_portfolio_updates to maintain backwards compatibility
        with existing callers expecting a position updates API.
        """
        await self.subscribe_portfolio_updates(handler, subscription_name)
        
    async def subscribe_order_requests(self, handler: Callable[[OrderRequestMessage], None], subscription_name: str = "order-consumer"):
        """Subscribe to order request messages."""
        topic = f'persistent://trading/{self.settings.environment}/order-requests'
        await self._subscribe(
            'order_requests',
            topic,
            subscription_name,
            AvroSchema(OrderRequestMessage),
            handler
        )

    async def subscribe_broker_updates(self, handler: Callable[[Any], None], subscription_name: str = "broker-updates-consumer"):
        """Subscribe to broker updates messages (JSON string payload).

        This is a flexible channel intended for execution/broker status updates. The schema is a JSON string.
        """
        topic = f'persistent://trading/{self.settings.environment}/broker-updates'
        await self._subscribe(
            'broker_updates',
            topic,
            subscription_name,
            StringSchema(),
            handler
        )

    async def subscribe_indicator_analysis(self, handler: Callable[[Any], None], subscription_name: str = "indicator-analysis-consumer"):
        """Subscribe to indicator analysis messages.

        This channel carries JSON-encoded analysis payloads as strings to keep the schema flexible:
        {
          "symbol": "AAPL",
          "timestamp": "2025-09-10T04:11:00Z",
          "indicators": { ... },
          "overall_signal": "BUY|SELL|HOLD",
          "signal_strength": 0.0-1.0,
          "confidence": 0.0-1.0
        }
        """
        topic = f'persistent://trading/{self.settings.environment}/indicator-analysis'
        await self._subscribe(
            'indicator_analysis',
            topic,
            subscription_name,
            StringSchema(),
            handler
        )
        
    async def _subscribe(self, name: str, topic: str, subscription_name: str, schema, handler: Callable):
        """Subscribe to a topic with message handler and retries."""
        max_retries = int(os.getenv("MSG_SUBSCRIBE_MAX_ATTEMPTS", "6"))
        base_delay = float(os.getenv("MSG_SUBSCRIBE_BASE_DELAY", "1.0"))

        for attempt in range(1, max_retries + 1):
            try:
                consumer = self.client.subscribe(
                    topic,
                    subscription_name=subscription_name,
                    schema=schema,
                    consumer_type=pulsar.ConsumerType.Shared,
                    receiver_queue_size=1000,
                    max_total_receiver_queue_size_across_partitions=50000,
                    initial_position=pulsar.InitialPosition.Latest,
                    replicate_subscription_state_enabled=True,
                )
                self.consumers[name] = consumer
                self.message_handlers[name] = handler
                logger.info(f"Subscribed to {name} on topic {topic}")
                return
            except Exception as e:
                logger.error(f"Failed to subscribe to {name} (attempt {attempt}/{max_retries}): {e}")
                if attempt >= max_retries:
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                delay = min(delay, 20.0)
                delay += random.uniform(0, 0.5)
                await asyncio.sleep(delay)
            
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
            
        loop = asyncio.get_event_loop()
        
        while self._running:
            try:
                # Run the blocking receive in executor to avoid blocking async loop
                msg = await loop.run_in_executor(
                    self._executor,
                    lambda: self._receive_with_timeout(consumer, 1000)
                )
                
                if msg:
                    try:
                        # Process message
                        await self._process_message(msg, handler)
                        # Acknowledge successful processing
                        await loop.run_in_executor(
                            self._executor,
                            consumer.acknowledge,
                            msg
                        )
                        logger.debug(f"Processed message from {name}: {msg.message_id()}")
                    except Exception as e:
                        logger.error(f"Failed to process message from {name}: {e}")
                        # Negative acknowledge to trigger redelivery
                        await loop.run_in_executor(
                            self._executor,
                            consumer.negative_acknowledge,
                            msg
                        )
                else:
                    # No message received, small delay
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                # Only log non-timeout errors
                error_str = str(e)
                if "Timeout" not in error_str and "TimeOut" not in error_str:
                    logger.warning(f"Consumer {name} error: {e}")
                    # Longer sleep on real errors
                    await asyncio.sleep(1.0)
                else:
                    # Very short sleep on timeouts to keep responsive
                    await asyncio.sleep(0.01)
    
    def _receive_with_timeout(self, consumer: pulsar.Consumer, timeout_ms: int):
        """Synchronous receive with timeout handling."""
        try:
            return consumer.receive(timeout_millis=timeout_ms)
        except Exception as e:
            # Silently return None on timeout (normal for empty topics)
            # Only log non-timeout errors
            error_str = str(e)
            if "Timeout" not in error_str and "TimeOut" not in error_str:
                logger.warning(f"Consumer receive error: {e}")
            return None
                
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
        
        # Shutdown executor
        self._executor.shutdown(wait=False)


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
            connection_timeout_ms=30000,  # Increased to 30 seconds
            operation_timeout_seconds=60,  # Increased to 60 seconds
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