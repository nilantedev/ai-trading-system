#!/usr/bin/env python3
"""
Reinforcement Learning Engine for Trading - Continuous learning and adaptation
Uses Q-Learning with experience replay for optimal trading decisions.
"""

import asyncio
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import random
import pickle
import os

from trading_common import MarketData, get_settings, get_logger
from trading_common.cache import get_trading_cache

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class TradingState:
    """Represents current trading state for RL agent."""
    symbol: str
    current_price: float
    price_change_1m: float  # 1 minute price change
    price_change_5m: float  # 5 minute price change
    price_change_15m: float  # 15 minute price change
    volume_ratio: float  # Current volume vs average
    rsi: float  # RSI indicator
    macd: float  # MACD signal
    bollinger_position: float  # Position within Bollinger Bands (0-1)
    portfolio_exposure: float  # Current exposure to this symbol (0-1)
    unrealized_pnl: float  # Unrealized P&L for current position
    market_session: int  # 0=premarket, 1=open, 2=midday, 3=close, 4=afterhours
    volatility_regime: int  # 0=low, 1=medium, 2=high volatility
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for RL model."""
        return np.array([
            self.current_price / 1000,  # Normalize price
            self.price_change_1m,
            self.price_change_5m, 
            self.price_change_15m,
            self.volume_ratio,
            self.rsi / 100,  # Normalize RSI
            self.macd,
            self.bollinger_position,
            self.portfolio_exposure,
            self.unrealized_pnl,
            self.market_session / 4,  # Normalize session
            self.volatility_regime / 2  # Normalize volatility
        ], dtype=np.float32)


@dataclass
class TradingAction:
    """Represents trading action for RL agent."""
    action_type: int  # 0=hold, 1=buy, 2=sell
    position_size: float  # Size of position (0-1)
    confidence: float  # Confidence in action (0-1)
    
    @classmethod
    def from_index(cls, action_index: int) -> 'TradingAction':
        """Create action from discrete action index."""
        # 15 discrete actions: 5 hold sizes, 5 buy sizes, 5 sell sizes
        if action_index < 5:  # Hold with different sizes
            return cls(0, action_index * 0.2, 0.5)
        elif action_index < 10:  # Buy with different sizes
            size = (action_index - 4) * 0.2  # 0.2, 0.4, 0.6, 0.8, 1.0
            return cls(1, size, 0.7)
        else:  # Sell with different sizes
            size = (action_index - 9) * 0.2
            return cls(2, size, 0.7)


@dataclass 
class Experience:
    """Experience tuple for replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    timestamp: datetime


class DQNAgent:
    """Deep Q-Network agent for trading decisions."""
    
    def __init__(self, state_size: int = 12, action_size: int = 15, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Hyperparameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        self.batch_size = 32
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        
        # Simple neural network weights (would use proper neural network in production)
        self.q_table = np.random.randn(state_size, action_size) * 0.1
        
        # Performance tracking
        self.total_rewards = 0
        self.episode_count = 0
        self.win_rate = 0.5
        
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        experience = Experience(
            state=state,
            action=action, 
            reward=reward,
            next_state=next_state,
            done=done,
            timestamp=datetime.utcnow()
        )
        self.memory.append(experience)
    
    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy."""
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.action_size)  # Random action
        
        # Use simplified Q-value calculation
        state_hash = hash(state.tobytes()) % 1000
        q_values = self.q_table[state_hash % self.state_size]
        return np.argmax(q_values)
    
    def replay(self) -> float:
        """Train the agent on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return 0.0
            
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        
        total_loss = 0.0
        
        for experience in batch:
            state = experience.state
            action = experience.action
            reward = experience.reward
            next_state = experience.next_state
            done = experience.done
            
            # Calculate target Q-value
            if done:
                target = reward
            else:
                next_state_hash = hash(next_state.tobytes()) % 1000
                next_q_values = self.q_table[next_state_hash % self.state_size]
                target = reward + self.gamma * np.max(next_q_values)
            
            # Update Q-table (simplified)
            state_hash = hash(state.tobytes()) % 1000
            current_q = self.q_table[state_hash % self.state_size, action]
            
            # Q-learning update
            self.q_table[state_hash % self.state_size, action] = (
                current_q + self.learning_rate * (target - current_q)
            )
            
            total_loss += abs(target - current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return total_loss / self.batch_size
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        model_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'total_rewards': self.total_rewards,
            'episode_count': self.episode_count,
            'win_rate': self.win_rate
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"RL model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = model_data['q_table']
            self.epsilon = model_data['epsilon']
            self.total_rewards = model_data['total_rewards']
            self.episode_count = model_data['episode_count']
            self.win_rate = model_data['win_rate']
            
            logger.info(f"RL model loaded from {filepath}")
        except Exception as e:
            logger.warning(f"Failed to load RL model: {e}")


class ReinforcementLearningEngine:
    """Main RL engine for trading system."""
    
    def __init__(self):
        self.agent = DQNAgent()
        self.cache = None
        
        # State tracking
        self.current_states = {}  # symbol -> TradingState
        self.previous_actions = {}  # symbol -> TradingAction
        self.position_entry_prices = {}  # symbol -> entry_price
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.learning_episodes = 0
        
        # Model persistence
        self.model_path = "/tmp/trading_rl_model.pkl"
        
        # Training configuration
        self.training_mode = True
        self.reward_lookback_minutes = 5  # Look back 5 minutes for reward calculation
        
    async def initialize(self):
        """Initialize the RL engine."""
        self.cache = get_trading_cache()
        
        # Try to load existing model
        if os.path.exists(self.model_path):
            self.agent.load_model(self.model_path)
            logger.info("Loaded existing RL model")
        else:
            logger.info("Starting with fresh RL model")
        
        logger.info("Reinforcement Learning Engine initialized")
    
    async def update_state(self, symbol: str, market_data: MarketData, 
                          indicators: Dict[str, float], portfolio_data: Dict[str, Any]):
        """Update the current state for a symbol."""
        
        # Calculate price changes
        price_changes = await self._calculate_price_changes(symbol, market_data)
        
        # Determine market session
        market_session = self._get_market_session(market_data.timestamp)
        
        # Determine volatility regime
        volatility_regime = self._get_volatility_regime(indicators.get('volatility', 0.02))
        
        # Create trading state
        state = TradingState(
            symbol=symbol,
            current_price=market_data.close,
            price_change_1m=price_changes.get('1m', 0.0),
            price_change_5m=price_changes.get('5m', 0.0),
            price_change_15m=price_changes.get('15m', 0.0),
            volume_ratio=indicators.get('volume_ratio', 1.0),
            rsi=indicators.get('rsi', 50.0),
            macd=indicators.get('macd', 0.0),
            bollinger_position=indicators.get('bollinger_position', 0.5),
            portfolio_exposure=portfolio_data.get('exposure', 0.0),
            unrealized_pnl=portfolio_data.get('unrealized_pnl', 0.0),
            market_session=market_session,
            volatility_regime=volatility_regime
        )
        
        # If we have a previous state and action, calculate reward and learn
        if symbol in self.current_states and symbol in self.previous_actions:
            await self._process_experience(symbol, state, market_data)
        
        # Update current state
        self.current_states[symbol] = state
    
    async def get_trading_recommendation(self, symbol: str) -> Optional[TradingAction]:
        """Get RL-based trading recommendation."""
        if symbol not in self.current_states:
            logger.warning(f"No state available for {symbol}")
            return None
        
        state = self.current_states[symbol]
        state_array = state.to_array()
        
        # Get action from agent
        action_index = self.agent.act(state_array)
        action = TradingAction.from_index(action_index)
        
        # Store action for next reward calculation
        self.previous_actions[symbol] = action
        
        logger.debug(f"RL recommendation for {symbol}: {action.action_type} size={action.position_size:.2f}")
        
        return action
    
    async def record_trade_outcome(self, symbol: str, entry_price: float, 
                                 exit_price: float, position_size: float, 
                                 hold_time_minutes: int):
        """Record the outcome of a completed trade for learning."""
        
        # Calculate trade P&L
        if position_size > 0:  # Long position
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # Short position  
            pnl_pct = (entry_price - exit_price) / entry_price
            
        pnl_dollars = pnl_pct * abs(position_size) * entry_price
        
        # Update performance metrics
        self.total_trades += 1
        self.total_pnl += pnl_dollars
        
        if pnl_dollars > 0:
            self.winning_trades += 1
            
        self.agent.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.5
        
        # Calculate reward based on risk-adjusted return
        risk_free_rate = 0.02  # 2% annual risk-free rate
        time_factor = hold_time_minutes / (365 * 24 * 60)  # Convert to annual
        risk_adjusted_return = pnl_pct - (risk_free_rate * time_factor)
        
        # Reward function: emphasize risk-adjusted returns and trade efficiency
        base_reward = risk_adjusted_return * 100  # Scale up
        time_penalty = -0.1 * (hold_time_minutes / 60)  # Slight penalty for holding too long
        
        reward = base_reward + time_penalty
        
        logger.info(f"Trade completed for {symbol}: P&L=${pnl_dollars:.2f} ({pnl_pct*100:.2f}%), "
                   f"hold_time={hold_time_minutes}min, reward={reward:.2f}")
        
        # Cache performance metrics
        if self.cache:
            performance_data = {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': self.agent.win_rate,
                'total_pnl': self.total_pnl,
                'avg_pnl_per_trade': self.total_pnl / self.total_trades if self.total_trades > 0 else 0,
                'learning_episodes': self.learning_episodes,
                'epsilon': self.agent.epsilon
            }
            await self.cache.set_json("rl_performance", performance_data, ttl=300)
    
    async def _process_experience(self, symbol: str, current_state: TradingState, 
                                market_data: MarketData):
        """Process experience and train the agent."""
        
        previous_state = self.current_states[symbol]
        previous_action = self.previous_actions[symbol]
        
        # Calculate short-term reward based on price movement
        price_change = (current_state.current_price - previous_state.current_price) / previous_state.current_price
        
        # Reward based on action alignment with price movement
        if previous_action.action_type == 1:  # Buy action
            reward = price_change * 100  # Reward for price going up after buy
        elif previous_action.action_type == 2:  # Sell action  
            reward = -price_change * 100  # Reward for price going down after sell
        else:  # Hold action
            reward = -abs(price_change) * 10  # Small penalty for unnecessary holding during movement
        
        # Add volume and volatility considerations
        if current_state.volume_ratio > 1.5:  # High volume
            reward *= 1.1  # Boost reward for high volume moves
            
        # Store experience
        previous_state_array = previous_state.to_array()
        current_state_array = current_state.to_array()
        
        action_index = self._action_to_index(previous_action)
        
        self.agent.remember(
            state=previous_state_array,
            action=action_index,
            reward=reward,
            next_state=current_state_array,
            done=False  # Continuous trading, never truly "done"
        )
        
        # Train if we have enough experiences
        if len(self.agent.memory) >= self.agent.batch_size:
            loss = self.agent.replay()
            self.learning_episodes += 1
            
            if self.learning_episodes % 100 == 0:  # Save model every 100 episodes
                self.agent.save_model(self.model_path)
                logger.info(f"RL training episode {self.learning_episodes}, loss={loss:.4f}, "
                           f"epsilon={self.agent.epsilon:.3f}")
    
    def _action_to_index(self, action: TradingAction) -> int:
        """Convert TradingAction to discrete action index."""
        size_index = int(action.position_size * 5)  # 0-4
        
        if action.action_type == 0:  # Hold
            return min(size_index, 4)
        elif action.action_type == 1:  # Buy
            return min(5 + size_index, 9)
        else:  # Sell
            return min(10 + size_index, 14)
    
    async def _calculate_price_changes(self, symbol: str, current_data: MarketData) -> Dict[str, float]:
        """Calculate price changes over different timeframes."""
        changes = {}
        
        if self.cache:
            try:
                # Get historical data from cache
                for timeframe in ['1m', '5m', '15m']:
                    cache_key = f"price_history:{symbol}:{timeframe}"
                    historical_price = await self.cache.get(cache_key)
                    
                    if historical_price:
                        historical_price = float(historical_price)
                        changes[timeframe] = (current_data.close - historical_price) / historical_price
                    else:
                        changes[timeframe] = 0.0
                    
                    # Update cache with current price
                    await self.cache.set(cache_key, str(current_data.close), ttl=int(timeframe[:-1]) * 60)
                    
            except Exception as e:
                logger.warning(f"Failed to calculate price changes for {symbol}: {e}")
        
        return changes
    
    def _get_market_session(self, timestamp: datetime) -> int:
        """Determine market session based on time."""
        hour = timestamp.hour
        
        if 4 <= hour < 9:  # Pre-market
            return 0
        elif 9 <= hour < 11:  # Market open
            return 1
        elif 11 <= hour < 15:  # Mid-day
            return 2
        elif 15 <= hour < 16:  # Market close
            return 3
        else:  # After-hours
            return 4
    
    def _get_volatility_regime(self, volatility: float) -> int:
        """Determine volatility regime."""
        if volatility < 0.015:  # 1.5% daily volatility
            return 0  # Low
        elif volatility < 0.035:  # 3.5% daily volatility
            return 1  # Medium
        else:
            return 2  # High
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get RL engine performance metrics."""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.agent.win_rate,
            'total_pnl': self.total_pnl,
            'avg_pnl_per_trade': self.total_pnl / self.total_trades if self.total_trades > 0 else 0,
            'learning_episodes': self.learning_episodes,
            'exploration_rate': self.agent.epsilon,
            'memory_size': len(self.agent.memory),
            'training_mode': self.training_mode
        }
    
    async def set_training_mode(self, enabled: bool):
        """Enable or disable training mode."""
        self.training_mode = enabled
        if not enabled:
            self.agent.epsilon = 0.01  # Minimal exploration in production
        logger.info(f"RL training mode {'enabled' if enabled else 'disabled'}")


# Global RL engine instance
rl_engine: Optional[ReinforcementLearningEngine] = None


async def get_rl_engine() -> ReinforcementLearningEngine:
    """Get or create RL engine instance."""
    global rl_engine
    if rl_engine is None:
        rl_engine = ReinforcementLearningEngine()
        await rl_engine.initialize()
    return rl_engine