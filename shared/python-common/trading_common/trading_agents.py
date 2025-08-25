"""OpenAI Swarm-based trading agents for coordinated decision making."""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from swarm import Swarm, Agent
from .config import get_settings
from .ai_models import get_model_router, ModelType

logger = logging.getLogger(__name__)


@dataclass
class TradingContext:
    """Context information for trading decisions."""
    symbol: str
    current_price: float
    market_data: Dict[str, Any]
    portfolio_state: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    news_sentiment: Optional[Dict[str, Any]] = None


@dataclass
class TradingDecision:
    """Trading decision from agent coordination."""
    action: str  # 'buy', 'sell', 'hold'
    symbol: str
    confidence: float
    reasoning: str
    position_size: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    agent_consensus: Dict[str, str] = None


class TradingAgentOrchestrator:
    """Coordinates multiple trading agents using OpenAI Swarm."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize Swarm client if API key available
        if self.settings.ai.openai_api_key:
            self.client = Swarm()
        else:
            self.client = None
            logger.info("OpenAI API key not available, using mock responses")
            
        self.agents = self._create_agents()
        
    def _create_agents(self):
        """Create specialized trading agents."""
        
        # Market Analysis Agent
        market_analyst = Agent(
            name="Market Analyst",
            instructions="""
            You are a senior market analyst specializing in technical and fundamental analysis.
            
            Your responsibilities:
            - Analyze market data, price movements, and trading volumes
            - Identify trends, support/resistance levels, and chart patterns
            - Assess market sentiment and momentum indicators
            - Provide clear, data-driven market insights
            
            Always provide:
            1. Current market trend assessment
            2. Key technical levels (support/resistance)
            3. Volume analysis and momentum indicators
            4. Market sentiment evaluation
            5. Confidence level (0-100%) for your analysis
            
            Be concise but thorough. Focus on actionable insights.
            """,
            model="gpt-4o-mini"  # Use OpenAI for market analysis
        )
        
        # Risk Management Agent
        risk_manager = Agent(
            name="Risk Manager",
            instructions="""
            You are a conservative risk management specialist focused on capital preservation.
            
            Your responsibilities:
            - Assess portfolio risk exposure and position sizing
            - Calculate risk metrics (VaR, Sharpe ratio, drawdown)
            - Set stop-loss and take-profit levels
            - Monitor correlation risks and concentration limits
            - Ensure compliance with risk management rules
            
            Always provide:
            1. Risk assessment for proposed trades (1-10 risk score)
            2. Recommended position size based on risk tolerance
            3. Stop-loss and take-profit recommendations
            4. Portfolio impact analysis
            5. Risk-adjusted return expectations
            
            Never approve trades that violate risk limits. Be conservative.
            """,
            model="claude-3-haiku-20240307"  # Use Claude for risk analysis
        )
        
        # Signal Generation Agent
        signal_generator = Agent(
            name="Signal Generator",
            instructions="""
            You are a quantitative trading signal generator using advanced algorithms.
            
            Your responsibilities:
            - Generate buy/sell/hold signals based on multiple indicators
            - Combine technical analysis with sentiment data
            - Time entry and exit points for optimal execution
            - Assess signal strength and probability of success
            
            Always provide:
            1. Clear trading signal (BUY/SELL/HOLD)
            2. Signal strength (1-100%)
            3. Entry price target and timing
            4. Expected price movement and timeframe
            5. Supporting technical indicators
            
            Be decisive but realistic. Explain your signal logic clearly.
            """,
            model="gpt-4o-mini"  # Use OpenAI for signal generation
        )
        
        # Portfolio Management Agent
        portfolio_manager = Agent(
            name="Portfolio Manager",
            instructions="""
            You are a portfolio manager responsible for execution and allocation decisions.
            
            Your responsibilities:
            - Make final trading decisions based on team input
            - Manage position sizing and portfolio allocation
            - Consider liquidity, market conditions, and timing
            - Balance risk and reward across the portfolio
            
            Always provide:
            1. Final trading recommendation (action + position size)
            2. Execution strategy and timing
            3. Portfolio impact assessment
            4. Alternative scenarios consideration
            5. Success probability estimate
            
            Consider all team input but make the final decision. Be decisive.
            """,
            model="gpt-4"  # Use GPT-4 for complex portfolio decisions
        )
        
        return {
            "market_analyst": market_analyst,
            "risk_manager": risk_manager, 
            "signal_generator": signal_generator,
            "portfolio_manager": portfolio_manager
        }
    
    async def analyze_trading_opportunity(self, context: TradingContext) -> TradingDecision:
        """Coordinate agents to make a trading decision."""
        try:
            # Prepare context for agents
            context_str = self._format_context(context)
            
            # Step 1: Market Analysis
            market_analysis = await self._get_agent_input(
                "market_analyst", 
                f"Analyze the current market situation for {context.symbol}:\n{context_str}"
            )
            
            # Step 2: Generate Trading Signal
            signal_prompt = f"""
            Based on market analysis: {market_analysis}
            
            Generate a trading signal for {context.symbol}:
            {context_str}
            """
            
            trading_signal = await self._get_agent_input("signal_generator", signal_prompt)
            
            # Step 3: Risk Assessment
            risk_prompt = f"""
            Assess the risk for this trading opportunity:
            
            Market Analysis: {market_analysis}
            Trading Signal: {trading_signal}
            Context: {context_str}
            """
            
            risk_assessment = await self._get_agent_input("risk_manager", risk_prompt)
            
            # Step 4: Final Portfolio Decision
            decision_prompt = f"""
            Make the final trading decision based on team analysis:
            
            Market Analysis: {market_analysis}
            Trading Signal: {trading_signal}
            Risk Assessment: {risk_assessment}
            
            Provide your final decision in JSON format:
            {{
                "action": "buy|sell|hold",
                "confidence": 0-100,
                "position_size": "percentage of portfolio",
                "reasoning": "brief explanation",
                "stop_loss": "price level",
                "take_profit": "price level"
            }}
            """
            
            final_decision = await self._get_agent_input("portfolio_manager", decision_prompt)
            
            # Parse and validate decision
            return self._parse_decision(
                final_decision, 
                context.symbol,
                {
                    "market_analysis": market_analysis,
                    "signal": trading_signal,
                    "risk_assessment": risk_assessment
                }
            )
            
        except Exception as e:
            logger.error(f"Error in agent coordination: {e}")
            # Return safe default
            return TradingDecision(
                action="hold",
                symbol=context.symbol,
                confidence=0.0,
                reasoning=f"Error in analysis: {str(e)}",
                agent_consensus={"error": str(e)}
            )
    
    async def _get_agent_input(self, agent_name: str, prompt: str) -> str:
        """Get input from a specific agent."""
        try:
            agent = self.agents[agent_name]
            
            # For development, use mock responses if no API keys
            if not self.settings.ai.openai_api_key:
                return self._get_mock_response(agent_name, prompt)
            
            response = self.client.run(
                agent=agent,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.messages[-1]["content"]
            
        except Exception as e:
            logger.warning(f"Agent {agent_name} failed: {e}")
            return self._get_mock_response(agent_name, prompt)
    
    def _get_mock_response(self, agent_name: str, prompt: str) -> str:
        """Generate mock responses for development/testing."""
        mock_responses = {
            "market_analyst": """
            Market Analysis for {symbol}:
            - Trend: Bullish short-term, consolidating
            - Support: $145.20, Resistance: $152.80
            - Volume: Above average, indicating interest
            - RSI: 58 (neutral territory)
            - Moving averages: Price above 20-day MA
            - Sentiment: Cautiously optimistic
            - Confidence: 72%
            """,
            
            "risk_manager": """
            Risk Assessment:
            - Risk Score: 5/10 (moderate risk)
            - Recommended position: 2-3% of portfolio
            - Stop-loss: 3% below entry
            - Take-profit: 6% above entry
            - Max drawdown risk: 1.5%
            - Portfolio correlation: Low
            - Liquidity: High
            """,
            
            "signal_generator": """
            Trading Signal: BUY
            - Signal strength: 68%
            - Entry target: $150.50
            - Expected move: +4-6% over 5-10 days
            - Technical confluence: MA crossover + volume spike
            - Momentum: Positive
            - Pattern: Cup and handle formation
            """,
            
            "portfolio_manager": json.dumps({
                "action": "buy",
                "confidence": 65,
                "position_size": "2.5%",
                "reasoning": "Moderate bullish setup with good risk/reward",
                "stop_loss": "146.00", 
                "take_profit": "158.00"
            })
        }
        
        template = mock_responses.get(agent_name, "Analysis in progress...")
        if "{symbol}" in template and hasattr(self, '_current_symbol'):
            template = template.format(symbol=self._current_symbol)
        return template
    
    def _format_context(self, context: TradingContext) -> str:
        """Format trading context for agents."""
        self._current_symbol = context.symbol  # Store for mock responses
        
        return f"""
        Symbol: {context.symbol}
        Current Price: ${context.current_price:.2f}
        
        Market Data:
        - Volume: {context.market_data.get('volume', 'N/A')}
        - Price Change: {context.market_data.get('change', 'N/A')}
        - High: {context.market_data.get('high', 'N/A')}
        - Low: {context.market_data.get('low', 'N/A')}
        
        Portfolio State:
        - Available Cash: {context.portfolio_state.get('cash', 'N/A')}
        - Current Position: {context.portfolio_state.get('position', 'None')}
        - Total Value: {context.portfolio_state.get('total_value', 'N/A')}
        
        Risk Metrics:
        - Portfolio Beta: {context.risk_metrics.get('beta', 'N/A')}
        - VaR: {context.risk_metrics.get('var', 'N/A')}
        - Sharpe Ratio: {context.risk_metrics.get('sharpe', 'N/A')}
        
        News Sentiment: {context.news_sentiment.get('score', 'Neutral') if context.news_sentiment else 'N/A'}
        """
    
    def _parse_decision(self, decision_text: str, symbol: str, agent_inputs: Dict[str, str]) -> TradingDecision:
        """Parse agent decision into structured format."""
        try:
            # Try to parse JSON from decision
            if "{" in decision_text:
                json_start = decision_text.find("{")
                json_end = decision_text.rfind("}") + 1
                json_str = decision_text[json_start:json_end]
                decision_data = json.loads(json_str)
                
                return TradingDecision(
                    action=decision_data.get("action", "hold").lower(),
                    symbol=symbol,
                    confidence=float(decision_data.get("confidence", 50)),
                    reasoning=decision_data.get("reasoning", "No reasoning provided"),
                    position_size=self._parse_position_size(decision_data.get("position_size")),
                    stop_loss=self._parse_price(decision_data.get("stop_loss")),
                    take_profit=self._parse_price(decision_data.get("take_profit")),
                    agent_consensus=agent_inputs
                )
        except Exception as e:
            logger.warning(f"Failed to parse decision JSON: {e}")
        
        # Fallback to text parsing
        action = "hold"
        if "buy" in decision_text.lower():
            action = "buy"
        elif "sell" in decision_text.lower():
            action = "sell"
        
        return TradingDecision(
            action=action,
            symbol=symbol,
            confidence=50.0,
            reasoning=decision_text[:200] + "..." if len(decision_text) > 200 else decision_text,
            agent_consensus=agent_inputs
        )
    
    def _parse_position_size(self, size_str: str) -> Optional[float]:
        """Parse position size from string."""
        if not size_str:
            return None
        try:
            # Remove % and convert to decimal
            size_str = str(size_str).replace("%", "").strip()
            return float(size_str) / 100.0
        except:
            return None
    
    def _parse_price(self, price_str: str) -> Optional[float]:
        """Parse price from string."""
        if not price_str:
            return None
        try:
            # Remove $ and convert to float
            price_str = str(price_str).replace("$", "").strip()
            return float(price_str)
        except:
            return None
    
    async def get_agent_status(self) -> Dict[str, bool]:
        """Check status of all agents."""
        status = {}
        for name, agent in self.agents.items():
            try:
                # Simple test to verify agent is working
                test_response = await self._get_agent_input(name, "Status check")
                status[name] = bool(test_response)
            except Exception as e:
                logger.warning(f"Agent {name} status check failed: {e}")
                status[name] = False
        return status


# Global orchestrator instance
_orchestrator: Optional[TradingAgentOrchestrator] = None


async def get_trading_orchestrator() -> TradingAgentOrchestrator:
    """Get or create global trading agent orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TradingAgentOrchestrator()
    return _orchestrator


async def make_trading_decision(context: TradingContext) -> TradingDecision:
    """Make a coordinated trading decision using all agents."""
    orchestrator = await get_trading_orchestrator()
    return await orchestrator.analyze_trading_opportunity(context)


async def check_agents_health() -> Dict[str, bool]:
    """Check health of all trading agents."""
    orchestrator = await get_trading_orchestrator()
    return await orchestrator.get_agent_status()