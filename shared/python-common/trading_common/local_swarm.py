"""Local Swarm implementation using Ollama models instead of OpenAI API.

This provides Swarm-like agent coordination using only local models,
maintaining the same orchestration patterns without API costs.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)


@dataclass
class LocalAgent:
    """Local agent that uses Ollama models."""
    name: str
    instructions: str
    model: str = "mixtral:8x22b"  # Default fast model
    functions: List[Callable] = field(default_factory=list)
    
    async def execute(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Execute agent with given prompt and context."""
        # Combine instructions with prompt
        full_prompt = f"""You are {self.name}.

{self.instructions}

Context:
{json.dumps(context, indent=2) if context else 'No additional context'}

Task: {prompt}

Response:"""
        
        # Call Ollama
        client = httpx.AsyncClient(timeout=60.0)
        try:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 1000
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            logger.error(f"Ollama execution failed for {self.name}: {e}")
            return f"Error: {str(e)}"
        finally:
            await client.aclose()


@dataclass
class AgentResponse:
    """Response from agent execution."""
    agent: str
    content: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LocalSwarm:
    """Local implementation of Swarm-like coordination using Ollama."""
    
    def __init__(self):
        self.agents: Dict[str, LocalAgent] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        
    def add_agent(self, agent: LocalAgent):
        """Add an agent to the swarm."""
        self.agents[agent.name] = agent
        
    async def run(
        self,
        agent: LocalAgent,
        messages: List[Dict[str, str]],
        context_variables: Dict[str, Any] = None,
        max_turns: int = 10
    ) -> AgentResponse:
        """Run agent with messages and context."""
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Execute agent
        response = await agent.execute(prompt, context_variables)
        
        # Record in history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent.name,
            "prompt": prompt,
            "response": response,
            "context": context_variables
        })
        
        return AgentResponse(
            agent=agent.name,
            content=response,
            confidence=self._extract_confidence(response),
            metadata={"model": agent.model, "timestamp": datetime.now().isoformat()}
        )
    
    async def run_multiple(
        self,
        agents: List[LocalAgent],
        messages: List[Dict[str, str]],
        context_variables: Dict[str, Any] = None,
        aggregation: str = "consensus"
    ) -> Dict[str, Any]:
        """Run multiple agents and aggregate responses."""
        tasks = []
        for agent in agents:
            tasks.append(self.run(agent, messages, context_variables))
        
        responses = await asyncio.gather(*tasks)
        
        if aggregation == "consensus":
            return self._aggregate_consensus(responses)
        elif aggregation == "weighted":
            return self._aggregate_weighted(responses)
        else:
            return {"responses": [r.__dict__ for r in responses]}
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert message list to prompt string."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role.capitalize()}: {content}")
        return "\n".join(prompt_parts)
    
    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response if present."""
        # Look for confidence patterns in response
        import re
        patterns = [
            r"confidence[:\s]+(\d+(?:\.\d+)?)",
            r"confidence.*?(\d+)%",
            r"(\d+)%\s+confident"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                try:
                    conf = float(match.group(1))
                    return conf / 100 if conf > 1 else conf
                except ValueError:
                    pass
        
        return 0.5  # Default confidence
    
    def _aggregate_consensus(self, responses: List[AgentResponse]) -> Dict[str, Any]:
        """Aggregate responses by consensus."""
        # Extract key decisions from responses
        decisions = {}
        for resp in responses:
            # Simple keyword extraction for trading decisions
            content_lower = resp.content.lower()
            if "buy" in content_lower:
                decisions[resp.agent] = "buy"
            elif "sell" in content_lower:
                decisions[resp.agent] = "sell"
            else:
                decisions[resp.agent] = "hold"
        
        # Find consensus
        from collections import Counter
        decision_counts = Counter(decisions.values())
        consensus = decision_counts.most_common(1)[0][0] if decision_counts else "hold"
        
        return {
            "consensus": consensus,
            "confidence": sum(r.confidence for r in responses) / len(responses),
            "agent_decisions": decisions,
            "full_responses": [r.__dict__ for r in responses]
        }
    
    def _aggregate_weighted(self, responses: List[AgentResponse]) -> Dict[str, Any]:
        """Aggregate responses with confidence weighting."""
        weighted_sum = {}
        total_confidence = sum(r.confidence for r in responses)
        
        for resp in responses:
            weight = resp.confidence / total_confidence if total_confidence > 0 else 1.0 / len(responses)
            weighted_sum[resp.agent] = {
                "response": resp.content,
                "weight": weight,
                "confidence": resp.confidence
            }
        
        return {
            "weighted_responses": weighted_sum,
            "average_confidence": total_confidence / len(responses),
            "full_responses": [r.__dict__ for r in responses]
        }


# Best open-source models for specific trading tasks (aligned with installed set)
# Installed models: phi3:14b, solar:10.7b, yi:34b, command-r-plus:104b, deepseek-v3:latest, mixtral:8x22b, qwen2.5:72b
RECOMMENDED_MODELS = {
    "market_analysis": {
        "model": "qwen2.5:72b",  # Strong analytical reasoning
        "fallback": "mixtral:8x22b"
    },
    "risk_assessment": {
        "model": "deepseek-v3:latest",  # Strong reasoning capabilities
        "fallback": "qwen2.5:72b"
    },
    "strategy_generation": {
        "model": "qwen2.5:72b",  # Creative and strategic
        "fallback": "phi3:14b"
    },
    "sentiment_analysis": {
        "model": "phi3:14b",  # Fast and efficient
        "fallback": "solar:10.7b"
    },
    "news_summarization": {
        "model": "phi3:14b",  # Good at summarization & concise output
        "fallback": "mixtral:8x22b"
    },
    "portfolio_optimization": {
        "model": "deepseek-v3:latest",  # Reasoning & math
        "fallback": "yi:34b"
    },
    "execution_planning": {
        "model": "mixtral:8x22b",  # Fast decision making
        "fallback": "phi3:14b"
    }
}


def create_trading_agents() -> Dict[str, LocalAgent]:
    """Create specialized trading agents using local models."""
    
    agents = {}
    
    # Market Analysis Agent
    agents["market_analyst"] = LocalAgent(
        name="Market Analyst",
        model=RECOMMENDED_MODELS["market_analysis"]["model"],
        instructions="""You are a senior market analyst specializing in technical and fundamental analysis.
        
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

Be concise but thorough. Focus on actionable insights."""
    )
    
    # Risk Manager Agent
    agents["risk_manager"] = LocalAgent(
        name="Risk Manager",
        model=RECOMMENDED_MODELS["risk_assessment"]["model"],
        instructions="""You are a conservative risk management specialist focused on capital preservation.

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

Be conservative and prioritize capital preservation."""
    )
    
    # Strategy Generator Agent
    agents["strategy_generator"] = LocalAgent(
        name="Strategy Generator",
        model=RECOMMENDED_MODELS["strategy_generation"]["model"],
        instructions="""You are a quantitative strategist developing trading strategies.

Your responsibilities:
- Design trading strategies based on market conditions
- Optimize entry and exit points
- Develop hedging strategies
- Create portfolio allocation recommendations
- Backtest strategy performance

Always provide:
1. Clear strategy description
2. Entry/exit criteria
3. Expected return and risk metrics
4. Market conditions where strategy works best
5. Confidence in strategy success

Focus on systematic, rule-based approaches."""
    )
    
    # Sentiment Analyst Agent
    agents["sentiment_analyst"] = LocalAgent(
        name="Sentiment Analyst",
        model=RECOMMENDED_MODELS["sentiment_analysis"]["model"],
        instructions="""You are a sentiment analysis specialist monitoring market psychology.

Your responsibilities:
- Analyze news sentiment and social media trends
- Assess market fear and greed levels
- Identify sentiment divergences
- Monitor institutional positioning
- Track retail investor behavior

Always provide:
1. Overall market sentiment score (-100 to +100)
2. Key sentiment drivers
3. Sentiment trend direction
4. Contrarian opportunities
5. Confidence in sentiment assessment

Be objective and data-driven."""
    )
    
    # Execution Planner Agent
    agents["execution_planner"] = LocalAgent(
        name="Execution Planner",
        model=RECOMMENDED_MODELS["execution_planning"]["model"],
        instructions="""You are an execution specialist optimizing trade implementation.

Your responsibilities:
- Plan optimal trade execution timing
- Minimize market impact and slippage
- Choose appropriate order types
- Monitor liquidity conditions
- Coordinate multi-leg strategies

Always provide:
1. Recommended execution timing
2. Order type and size recommendations
3. Expected slippage and costs
4. Liquidity assessment
5. Execution confidence score

Focus on minimizing costs and market impact."""
    )
    
    return agents


async def test_local_swarm():
    """Test the local swarm implementation."""
    swarm = LocalSwarm()
    agents = create_trading_agents()
    
    # Add agents to swarm
    for agent in agents.values():
        swarm.add_agent(agent)
    
    # Test with a trading scenario
    messages = [
        {"role": "user", "content": "Should I buy AAPL at current price of $195?"}
    ]
    
    context = {
        "symbol": "AAPL",
        "current_price": 195.0,
        "volume": 50000000,
        "rsi": 65,
        "macd": "bullish"
    }
    
    # Run multiple agents
    result = await swarm.run_multiple(
        list(agents.values()),
        messages,
        context,
        aggregation="consensus"
    )
    
    return result