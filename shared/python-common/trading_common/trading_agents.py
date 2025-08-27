"""Local AI agent orchestration for trading decisions using only free, local models.

This replaces OpenAI Swarm with a local implementation that maintains
the same sophisticated multi-agent coordination without any API costs.
"""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from .local_swarm import LocalSwarm, LocalAgent, create_trading_agents
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
    """Coordinates multiple trading agents using local models only."""
    
    def __init__(self):
        self.settings = get_settings()
        self.swarm = LocalSwarm()
        self.agents = create_trading_agents()
        
        # Add all agents to swarm
        for agent in self.agents.values():
            self.swarm.add_agent(agent)
            
        logger.info("Trading Agent Orchestrator initialized with local models only")
        self._log_model_configuration()
    
    def _log_model_configuration(self):
        """Log the model configuration for transparency."""
        logger.info("=== Local Model Configuration ===")
        logger.info("All models run locally via Ollama - NO API COSTS")
        logger.info("Models selected for optimal performance:")
        logger.info("- Market Analysis: qwen2.5:72b (or mixtral:8x7b fallback)")
        logger.info("- Risk Assessment: deepseek-r1:70b (or llama3.1:70b fallback)")
        logger.info("- Strategy: llama3.1:70b (or mistral:7b fallback)")
        logger.info("- Sentiment: phi3:medium (fast, efficient)")
        logger.info("- Execution: mixtral:8x7b (fast decisions)")
        logger.info("================================")
    
    async def make_trading_decision(self, context: TradingContext) -> TradingDecision:
        """Coordinate agents to make a trading decision."""
        
        # Prepare context for agents
        agent_context = {
            "symbol": context.symbol,
            "current_price": context.current_price,
            "market_data": context.market_data,
            "portfolio": context.portfolio_state,
            "risk_metrics": context.risk_metrics,
            "news_sentiment": context.news_sentiment
        }
        
        # Prepare messages for agents
        messages = [
            {
                "role": "user",
                "content": f"""Analyze the following trading opportunity:
                
Symbol: {context.symbol}
Current Price: ${context.current_price}
Market Data: {json.dumps(context.market_data, indent=2)}
Portfolio State: {json.dumps(context.portfolio_state, indent=2)}
Risk Metrics: {json.dumps(context.risk_metrics, indent=2)}
News Sentiment: {json.dumps(context.news_sentiment, indent=2) if context.news_sentiment else 'N/A'}

Provide your analysis and recommendation (BUY/SELL/HOLD) with confidence percentage."""
            }
        ]
        
        # Run all agents in parallel for speed
        result = await self.swarm.run_multiple(
            list(self.agents.values()),
            messages,
            agent_context,
            aggregation="consensus"
        )
        
        # Extract decision from consensus
        decision = self._parse_decision(result, context)
        
        return decision
    
    def _parse_decision(self, result: Dict[str, Any], context: TradingContext) -> TradingDecision:
        """Parse agent responses into a trading decision."""
        
        consensus = result.get("consensus", "hold")
        confidence = result.get("confidence", 0.5)
        agent_decisions = result.get("agent_decisions", {})
        
        # Build reasoning from agent responses
        reasoning_parts = []
        for agent_name, response_data in result.get("weighted_responses", {}).items():
            if isinstance(response_data, dict):
                reasoning_parts.append(f"{agent_name}: {response_data.get('response', '')[:200]}...")
        
        reasoning = "\n".join(reasoning_parts) if reasoning_parts else "Based on multi-agent analysis"
        
        # Calculate position size based on confidence and risk
        position_size = self._calculate_position_size(confidence, context)
        
        # Set stop loss and take profit
        stop_loss, take_profit = self._calculate_risk_levels(consensus, context)
        
        return TradingDecision(
            action=consensus,
            symbol=context.symbol,
            confidence=confidence,
            reasoning=reasoning,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            agent_consensus=agent_decisions
        )
    
    def _calculate_position_size(self, confidence: float, context: TradingContext) -> float:
        """Calculate position size based on confidence and risk management."""
        
        # Kelly Criterion-inspired sizing
        portfolio_value = context.portfolio_state.get("total_value", 100000)
        max_risk_per_trade = 0.02  # 2% max risk
        
        # Adjust by confidence
        risk_adjusted_size = max_risk_per_trade * confidence
        
        # Apply portfolio constraints
        max_position_size = portfolio_value * risk_adjusted_size
        
        # Round to reasonable trading size
        shares = int(max_position_size / context.current_price)
        
        return max(1, shares)  # At least 1 share
    
    def _calculate_risk_levels(
        self, 
        action: str, 
        context: TradingContext
    ) -> tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels."""
        
        if action == "hold":
            return None, None
        
        current_price = context.current_price
        atr = context.market_data.get("atr", current_price * 0.02)  # Default 2% if no ATR
        
        if action == "buy":
            stop_loss = current_price - (2 * atr)  # 2 ATR stop
            take_profit = current_price + (3 * atr)  # 3 ATR target
        else:  # sell
            stop_loss = current_price + (2 * atr)
            take_profit = current_price - (3 * atr)
        
        return round(stop_loss, 2), round(take_profit, 2)
    
    async def analyze_market_regime(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze overall market regime using agent consensus."""
        
        messages = [
            {
                "role": "user",
                "content": f"""Analyze the current market regime for these symbols: {', '.join(symbols)}
                
Determine:
1. Market trend (bullish/bearish/neutral)
2. Volatility regime (low/medium/high)
3. Risk appetite (risk-on/risk-off)
4. Recommended portfolio stance
5. Key risks to monitor

Provide confidence scores for each assessment."""
            }
        ]
        
        # Use market analyst and risk manager for regime analysis
        agents = [self.agents["market_analyst"], self.agents["risk_manager"]]
        
        result = await self.swarm.run_multiple(
            agents,
            messages,
            context_variables={"symbols": symbols},
            aggregation="weighted"
        )
        
        return {
            "regime_analysis": result,
            "timestamp": datetime.now().isoformat(),
            "symbols_analyzed": symbols
        }
    
    async def generate_portfolio_recommendations(
        self, 
        portfolio: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate portfolio optimization recommendations."""
        
        messages = [
            {
                "role": "user", 
                "content": f"""Given the current portfolio and market conditions, provide recommendations:

Portfolio: {json.dumps(portfolio, indent=2)}
Market Conditions: {json.dumps(market_conditions, indent=2)}

Recommend:
1. Rebalancing actions
2. New positions to consider
3. Positions to exit
4. Risk adjustments
5. Hedging strategies

Provide specific, actionable recommendations with confidence scores."""
            }
        ]
        
        # Use strategy and risk agents for portfolio recommendations
        agents = [
            self.agents["strategy_generator"],
            self.agents["risk_manager"],
            self.agents["execution_planner"]
        ]
        
        result = await self.swarm.run_multiple(
            agents,
            messages,
            context_variables={
                "portfolio": portfolio,
                "market_conditions": market_conditions
            },
            aggregation="consensus"
        )
        
        return {
            "recommendations": result,
            "generated_at": datetime.now().isoformat()
        }
    
    async def process_news_impact(
        self,
        news_items: List[Dict[str, Any]],
        positions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze news impact on current positions."""
        
        messages = [
            {
                "role": "user",
                "content": f"""Analyze the impact of recent news on our positions:

News Items: {json.dumps(news_items[:5], indent=2)}  # First 5 items
Current Positions: {json.dumps(positions, indent=2)}

Assess:
1. Sentiment impact on each position
2. Urgency of action required (1-10)
3. Recommended actions
4. Risk adjustments needed
5. Opportunities created

Be specific about which news affects which positions."""
            }
        ]
        
        # Use sentiment analyst primarily
        result = await self.swarm.run(
            self.agents["sentiment_analyst"],
            messages,
            context_variables={
                "news": news_items,
                "positions": positions
            }
        )
        
        return {
            "impact_analysis": result.content,
            "confidence": result.confidence,
            "analyzed_at": datetime.now().isoformat()
        }
    
    async def validate_trade_idea(
        self,
        trade_idea: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a trade idea through multi-agent review."""
        
        messages = [
            {
                "role": "user",
                "content": f"""Validate this trade idea:

{json.dumps(trade_idea, indent=2)}

Evaluate:
1. Technical merit (1-10)
2. Risk/reward ratio
3. Market timing appropriateness
4. Portfolio fit
5. Execution feasibility

Provide GO/NO-GO decision with detailed reasoning."""
            }
        ]
        
        # Run all agents for comprehensive validation
        result = await self.swarm.run_multiple(
            list(self.agents.values()),
            messages,
            context_variables=trade_idea,
            aggregation="consensus"
        )
        
        # Extract validation decision
        go_no_go = "GO" if result.get("confidence", 0) > 0.6 else "NO-GO"
        
        return {
            "decision": go_no_go,
            "confidence": result.get("confidence", 0),
            "agent_feedback": result.get("agent_decisions", {}),
            "detailed_analysis": result,
            "validated_at": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all agents and models."""
        
        health_status = {}
        
        # Check each agent's model availability
        for agent_name, agent in self.agents.items():
            try:
                # Quick test prompt
                response = await agent.execute("Return 'OK' if you're working", {})
                health_status[agent_name] = {
                    "status": "healthy" if "OK" in response else "degraded",
                    "model": agent.model,
                    "response_sample": response[:100]
                }
            except Exception as e:
                health_status[agent_name] = {
                    "status": "unhealthy",
                    "model": agent.model,
                    "error": str(e)
                }
        
        # Overall health
        unhealthy_count = sum(1 for s in health_status.values() if s["status"] == "unhealthy")
        overall_health = "healthy" if unhealthy_count == 0 else "degraded" if unhealthy_count < 3 else "unhealthy"
        
        return {
            "overall_health": overall_health,
            "agents": health_status,
            "checked_at": datetime.now().isoformat(),
            "using_local_models": True,
            "api_costs": "$0.00 - All models run locally"
        }


# Global orchestrator instance
_orchestrator: Optional[TradingAgentOrchestrator] = None


async def get_orchestrator() -> TradingAgentOrchestrator:
    """Get or create global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TradingAgentOrchestrator()
    return _orchestrator


async def make_trading_decision(context: TradingContext) -> TradingDecision:
    """Make a trading decision using agent orchestration."""
    orchestrator = await get_orchestrator()
    return await orchestrator.make_trading_decision(context)