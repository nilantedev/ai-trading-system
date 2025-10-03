#!/usr/bin/env python3
"""
Production LLM Service - Integrates state-of-the-art language models
for financial analysis and trading decisions.
"""

import os
import asyncio
import yaml
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import aiohttp
import json
from datetime import datetime

# Ollama integration
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# OpenAI integration  
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Anthropic integration
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported model providers."""
    LOCAL = "local"
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"


class AnalysisType(Enum):
    """Types of financial analysis."""
    TECHNICAL = "technical_analysis"
    FUNDAMENTAL = "fundamental_analysis"
    SENTIMENT = "sentiment_analysis"
    RISK = "risk_assessment"
    OPTIONS = "options_pricing"
    NEWS = "news_analysis"
    FORECAST = "time_series_forecast"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    provider: ModelProvider
    model_id: str
    capabilities: List[str]
    requirements: Dict[str, Any]
    quantization: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7


@dataclass
class AnalysisRequest:
    """Request for financial analysis."""
    type: AnalysisType
    data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    urgency: str = "normal"  # "urgent", "normal", "background"
    require_confidence: bool = True


@dataclass
class AnalysisResult:
    """Result from financial analysis."""
    request_type: AnalysisType
    analysis: str
    confidence: float
    model_used: str
    reasoning: Optional[str] = None
    recommendations: List[str] = None
    risks: List[str] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None


class ProductionLLMService:
    """
    Production-ready LLM service with state-of-the-art models
    for financial analysis and trading decisions.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the LLM service."""
        # Try multiple possible config paths
        # Allow environment overrides first
        env_paths = [
            os.getenv("MODEL_CONFIG_PATH"),
            os.getenv("LLM_CONFIG_PATH"),
        ]
        possible_paths = [p for p in env_paths if p] + [
            "/app/models/production-models.yaml",  # path as mounted by docker-compose
            "/app/infrastructure/ai-models/production-models.yaml",
            "/app/services/ml/production-models.yaml",
            "./infrastructure/ai-models/production-models.yaml",
            "infrastructure/ai-models/production-models.yaml",
        ]
        
        if config_path:
            possible_paths.insert(0, config_path)
        
        self.config_path = None
        for path in possible_paths:
            if os.path.exists(path):
                self.config_path = path
                logger.info(f"Found config at: {path}")
                break
        
        if not self.config_path:
            logger.warning(f"No config file found in: {possible_paths}")
            self.config_path = possible_paths[0]  # Use default for error messages
        
        self.models: Dict[str, ModelConfig] = {}
        self.routing_table: Dict[AnalysisType, List[str]] = {}
        self.active_models: Dict[str, Any] = {}
        
        # Load configuration
        self._load_configuration()
        
        # Initialize model clients
        self._initialize_clients()
        
    def _load_configuration(self):
        """Load model configuration from YAML."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Preferred: new production schema (models/model_overrides/routing_rules)
            if 'models' in config or 'routing_rules' in config:
                defaults = config.get('default_settings', {})
                model_overrides = config.get('model_overrides', {})

                def _apply_overrides(mc: ModelConfig) -> ModelConfig:
                    ov = model_overrides.get(mc.model_id) or model_overrides.get(mc.name)
                    if ov:
                        mc.temperature = float(ov.get('temperature', mc.temperature))
                        mc.max_tokens = int(ov.get('max_tokens', mc.max_tokens))
                    # global defaults if not overridden
                    if 'temperature' in defaults and mc.temperature == 0.7:
                        try:
                            mc.temperature = float(defaults['temperature'])
                        except Exception:
                            pass
                    if 'max_tokens' in defaults and mc.max_tokens == 4096:
                        try:
                            mc.max_tokens = int(defaults['max_tokens'])
                        except Exception:
                            pass
                    return mc

                def _ensure_model(name: str, capability_hint: Optional[str] = None):
                    if not name:
                        return
                    if name not in self.models:
                        mc = ModelConfig(
                            name=name,
                            provider=ModelProvider.OLLAMA,
                            model_id=name,
                            capabilities=[capability_hint] if capability_hint else [],
                            requirements={}
                        )
                        self.models[name] = _apply_overrides(mc)

                def _extend_route(atype: AnalysisType, lst):
                    if not lst:
                        return
                    seq = [m for m in lst if m]
                    for m in seq:
                        _ensure_model(m)
                    existing = self.routing_table.get(atype, [])
                    # preserve order, de-dup
                    ordered = []
                    for m in existing + seq:
                        if m and m not in ordered:
                            ordered.append(m)
                    self.routing_table[atype] = ordered

                models_cfg = config.get('models', {})
                # Analysis section
                analysis_cfg = models_cfg.get('analysis', {})
                if analysis_cfg:
                    prim = analysis_cfg.get('primary')
                    fb = analysis_cfg.get('fallback')
                    spec = analysis_cfg.get('specialized', {})
                    # Map specialized keys to AnalysisType
                    spec_map = {
                        'technical': AnalysisType.TECHNICAL,
                        'fundamental': AnalysisType.FUNDAMENTAL,
                        'sentiment': AnalysisType.SENTIMENT,
                        'news': AnalysisType.NEWS,
                        'risk': AnalysisType.RISK,
                        'options': AnalysisType.OPTIONS,
                        'forecast': AnalysisType.FORECAST,
                    }
                    for key, atype in spec_map.items():
                        model_name = spec.get(key)
                        if model_name or prim or fb:
                            _extend_route(atype, [model_name, prim, fb])

                # Trading section: risk/execution/strategy hints
                trading_cfg = models_cfg.get('trading', {})
                if trading_cfg:
                    prim = trading_cfg.get('primary')
                    fb = trading_cfg.get('fallback')
                    spec = trading_cfg.get('specialized', {})
                    _extend_route(AnalysisType.RISK, [spec.get('risk'), prim, fb])
                    # Map strategy -> TECHNICAL as a general analysis fallback for strategy selection prompts
                    _extend_route(AnalysisType.TECHNICAL, [spec.get('strategy'), prim, fb])

                # Research section: market/news/reports hints
                research_cfg = models_cfg.get('research', {})
                if research_cfg:
                    prim = research_cfg.get('primary')
                    fb = research_cfg.get('fallback')
                    spec = research_cfg.get('specialized', {})
                    _extend_route(AnalysisType.NEWS, [spec.get('news'), prim, fb])
                    _extend_route(AnalysisType.FUNDAMENTAL, [spec.get('market'), prim, fb])

                # Optional: routing_rules.task_routing (map task names to AnalysisType when possible)
                rr = config.get('routing_rules', {})
                tr = rr.get('task_routing', {})
                # map common aliases to AnalysisType values
                alias_map = {
                    'market_analysis': AnalysisType.TECHNICAL,
                    'risk_assessment': AnalysisType.RISK,
                    'trade_execution': AnalysisType.TECHNICAL,
                    'strategy_optimization': AnalysisType.TECHNICAL,
                    'sentiment_analysis': AnalysisType.SENTIMENT,
                    'report_generation': AnalysisType.FORECAST,
                    'options_pricing': AnalysisType.OPTIONS,
                    'technical_analysis': AnalysisType.TECHNICAL,
                    'fundamental_analysis': AnalysisType.FUNDAMENTAL,
                    'news_analysis': AnalysisType.NEWS,
                    'time_series_forecast': AnalysisType.FORECAST,
                }
                for task_name, model_list in tr.items():
                    atype = alias_map.get(task_name)
                    if atype:
                        _extend_route(atype, model_list)

                logger.info(f"Loaded {len(self.models)} model configurations (production schema)")

            else:
                # Legacy schema support: llms + selection_strategy
                # Load LLM configurations
                for category in ['primary', 'specialized', 'cloud_backup']:
                    if category in config.get('llms', {}):
                        for model_name, model_config in config['llms'][category].items():
                            self.models[model_name] = ModelConfig(
                                name=model_name,
                                provider=ModelProvider(model_config.get('provider', 'local')),
                                model_id=model_config.get('model', model_name),
                                capabilities=model_config.get('capabilities', []),
                                requirements=model_config.get('requirements', {}),
                                quantization=model_config.get('quantization'),
                                api_key=os.getenv(model_config.get('api_key', '').strip('${}'))
                            )
                # Load routing configuration
                if 'selection_strategy' in config:
                    routing = config['selection_strategy'].get('routing', {})
                    for task, models in routing.items():
                        try:
                            self.routing_table[AnalysisType(task)] = models
                        except ValueError:
                            logger.warning(f"Unknown analysis type in routing: {task}")
                logger.info(f"Loaded {len(self.models)} model configurations (legacy schema)")
            
            # If no models were loaded at all, use fallback
            if not self.models:
                logger.warning("No models loaded from config; falling back to built-in defaults")
                self._use_fallback_config()

        except Exception as e:
            logger.warning(f"Configuration file not found, using fallback: {e}")
            self._use_fallback_config()
    
    def _use_fallback_config(self):
        """Use fallback configuration if main config fails."""
        logger.info("Using production fallback configuration")
        
        # Production models configuration
        if OLLAMA_AVAILABLE:
            # Main analysis models
            self.models['qwen2.5-72b'] = ModelConfig(
                name="qwen2.5-72b",
                provider=ModelProvider.OLLAMA,
                model_id="qwen2.5:72b",
                capabilities=["general", "analysis", "technical"],
                requirements={"ram": "48GB"}
            )
            self.models['mixtral-8x22b'] = ModelConfig(
                name="mixtral-8x22b",
                provider=ModelProvider.OLLAMA,
                model_id="mixtral:8x22b",
                capabilities=["trading", "analysis", "strategy"],
                requirements={"ram": "80GB"}
            )
            self.models['deepseek-v3'] = ModelConfig(
                name="deepseek-v3",
                provider=ModelProvider.OLLAMA,
                model_id="deepseek-v3:latest",
                capabilities=["technical", "strategy", "deep_analysis"],
                requirements={"ram": "256GB"}
            )
            self.models['yi-34b'] = ModelConfig(
                name="yi-34b",
                provider=ModelProvider.OLLAMA,
                model_id="yi:34b",
                capabilities=["fundamental", "research"],
                requirements={"ram": "24GB"}
            )
            self.models['command-r-plus'] = ModelConfig(
                name="command-r-plus",
                provider=ModelProvider.OLLAMA,
                model_id="command-r-plus:104b",
                capabilities=["sentiment", "research", "reports"],
                requirements={"ram": "64GB"}
            )
            self.models['solar-10.7b'] = ModelConfig(
                name="solar-10.7b",
                provider=ModelProvider.OLLAMA,
                model_id="solar:10.7b",
                capabilities=["risk", "news", "fast_analysis"],
                requirements={"ram": "8GB"}
            )
            self.models['phi3-14b'] = ModelConfig(
                name="phi3-14b",
                provider=ModelProvider.OLLAMA,
                model_id="phi3:14b",
                capabilities=["execution", "risk", "quick_decisions"],
                requirements={"ram": "10GB"}
            )
        
        # Set routing based on task type
        self.routing_table[AnalysisType.TECHNICAL] = ["deepseek-v3", "qwen2.5-72b", "mixtral-8x22b"]
        self.routing_table[AnalysisType.FUNDAMENTAL] = ["yi-34b", "qwen2.5-72b", "command-r-plus"]
        self.routing_table[AnalysisType.SENTIMENT] = ["command-r-plus", "solar-10.7b", "yi-34b"]
        self.routing_table[AnalysisType.RISK] = ["solar-10.7b", "phi3-14b", "deepseek-v3"]
        # Note: STRATEGY is not an explicit AnalysisType; route strategy-like prompts via TECHNICAL
        # to preserve behavior without expanding the enum here.
        # self.routing_table[AnalysisType.STRATEGY] = ["deepseek-v3", "mixtral-8x22b", "qwen2.5-72b"]
        
        # Default routing for any unspecified types
        for analysis_type in AnalysisType:
            if analysis_type not in self.routing_table:
                self.routing_table[analysis_type] = ["qwen2.5-72b", "mixtral-8x22b"]
    
    def _initialize_clients(self):
        """Initialize API clients for different providers."""
        self.clients = {}
        
        # Ollama client
        if OLLAMA_AVAILABLE:
            try:
                self.clients['ollama'] = ollama.Client(
                    host=os.getenv('OLLAMA_HOST', 'http://localhost:11434')
                )
                logger.info("Ollama client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama client: {e}")
        
        # OpenAI client
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            try:
                openai.api_key = os.getenv('OPENAI_API_KEY')
                self.clients['openai'] = openai
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        # Anthropic client
        if ANTHROPIC_AVAILABLE and os.getenv('ANTHROPIC_API_KEY'):
            try:
                self.clients['anthropic'] = anthropic.Anthropic(
                    api_key=os.getenv('ANTHROPIC_API_KEY')
                )
                logger.info("Anthropic client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")
    
    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Perform financial analysis using appropriate LLM.
        """
        # Select best model for the task
        model_name = self._select_model(request.type, request.urgency)
        if not model_name:
            raise ValueError(f"No model available for {request.type}")
        
        model_config = self.models[model_name]
        
        # Build prompt based on analysis type
        prompt = self._build_prompt(request)
        
        # Call appropriate model
        try:
            if model_config.provider == ModelProvider.OLLAMA:
                result = await self._call_ollama(model_config, prompt)
            elif model_config.provider == ModelProvider.OPENAI:
                result = await self._call_openai(model_config, prompt)
            elif model_config.provider == ModelProvider.ANTHROPIC:
                result = await self._call_anthropic(model_config, prompt)
            else:
                result = await self._call_local(model_config, prompt)
            
            # Parse and structure the result
            return self._parse_result(result, request, model_name)
            
        except Exception as e:
            logger.error(f"Model {model_name} failed: {e}")
            # Try fallback model
            return await self._fallback_analysis(request, exclude=[model_name])
    
    def _select_model(self, analysis_type: AnalysisType, urgency: str) -> Optional[str]:
        """Select the best available model for the task."""
        # Get candidate models from routing table
        candidates = self.routing_table.get(analysis_type, [])
        
        # Filter by availability
        available = []
        for model_name in candidates:
            if model_name in self.models:
                model = self.models[model_name]
                # Check if we have the necessary client
                if model.provider == ModelProvider.OLLAMA and 'ollama' in self.clients:
                    available.append(model_name)
                elif model.provider == ModelProvider.OPENAI and 'openai' in self.clients:
                    available.append(model_name)
                elif model.provider == ModelProvider.ANTHROPIC and 'anthropic' in self.clients:
                    available.append(model_name)
                elif model.provider == ModelProvider.LOCAL:
                    available.append(model_name)
        
        if not available:
            # Use any available model as fallback
            available = [name for name in self.models.keys() 
                        if self._is_model_available(name)]
        
        # For urgent requests, prefer faster models
        if urgency == "urgent" and available:
            # Prefer smaller, faster models for urgent requests
            for fast_model in ['solar', 'phi3', 'llama3.2']:
                if fast_model in available:
                    return fast_model
        
        # Return first available model
        return available[0] if available else None
    
    def _is_model_available(self, model_name: str) -> bool:
        """Check if a model is available for use."""
        if model_name not in self.models:
            return False
        
        model = self.models[model_name]
        
        if model.provider == ModelProvider.OLLAMA:
            return 'ollama' in self.clients
        elif model.provider == ModelProvider.OPENAI:
            return 'openai' in self.clients and model.api_key
        elif model.provider == ModelProvider.ANTHROPIC:
            return 'anthropic' in self.clients and model.api_key
        elif model.provider == ModelProvider.LOCAL:
            return True  # Assume local models are available
        
        return False
    
    def _build_prompt(self, request: AnalysisRequest) -> str:
        """Build a prompt for the LLM based on the analysis request."""
        base_prompts = {
            AnalysisType.TECHNICAL: """
                Perform technical analysis on the following market data:
                {data}
                
                Analyze:
                1. Key chart patterns and indicators
                2. Support and resistance levels
                3. Momentum and trend direction
                4. Volume analysis
                5. Trading signals and entry/exit points
                
                Provide specific, actionable recommendations with confidence levels.
            """,
            
            AnalysisType.FUNDAMENTAL: """
                Perform fundamental analysis on the following financial data:
                {data}
                
                Evaluate:
                1. Financial health and key ratios
                2. Revenue and earnings trends
                3. Competitive position
                4. Growth prospects
                5. Valuation metrics
                
                Provide investment recommendation with detailed reasoning.
            """,
            
            AnalysisType.SENTIMENT: """
                Analyze market sentiment from the following data:
                {data}
                
                Determine:
                1. Overall sentiment (bullish/bearish/neutral)
                2. Sentiment strength and confidence
                3. Key themes and concerns
                4. Potential sentiment shifts
                5. Contrarian opportunities
                
                Quantify sentiment on a -1 to +1 scale with explanation.
            """,
            
            AnalysisType.RISK: """
                Perform risk assessment on the following portfolio/position:
                {data}
                
                Calculate and analyze:
                1. Value at Risk (VaR) estimates
                2. Maximum drawdown scenarios
                3. Correlation risks
                4. Black swan events
                5. Risk mitigation strategies
                
                Provide specific risk metrics and hedging recommendations.
            """,
            
            AnalysisType.OPTIONS: """
                Analyze options trading opportunities:
                {data}
                
                Evaluate:
                1. Implied volatility analysis
                2. Greeks (Delta, Gamma, Theta, Vega)
                3. Optimal strike selection
                4. Spread strategies
                5. Risk/reward ratios
                
                Recommend specific options strategies with entry/exit points.
            """,
            
            AnalysisType.NEWS: """
                Analyze the impact of the following news:
                {data}
                
                Assess:
                1. Market impact (immediate and long-term)
                2. Affected sectors and stocks
                3. Trading opportunities
                4. Risk factors
                5. Similar historical events
                
                Provide actionable trading recommendations based on the news.
            """,
            
            AnalysisType.FORECAST: """
                Generate time series forecast for:
                {data}
                
                Predict:
                1. Next period price targets
                2. Confidence intervals
                3. Key inflection points
                4. Seasonal patterns
                5. Risk factors
                
                Provide specific price targets with probabilities.
            """
        }
        
        # Get base prompt
        prompt = base_prompts.get(
            request.type,
            "Analyze the following financial data: {data}"
        )
        
        # Format with data
        prompt = prompt.format(data=json.dumps(request.data, indent=2))
        
        # Add context if provided
        if request.context:
            prompt += f"\n\nAdditional context:\n{json.dumps(request.context, indent=2)}"
        
        # Add confidence requirement
        if request.require_confidence:
            prompt += "\n\nProvide confidence scores (0-100%) for all recommendations."
        
        return prompt
    
    async def _call_ollama(self, model: ModelConfig, prompt: str) -> str:
        """Call Ollama model."""
        if 'ollama' not in self.clients:
            raise ValueError("Ollama client not available")
        
        try:
            response = self.clients['ollama'].chat(
                model=model.model_id,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are an expert financial analyst and trader with deep knowledge of markets, technical analysis, and risk management.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': model.temperature,
                    'max_tokens': model.max_tokens,
                }
            )
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise
    
    async def _call_openai(self, model: ModelConfig, prompt: str) -> str:
        """Call OpenAI model."""
        if 'openai' not in self.clients:
            raise ValueError("OpenAI client not available")
        
        try:
            response = await self.clients['openai'].ChatCompletion.acreate(
                model=model.model_id,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial analyst and trader with deep knowledge of markets, technical analysis, and risk management."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=model.temperature,
                max_tokens=model.max_tokens,
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            raise
    
    async def _call_anthropic(self, model: ModelConfig, prompt: str) -> str:
        """Call Anthropic model."""
        if 'anthropic' not in self.clients:
            raise ValueError("Anthropic client not available")
        
        try:
            message = self.clients['anthropic'].messages.create(
                model=model.model_id,
                max_tokens=model.max_tokens,
                temperature=model.temperature,
                system="You are an expert financial analyst and trader with deep knowledge of markets, technical analysis, and risk management.",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic call failed: {e}")
            raise
    
    async def _call_local(self, model: ModelConfig, prompt: str) -> str:
        """Call local model (placeholder for custom implementation)."""
        # This would integrate with vLLM, TGI, or custom serving
        logger.warning(f"Local model {model.name} not implemented, using mock response")
        return f"Mock analysis for {model.name}: Analysis would go here"
    
    def _parse_result(self, raw_result: str, request: AnalysisRequest, model_name: str) -> AnalysisResult:
        """Parse the model output into structured result."""
        # Extract confidence if present
        confidence = 0.75  # Default confidence
        import re
        confidence_match = re.search(r'confidence:?\s*(\d+)%?', raw_result, re.IGNORECASE)
        if confidence_match:
            confidence = float(confidence_match.group(1)) / 100
        
        # Extract recommendations (lines starting with - or *)
        recommendations = []
        for line in raw_result.split('\n'):
            if line.strip().startswith(('-', '*', '•')):
                recommendations.append(line.strip()[1:].strip())
        
        # Extract risk mentions
        risks = []
        risk_section = re.search(r'risks?:?(.*?)(?:\n\n|$)', raw_result, re.IGNORECASE | re.DOTALL)
        if risk_section:
            risk_text = risk_section.group(1)
            for line in risk_text.split('\n'):
                if line.strip():
                    risks.append(line.strip())
        
        return AnalysisResult(
            request_type=request.type,
            analysis=raw_result,
            confidence=confidence,
            model_used=model_name,
            recommendations=recommendations[:5],  # Top 5 recommendations
            risks=risks[:3],  # Top 3 risks
            metadata={
                'urgency': request.urgency,
                'model_provider': self.models[model_name].provider.value,
                'tokens_used': len(raw_result.split())  # Approximate
            },
            timestamp=datetime.now()
        )
    
    async def _fallback_analysis(self, request: AnalysisRequest, exclude: List[str]) -> AnalysisResult:
        """Fallback analysis when primary model fails."""
        # Find alternative model
        for model_name in self.models.keys():
            if model_name not in exclude and self._is_model_available(model_name):
                try:
                    logger.info(f"Attempting fallback with {model_name}")
                    model_config = self.models[model_name]
                    prompt = self._build_prompt(request)
                    
                    if model_config.provider == ModelProvider.OLLAMA:
                        result = await self._call_ollama(model_config, prompt)
                    else:
                        result = f"Fallback analysis using {model_name}"
                    
                    return self._parse_result(result, request, model_name)
                    
                except Exception as e:
                    logger.error(f"Fallback model {model_name} also failed: {e}")
                    continue
        
        # If all models fail, return error result
        return AnalysisResult(
            request_type=request.type,
            analysis="Analysis failed - all models unavailable",
            confidence=0.0,
            model_used="none",
            metadata={'error': 'All models failed'},
            timestamp=datetime.now()
        )
    
    async def batch_analyze(self, requests: List[AnalysisRequest]) -> List[AnalysisResult]:
        """Perform batch analysis for multiple requests."""
        tasks = [self.analyze(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all configured models."""
        status = {
            'total_models': len(self.models),
            'available_models': sum(1 for m in self.models if self._is_model_available(m)),
            'models': {}
        }
        
        for model_name, model_config in self.models.items():
            status['models'][model_name] = {
                'provider': model_config.provider.value,
                'available': self._is_model_available(model_name),
                'capabilities': model_config.capabilities,
                'requirements': model_config.requirements
            }
        
        return status


# Example usage
async def main():
    """Example usage of the Production LLM Service."""
    service = ProductionLLMService()
    
    # Example technical analysis request
    request = AnalysisRequest(
        type=AnalysisType.TECHNICAL,
        data={
            'symbol': 'AAPL',
            'price': 195.89,
            'volume': 58932120,
            'rsi': 68,
            'macd': {'macd': 2.15, 'signal': 1.98},
            'sma_50': 188.45,
            'sma_200': 175.30
        },
        urgency='normal'
    )
    
    result = await service.analyze(request)
    print(f"Analysis Result:")
    print(f"Model Used: {result.model_used}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Recommendations: {result.recommendations}")
    print(f"Risks: {result.risks}")


if __name__ == "__main__":
    asyncio.run(main())