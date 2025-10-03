#!/usr/bin/env python3
"""
Model Router - Intelligent routing for Ollama models based on task type and urgency
"""

import asyncio
import time
import os
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import json
import httpx
from datetime import datetime, timedelta
import random
from zoneinfo import ZoneInfo
from prometheus_client import Gauge

def _parse_hhmm(s: str) -> tuple[int, int]:
    try:
        hh, mm = s.strip().split(":", 1)
        return int(hh), int(mm)
    except Exception:
        return 0, 0

class TaskUrgency(Enum):
    REALTIME = 1  # <100ms
    FAST = 2      # <1s
    NORMAL = 3    # <5s
    BATCH = 4     # <30s
    DEEP = 5      # <2min

class TaskType(Enum):
    SIGNAL_GENERATION = "signal"
    RISK_ASSESSMENT = "risk"
    MARKET_ANALYSIS = "market"
    DOCUMENT_ANALYSIS = "document"
    FINANCIAL_METRICS = "metrics"
    OPTIONS_PRICING = "options"
    STRATEGY_SELECTION = "strategy"
    SENTIMENT_ANALYSIS = "sentiment"
    TECHNICAL_ANALYSIS = "technical"
    NEWS_PROCESSING = "news"

@dataclass
class ModelInfo:
    name: str
    memory_gb: int
    specialization: List[TaskType]
    avg_latency_ms: int
    max_context: int
    is_available: bool = True
    current_load: float = 0.0

@dataclass
class RoutingDecision:
    model: str
    estimated_latency_ms: int
    confidence: float
    fallback_model: Optional[str] = None

class ModelRouter:
    def __init__(self, ollama_host: str = None):
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.models = self._initialize_models()
        self.usage_stats = {}
        self.cache_ttl = timedelta(minutes=5)
        self.last_health_check = None
        # Market hours gating to keep CPU token rate high during trading
        self.market_tz = os.getenv("MARKET_HOURS_TZ", os.getenv("TRADING_HOURS_TZ", "America/New_York"))
        self.market_open = os.getenv("MARKET_HOURS_OPEN", os.getenv("TRADING_HOURS_OPEN", "09:30"))
        self.market_close = os.getenv("MARKET_HOURS_CLOSE", os.getenv("TRADING_HOURS_CLOSE", "16:00"))
        self.disable_heavy_during_market = os.getenv("ROUTER_MARKET_DISABLE_HEAVY", "true").lower() in ("1","true","yes","on")
        # Models considered "heavy" (large CPU footprint) filtered during market hours for FAST/NORMAL tasks
        try:
            self.heavy_memory_threshold_gb = int(os.getenv("ROUTER_HEAVY_MEMORY_GB", "40"))
        except Exception:
            self.heavy_memory_threshold_gb = 40
        # Optional explicit heavy list override
        raw = os.getenv("ROUTER_HEAVY_MODELS", "")
        self.heavy_models_override = [m.strip() for m in raw.split(',') if m.strip()] if raw else []
        # CPU-only environment hints: larger models incur high latency but can be used for DEEP tasks.
        # These weights are used in scoring to bias selection by urgency and task type.
        self.bias = {
            "llama_pref_deep": 2.5,
            "llama_penalize_fast": 2.5,
            "strategy_mixtral_bonus": 2.0,
            "doc_qwen_bonus": 1.5,
        }
        # Prometheus gauges for token throughput
        try:
            self.g_tokens_sec = Gauge('router_model_tokens_per_sec', 'Observed tokens/sec per model from last call', ['model'])
            self.g_latency_ms = Gauge('router_model_last_latency_ms', 'Last latency (ms) per model', ['model'])
        except Exception:
            self.g_tokens_sec = None
            self.g_latency_ms = None
        
    def _initialize_models(self) -> Dict[str, ModelInfo]:
        """Initialize model configurations based on production setup"""
        return {
            # Strong generalist with large context; good fallback for most tasks
            "qwen2.5:72b": ModelInfo(
                name="qwen2.5:72b",
                memory_gb=47,
                specialization=[
                    TaskType.MARKET_ANALYSIS,
                    TaskType.TECHNICAL_ANALYSIS,
                    TaskType.SIGNAL_GENERATION
                ],
                avg_latency_ms=2000,
                max_context=32768
            ),
            # MoE model with strong reasoning; great for strategy/risk planning
            "mixtral:8x22b": ModelInfo(
                name="mixtral:8x22b",
                memory_gb=79,
                specialization=[
                    TaskType.STRATEGY_SELECTION,
                    TaskType.RISK_ASSESSMENT,
                    TaskType.OPTIONS_PRICING
                ],
                avg_latency_ms=3000,
                max_context=65536
            ),
            # Llama 3.1 70B - preferred deep analysis model on CPU for long-form analytics
            "llama3.1:70b": ModelInfo(
                name="llama3.1:70b",
                memory_gb=42,
                specialization=[
                    TaskType.MARKET_ANALYSIS,
                    TaskType.DOCUMENT_ANALYSIS,
                    TaskType.NEWS_PROCESSING,
                    TaskType.STRATEGY_SELECTION
                ],
                avg_latency_ms=5000,
                max_context=131072
            ),
            # Strong document and tool-use model; excellent for news/doc processing and summarization
            "command-r-plus:104b": ModelInfo(
                name="command-r-plus:104b",
                memory_gb=59,
                specialization=[
                    TaskType.DOCUMENT_ANALYSIS,
                    TaskType.NEWS_PROCESSING,
                    TaskType.SENTIMENT_ANALYSIS
                ],
                avg_latency_ms=2500,
                max_context=128000
            ),
            # Lightweight, fast model for quick metrics and signal sketches
            "solar:10.7b": ModelInfo(
                name="solar:10.7b",
                memory_gb=6,
                specialization=[
                    TaskType.FINANCIAL_METRICS,
                    TaskType.SIGNAL_GENERATION
                ],
                avg_latency_ms=500,
                max_context=4096
            ),
            # Balanced model; good generalist with decent context; strong for market narratives
            "yi:34b": ModelInfo(
                name="yi:34b",
                memory_gb=19,
                specialization=[
                    TaskType.DOCUMENT_ANALYSIS,
                    TaskType.MARKET_ANALYSIS
                ],
                avg_latency_ms=1500,
                max_context=200000
            ),
            # Very fast for brief outputs; use for real-time style prompts
            "phi3:14b": ModelInfo(
                name="phi3:14b",
                memory_gb=8,
                specialization=[
                    TaskType.SIGNAL_GENERATION,
                    TaskType.RISK_ASSESSMENT
                ],
                avg_latency_ms=300,
                max_context=4096
            )
        }
    
    async def check_model_availability(self) -> Dict[str, bool]:
        """Check which models are currently available"""
        availability = {}
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                hosts = []
                cur = self.ollama_host
                if cur:
                    hosts.append(cur)
                # Fallback candidates inside Docker network
                for h in ("http://ollama:11434", "http://trading-ollama:11434", "http://localhost:11434"):
                    if h not in hosts:
                        hosts.append(h)
                last_err = None
                for h in hosts:
                    try:
                        r = await client.get(f"{h}/api/tags")
                        if r.status_code == 200:
                            data = r.json() or {}
                            model_names = {m.get("name") for m in data.get("models", []) if isinstance(m, dict)}
                            for model_key in self.models.keys():
                                is_available = model_key in model_names
                                self.models[model_key].is_available = is_available
                                availability[model_key] = is_available
                            # Update selected host to the working one
                            self.ollama_host = h
                            break
                        else:
                            last_err = f"HTTP {r.status_code}"
                    except Exception as e:  # noqa: BLE001
                        last_err = str(e)
                        continue
                # If none succeeded, mark all as unavailable
                if not availability:
                    for model_key in self.models.keys():
                        availability[model_key] = False
                        self.models[model_key].is_available = False
        except Exception as e:
            print(f"Error checking model availability: {e}")
            for model_key in self.models:
                availability[model_key] = False
                self.models[model_key].is_available = False
        
        self.last_health_check = datetime.utcnow()
        return availability
    
    def estimate_latency(self, model: ModelInfo, prompt_length: int) -> int:
        """Estimate latency based on model and prompt length"""
        base_latency = model.avg_latency_ms
        
        # Adjust for prompt length
        if prompt_length > 10000:
            base_latency *= 1.5
        elif prompt_length > 50000:
            base_latency *= 2.0
        
        # Adjust for current load
        if model.current_load > 0.8:
            base_latency *= 2.0
        elif model.current_load > 0.5:
            base_latency *= 1.3
        
        return int(base_latency)
    
    def route_request(
        self,
        task_type: TaskType,
        urgency: TaskUrgency,
        prompt_length: int = 1000,
        require_long_context: bool = False
    ) -> RoutingDecision:
        """Route request to appropriate model based on task and urgency"""
        
        # Filter available models
        available_models = [
            model for model in self.models.values() 
            if model.is_available
        ]
        
        if not available_models:
            return RoutingDecision(
                model="qwen2.5:72b",  # Default fallback
                estimated_latency_ms=5000,
                confidence=0.1
            )
        
        # Filter by context requirements
        if require_long_context:
            available_models = [
                m for m in available_models 
                if m.max_context >= 32768
            ]

        # Optional market-hours gating to prioritize smaller, faster models during trading
        if self.disable_heavy_during_market and urgency in (TaskUrgency.REALTIME, TaskUrgency.FAST, TaskUrgency.NORMAL):
            try:
                tz = ZoneInfo(self.market_tz)
                now_local = datetime.now(tz)
                if now_local.weekday() < 5:  # Mon-Fri
                    oh, om = _parse_hhmm(self.market_open)
                    ch, cm = _parse_hhmm(self.market_close)
                    open_t = now_local.replace(hour=oh, minute=om, second=0, microsecond=0)
                    close_t = now_local.replace(hour=ch, minute=cm, second=0, microsecond=0)
                    if open_t <= now_local <= close_t:
                        heavy_set = set(self.heavy_models_override) if self.heavy_models_override else None
                        def _is_heavy(m: ModelInfo) -> bool:
                            if heavy_set is not None:
                                return m.name in heavy_set
                            return m.memory_gb >= self.heavy_memory_threshold_gb
                        filtered = [m for m in available_models if not _is_heavy(m)]
                        # Only apply if it leaves at least one model; otherwise keep originals
                        if filtered:
                            available_models = filtered
            except Exception:
                # On any error, skip gating
                pass
        
        # Score models based on criteria
        scored_models = []
        for model in available_models:
            score = 0.0
            
            # Task specialization score
            if task_type in model.specialization:
                score += 3.0
            
            # Urgency matching score
            estimated_latency = self.estimate_latency(model, prompt_length)
            
            if urgency == TaskUrgency.REALTIME:
                if estimated_latency < 100:
                    score += 5.0
                elif estimated_latency < 500:
                    score += 2.0
            elif urgency == TaskUrgency.FAST:
                if estimated_latency < 1000:
                    score += 4.0
                elif estimated_latency < 2000:
                    score += 2.0
            elif urgency == TaskUrgency.NORMAL:
                if estimated_latency < 5000:
                    score += 3.0
            
            # Penalize high load
            score -= model.current_load * 2.0
            
            # Memory efficiency bonus for smaller models
            if model.memory_gb < 20:
                score += 1.0

            # CPU-only heuristic adjustments:
            # - Prefer Llama 70B for DEEP long-form analytics (market/doc/news)
            if model.name.startswith("llama3.1:") or model.name.startswith("llama3:"):
                if urgency == TaskUrgency.DEEP and task_type in (
                    TaskType.MARKET_ANALYSIS,
                    TaskType.DOCUMENT_ANALYSIS,
                    TaskType.NEWS_PROCESSING,
                ):
                    score += self.bias["llama_pref_deep"]
                # Penalize for fast/realtime tasks (high latency on CPU)
                if urgency in (TaskUrgency.REALTIME, TaskUrgency.FAST):
                    score -= self.bias["llama_penalize_fast"]

            # Strategy/risk: favor Mixtral slightly
            if model.name.startswith("mixtral") and task_type in (
                TaskType.STRATEGY_SELECTION,
                TaskType.RISK_ASSESSMENT,
                TaskType.OPTIONS_PRICING,
            ):
                score += self.bias["strategy_mixtral_bonus"]

            # Document/news processing: add a small bias for Qwen
            if model.name.startswith("qwen") and task_type in (
                TaskType.DOCUMENT_ANALYSIS,
                TaskType.NEWS_PROCESSING,
            ):
                score += self.bias["doc_qwen_bonus"]
            
            scored_models.append((model, score, estimated_latency))
        
        # Sort by score
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        if not scored_models:
            return RoutingDecision(
                model="qwen2.5:72b",
                estimated_latency_ms=5000,
                confidence=0.1
            )
        
        best_model, best_score, best_latency = scored_models[0]
        
        # Determine fallback
        fallback_model = None
        if len(scored_models) > 1:
            fallback_model = scored_models[1][0].name
        
        # Calculate confidence
        confidence = min(best_score / 10.0, 1.0)
        
        return RoutingDecision(
            model=best_model.name,
            estimated_latency_ms=best_latency,
            confidence=confidence,
            fallback_model=fallback_model
        )
    
    async def execute_with_model(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """Execute prompt with specified model"""
        
        start_time = time.time()
        # Determine HTTP client timeout (model-aware)
        try:
            base_timeout = float(os.getenv("OLLAMA_CLIENT_TIMEOUT_SECONDS", "120"))
        except Exception:
            base_timeout = 120.0
        timeout_seconds = base_timeout
        # Allow longer timeouts for very large models when generating long outputs
        heavy_prefixes = ("llama3.1:", "mixtral", "qwen2.5:", "command-r-plus:", "yi:")
        if any(model.startswith(p) for p in heavy_prefixes):
            try:
                timeout_seconds = float(os.getenv("OLLAMA_CLIENT_TIMEOUT_HEAVY_SECONDS", str(max(base_timeout, 600))))
            except Exception:
                timeout_seconds = max(base_timeout, 600.0)
        # For very long-context requests (heuristic if caller passes large max_tokens), allow more time
        if max_tokens and max_tokens > 4096:
            timeout_seconds = max(timeout_seconds, base_timeout * 2)

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": float(temperature),
                            "num_predict": int(max_tokens)
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    
                    # Update usage stats
                    if model not in self.usage_stats:
                        self.usage_stats[model] = {
                            "calls": 0,
                            "total_latency": 0,
                            "errors": 0
                        }
                    
                    self.usage_stats[model]["calls"] += 1
                    self.usage_stats[model]["total_latency"] += elapsed_ms
                    
                    # Update model load estimation
                    if model in self.models:
                        self.models[model].current_load = min(
                            self.models[model].current_load + 0.1, 1.0
                        )
                        # Schedule load decrease
                        asyncio.create_task(self._decrease_load(model))
                    
                    # Prometheus: tokens/sec and latency
                    try:
                        eval_count = float(result.get("eval_count", 0) or 0)
                        eval_ns = float(result.get("eval_duration", 0) or 0)
                        tps = (eval_count / (eval_ns / 1e9)) if eval_ns > 0 else 0.0
                        if self.g_tokens_sec:
                            self.g_tokens_sec.labels(model=model).set(tps)
                        if self.g_latency_ms:
                            self.g_latency_ms.labels(model=model).set(elapsed_ms)
                    except Exception:
                        tps = 0.0

                    return {
                        "success": True,
                        "model": model,
                        "response": result.get("response", ""),
                        "latency_ms": elapsed_ms,
                        "tokens_used": result.get("eval_count", 0),
                        "eval_duration_ms": int((result.get("eval_duration", 0) or 0) / 1e6),
                        "total_duration_ms": int((result.get("total_duration", 0) or 0) / 1e6),
                        "tokens_per_sec": tps
                    }
                else:
                    return {
                        "success": False,
                        "model": model,
                        "error": f"HTTP {response.status_code}",
                        "latency_ms": int((time.time() - start_time) * 1000)
                    }
                    
        except Exception as e:
            if model in self.usage_stats:
                self.usage_stats[model]["errors"] += 1
            
            return {
                "success": False,
                "model": model,
                "error": str(e),
                "latency_ms": int((time.time() - start_time) * 1000)
            }
    
    async def _decrease_load(self, model: str):
        """Gradually decrease load estimation"""
        await asyncio.sleep(10)
        if model in self.models:
            self.models[model].current_load = max(
                self.models[model].current_load - 0.1, 0.0
            )
    
    async def smart_execute(
        self,
        task_type: TaskType,
        urgency: TaskUrgency,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        require_long_context: bool = False
    ) -> Dict[str, Any]:
        """Execute with intelligent routing and fallback"""
        
        # Check model availability if needed
        if (self.last_health_check is None or 
            datetime.utcnow() - self.last_health_check > timedelta(minutes=5)):
            await self.check_model_availability()
        
        # Route the request
        routing = self.route_request(
            task_type=task_type,
            urgency=urgency,
            prompt_length=len(prompt),
            require_long_context=require_long_context
        )
        
        # Try primary model
        result = await self.execute_with_model(
            model=routing.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # If failed and fallback available, try fallback
        if not result["success"] and routing.fallback_model:
            print(f"Primary model {routing.model} failed, trying fallback {routing.fallback_model}")
            result = await self.execute_with_model(
                model=routing.fallback_model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            result["used_fallback"] = True
        
        result["routing_decision"] = {
            "primary_model": routing.model,
            "estimated_latency_ms": routing.estimated_latency_ms,
            "confidence": routing.confidence,
            "fallback_model": routing.fallback_model
        }
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics"""
        stats = {
            "models": {},
            "usage": self.usage_stats,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None
        }
        
        for model_name, model_info in self.models.items():
            stats["models"][model_name] = {
                "available": model_info.is_available,
                "memory_gb": model_info.memory_gb,
                "current_load": model_info.current_load,
                "specializations": [t.value for t in model_info.specialization]
            }
        
        return stats

# Example usage
async def main():
    router = ModelRouter()
    
    # Check availability
    availability = await router.check_model_availability()
    print("Model Availability:", availability)
    
    # Example: Fast signal generation
    result = await router.smart_execute(
        task_type=TaskType.SIGNAL_GENERATION,
        urgency=TaskUrgency.FAST,
        prompt="Generate trading signal for SPY based on current technical indicators",
        temperature=0.3,
        max_tokens=500
    )
    print("Signal Generation Result:", result)
    
    # Example: Deep market analysis
    result = await router.smart_execute(
        task_type=TaskType.MARKET_ANALYSIS,
        urgency=TaskUrgency.DEEP,
        prompt="Provide comprehensive market analysis for technology sector",
        temperature=0.7,
        max_tokens=2000,
        require_long_context=True
    )
    print("Market Analysis Result:", result)
    
    # Get stats
    stats = router.get_stats()
    print("Router Stats:", json.dumps(stats, indent=2))

if __name__ == "__main__":
    asyncio.run(main())