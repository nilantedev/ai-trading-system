"""AI model infrastructure for trading system with local model support only."""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass
import os
from enum import Enum

import httpx
# Removed paid API imports - using only local models

from .config import get_settings

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""
    LOCAL_OLLAMA = "local_ollama"
    # Paid APIs removed - using only local models


@dataclass
class ModelResponse:
    """Standardized model response."""
    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseModelClient:
    """Base class for model clients."""
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from model."""
        raise NotImplementedError
    
    async def health_check(self) -> bool:
        """Check if model service is healthy."""
        raise NotImplementedError


class OllamaClient(BaseModelClient):
    """Ollama local model client with advanced models."""
    
    def __init__(self, base_url: str = None):
        # Prefer explicit env; fall back to container DNS (ollama)
        if base_url is None:
            env_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST")
            if env_url and env_url.startswith("http"):
                base_url = env_url
            elif env_url:
                base_url = f"http://{env_url}:11434"
            else:
                base_url = "http://ollama:11434"
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)  # Increased timeout for large models
        # Default to powerful open-source models
        # Default model mapping aligned with installed models (2025-09-05)
        self.default_models = {
            "analysis": "qwen2.5:72b",          # Primary deep analysis
            "risk": "llama3.1:70b",            # Long-form reasoning
            "strategy": "mixtral:8x22b",       # Strategy generation
            "default": "phi3:14b"               # Fast general fallback
        }
        
    async def generate(self, prompt: str, model: str = "phi3:14b", **kwargs) -> ModelResponse:
        """Generate response using local Ollama model."""
        start_time = datetime.now()
        
        try:
            # Prepare request
            request_data = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                **kwargs
            }
            
            # Make request
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=request_data
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            return ModelResponse(
                content=result.get("response", ""),
                model=f"ollama/{model}",
                latency_ms=latency,
                metadata={
                    "context": result.get("context"),
                    "done": result.get("done", False),
                    "total_duration": result.get("total_duration"),
                    "load_duration": result.get("load_duration"),
                    "prompt_eval_count": result.get("prompt_eval_count"),
                    "eval_count": result.get("eval_count")
                }
            )
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check Ollama service health."""
        try:
            response = await self.client.get(f"{self.base_url}/api/version")
            return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []


# OpenAI client removed - using only local models for cost-free operation# Anthropic client removed - using only local models for cost-free operation


class ModelRouter:
    """Routes requests to appropriate model based on requirements and availability."""
    
    def __init__(self):
        self.clients = {}
        self.settings = get_settings()
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize available model clients."""
        # Only initialize Ollama for local models (cost-free operation)
        self.clients[ModelType.LOCAL_OLLAMA] = OllamaClient()
        logger.info("AI Model Router initialized with local Ollama models only")
    
    async def generate(
        self, 
        prompt: str, 
        model_preference: Optional[List[ModelType]] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response with automatic failover."""
        # Only use local models for cost-free operation
        if model_preference is None:
            model_preference = [ModelType.LOCAL_OLLAMA]
        
        last_error = None
        
        for model_type in model_preference:
            client = self.clients.get(model_type)
            if not client:
                continue
                
            try:
                # Check if client is healthy
                if not await client.health_check():
                    logger.warning(f"{model_type.value} client unhealthy, trying next")
                    continue
                
                # Generate response
                return await client.generate(prompt, **kwargs)
                
            except Exception as e:
                last_error = e
                logger.warning(f"Model {model_type.value} failed: {e}, trying next")
                continue
        
        # All models failed
        if last_error:
            raise last_error
        else:
            raise RuntimeError("No available models")
    
    async def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models by type."""
        available = {}
        
        for model_type, client in self.clients.items():
            try:
                if await client.health_check():
                    if hasattr(client, 'list_models'):
                        available[model_type.value] = await client.list_models()
                    else:
                        # Only local models available
                        available[model_type.value] = []
            except Exception as e:
                logger.warning(f"Failed to get models for {model_type.value}: {e}")
        
        return available
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all model clients."""
        health_status = {}
        
        for model_type, client in self.clients.items():
            try:
                health_status[model_type.value] = await client.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {model_type.value}: {e}")
                health_status[model_type.value] = False
        
        return health_status


# Global model router instance
_model_router: Optional[ModelRouter] = None


async def get_model_router() -> ModelRouter:
    """Get or create global model router."""
    global _model_router
    if _model_router is None:
        _model_router = ModelRouter()
    return _model_router


async def generate_response(
    prompt: str, 
    model_preference: Optional[List[ModelType]] = None,
    **kwargs
) -> ModelResponse:
    """Generate AI response with automatic model selection."""
    router = await get_model_router()
    return await router.generate(prompt, model_preference, **kwargs)


async def check_ai_health() -> Dict[str, bool]:
    """Check health of all AI services."""
    router = await get_model_router()
    return await router.health_check_all()