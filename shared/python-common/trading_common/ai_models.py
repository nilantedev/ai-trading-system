"""AI model infrastructure for trading system with local and cloud model support."""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import httpx
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from .config import get_settings

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""
    LOCAL_OLLAMA = "local_ollama"
    OPENAI = "openai" 
    ANTHROPIC = "anthropic"


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
    """Ollama local model client."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def generate(self, prompt: str, model: str = "phi3:mini", **kwargs) -> ModelResponse:
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


class OpenAIClient(BaseModelClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=api_key or settings.ai.openai_api_key
        )
        
    async def generate(self, prompt: str, model: str = "gpt-4o-mini", **kwargs) -> ModelResponse:
        """Generate response using OpenAI model."""
        start_time = datetime.now()
        
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Make request
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            return ModelResponse(
                content=response.choices[0].message.content,
                model=f"openai/{model}",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None,
                latency_ms=latency,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "id": response.id,
                    "created": response.created
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check OpenAI API health."""
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False


class AnthropicClient(BaseModelClient):
    """Anthropic Claude API client."""
    
    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings()
        self.client = AsyncAnthropic(
            api_key=api_key or settings.ai.anthropic_api_key
        )
        
    async def generate(self, prompt: str, model: str = "claude-3-haiku-20240307", **kwargs) -> ModelResponse:
        """Generate response using Anthropic model."""
        start_time = datetime.now()
        
        try:
            # Make request
            response = await self.client.messages.create(
                model=model,
                max_tokens=kwargs.get("max_tokens", 1000),
                messages=[{"role": "user", "content": prompt}],
                **{k: v for k, v in kwargs.items() if k != "max_tokens"}
            )
            
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            return ModelResponse(
                content=response.content[0].text if response.content else "",
                model=f"anthropic/{model}",
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                } if response.usage else None,
                latency_ms=latency,
                metadata={
                    "id": response.id,
                    "role": response.role,
                    "stop_reason": response.stop_reason
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check Anthropic API health."""
        try:
            # Simple test request
            await self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}]
            )
            return True
        except Exception:
            return False


class ModelRouter:
    """Routes requests to appropriate model based on requirements and availability."""
    
    def __init__(self):
        self.clients = {}
        self.settings = get_settings()
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize available model clients."""
        # Always initialize Ollama for local models
        self.clients[ModelType.LOCAL_OLLAMA] = OllamaClient()
        
        # Initialize cloud clients if API keys are available
        if self.settings.ai.openai_api_key:
            self.clients[ModelType.OPENAI] = OpenAIClient()
            
        if self.settings.ai.anthropic_api_key:
            self.clients[ModelType.ANTHROPIC] = AnthropicClient()
    
    async def generate(
        self, 
        prompt: str, 
        model_preference: Optional[List[ModelType]] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response with automatic failover."""
        # Default preference: local first, then cloud
        if model_preference is None:
            model_preference = [ModelType.LOCAL_OLLAMA, ModelType.OPENAI, ModelType.ANTHROPIC]
        
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
                        # Default models for cloud services
                        if model_type == ModelType.OPENAI:
                            available[model_type.value] = ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
                        elif model_type == ModelType.ANTHROPIC:
                            available[model_type.value] = ["claude-3-haiku-20240307", "claude-3-sonnet-20240229"]
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