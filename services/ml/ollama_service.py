#!/usr/bin/env python3
"""
Ollama Service - Production integration with Ollama API
"""

import asyncio
import httpx
import json
import os
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
from pathlib import Path

@dataclass
class ModelResponse:
    model: str
    response: str
    tokens_used: int
    latency_ms: int
    cached: bool = False
    timestamp: datetime = None

class OllamaService:
    def __init__(
        self,
        host: str = None,
        cache_dir: str = "/tmp/ollama_cache",
        cache_ttl_minutes: int = 30
    ):
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.client = None
        self.available_models = set()
        
    async def initialize(self):
        """Initialize the service and check connectivity"""
        try:
            self.client = httpx.AsyncClient(timeout=120.0)
            await self.refresh_models()
            return True
        except Exception as e:
            print(f"Failed to initialize Ollama service: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            await self.client.aclose()
    
    async def refresh_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = await self.client.get(f"{self.host}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                self.available_models = {m["name"] for m in models}
                return list(self.available_models)
        except Exception as e:
            print(f"Error refreshing models: {e}")
        return []
    
    def _get_cache_key(self, model: str, prompt: str, **kwargs) -> str:
        """Generate cache key for request"""
        cache_data = {
            "model": model,
            "prompt": prompt,
            **kwargs
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[ModelResponse]:
        """Retrieve cached response if valid"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                timestamp = datetime.fromisoformat(data["timestamp"])
                if datetime.utcnow() - timestamp < self.cache_ttl:
                    return ModelResponse(
                        model=data["model"],
                        response=data["response"],
                        tokens_used=data["tokens_used"],
                        latency_ms=data["latency_ms"],
                        cached=True,
                        timestamp=timestamp
                    )
            except Exception as e:
                print(f"Error reading cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, response: ModelResponse):
        """Save response to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            data = {
                "model": response.model,
                "response": response.response,
                "tokens_used": response.tokens_used,
                "latency_ms": response.latency_ms,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving to cache: {e}")
    
    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        use_cache: bool = True,
        stream: bool = False,
        **kwargs
    ) -> ModelResponse:
        """Generate response from model"""
        
        # Check cache if enabled
        if use_cache and not stream:
            cache_key = self._get_cache_key(model, prompt, temperature=temperature)
            cached = self._get_cached_response(cache_key)
            if cached:
                return cached
        
        # Validate model availability
        if model not in self.available_models:
            await self.refresh_models()
            if model not in self.available_models:
                raise ValueError(f"Model {model} not available. Available: {self.available_models}")
        
        start_time = datetime.utcnow()
        
        try:
            request_data = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "options": {
                    "num_predict": max_tokens,
                    **kwargs
                },
                "stream": stream
            }
            
            response = await self.client.post(
                f"{self.host}/api/generate",
                json=request_data
            )
            
            if response.status_code == 200:
                if stream:
                    # Return async generator for streaming
                    return self._handle_stream(response)
                else:
                    result = response.json()
                    
                    elapsed_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                    
                    model_response = ModelResponse(
                        model=model,
                        response=result.get("response", ""),
                        tokens_used=result.get("eval_count", 0) + result.get("prompt_eval_count", 0),
                        latency_ms=elapsed_ms,
                        timestamp=datetime.utcnow()
                    )
                    
                    # Cache the response
                    if use_cache:
                        cache_key = self._get_cache_key(model, prompt, temperature=temperature)
                        self._save_to_cache(cache_key, model_response)
                    
                    return model_response
            else:
                raise Exception(f"API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Failed to generate with {model}: {e}")
    
    async def _handle_stream(self, response) -> AsyncGenerator[str, None]:
        """Handle streaming response"""
        async for line in response.aiter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                except json.JSONDecodeError:
                    continue
    
    async def embeddings(
        self,
        model: str,
        prompt: str
    ) -> List[float]:
        """Generate embeddings for text"""
        
        try:
            response = await self.client.post(
                f"{self.host}/api/embeddings",
                json={
                    "model": model,
                    "prompt": prompt
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("embedding", [])
            else:
                raise Exception(f"API error: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {e}")
    
    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> ModelResponse:
        """Chat completion with conversation history"""
        
        start_time = datetime.utcnow()
        
        try:
            request_data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "options": {
                    "num_predict": max_tokens
                },
                "stream": stream
            }
            
            response = await self.client.post(
                f"{self.host}/api/chat",
                json=request_data
            )
            
            if response.status_code == 200:
                if stream:
                    return self._handle_stream(response)
                else:
                    result = response.json()
                    elapsed_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                    
                    return ModelResponse(
                        model=model,
                        response=result.get("message", {}).get("content", ""),
                        tokens_used=result.get("eval_count", 0) + result.get("prompt_eval_count", 0),
                        latency_ms=elapsed_ms,
                        timestamp=datetime.utcnow()
                    )
            else:
                raise Exception(f"API error: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Failed chat completion: {e}")
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry"""
        try:
            response = await self.client.post(
                f"{self.host}/api/pull",
                json={"name": model_name},
                timeout=3600.0  # 1 hour timeout for large models
            )
            
            if response.status_code == 200:
                await self.refresh_models()
                return True
            
        except Exception as e:
            print(f"Failed to pull model {model_name}: {e}")
        
        return False
    
    async def delete_model(self, model_name: str) -> bool:
        """Delete a model"""
        try:
            response = await self.client.delete(
                f"{self.host}/api/delete",
                json={"name": model_name}
            )
            
            if response.status_code == 200:
                await self.refresh_models()
                return True
                
        except Exception as e:
            print(f"Failed to delete model {model_name}: {e}")
        
        return False
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        try:
            response = await self.client.post(
                f"{self.host}/api/show",
                json={"name": model_name}
            )
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            print(f"Failed to get model info: {e}")
        
        return {}
    
    def clear_cache(self, older_than_hours: int = 24):
        """Clear old cache entries"""
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        cleared = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                timestamp = datetime.fromisoformat(data["timestamp"])
                if timestamp < cutoff_time:
                    cache_file.unlink()
                    cleared += 1
            except Exception:
                pass
        
        return cleared
    
    async def batch_generate(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = 3
    ) -> List[ModelResponse]:
        """Process multiple requests concurrently"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_request(req):
            async with semaphore:
                return await self.generate(**req)
        
        tasks = [process_request(req) for req in requests]
        return await asyncio.gather(*tasks)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            response = await self.client.get(f"{self.host}/api/tags")
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                
                return {
                    "status": "healthy",
                    "host": self.host,
                    "available_models": len(models),
                    "models": [m["name"] for m in models],
                    "cache_size": len(list(self.cache_dir.glob("*.json"))),
                    "timestamp": datetime.utcnow().isoformat()
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "host": self.host,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

# Example usage
async def main():
    service = OllamaService()
    
    if await service.initialize():
        print("Ollama service initialized")
        
        # Check health
        health = await service.health_check()
        print("Health:", json.dumps(health, indent=2))
        
        # Generate response
        response = await service.generate(
            model="phi3:14b",
            prompt="What is the current market sentiment?",
            temperature=0.7,
            max_tokens=200
        )
        print(f"Response: {response.response[:100]}...")
        print(f"Latency: {response.latency_ms}ms")
        
        # Chat example
        messages = [
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": "Analyze SPY technical indicators"}
        ]
        
        chat_response = await service.chat(
            model="qwen2.5:72b",
            messages=messages
        )
        print(f"Chat: {chat_response.response[:100]}...")
        
        await service.cleanup()

if __name__ == "__main__":
    asyncio.run(main())