# Create local LLM provider for Ollama
local_provider_content = '''"""
Local LLM Provider implementation supporting Ollama and other local model servers.
Provides privacy-focused local model execution for sensitive legal work.
"""

import asyncio
import time
import json
import httpx
from typing import Dict, List, Any, Optional, AsyncGenerator

from .base_provider import BaseLLMProvider, ProviderType, ModelCapability, ModelInfo, LLMRequest, LLMResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)


class LocalProvider(BaseLLMProvider):
    """
    Local LLM Provider supporting Ollama and other local model servers.
    
    Designed for maximum privacy and security when handling sensitive legal documents.
    Supports models like Llama, Mistral, Code Llama, and other open-source models.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        default_model: str = "llama3.1:8b",
        timeout: int = 300,
        **kwargs
    ):
        super().__init__(
            provider_type=ProviderType.LOCAL,
            base_url=host,
            default_model=default_model,
            **kwargs
        )
        
        self.host = host
        self.timeout = timeout
        
        # HTTP client for API calls
        self.client = httpx.AsyncClient(
            timeout=timeout,
            base_url=host
        )
        
        # Initialize available models cache
        self._models_cache_time = 0
        self._models_cache_duration = 300  # 5 minutes
        
        logger.info(f"Local provider initialized with host: {host}, default model: {default_model}")

    async def _refresh_models_cache(self) -> None:
        """Refresh the cache of available models."""
        current_time = time.time()
        if current_time - self._models_cache_time < self._models_cache_duration:
            return
        
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            
            models_data = response.json()
            self._available_models = {}
            
            for model_data in models_data.get("models", []):
                model_name = model_data.get("name", "")
                size = model_data.get("size", 0)
                
                # Determine capabilities based on model name
                capabilities = [ModelCapability.TEXT_GENERATION, ModelCapability.CHAT]
                
                if "code" in model_name.lower():
                    capabilities.append(ModelCapability.CODE_GENERATION)
                
                # Estimate context length based on model size
                context_length = self._estimate_context_length(model_name, size)
                
                self._available_models[model_name] = ModelInfo(
                    name=model_name,
                    provider=ProviderType.LOCAL,
                    capabilities=capabilities,
                    context_length=context_length,
                    is_local=True,
                    supports_streaming=True,
                    supports_function_calling=False,
                    description=f"Local model: {model_name}"
                )
            
            self._models_cache_time = current_time
            logger.info(f"Refreshed local models cache with {len(self._available_models)} models")
            
        except Exception as e:
            logger.error(f"Error refreshing models cache: {e}")

    def _estimate_context_length(self, model_name: str, size: int) -> int:
        """Estimate context length based on model name and size."""
        model_name_lower = model_name.lower()
        
        # Common context lengths for different models
        if "llama3.1" in model_name_lower:
            return 128000
        elif "llama3" in model_name_lower or "llama2" in model_name_lower:
            return 8192
        elif "mistral" in model_name_lower:
            return 32768
        elif "codellama" in model_name_lower:
            return 16384
        elif "phi" in model_name_lower:
            return 4096
        else:
            # Default estimation based on size
            if size > 10 * 1024 * 1024 * 1024:  # > 10GB
                return 32768
            elif size > 4 * 1024 * 1024 * 1024:  # > 4GB
                return 8192
            else:
                return 4096

    async def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate a text response using local models."""
        try:
            await self._check_rate_limits()
            
            model_name = model or self.default_model
            
            # Prepare request payload
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                    "num_predict": kwargs.get("max_tokens", 1000),
                }
            }
            
            # Add system prompt if provided
            if kwargs.get("system_prompt"):
                payload["system"] = kwargs["system_prompt"]
            
            # Add stop sequences if provided
            if kwargs.get("stop"):
                payload["options"]["stop"] = kwargs["stop"]
            
            # Make API call
            response = await self.client.post("/api/generate", json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result.get("response", "")
            
            self._log_interaction(prompt, content, model_name)
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating local response: {e}")
            raise

    async def generate_response_structured(
        self,
        request: LLMRequest
    ) -> LLMResponse:
        """Generate a structured response with metadata."""
        try:
            await self._check_rate_limits()
            
            model_name = request.model or self.default_model
            
            # Prepare prompt from messages if provided
            if request.messages:
                prompt = self._convert_messages_to_prompt(request.messages)
            else:
                prompt = request.prompt
            
            # Add system prompt
            if request.system_prompt:
                prompt = f"System: {request.system_prompt}\\n\\nHuman: {prompt}\\n\\nAssistant:"
            
            # Prepare request payload
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "num_predict": request.max_tokens,
                }
            }
            
            if request.stop:
                payload["options"]["stop"] = request.stop
            
            # Make API call
            response = await self.client.post("/api/generate", json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result.get("response", "")
            
            # Create structured response
            llm_response = LLMResponse(
                content=content,
                model=model_name,
                provider=self.provider_type.value,
                usage={
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                },
                metadata={
                    **request.metadata,
                    "eval_duration": result.get("eval_duration", 0),
                    "load_duration": result.get("load_duration", 0)
                },
                finish_reason=result.get("done_reason", "completed")
            )
            
            self._log_interaction(request.prompt, content, model_name)
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Error generating structured local response: {e}")
            raise

    async def stream_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response generation for real-time feedback."""
        try:
            await self._check_rate_limits()
            
            model_name = model or self.default_model
            
            # Prepare request payload
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                    "num_predict": kwargs.get("max_tokens", 1000),
                }
            }
            
            # Add system prompt if provided
            if kwargs.get("system_prompt"):
                payload["system"] = kwargs["system_prompt"]
            
            # Make streaming API call
            full_response = ""
            async with self.client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                chunk = data["response"]
                                full_response += chunk
                                yield chunk
                                
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
            
            # Log the complete interaction
            self._log_interaction(prompt, full_response, model_name)
            
        except Exception as e:
            logger.error(f"Error streaming local response: {e}")
            yield f"Error: {e}"

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Handle chat-style completions using Ollama's chat API."""
        try:
            await self._check_rate_limits()
            
            model_name = model or self.default_model
            
            # Prepare request payload for chat API
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                    "num_predict": kwargs.get("max_tokens", 1000),
                }
            }
            
            # Try chat API first, fall back to generate API
            try:
                response = await self.client.post("/api/chat", json=payload)
                response.raise_for_status()
                
                result = response.json()
                content = result.get("message", {}).get("content", "")
                
            except httpx.HTTPStatusError:
                # Fall back to generate API
                prompt = self._convert_messages_to_prompt(messages)
                content = await self.generate_response(prompt, model, **kwargs)
            
            return content
            
        except Exception as e:
            logger.error(f"Error with local chat completion: {e}")
            raise

    async def get_available_models(self) -> List[ModelInfo]:
        """Get list of available local models."""
        await self._refresh_models_cache()
        return list(self._available_models.values())

    async def validate_model(self, model_name: str) -> bool:
        """Validate that a model is available locally."""
        try:
            # Check if model exists in Ollama
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            
            models_data = response.json()
            available_models = [m.get("name", "") for m in models_data.get("models", [])]
            
            return model_name in available_models
            
        except Exception as e:
            logger.warning(f"Model validation failed for {model_name}: {e}")
            return False

    async def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Pull/download a model to the local Ollama instance."""
        try:
            payload = {"name": model_name}
            
            # Stream the pull process
            pull_status = {"status": "starting", "progress": 0}
            
            async with self.client.stream("POST", "/api/pull", json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            pull_status.update(data)
                            
                            if data.get("status") == "success":
                                logger.info(f"Successfully pulled model: {model_name}")
                                # Refresh models cache
                                self._models_cache_time = 0
                                await self._refresh_models_cache()
                                break
                                
                        except json.JSONDecodeError:
                            continue
            
            return pull_status
            
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            raise

    async def delete_model(self, model_name: str) -> bool:
        """Delete a model from the local Ollama instance."""
        try:
            payload = {"name": model_name}
            
            response = await self.client.delete("/api/delete", json=payload)
            response.raise_for_status()
            
            logger.info(f"Successfully deleted model: {model_name}")
            
            # Refresh models cache
            self._models_cache_time = 0
            await self._refresh_models_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            return False

    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model."""
        try:
            payload = {"name": model_name}
            
            response = await self.client.post("/api/show", json=payload)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return {}

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the local Ollama service."""
        try:
            # Check if Ollama is running
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            
            models_data = response.json()
            model_count = len(models_data.get("models", []))
            
            return {
                "status": "healthy",
                "provider": self.provider_type.value,
                "host": self.host,
                "available_models": model_count,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_type.value,
                "host": self.host,
                "error": str(e),
                "timestamp": time.time()
            }

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            asyncio.create_task(self.close())
        except Exception:
            pass
'''

with open("legal_agent_orchestrator/llm_providers/local_provider.py", "w") as f:
    f.write(local_provider_content)

print("Local LLM provider created!")