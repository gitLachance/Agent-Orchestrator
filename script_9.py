# Fix the OpenAI provider with missing imports
openai_provider_content_fixed = '''"""
OpenAI LLM Provider implementation supporting GPT models and function calling.
Includes support for Azure OpenAI endpoints.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, AsyncGenerator
import json
import openai
from openai import AsyncOpenAI

from .base_provider import BaseLLMProvider, ProviderType, ModelCapability, ModelInfo, LLMRequest, LLMResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM Provider supporting GPT-3.5, GPT-4, and other OpenAI models.
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        default_model: str = "gpt-4-turbo-preview",
        organization: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            provider_type=ProviderType.OPENAI,
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            **kwargs
        )
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization
        )
        
        # Define available models
        self._initialize_models()
        
        logger.info(f"OpenAI provider initialized with default model: {default_model}")

    def _initialize_models(self) -> None:
        """Initialize available OpenAI models."""
        self._available_models = {
            "gpt-4-turbo-preview": ModelInfo(
                name="gpt-4-turbo-preview",
                provider=ProviderType.OPENAI,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                    ModelCapability.FUNCTION_CALLING
                ],
                context_length=128000,
                input_cost_per_token=0.00001,
                output_cost_per_token=0.00003,
                supports_streaming=True,
                supports_function_calling=True,
                description="Latest GPT-4 Turbo model with improved performance"
            ),
            "gpt-4": ModelInfo(
                name="gpt-4",
                provider=ProviderType.OPENAI,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                    ModelCapability.FUNCTION_CALLING
                ],
                context_length=8192,
                input_cost_per_token=0.00003,
                output_cost_per_token=0.00006,
                supports_streaming=True,
                supports_function_calling=True,
                description="Original GPT-4 model with excellent reasoning"
            ),
            "gpt-3.5-turbo": ModelInfo(
                name="gpt-3.5-turbo",
                provider=ProviderType.OPENAI,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                    ModelCapability.FUNCTION_CALLING
                ],
                context_length=16384,
                input_cost_per_token=0.0000005,
                output_cost_per_token=0.0000015,
                supports_streaming=True,
                supports_function_calling=True,
                description="Fast and efficient GPT-3.5 model"
            ),
            "text-embedding-3-large": ModelInfo(
                name="text-embedding-3-large",
                provider=ProviderType.OPENAI,
                capabilities=[ModelCapability.EMBEDDING],
                context_length=8191,
                input_cost_per_token=0.00000013,
                supports_streaming=False,
                supports_function_calling=False,
                description="High-dimensional text embeddings"
            )
        }

    async def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate a text response using OpenAI models."""
        try:
            await self._check_rate_limits()
            
            model_name = model or self.default_model
            
            # Prepare request parameters
            request_params = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                "presence_penalty": kwargs.get("presence_penalty", 0.0),
            }
            
            # Add stop sequences if provided
            if kwargs.get("stop"):
                request_params["stop"] = kwargs["stop"]
            
            # Add system prompt if provided
            if kwargs.get("system_prompt"):
                request_params["messages"].insert(0, {
                    "role": "system",
                    "content": kwargs["system_prompt"]
                })
            
            # Make API call
            response = await self.client.chat.completions.create(**request_params)
            
            # Extract response content
            content = response.choices[0].message.content
            
            # Log token usage
            if response.usage:
                current_time = time.time()
                total_tokens = response.usage.total_tokens
                self._token_usage.append((current_time, total_tokens))
            
            self._log_interaction(prompt, content, model_name)
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            raise

    async def generate_response_structured(
        self,
        request: LLMRequest
    ) -> LLMResponse:
        """Generate a structured response with metadata."""
        try:
            await self._check_rate_limits()
            
            model_name = request.model or self.default_model
            
            # Prepare messages
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            
            if request.messages:
                messages.extend(request.messages)
            else:
                messages.append({"role": "user", "content": request.prompt})
            
            # Prepare request parameters
            request_params = {
                "model": model_name,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "stream": request.stream
            }
            
            if request.stop:
                request_params["stop"] = request.stop
            
            # Make API call
            response = await self.client.chat.completions.create(**request_params)
            
            # Create structured response
            llm_response = LLMResponse(
                content=response.choices[0].message.content,
                model=model_name,
                provider=self.provider_type.value,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                },
                metadata=request.metadata,
                request_id=response.id,
                finish_reason=response.choices[0].finish_reason
            )
            
            # Log interaction
            self._log_interaction(request.prompt, llm_response.content, model_name)
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Error generating structured OpenAI response: {e}")
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
            
            # Prepare request parameters
            request_params = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7),
                "stream": True
            }
            
            # Add system prompt if provided
            if kwargs.get("system_prompt"):
                request_params["messages"].insert(0, {
                    "role": "system",
                    "content": kwargs["system_prompt"]
                })
            
            # Make streaming API call
            stream = await self.client.chat.completions.create(**request_params)
            
            full_response = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content_chunk = chunk.choices[0].delta.content
                    full_response += content_chunk
                    yield content_chunk
            
            # Log the complete interaction
            self._log_interaction(prompt, full_response, model_name)
            
        except Exception as e:
            logger.error(f"Error streaming OpenAI response: {e}")
            yield f"Error: {e}"

    async def function_calling(
        self,
        prompt: str,
        functions: List[Dict[str, Any]],
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle function calling with OpenAI models."""
        try:
            await self._check_rate_limits()
            
            model_name = model or self.default_model
            
            # Validate model supports function calling
            model_info = await self._get_model_info(model_name)
            if not model_info or not model_info.supports_function_calling:
                raise ValueError(f"Model {model_name} does not support function calling")
            
            # Prepare request
            request_params = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "functions": functions,
                "function_call": "auto",
                "temperature": kwargs.get("temperature", 0.7)
            }
            
            # Make API call
            response = await self.client.chat.completions.create(**request_params)
            
            message = response.choices[0].message
            
            if message.function_call:
                return {
                    "function_call": {
                        "name": message.function_call.name,
                        "arguments": json.loads(message.function_call.arguments)
                    },
                    "content": message.content,
                    "usage": response.usage._asdict() if response.usage else {}
                }
            else:
                return {
                    "content": message.content,
                    "usage": response.usage._asdict() if response.usage else {}
                }
                
        except Exception as e:
            logger.error(f"Error with OpenAI function calling: {e}")
            raise

    async def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = "text-embedding-3-large"
    ) -> List[List[float]]:
        """Get embeddings for texts using OpenAI embedding models."""
        try:
            await self._check_rate_limits()
            
            model_name = model or "text-embedding-3-large"
            
            # Make API call
            response = await self.client.embeddings.create(
                model=model_name,
                input=texts
            )
            
            # Extract embeddings
            embeddings = [data.embedding for data in response.data]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting OpenAI embeddings: {e}")
            raise

    async def get_available_models(self) -> List[ModelInfo]:
        """Get list of available OpenAI models."""
        try:
            # Try to fetch models from API
            models_response = await self.client.models.list()
            
            # Filter for relevant models and update our cache
            available_model_names = [model.id for model in models_response.data]
            
            available_models = []
            for model_name, model_info in self._available_models.items():
                if model_name in available_model_names:
                    available_models.append(model_info)
            
            return available_models
            
        except Exception as e:
            logger.warning(f"Error fetching OpenAI models: {e}")
            # Return cached models
            return list(self._available_models.values())

    async def validate_model(self, model_name: str) -> bool:
        """Validate that a model is available and accessible."""
        try:
            # Try a simple test request
            await self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.warning(f"Model validation failed for {model_name}: {e}")
            return False

    async def estimate_cost(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1000
    ) -> Dict[str, float]:
        """Estimate the cost of a request."""
        model_name = model or self.default_model
        model_info = await self._get_model_info(model_name)
        
        if not model_info:
            return {"error": "Model not found"}
        
        # Rough token estimation (OpenAI uses ~4 chars per token)
        prompt_tokens = len(prompt) // 4
        estimated_total_tokens = prompt_tokens + max_tokens
        
        input_cost = prompt_tokens * (model_info.input_cost_per_token or 0)
        output_cost = max_tokens * (model_info.output_cost_per_token or 0)
        total_cost = input_cost + output_cost
        
        return {
            "estimated_prompt_tokens": prompt_tokens,
            "estimated_completion_tokens": max_tokens,
            "estimated_total_tokens": estimated_total_tokens,
            "estimated_input_cost": input_cost,
            "estimated_output_cost": output_cost,
            "estimated_total_cost": total_cost,
            "currency": "USD"
        }
'''

with open("legal_agent_orchestrator/llm_providers/openai_provider.py", "w") as f:
    f.write(openai_provider_content_fixed)

print("Fixed OpenAI provider imports!")