# Create the base LLM provider class
base_provider_content = '''"""
Base LLM Provider class defining the interface for all language model providers.
Supports both local and cloud-based LLM services with security and compliance features.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncGenerator, Union
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import time
from datetime import datetime

from ..utils.logging import get_logger
from ..utils.security import SecurityValidator

logger = get_logger(__name__)


class ProviderType(str, Enum):
    """Types of LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    LOCAL = "local"
    GOOGLE = "google"
    COHERE = "cohere"


class ModelCapability(str, Enum):
    """Capabilities that models can have."""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    EMBEDDING = "embedding"
    CHAT = "chat"
    COMPLETION = "completion"


class LLMRequest(BaseModel):
    """Request structure for LLM interactions."""
    prompt: str
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[List[str]] = None
    stream: bool = False
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Response structure from LLM interactions."""
    content: str
    model: str
    provider: str
    usage: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None
    finish_reason: Optional[str] = None


class ModelInfo(BaseModel):
    """Information about a model."""
    name: str
    provider: ProviderType
    capabilities: List[ModelCapability]
    context_length: int
    input_cost_per_token: Optional[float] = None
    output_cost_per_token: Optional[float] = None
    supports_streaming: bool = True
    supports_function_calling: bool = False
    is_local: bool = False
    description: Optional[str] = None


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    Provides a unified interface for interacting with different language models
    while maintaining security, compliance, and performance standards required
    for legal applications.
    """

    def __init__(
        self,
        provider_type: ProviderType,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        security_validator: Optional[SecurityValidator] = None,
        rate_limit_requests_per_minute: int = 60,
        rate_limit_tokens_per_minute: int = 100000
    ):
        self.provider_type = provider_type
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        self.security_validator = security_validator or SecurityValidator()
        
        # Rate limiting
        self.rate_limit_requests_per_minute = rate_limit_requests_per_minute
        self.rate_limit_tokens_per_minute = rate_limit_tokens_per_minute
        self._request_timestamps = []
        self._token_usage = []
        
        # Model information cache
        self._available_models: Dict[str, ModelInfo] = {}
        
        # Request history for monitoring
        self.request_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized {provider_type.value} LLM provider")

    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate a text response from the LLM."""
        pass

    @abstractmethod
    async def generate_response_structured(
        self,
        request: LLMRequest
    ) -> LLMResponse:
        """Generate a structured response with metadata."""
        pass

    @abstractmethod
    async def stream_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response generation for real-time feedback."""
        pass

    @abstractmethod
    async def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models from this provider."""
        pass

    @abstractmethod
    async def validate_model(self, model_name: str) -> bool:
        """Validate that a model is available and accessible."""
        pass

    async def generate_response_with_validation(
        self,
        prompt: str,
        model: Optional[str] = None,
        validate_content: bool = True,
        **kwargs
    ) -> str:
        """Generate response with security validation."""
        try:
            # Security validation
            if validate_content:
                is_safe = await self.security_validator.validate_input(prompt)
                if not is_safe:
                    raise ValueError("Input failed security validation")
            
            # Rate limiting check
            await self._check_rate_limits()
            
            # Generate response
            response = await self.generate_response(prompt, model, **kwargs)
            
            # Validate output if required
            if validate_content:
                is_safe_output = await self.security_validator.validate_output(response)
                if not is_safe_output:
                    logger.warning("Generated content flagged by security validator")
                    response = "[Content flagged by security validator]"
            
            # Log the interaction
            self._log_interaction(prompt, response, model)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Handle chat-style completions."""
        # Convert messages to a single prompt for providers that don't support chat natively
        if not await self._supports_chat_format(model):
            prompt = self._convert_messages_to_prompt(messages)
            return await self.generate_response(prompt, model, **kwargs)
        else:
            # Use native chat format
            request = LLMRequest(
                messages=messages,
                model=model or self.default_model,
                **kwargs
            )
            response = await self.generate_response_structured(request)
            return response.content

    async def function_calling(
        self,
        prompt: str,
        functions: List[Dict[str, Any]],
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle function calling if supported by the model."""
        model_info = await self._get_model_info(model or self.default_model)
        
        if not model_info or not model_info.supports_function_calling:
            raise NotImplementedError(f"Function calling not supported by {model or self.default_model}")
        
        # This should be implemented by specific providers
        raise NotImplementedError("Function calling must be implemented by specific providers")

    async def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Get embeddings for texts if supported."""
        # This should be implemented by providers that support embeddings
        raise NotImplementedError("Embeddings must be implemented by specific providers")

    async def _check_rate_limits(self) -> None:
        """Check and enforce rate limits."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old timestamps
        self._request_timestamps = [ts for ts in self._request_timestamps if ts > minute_ago]
        self._token_usage = [(ts, tokens) for ts, tokens in self._token_usage if ts > minute_ago]
        
        # Check request rate limit
        if len(self._request_timestamps) >= self.rate_limit_requests_per_minute:
            wait_time = self._request_timestamps[0] + 60 - current_time
            logger.warning(f"Rate limit reached, waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
        
        # Check token rate limit
        total_tokens = sum(tokens for _, tokens in self._token_usage)
        if total_tokens >= self.rate_limit_tokens_per_minute:
            wait_time = self._token_usage[0][0] + 60 - current_time
            logger.warning(f"Token rate limit reached, waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
        
        # Record this request
        self._request_timestamps.append(current_time)

    def _log_interaction(
        self,
        prompt: str,
        response: str,
        model: Optional[str]
    ) -> None:
        """Log interaction for monitoring and compliance."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "provider": self.provider_type.value,
            "model": model or self.default_model,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "prompt_hash": hash(prompt),  # For deduplication without storing content
        }
        
        self.request_history.append(interaction)
        
        # Keep only recent history (last 1000 interactions)
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]

    async def _get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get cached model information."""
        if model_name not in self._available_models:
            # Refresh model cache
            await self.get_available_models()
        
        return self._available_models.get(model_name)

    async def _supports_chat_format(self, model: Optional[str]) -> bool:
        """Check if model supports chat message format."""
        model_info = await self._get_model_info(model or self.default_model)
        return model_info and ModelCapability.CHAT in model_info.capabilities

    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt."""
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\\n\\n".join(prompt_parts)

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for monitoring."""
        if not self.request_history:
            return {"total_requests": 0}
        
        recent_requests = [
            req for req in self.request_history
            if (datetime.now() - datetime.fromisoformat(req["timestamp"])).days < 30
        ]
        
        return {
            "total_requests": len(self.request_history),
            "recent_requests": len(recent_requests),
            "models_used": list(set(req["model"] for req in recent_requests)),
            "avg_prompt_length": sum(req["prompt_length"] for req in recent_requests) / max(len(recent_requests), 1),
            "avg_response_length": sum(req["response_length"] for req in recent_requests) / max(len(recent_requests), 1),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the provider."""
        try:
            # Simple test request
            test_response = await self.generate_response(
                "Hello", 
                max_tokens=10
            )
            
            return {
                "status": "healthy",
                "provider": self.provider_type.value,
                "test_response_length": len(test_response),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_type.value,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def __str__(self) -> str:
        return f"{self.provider_type.value}Provider(model={self.default_model})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider_type={self.provider_type}, default_model={self.default_model})"
'''

with open("legal_agent_orchestrator/llm_providers/base_provider.py", "w") as f:
    f.write(base_provider_content)

print("Base LLM provider created!")