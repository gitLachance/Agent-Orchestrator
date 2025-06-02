# Create configuration management
config_content = '''"""
Configuration management for the Legal Agent Orchestrator.
Supports multiple environments and secure credential handling.
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from pathlib import Path

from .logging import get_logger

logger = get_logger(__name__)


class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    enabled: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_model: Optional[str] = None
    rate_limit_requests_per_minute: int = 60
    rate_limit_tokens_per_minute: int = 100000
    models: List[str] = Field(default_factory=list)
    timeout: int = 300
    
    @validator('api_key')
    def validate_api_key(cls, v, values):
        if values.get('enabled') and not v:
            logger.warning("Enabled LLM provider missing API key")
        return v


class DatabaseConfig(BaseModel):
    """Database configuration."""
    url: str = "sqlite:///./legal_agent.db"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20


class SecurityConfig(BaseModel):
    """Security configuration."""
    secret_key: str
    encryption_key: str
    token_expiry_hours: int = 24
    password_hash_rounds: int = 12
    enable_content_validation: bool = True
    max_file_size_mb: int = 100
    allowed_file_extensions: List[str] = Field(default_factory=lambda: [".pdf", ".docx", ".txt", ".md"])


class RAGConfig(BaseModel):
    """RAG system configuration."""
    vector_store_type: str = "chroma"
    embedding_model: str = "text-embedding-3-large"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retrieval_results: int = 10
    similarity_threshold: float = 0.7
    collection_name: str = "legal_documents"


class AgentConfig(BaseModel):
    """Agent configuration."""
    memory_limit: int = 100
    max_reasoning_steps: int = 50
    enable_collaboration: bool = True
    default_temperature: float = 0.7
    max_tokens: int = 2000


class DashboardConfig(BaseModel):
    """Dashboard configuration."""
    host: str = "0.0.0.0"
    port: int = 8501
    debug: bool = False
    theme: str = "light"
    enable_authentication: bool = True


class APIConfig(BaseModel):
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    enable_docs: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 5


class Config(BaseSettings):
    """Main configuration class for the Legal Agent Orchestrator."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # LLM Providers
    llm_providers: Dict[str, LLMProviderConfig] = Field(default_factory=dict)
    
    # Components
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    security: Optional[SecurityConfig] = None
    rag: RAGConfig = Field(default_factory=RAGConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Data directories
    data_dir: str = Field(default="./data", env="DATA_DIR")
    documents_dir: str = Field(default="./data/documents", env="DOCUMENTS_DIR")
    embeddings_dir: str = Field(default="./data/embeddings", env="EMBEDDINGS_DIR")
    cache_dir: str = Field(default="./data/cache", env="CACHE_DIR")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        
    def __init__(self, config_file: Optional[str] = None, **kwargs):
        """Initialize configuration with optional config file."""
        # Load from file if provided
        if config_file:
            config_data = self._load_config_file(config_file)
            kwargs.update(config_data)
        
        super().__init__(**kwargs)
        
        # Ensure required security config
        if not self.security:
            self.security = self._create_default_security_config()
        
        # Create data directories
        self._create_directories()
        
        # Initialize LLM providers from environment
        self._initialize_llm_providers()
        
        logger.info(f"Configuration loaded for environment: {self.environment}")

    def _load_config_file(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_file}")
                return {}
            
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from: {config_file}")
            return config_data
            
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}")
            return {}

    def _create_default_security_config(self) -> SecurityConfig:
        """Create default security configuration."""
        import secrets
        
        return SecurityConfig(
            secret_key=os.getenv("SECRET_KEY", secrets.token_urlsafe(32)),
            encryption_key=os.getenv("ENCRYPTION_KEY", secrets.token_urlsafe(32))
        )

    def _create_directories(self) -> None:
        """Create necessary data directories."""
        directories = [
            self.data_dir,
            self.documents_dir,
            self.embeddings_dir,
            self.cache_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _initialize_llm_providers(self) -> None:
        """Initialize LLM provider configurations from environment variables."""
        # OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.llm_providers["openai"] = LLMProviderConfig(
                api_key=openai_key,
                default_model="gpt-4-turbo-preview",
                models=["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"]
            )
        
        # Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            self.llm_providers["anthropic"] = LLMProviderConfig(
                api_key=anthropic_key,
                default_model="claude-3-sonnet-20240229",
                models=["claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
            )
        
        # Azure OpenAI
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        if azure_endpoint and azure_key:
            self.llm_providers["azure"] = LLMProviderConfig(
                api_key=azure_key,
                base_url=azure_endpoint,
                default_model="gpt-4",
                models=["gpt-4", "gpt-35-turbo"]
            )
        
        # Local (Ollama)
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.llm_providers["local"] = LLMProviderConfig(
            enabled=True,
            base_url=ollama_host,
            default_model="llama3.1:8b",
            models=["llama3.1:8b", "llama3.1:70b", "mistral:7b", "codellama:7b"]
        )

    def get_llm_provider_config(self, provider_name: str) -> Optional[LLMProviderConfig]:
        """Get configuration for a specific LLM provider."""
        return self.llm_providers.get(provider_name)

    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled LLM providers."""
        return [
            name for name, config in self.llm_providers.items()
            if config.enabled
        ]

    def save_to_file(self, file_path: str) -> None:
        """Save current configuration to a YAML file."""
        try:
            config_dict = self.dict()
            
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to: {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {e}")

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check for enabled providers with API keys
        for name, provider_config in self.llm_providers.items():
            if provider_config.enabled and name != "local":
                if not provider_config.api_key:
                    issues.append(f"Provider '{name}' is enabled but missing API key")
        
        # Check security configuration
        if self.security:
            if len(self.security.secret_key) < 32:
                issues.append("Security secret key should be at least 32 characters")
            if len(self.security.encryption_key) < 32:
                issues.append("Security encryption key should be at least 32 characters")
        
        # Check data directories
        for directory in [self.data_dir, self.documents_dir, self.embeddings_dir, self.cache_dir]:
            if not Path(directory).exists():
                issues.append(f"Data directory does not exist: {directory}")
        
        return issues

    def update_provider_config(self, provider_name: str, **updates) -> None:
        """Update configuration for a specific provider."""
        if provider_name in self.llm_providers:
            current_config = self.llm_providers[provider_name]
            updated_config = current_config.copy(update=updates)
            self.llm_providers[provider_name] = updated_config
            logger.info(f"Updated configuration for provider: {provider_name}")
        else:
            logger.warning(f"Provider not found for update: {provider_name}")

    def __str__(self) -> str:
        enabled_providers = self.get_enabled_providers()
        return f"Config(env={self.environment}, providers={enabled_providers})"


def load_config(
    config_file: Optional[str] = None,
    environment: Optional[str] = None
) -> Config:
    """Load configuration with automatic environment detection."""
    
    # Determine environment
    env = environment or os.getenv("ENVIRONMENT", "development")
    
    # Determine config file
    if not config_file:
        config_files = [
            f"config/{env}.yaml",
            f"config/{env}.yml",
            "config/local.yaml",
            "config/local.yml"
        ]
        
        for file_path in config_files:
            if Path(file_path).exists():
                config_file = file_path
                break
    
    # Load configuration
    config = Config(config_file=config_file, environment=env)
    
    # Validate configuration
    issues = config.validate_config()
    if issues:
        logger.warning(f"Configuration issues found: {issues}")
    
    return config


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _global_config
    _global_config = config
'''

with open("legal_agent_orchestrator/utils/config.py", "w") as f:
    f.write(config_content)

print("Configuration management created!")