# Create the main FastAPI application
main_api_content = '''"""
Main FastAPI application for the Legal Agent Orchestrator.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .agents.orchestrator import LegalOrchestrator
from .llm_providers.openai_provider import OpenAIProvider
from .llm_providers.local_provider import LocalProvider
from .llm_providers.anthropic_provider import AnthropicProvider
from .rag.retriever import DocumentRetriever
from .utils.config import get_config, Config
from .utils.logging import setup_logging, get_logger, log_security_event
from .utils.security import SecurityValidator, AccessControl
from .api.routes import router as api_router
from .api.middleware import SecurityMiddleware, LoggingMiddleware

# Global instances
_orchestrator: Optional[LegalOrchestrator] = None
logger = get_logger(__name__)


def get_orchestrator_instance() -> LegalOrchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        raise RuntimeError("Orchestrator not initialized")
    return _orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _orchestrator
    
    # Startup
    logger.info("Starting Legal Agent Orchestrator")
    
    try:
        config = get_config()
        
        # Initialize LLM providers
        llm_providers = await initialize_llm_providers(config)
        
        # Initialize document retriever
        document_retriever = DocumentRetriever(config)
        await document_retriever.initialize()
        
        # Initialize orchestrator
        _orchestrator = LegalOrchestrator(
            config=config,
            llm_providers=llm_providers,
            document_retriever=document_retriever
        )
        
        logger.info("Legal Agent Orchestrator started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down Legal Agent Orchestrator")
    
    if _orchestrator:
        await _orchestrator.shutdown()
    
    # Close LLM provider connections
    for provider in llm_providers.values():
        if hasattr(provider, 'close'):
            await provider.close()
    
    logger.info("Legal Agent Orchestrator shut down complete")


async def initialize_llm_providers(config: Config) -> Dict[str, Any]:
    """Initialize LLM providers based on configuration."""
    providers = {}
    
    # OpenAI
    openai_config = config.get_llm_provider_config("openai")
    if openai_config and openai_config.enabled and openai_config.api_key:
        try:
            providers["openai"] = OpenAIProvider(
                api_key=openai_config.api_key,
                base_url=openai_config.base_url,
                default_model=openai_config.default_model,
                rate_limit_requests_per_minute=openai_config.rate_limit_requests_per_minute,
                rate_limit_tokens_per_minute=openai_config.rate_limit_tokens_per_minute
            )
            logger.info("OpenAI provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI provider: {e}")
    
    # Anthropic
    anthropic_config = config.get_llm_provider_config("anthropic")
    if anthropic_config and anthropic_config.enabled and anthropic_config.api_key:
        try:
            providers["anthropic"] = AnthropicProvider(
                api_key=anthropic_config.api_key,
                default_model=anthropic_config.default_model,
                rate_limit_requests_per_minute=anthropic_config.rate_limit_requests_per_minute,
                rate_limit_tokens_per_minute=anthropic_config.rate_limit_tokens_per_minute
            )
            logger.info("Anthropic provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic provider: {e}")
    
    # Local (Ollama)
    local_config = config.get_llm_provider_config("local")
    if local_config and local_config.enabled:
        try:
            providers["local"] = LocalProvider(
                host=local_config.base_url or "http://localhost:11434",
                default_model=local_config.default_model or "llama3.1:8b",
                timeout=local_config.timeout
            )
            
            # Test connection
            health = await providers["local"].health_check()
            if health["status"] == "healthy":
                logger.info("Local provider initialized and connected")
            else:
                logger.warning("Local provider initialized but not healthy")
                
        except Exception as e:
            logger.warning(f"Failed to initialize local provider: {e}")
    
    if not providers:
        raise RuntimeError("No LLM providers could be initialized")
    
    # Set default provider
    if "openai" in providers:
        providers["default"] = providers["openai"]
    elif "local" in providers:
        providers["default"] = providers["local"]
    else:
        providers["default"] = list(providers.values())[0]
    
    return providers


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_config()
    
    # Setup logging
    setup_logging(
        level=config.logging.level,
        log_file=config.logging.file,
        max_file_size_mb=config.logging.max_file_size_mb,
        backup_count=config.logging.backup_count,
        enable_rich_console=config.debug,
        enable_json_formatting=not config.debug
    )
    
    # Create FastAPI app
    app = FastAPI(
        title="Legal Agent Orchestrator",
        description="AI-powered legal assistant with agent orchestration",
        version="0.1.0",
        docs_url="/docs" if config.api.enable_docs else None,
        redoc_url="/redoc" if config.api.enable_docs else None,
        lifespan=lifespan
    )
    
    # Add security middleware
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(LoggingMiddleware)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add trusted host middleware for production
    if config.environment == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["yourdomain.com", "www.yourdomain.com"]
        )
    
    # Include API routes
    app.include_router(api_router, prefix="/api/v1")
    
    # Error handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "timestamp": str(datetime.now()),
                "path": request.url.path
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        log_security_event(
            "unhandled_exception",
            "HIGH",
            {
                "exception_type": type(exc).__name__,
                "path": request.url.path,
                "method": request.method
            }
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "status_code": 500,
                "timestamp": str(datetime.now()),
                "path": request.url.path
            }
        )
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": "Legal Agent Orchestrator",
            "version": "0.1.0",
            "status": "running",
            "docs_url": "/docs" if config.api.enable_docs else None
        }
    
    return app


# Create the app instance
app = create_app()


def run_server():
    """Run the development server."""
    config = get_config()
    
    uvicorn.run(
        "legal_agent_orchestrator.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.debug,
        log_level=config.logging.level.lower(),
        access_log=True
    )


if __name__ == "__main__":
    run_server()
'''

# Also need to create the main __init__.py file
main_init_content = '''"""
Legal Agent Orchestrator - AI-powered legal assistant with agent orchestration.
"""

__version__ = "0.1.0"
__author__ = "Legal AI Developer"
__email__ = "developer@legalai.com"

from .agents.orchestrator import LegalOrchestrator
from .utils.config import get_config, load_config
from .utils.logging import setup_logging

__all__ = [
    "LegalOrchestrator",
    "get_config",
    "load_config", 
    "setup_logging"
]
'''

with open("legal_agent_orchestrator/main.py", "w") as f:
    f.write(main_api_content)

with open("legal_agent_orchestrator/__init__.py", "w") as f:
    f.write(main_init_content)

print("Main application files created!")