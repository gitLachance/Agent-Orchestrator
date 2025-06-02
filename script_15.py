# Create the main FastAPI application
api_routes_content = '''"""
FastAPI routes for the Legal Agent Orchestrator API.
Provides REST endpoints for agent management and task execution.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import asyncio
from datetime import datetime

from ..agents.orchestrator import LegalOrchestrator, WorkflowType, TaskPriority
from ..utils.config import get_config
from ..utils.logging import get_logger, log_security_event
from ..utils.security import AccessControl
from .models import *

logger = get_logger(__name__)
router = APIRouter()
security = HTTPBearer()


# Dependency for getting the orchestrator
async def get_orchestrator() -> LegalOrchestrator:
    """Get the legal orchestrator instance."""
    # This would be injected from the main app
    from ..main import get_orchestrator_instance
    return get_orchestrator_instance()


# Dependency for authentication
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Authenticate and get current user."""
    config = get_config()
    access_control = AccessControl(config.security.secret_key)
    
    user_data = access_control.validate_access_token(credentials.credentials)
    if not user_data:
        log_security_event(
            "invalid_token_used",
            "MEDIUM",
            {"token_prefix": credentials.credentials[:10]}
        )
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return user_data


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Legal Agent Orchestrator"
    }


@router.post("/auth/login")
async def login(credentials: LoginRequest) -> LoginResponse:
    """Authenticate user and return access token."""
    config = get_config()
    access_control = AccessControl(config.security.secret_key)
    
    # TODO: Implement actual user authentication
    # For now, using a simple demo user
    if credentials.username == "demo" and credentials.password == "demo":
        token = access_control.create_access_token(
            user_id="demo_user",
            permissions=["read", "write", "admin"],
            expiry_hours=config.security.token_expiry_hours
        )
        
        return LoginResponse(
            access_token=token,
            token_type="bearer",
            expires_in=config.security.token_expiry_hours * 3600
        )
    else:
        log_security_event(
            "failed_login_attempt",
            "MEDIUM",
            {"username": credentials.username}
        )
        raise HTTPException(status_code=401, detail="Invalid credentials")


@router.get("/agents/status")
async def get_agents_status(
    orchestrator: LegalOrchestrator = Depends(get_orchestrator),
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get status of all agents."""
    return orchestrator.get_agent_status()


@router.get("/tasks")
async def get_tasks(
    orchestrator: LegalOrchestrator = Depends(get_orchestrator),
    current_user: Dict = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get all tasks."""
    return orchestrator.get_all_tasks()


@router.get("/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    orchestrator: LegalOrchestrator = Depends(get_orchestrator),
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get status of a specific task."""
    status = orchestrator.get_task_status(task_id)
    if "error" in status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status


@router.post("/research")
async def execute_research(
    request: ResearchRequest,
    orchestrator: LegalOrchestrator = Depends(get_orchestrator),
    current_user: Dict = Depends(get_current_user)
) -> TaskResponse:
    """Execute a legal research workflow."""
    try:
        logger.info(f"Research request: {request.query}")
        
        result = await orchestrator.execute_research_workflow(
            query=request.query,
            jurisdiction=request.jurisdiction,
            case_type=request.case_type
        )
        
        return TaskResponse(
            success=True,
            result=result,
            message="Research completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Research workflow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/contract-analysis")
async def execute_contract_analysis(
    request: ContractAnalysisRequest,
    orchestrator: LegalOrchestrator = Depends(get_orchestrator),
    current_user: Dict = Depends(get_current_user)
) -> TaskResponse:
    """Execute a contract analysis workflow."""
    try:
        logger.info(f"Contract analysis request: {request.analysis_type}")
        
        result = await orchestrator.execute_contract_analysis_workflow(
            contract_text=request.contract_text,
            analysis_type=request.analysis_type
        )
        
        return TaskResponse(
            success=True,
            result=result,
            message="Contract analysis completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Contract analysis workflow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/brief-writing")
async def execute_brief_writing(
    request: BriefWritingRequest,
    orchestrator: LegalOrchestrator = Depends(get_orchestrator),
    current_user: Dict = Depends(get_current_user)
) -> TaskResponse:
    """Execute a brief writing workflow."""
    try:
        logger.info(f"Brief writing request: {request.brief_type}")
        
        result = await orchestrator.execute_brief_writing_workflow(
            case_facts=request.case_facts,
            legal_issue=request.legal_issue,
            brief_type=request.brief_type
        )
        
        return TaskResponse(
            success=True,
            result=result,
            message="Brief writing completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Brief writing workflow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collaborative-analysis")
async def execute_collaborative_analysis(
    request: CollaborativeAnalysisRequest,
    orchestrator: LegalOrchestrator = Depends(get_orchestrator),
    current_user: Dict = Depends(get_current_user)
) -> TaskResponse:
    """Execute collaborative analysis between multiple agents."""
    try:
        logger.info(f"Collaborative analysis request with {len(request.agent_ids)} agents")
        
        result = await orchestrator.collaborative_analysis(
            agents=request.agent_ids,
            prompt=request.prompt,
            max_rounds=request.max_rounds
        )
        
        return TaskResponse(
            success=True,
            result=result,
            message="Collaborative analysis completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Collaborative analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks")
async def create_task(
    request: TaskCreationRequest,
    orchestrator: LegalOrchestrator = Depends(get_orchestrator),
    current_user: Dict = Depends(get_current_user)
) -> TaskCreationResponse:
    """Create a new task."""
    try:
        task_id = await orchestrator.create_task(
            workflow_type=WorkflowType(request.workflow_type),
            description=request.description,
            inputs=request.inputs,
            priority=TaskPriority(request.priority) if request.priority else TaskPriority.MEDIUM
        )
        
        return TaskCreationResponse(
            task_id=task_id,
            message="Task created successfully"
        )
        
    except Exception as e:
        logger.error(f"Task creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/execute")
async def execute_task(
    task_id: str,
    orchestrator: LegalOrchestrator = Depends(get_orchestrator),
    current_user: Dict = Depends(get_current_user)
) -> TaskResponse:
    """Execute a specific task."""
    try:
        result = await orchestrator.execute_task(task_id)
        
        return TaskResponse(
            success=True,
            result=result,
            message="Task executed successfully"
        )
        
    except Exception as e:
        logger.error(f"Task execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = Form("general"),
    current_user: Dict = Depends(get_current_user)
) -> DocumentUploadResponse:
    """Upload a legal document for processing."""
    try:
        config = get_config()
        
        # Validate file
        if file.size > config.security.max_file_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {config.security.max_file_size_mb}MB"
            )
        
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in config.security.allowed_file_extensions:
            raise HTTPException(
                status_code=415,
                detail=f"File type not supported. Allowed: {config.security.allowed_file_extensions}"
            )
        
        # Read file content
        content = await file.read()
        
        # TODO: Process document and add to vector store
        # For now, just return success
        
        logger.info(f"Document uploaded: {file.filename} ({len(content)} bytes)")
        
        return DocumentUploadResponse(
            filename=file.filename,
            document_id=f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            size=len(content),
            document_type=document_type,
            message="Document uploaded successfully"
        )
        
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm-providers")
async def get_llm_providers(
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get available LLM providers and their status."""
    config = get_config()
    
    providers_status = {}
    for name, provider_config in config.llm_providers.items():
        providers_status[name] = {
            "enabled": provider_config.enabled,
            "default_model": provider_config.default_model,
            "models": provider_config.models,
            "rate_limits": {
                "requests_per_minute": provider_config.rate_limit_requests_per_minute,
                "tokens_per_minute": provider_config.rate_limit_tokens_per_minute
            }
        }
    
    return {"providers": providers_status}


@router.get("/system/info")
async def get_system_info(
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get system information and statistics."""
    config = get_config()
    
    return {
        "environment": config.environment,
        "version": "0.1.0",
        "enabled_providers": config.get_enabled_providers(),
        "data_directories": {
            "data_dir": config.data_dir,
            "documents_dir": config.documents_dir,
            "embeddings_dir": config.embeddings_dir,
            "cache_dir": config.cache_dir
        },
        "security": {
            "content_validation_enabled": config.security.enable_content_validation,
            "max_file_size_mb": config.security.max_file_size_mb,
            "allowed_extensions": config.security.allowed_file_extensions
        }
    }
'''

with open("legal_agent_orchestrator/api/routes.py", "w") as f:
    f.write(api_routes_content)

print("FastAPI routes created!")