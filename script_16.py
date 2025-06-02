# Create API models for request/response structures
api_models_content = '''"""
Pydantic models for API request and response structures.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class WorkflowTypeEnum(str, Enum):
    """Available workflow types."""
    LEGAL_RESEARCH = "legal_research"
    CONTRACT_ANALYSIS = "contract_analysis"
    BRIEF_WRITING = "brief_writing"
    DUE_DILIGENCE = "due_diligence"
    COMPLIANCE_REVIEW = "compliance_review"
    CASE_ANALYSIS = "case_analysis"
    DOCUMENT_REVIEW = "document_review"
    LEGAL_MEMO = "legal_memo"


class TaskPriorityEnum(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


# Authentication Models
class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# Task Models
class TaskCreationRequest(BaseModel):
    """Request to create a new task."""
    workflow_type: WorkflowTypeEnum
    description: str
    inputs: Dict[str, Any]
    priority: Optional[TaskPriorityEnum] = TaskPriorityEnum.MEDIUM


class TaskCreationResponse(BaseModel):
    """Response for task creation."""
    task_id: str
    message: str


class TaskResponse(BaseModel):
    """Generic task response model."""
    success: bool
    result: Optional[Dict[str, Any]] = None
    message: str
    error: Optional[str] = None


# Research Models
class ResearchRequest(BaseModel):
    """Legal research request model."""
    query: str = Field(..., description="Legal research query")
    jurisdiction: Optional[str] = Field(None, description="Legal jurisdiction")
    case_type: Optional[str] = Field(None, description="Type of case")
    include_secondary_sources: bool = Field(True, description="Include secondary sources")
    max_results: int = Field(10, description="Maximum number of results")


# Contract Analysis Models
class ContractAnalysisRequest(BaseModel):
    """Contract analysis request model."""
    contract_text: str = Field(..., description="Contract text to analyze")
    analysis_type: str = Field("comprehensive", description="Type of analysis")
    focus_areas: Optional[List[str]] = Field(None, description="Specific areas to focus on")


# Brief Writing Models
class BriefWritingRequest(BaseModel):
    """Brief writing request model."""
    case_facts: str = Field(..., description="Facts of the case")
    legal_issue: str = Field(..., description="Legal issue to address")
    brief_type: str = Field("motion", description="Type of brief")
    jurisdiction: Optional[str] = Field(None, description="Jurisdiction")
    court_level: Optional[str] = Field(None, description="Court level")


# Collaborative Analysis Models
class CollaborativeAnalysisRequest(BaseModel):
    """Collaborative analysis request model."""
    prompt: str = Field(..., description="Analysis prompt")
    agent_ids: List[str] = Field(..., description="Agent IDs to involve")
    max_rounds: int = Field(3, description="Maximum collaboration rounds")


# Document Models
class DocumentUploadResponse(BaseModel):
    """Document upload response model."""
    filename: str
    document_id: str
    size: int
    document_type: str
    message: str
    processed_at: datetime = Field(default_factory=datetime.now)


class DocumentMetadata(BaseModel):
    """Document metadata model."""
    document_id: str
    filename: str
    size: int
    document_type: str
    upload_date: datetime
    processed: bool
    embedding_count: Optional[int] = None


# Agent Models
class AgentStatus(BaseModel):
    """Agent status model."""
    agent_id: str
    role: str
    status: str
    current_task: Optional[str]
    memory_usage: int
    tools: List[str]


class AgentInteraction(BaseModel):
    """Agent interaction model."""
    agent_id: str
    interaction_type: str
    timestamp: datetime
    input_length: int
    output_length: int
    duration_seconds: Optional[float]


# LLM Provider Models
class LLMProviderInfo(BaseModel):
    """LLM provider information model."""
    name: str
    enabled: bool
    default_model: str
    available_models: List[str]
    rate_limits: Dict[str, int]
    status: str


class ModelInfo(BaseModel):
    """Model information model."""
    name: str
    provider: str
    capabilities: List[str]
    context_length: int
    supports_streaming: bool
    supports_function_calling: bool
    is_local: bool


# System Models
class SystemHealth(BaseModel):
    """System health model."""
    status: str
    timestamp: datetime
    services: Dict[str, str]
    memory_usage: Optional[Dict[str, Any]] = None
    disk_usage: Optional[Dict[str, Any]] = None


class SystemInfo(BaseModel):
    """System information model."""
    version: str
    environment: str
    enabled_providers: List[str]
    data_directories: Dict[str, str]
    security_config: Dict[str, Any]


# Error Models
class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None


# Search Models
class SearchRequest(BaseModel):
    """Document search request model."""
    query: str = Field(..., description="Search query")
    document_types: Optional[List[str]] = Field(None, description="Document types to search")
    limit: int = Field(10, description="Maximum results")
    similarity_threshold: float = Field(0.7, description="Similarity threshold")


class SearchResult(BaseModel):
    """Search result model."""
    document_id: str
    title: str
    content_excerpt: str
    similarity_score: float
    document_type: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    results: List[SearchResult]
    total_found: int
    search_time_seconds: float


# Workflow Models
class WorkflowStep(BaseModel):
    """Workflow step model."""
    step_number: int
    agent_role: str
    action: str
    status: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None


class WorkflowExecution(BaseModel):
    """Workflow execution model."""
    workflow_id: str
    workflow_type: WorkflowTypeEnum
    status: str
    steps: List[WorkflowStep]
    created_at: datetime
    completed_at: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None


# Reasoning Models
class ReasoningStep(BaseModel):
    """Individual reasoning step model."""
    step_type: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]


class ReasoningChain(BaseModel):
    """Complete reasoning chain model."""
    agent_id: str
    task_id: str
    steps: List[ReasoningStep]
    created_at: datetime
    completed_at: Optional[datetime] = None


# Analytics Models
class UsageStatistics(BaseModel):
    """Usage statistics model."""
    period: str  # "daily", "weekly", "monthly"
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    top_agents_used: List[Dict[str, Any]]
    top_workflows: List[Dict[str, Any]]


class PerformanceMetrics(BaseModel):
    """Performance metrics model."""
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    active_connections: int
    requests_per_minute: float
    average_request_duration: float


# Configuration Models
class ProviderConfiguration(BaseModel):
    """LLM provider configuration model."""
    enabled: bool
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_model: Optional[str] = None
    rate_limits: Dict[str, int]


class SecurityConfiguration(BaseModel):
    """Security configuration model."""
    content_validation_enabled: bool
    max_file_size_mb: int
    allowed_file_extensions: List[str]
    token_expiry_hours: int


# Batch Processing Models
class BatchProcessingRequest(BaseModel):
    """Batch processing request model."""
    documents: List[str] = Field(..., description="Document IDs to process")
    workflow_type: WorkflowTypeEnum
    parameters: Dict[str, Any] = Field(default_factory=dict)


class BatchProcessingResponse(BaseModel):
    """Batch processing response model."""
    batch_id: str
    status: str
    total_documents: int
    processed_documents: int
    failed_documents: int
    estimated_completion: Optional[datetime] = None


# Export Models
class ExportRequest(BaseModel):
    """Export request model."""
    format: str = Field("json", description="Export format (json, csv, pdf)")
    data_type: str = Field(..., description="Type of data to export")
    filters: Dict[str, Any] = Field(default_factory=dict)
    date_range: Optional[Dict[str, datetime]] = None


class ExportResponse(BaseModel):
    """Export response model."""
    export_id: str
    format: str
    file_url: str
    size_bytes: int
    created_at: datetime
    expires_at: datetime


# Notification Models
class NotificationSettings(BaseModel):
    """Notification settings model."""
    email_notifications: bool = True
    webhook_url: Optional[str] = None
    notification_types: List[str] = Field(default_factory=list)


class Notification(BaseModel):
    """Notification model."""
    id: str
    type: str
    title: str
    message: str
    created_at: datetime
    read: bool = False
    data: Optional[Dict[str, Any]] = None
'''

with open("legal_agent_orchestrator/api/models.py", "w") as f:
    f.write(api_models_content)

print("API models created!")