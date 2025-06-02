# Create the orchestrator class
orchestrator_content = '''"""
Agent Orchestrator for managing and coordinating multiple specialized legal agents.
Implements dynamic agent selection, task routing, and collaborative workflows.
"""

import asyncio
from typing import Dict, List, Any, Optional, Type, Union
from datetime import datetime
from enum import Enum
import uuid

from .base_agent import BaseAgent, Message, MessageType, AgentRole
from .research_agent import ResearchAgent
from .reasoning_agent import ReasoningAgent
from .drafting_agent import DraftingAgent
from .review_agent import ReviewAgent

from ..llm_providers.base_provider import BaseLLMProvider
from ..utils.logging import get_logger
from ..utils.config import Config
from ..rag.retriever import DocumentRetriever

logger = get_logger(__name__)


class WorkflowType(str, Enum):
    """Types of legal workflows that can be orchestrated."""
    LEGAL_RESEARCH = "legal_research"
    CONTRACT_ANALYSIS = "contract_analysis"
    BRIEF_WRITING = "brief_writing"
    DUE_DILIGENCE = "due_diligence"
    COMPLIANCE_REVIEW = "compliance_review"
    CASE_ANALYSIS = "case_analysis"
    DOCUMENT_REVIEW = "document_review"
    LEGAL_MEMO = "legal_memo"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Task(BaseModel):
    """Represents a task in the orchestration system."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: WorkflowType
    priority: TaskPriority = TaskPriority.MEDIUM
    description: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    assigned_agents: List[str] = Field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    results: Dict[str, Any] = Field(default_factory=dict)
    reasoning_chain: List[Dict[str, Any]] = Field(default_factory=list)


class LegalOrchestrator:
    """
    Central orchestrator for managing legal AI agents and workflows.
    
    Responsibilities:
    - Agent lifecycle management
    - Task routing and coordination
    - Workflow execution
    - Security and compliance oversight
    - Result aggregation and synthesis
    """

    def __init__(
        self,
        config: Config,
        llm_providers: Dict[str, BaseLLMProvider],
        document_retriever: DocumentRetriever
    ):
        self.config = config
        self.llm_providers = llm_providers
        self.document_retriever = document_retriever
        
        # Agent registry
        self.agents: Dict[str, BaseAgent] = {}
        self.active_tasks: Dict[str, Task] = {}
        
        # Workflow templates
        self.workflow_templates = self._initialize_workflow_templates()
        
        # Initialize default agents
        self._initialize_default_agents()
        
        logger.info("Legal Orchestrator initialized")

    def _initialize_default_agents(self) -> None:
        """Initialize the default set of legal agents."""
        default_llm = self.llm_providers.get("default") or list(self.llm_providers.values())[0]
        
        # Create specialized agents
        agents_config = [
            (ResearchAgent, AgentRole.RESEARCH, "research_01"),
            (ReasoningAgent, AgentRole.REASONING, "reasoning_01"),
            (DraftingAgent, AgentRole.DRAFTING, "drafting_01"),
            (ReviewAgent, AgentRole.REVIEW, "review_01")
        ]
        
        for agent_class, role, agent_id in agents_config:
            try:
                agent = agent_class(
                    agent_id=agent_id,
                    llm_provider=default_llm,
                    document_retriever=self.document_retriever
                )
                self.register_agent(agent)
                logger.info(f"Initialized {role.value} agent: {agent_id}")
            except Exception as e:
                logger.error(f"Failed to initialize {role.value} agent: {e}")

    def _initialize_workflow_templates(self) -> Dict[WorkflowType, Dict[str, Any]]:
        """Initialize workflow templates for different legal tasks."""
        return {
            WorkflowType.LEGAL_RESEARCH: {
                "agents": [AgentRole.RESEARCH, AgentRole.REASONING],
                "steps": [
                    {"agent": AgentRole.RESEARCH, "action": "search_case_law"},
                    {"agent": AgentRole.RESEARCH, "action": "search_statutes"},
                    {"agent": AgentRole.REASONING, "action": "analyze_precedents"},
                    {"agent": AgentRole.REASONING, "action": "synthesize_findings"}
                ]
            },
            WorkflowType.CONTRACT_ANALYSIS: {
                "agents": [AgentRole.RESEARCH, AgentRole.REASONING, AgentRole.REVIEW],
                "steps": [
                    {"agent": AgentRole.REVIEW, "action": "extract_key_terms"},
                    {"agent": AgentRole.RESEARCH, "action": "research_legal_standards"},
                    {"agent": AgentRole.REASONING, "action": "assess_risks"},
                    {"agent": AgentRole.REASONING, "action": "recommend_modifications"}
                ]
            },
            WorkflowType.BRIEF_WRITING: {
                "agents": [AgentRole.RESEARCH, AgentRole.REASONING, AgentRole.DRAFTING, AgentRole.REVIEW],
                "steps": [
                    {"agent": AgentRole.RESEARCH, "action": "research_legal_basis"},
                    {"agent": AgentRole.REASONING, "action": "construct_arguments"},
                    {"agent": AgentRole.DRAFTING, "action": "draft_brief"},
                    {"agent": AgentRole.REVIEW, "action": "review_and_refine"}
                ]
            },
            WorkflowType.LEGAL_MEMO: {
                "agents": [AgentRole.RESEARCH, AgentRole.REASONING, AgentRole.DRAFTING],
                "steps": [
                    {"agent": AgentRole.RESEARCH, "action": "gather_relevant_law"},
                    {"agent": AgentRole.REASONING, "action": "analyze_legal_issues"},
                    {"agent": AgentRole.DRAFTING, "action": "write_memo"}
                ]
            }
        }

    def register_agent(self, agent: BaseAgent) -> None:
        """Register a new agent with the orchestrator."""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id} (role: {agent.role.value})")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the orchestrator."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")

    def get_agents_by_role(self, role: AgentRole) -> List[BaseAgent]:
        """Get all agents with a specific role."""
        return [agent for agent in self.agents.values() if agent.role == role]

    async def create_task(
        self,
        workflow_type: WorkflowType,
        description: str,
        inputs: Dict[str, Any],
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> str:
        """Create a new task and return its ID."""
        task = Task(
            type=workflow_type,
            priority=priority,
            description=description,
            inputs=inputs
        )
        
        self.active_tasks[task.id] = task
        logger.info(f"Created task {task.id}: {description}")
        
        return task.id

    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """Execute a task using the appropriate workflow."""
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")
            
        task = self.active_tasks[task_id]
        task.status = "in_progress"
        
        try:
            # Get workflow template
            workflow = self.workflow_templates.get(task.type)
            if not workflow:
                raise ValueError(f"No workflow template for {task.type}")
            
            # Execute workflow steps
            results = await self._execute_workflow(task, workflow)
            
            # Update task
            task.status = "completed"
            task.completed_at = datetime.now()
            task.results = results
            
            logger.info(f"Task {task_id} completed successfully")
            return results
            
        except Exception as e:
            task.status = "failed"
            logger.error(f"Task {task_id} failed: {e}")
            raise

    async def _execute_workflow(self, task: Task, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow by coordinating multiple agents."""
        results = {}
        context = task.inputs.copy()
        
        for step in workflow["steps"]:
            agent_role = step["agent"]
            action = step["action"]
            
            # Find available agent with the required role
            agents = self.get_agents_by_role(agent_role)
            if not agents:
                raise RuntimeError(f"No agent available for role: {agent_role}")
            
            agent = agents[0]  # Use first available agent
            
            # Create message for the agent
            message = Message(
                sender="orchestrator",
                receiver=agent.agent_id,
                type=MessageType.TASK,
                content=f"Execute {action} for task: {task.description}",
                metadata={
                    "action": action,
                    "context": context,
                    "task_id": task.id
                }
            )
            
            # Send message to agent and get response
            response = await agent.process_message(message)
            
            # Store results and update context
            step_result = {
                "agent": agent.agent_id,
                "action": action,
                "response": response.content,
                "metadata": response.metadata
            }
            
            results[f"step_{len(results) + 1}"] = step_result
            context[f"{action}_result"] = response.content
            
            # Add to task reasoning chain
            task.reasoning_chain.extend(agent.get_reasoning_chain())
            
            logger.info(f"Completed step {action} with agent {agent.agent_id}")
        
        return results

    async def execute_research_workflow(
        self,
        query: str,
        jurisdiction: Optional[str] = None,
        case_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a legal research workflow."""
        task_id = await self.create_task(
            workflow_type=WorkflowType.LEGAL_RESEARCH,
            description=f"Legal research: {query}",
            inputs={
                "query": query,
                "jurisdiction": jurisdiction,
                "case_type": case_type
            }
        )
        
        return await self.execute_task(task_id)

    async def execute_contract_analysis_workflow(
        self,
        contract_text: str,
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Execute a contract analysis workflow."""
        task_id = await self.create_task(
            workflow_type=WorkflowType.CONTRACT_ANALYSIS,
            description=f"Contract analysis: {analysis_type}",
            inputs={
                "contract_text": contract_text,
                "analysis_type": analysis_type
            }
        )
        
        return await self.execute_task(task_id)

    async def execute_brief_writing_workflow(
        self,
        case_facts: str,
        legal_issue: str,
        brief_type: str = "motion"
    ) -> Dict[str, Any]:
        """Execute a brief writing workflow."""
        task_id = await self.create_task(
            workflow_type=WorkflowType.BRIEF_WRITING,
            description=f"Brief writing: {brief_type} for {legal_issue}",
            inputs={
                "case_facts": case_facts,
                "legal_issue": legal_issue,
                "brief_type": brief_type
            }
        )
        
        return await self.execute_task(task_id)

    async def collaborative_analysis(
        self,
        agents: List[str],
        prompt: str,
        max_rounds: int = 3
    ) -> Dict[str, Any]:
        """Enable collaborative analysis between multiple agents."""
        results = {"rounds": [], "final_synthesis": ""}
        
        current_context = prompt
        
        for round_num in range(max_rounds):
            round_results = {}
            
            # Each agent contributes to the analysis
            for agent_id in agents:
                if agent_id not in self.agents:
                    continue
                    
                agent = self.agents[agent_id]
                
                message = Message(
                    sender="orchestrator",
                    receiver=agent_id,
                    type=MessageType.TASK,
                    content=f"Round {round_num + 1} analysis: {current_context}",
                    metadata={"round": round_num + 1, "collaborative": True}
                )
                
                response = await agent.process_message(message)
                round_results[agent_id] = {
                    "role": agent.role.value,
                    "response": response.content,
                    "reasoning": agent.get_reasoning_chain()[-3:]  # Last 3 reasoning steps
                }
            
            results["rounds"].append(round_results)
            
            # Update context with all agent responses
            context_parts = [f"{agent_id}: {result['response']}" 
                           for agent_id, result in round_results.items()]
            current_context = "\\n".join(context_parts)
        
        # Final synthesis by reasoning agent
        reasoning_agents = self.get_agents_by_role(AgentRole.REASONING)
        if reasoning_agents:
            synthesis_message = Message(
                sender="orchestrator",
                receiver=reasoning_agents[0].agent_id,
                type=MessageType.TASK,
                content=f"Synthesize the following collaborative analysis: {current_context}",
                metadata={"synthesis": True}
            )
            
            synthesis_response = await reasoning_agents[0].process_message(synthesis_message)
            results["final_synthesis"] = synthesis_response.content
        
        return results

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a specific task."""
        if task_id not in self.active_tasks:
            return {"error": "Task not found"}
            
        task = self.active_tasks[task_id]
        return {
            "id": task.id,
            "type": task.type.value,
            "status": task.status,
            "priority": task.priority.value,
            "description": task.description,
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "progress": len(task.reasoning_chain)
        }

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get status of all tasks."""
        return [self.get_task_status(task_id) for task_id in self.active_tasks.keys()]

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all registered agents."""
        return {
            agent_id: {
                "role": agent.role.value,
                "status": agent.state.status,
                "current_task": agent.state.current_task,
                "memory_usage": len(agent.state.memory),
                "tools": agent.state.tools_available
            }
            for agent_id, agent in self.agents.items()
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown the orchestrator."""
        logger.info("Shutting down Legal Orchestrator")
        
        # Cancel any pending tasks
        for task in self.active_tasks.values():
            if task.status == "in_progress":
                task.status = "cancelled"
        
        # Clear agent states
        for agent in self.agents.values():
            agent.state.status = "shutdown"
        
        logger.info("Legal Orchestrator shutdown complete")
'''

with open("legal_agent_orchestrator/agents/orchestrator.py", "w") as f:
    f.write(orchestrator_content)

print("Orchestrator class created!")