# Create the base agent class
base_agent_content = '''"""
Base Agent class that all specialized agents inherit from.
Provides common functionality for LLM interaction, memory management, and tool usage.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncGenerator
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import uuid
from datetime import datetime

from ..llm_providers.base_provider import BaseLLMProvider
from ..utils.logging import get_logger
from ..utils.security import SecurityValidator

logger = get_logger(__name__)


class AgentRole(str, Enum):
    """Enumeration of agent roles in the legal system."""
    RESEARCH = "research"
    REASONING = "reasoning"
    DRAFTING = "drafting"
    REVIEW = "review"
    ORCHESTRATOR = "orchestrator"


class MessageType(str, Enum):
    """Types of messages in agent communication."""
    TASK = "task"
    RESPONSE = "response"
    OBSERVATION = "observation"
    THOUGHT = "thought"
    ACTION = "action"
    FINAL_ANSWER = "final_answer"


class Message(BaseModel):
    """Message structure for agent communication."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    sender: str
    receiver: Optional[str] = None
    type: MessageType
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    sensitive: bool = False  # Flag for attorney-client privileged content


class AgentState(BaseModel):
    """Current state of an agent."""
    agent_id: str
    role: AgentRole
    status: str = "idle"  # idle, working, completed, error
    current_task: Optional[str] = None
    memory: List[Message] = Field(default_factory=list)
    tools_available: List[str] = Field(default_factory=list)
    reasoning_chain: List[Dict[str, Any]] = Field(default_factory=list)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the legal orchestration system.
    
    Provides core functionality for:
    - LLM interaction with security validation
    - Memory management and message handling
    - Tool integration and execution
    - Reasoning chain construction
    - Security and compliance features
    """

    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        llm_provider: BaseLLMProvider,
        tools: Optional[List[Any]] = None,
        memory_limit: int = 100,
        security_validator: Optional[SecurityValidator] = None
    ):
        self.agent_id = agent_id
        self.role = role
        self.llm_provider = llm_provider
        self.tools = tools or []
        self.memory_limit = memory_limit
        self.security_validator = security_validator or SecurityValidator()
        
        self.state = AgentState(
            agent_id=agent_id,
            role=role,
            tools_available=[tool.__class__.__name__ for tool in self.tools]
        )
        
        logger.info(f"Initialized {role.value} agent with ID: {agent_id}")

    async def process_message(self, message: Message) -> Message:
        """
        Process an incoming message and generate a response.
        
        Args:
            message: The incoming message to process
            
        Returns:
            Response message from the agent
        """
        try:
            # Security validation for sensitive content
            if message.sensitive:
                is_valid = await self.security_validator.validate_privileged_content(
                    message.content
                )
                if not is_valid:
                    return self._create_error_message(
                        "Security validation failed for privileged content"
                    )
            
            # Add message to memory
            self._add_to_memory(message)
            
            # Update agent status
            self.state.status = "working"
            self.state.current_task = message.content[:100] + "..."
            
            # Process the message based on type
            if message.type == MessageType.TASK:
                response = await self._handle_task(message)
            elif message.type == MessageType.OBSERVATION:
                response = await self._handle_observation(message)
            else:
                response = await self._handle_generic_message(message)
                
            # Add response to memory
            self._add_to_memory(response)
            
            # Update status
            self.state.status = "completed"
            self.state.current_task = None
            
            logger.info(f"Agent {self.agent_id} completed message processing")
            return response
            
        except Exception as e:
            logger.error(f"Error processing message in agent {self.agent_id}: {e}")
            self.state.status = "error"
            return self._create_error_message(str(e))

    @abstractmethod
    async def _handle_task(self, message: Message) -> Message:
        """Handle task-specific processing. Must be implemented by subclasses."""
        pass

    async def _handle_observation(self, message: Message) -> Message:
        """Handle observation messages (default implementation)."""
        return Message(
            sender=self.agent_id,
            receiver=message.sender,
            type=MessageType.RESPONSE,
            content=f"Observation acknowledged: {message.content}",
            metadata={"original_message_id": message.id}
        )

    async def _handle_generic_message(self, message: Message) -> Message:
        """Handle generic messages."""
        prompt = self._build_prompt(message)
        response_content = await self.llm_provider.generate_response(prompt)
        
        return Message(
            sender=self.agent_id,
            receiver=message.sender,
            type=MessageType.RESPONSE,
            content=response_content,
            metadata={"original_message_id": message.id}
        )

    def _build_prompt(self, message: Message) -> str:
        """Build a prompt for the LLM based on the message and agent context."""
        system_prompt = self._get_system_prompt()
        context = self._get_context()
        
        prompt = f"""
{system_prompt}

Context from previous interactions:
{context}

Current message: {message.content}

Please provide a response as a {self.role.value} agent in the legal domain.
Consider legal accuracy, ethical implications, and professional standards.
"""
        return prompt

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent type. Must be implemented by subclasses."""
        pass

    def _get_context(self) -> str:
        """Get relevant context from agent memory."""
        recent_messages = self.state.memory[-5:]  # Last 5 messages
        context_parts = []
        
        for msg in recent_messages:
            context_parts.append(f"{msg.sender}: {msg.content[:200]}...")
            
        return "\\n".join(context_parts)

    def _add_to_memory(self, message: Message) -> None:
        """Add a message to agent memory with size management."""
        self.state.memory.append(message)
        
        # Manage memory size
        if len(self.state.memory) > self.memory_limit:
            # Keep the most recent messages and important ones
            important_messages = [
                msg for msg in self.state.memory 
                if msg.type in [MessageType.TASK, MessageType.FINAL_ANSWER]
            ]
            recent_messages = self.state.memory[-50:]  # Keep last 50 messages
            
            # Combine and deduplicate
            combined = important_messages + recent_messages
            seen_ids = set()
            self.state.memory = [
                msg for msg in combined 
                if msg.id not in seen_ids and not seen_ids.add(msg.id)
            ]

    def _create_error_message(self, error: str) -> Message:
        """Create an error response message."""
        return Message(
            sender=self.agent_id,
            type=MessageType.RESPONSE,
            content=f"Error: {error}",
            metadata={"error": True}
        )

    async def use_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Use a tool and return the result."""
        try:
            tool = next((t for t in self.tools if t.__class__.__name__ == tool_name), None)
            if not tool:
                raise ValueError(f"Tool {tool_name} not available")
                
            result = await tool.execute(**kwargs)
            
            # Log tool usage for audit trail
            self.state.reasoning_chain.append({
                "step": "tool_usage",
                "tool": tool_name,
                "input": kwargs,
                "output": result,
                "timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error using tool {tool_name}: {e}")
            raise

    def add_reasoning_step(self, step_type: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """Add a step to the reasoning chain for transparency."""
        self.state.reasoning_chain.append({
            "step_type": step_type,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })

    def get_reasoning_chain(self) -> List[Dict[str, Any]]:
        """Get the complete reasoning chain for this agent."""
        return self.state.reasoning_chain.copy()

    def clear_reasoning_chain(self) -> None:
        """Clear the reasoning chain (useful for new tasks)."""
        self.state.reasoning_chain = []

    async def stream_response(self, message: Message) -> AsyncGenerator[str, None]:
        """Stream response generation for real-time feedback."""
        try:
            prompt = self._build_prompt(message)
            async for chunk in self.llm_provider.stream_response(prompt):
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            yield f"Error: {e}"

    def get_state(self) -> AgentState:
        """Get current agent state."""
        return self.state.copy()

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of agent memory for debugging/monitoring."""
        return {
            "total_messages": len(self.state.memory),
            "message_types": {
                msg_type.value: sum(1 for msg in self.state.memory if msg.type == msg_type)
                for msg_type in MessageType
            },
            "sensitive_messages": sum(1 for msg in self.state.memory if msg.sensitive),
            "reasoning_steps": len(self.state.reasoning_chain)
        }
'''

with open("legal_agent_orchestrator/agents/base_agent.py", "w") as f:
    f.write(base_agent_content)

print("Base agent class created!")