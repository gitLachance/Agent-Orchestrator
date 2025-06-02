# Create the research agent
research_agent_content = '''"""
Legal Research Agent specialized in legal research, case law analysis, and statute lookup.
Integrates with legal databases and document retrieval systems.
"""

from typing import Dict, List, Any, Optional
import asyncio
import re

from .base_agent import BaseAgent, Message, MessageType, AgentRole
from ..llm_providers.base_provider import BaseLLMProvider
from ..rag.retriever import DocumentRetriever
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ResearchAgent(BaseAgent):
    """
    Specialized agent for legal research tasks including:
    - Case law research and analysis
    - Statute and regulation lookup
    - Legal precedent identification
    - Citation verification and validation
    - Legal database querying
    """

    def __init__(
        self,
        agent_id: str,
        llm_provider: BaseLLMProvider,
        document_retriever: DocumentRetriever,
        tools: Optional[List[Any]] = None,
        memory_limit: int = 150
    ):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.RESEARCH,
            llm_provider=llm_provider,
            tools=tools,
            memory_limit=memory_limit
        )
        self.document_retriever = document_retriever
        
        # Research-specific capabilities
        self.research_capabilities = [
            "case_law_search",
            "statute_lookup",
            "regulation_search", 
            "precedent_analysis",
            "citation_verification",
            "legal_trend_analysis"
        ]
        
        logger.info(f"Research Agent {agent_id} initialized with capabilities: {self.research_capabilities}")

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the research agent."""
        return """
You are a specialized Legal Research Agent with expertise in:

CORE COMPETENCIES:
- Legal research methodology and strategy
- Case law analysis and precedent identification  
- Statutory interpretation and regulatory research
- Citation format validation (Blue Book, ALWD)
- Legal database search optimization
- Jurisdictional analysis and forum shopping considerations

RESEARCH APPROACH:
1. Begin with broad research to understand the legal landscape
2. Identify relevant jurisdictions and applicable law
3. Narrow focus to most relevant cases and statutes
4. Analyze holdings, reasoning, and potential distinguishing factors
5. Synthesize findings into actionable research memos

ETHICAL CONSIDERATIONS:
- Verify accuracy of all citations and legal references
- Note any conflicts of law or jurisdictional issues
- Identify potential ethical concerns or malpractice risks
- Flag any time-sensitive deadlines or statute of limitations

OUTPUT FORMAT:
- Provide clear, well-organized research findings
- Include proper legal citations
- Distinguish between binding and persuasive authority
- Note any gaps in research or areas requiring further investigation
- Maintain objectivity and present both favorable and adverse authority

Always prioritize accuracy, thoroughness, and professional legal standards.
"""

    async def _handle_task(self, message: Message) -> Message:
        """Handle research-specific tasks."""
        try:
            metadata = message.metadata or {}
            action = metadata.get("action", "general_research")
            context = metadata.get("context", {})
            
            self.add_reasoning_step("task_received", f"Processing research task: {action}")
            
            if action == "search_case_law":
                response_content = await self._search_case_law(message.content, context)
            elif action == "search_statutes":
                response_content = await self._search_statutes(message.content, context)
            elif action == "analyze_precedents":
                response_content = await self._analyze_precedents(message.content, context)
            elif action == "verify_citations":
                response_content = await self._verify_citations(message.content, context)
            elif action == "research_legal_standards":
                response_content = await self._research_legal_standards(message.content, context)
            else:
                response_content = await self._conduct_general_research(message.content, context)
            
            return Message(
                sender=self.agent_id,
                receiver=message.sender,
                type=MessageType.RESPONSE,
                content=response_content,
                metadata={
                    "original_message_id": message.id,
                    "research_type": action,
                    "reasoning_chain": self.get_reasoning_chain()[-5:]  # Last 5 steps
                }
            )
            
        except Exception as e:
            logger.error(f"Error in research task handling: {e}")
            return self._create_error_message(f"Research task failed: {e}")

    async def _search_case_law(self, query: str, context: Dict[str, Any]) -> str:
        """Search for relevant case law."""
        self.add_reasoning_step("case_law_search", f"Searching case law for: {query}")
        
        # Extract jurisdiction and case type from context
        jurisdiction = context.get("jurisdiction", "federal")
        case_type = context.get("case_type", "general")
        
        # Use document retriever to find relevant cases
        search_results = await self.document_retriever.search_documents(
            query=f"case law {query} {jurisdiction}",
            doc_type="case_law",
            limit=10
        )
        
        # Construct search prompt for LLM analysis
        prompt = f"""
Analyze the following case law search results for the query: "{query}"
Jurisdiction: {jurisdiction}
Case Type: {case_type}

Search Results:
{self._format_search_results(search_results)}

Provide a comprehensive case law analysis including:
1. Most relevant cases with proper citations
2. Key holdings and legal principles
3. Binding vs. persuasive authority analysis
4. Factual distinctions and analogies
5. Evolution of the law in this area
6. Recommended cases for further review

Format your response as a professional legal research memo.
"""

        response = await self.llm_provider.generate_response(prompt)
        self.add_reasoning_step("case_analysis", "Completed case law analysis")
        
        return response

    async def _search_statutes(self, query: str, context: Dict[str, Any]) -> str:
        """Search for relevant statutes and regulations."""
        self.add_reasoning_step("statute_search", f"Searching statutes for: {query}")
        
        jurisdiction = context.get("jurisdiction", "federal")
        
        # Search for statutory materials
        search_results = await self.document_retriever.search_documents(
            query=f"statute regulation {query} {jurisdiction}",
            doc_type="statute",
            limit=8
        )
        
        prompt = f"""
Analyze the following statutory and regulatory materials for: "{query}"
Jurisdiction: {jurisdiction}

Search Results:
{self._format_search_results(search_results)}

Provide a comprehensive statutory analysis including:
1. Relevant statutes with proper citations and section numbers
2. Plain language interpretation of key provisions
3. Regulatory framework and implementing regulations
4. Recent amendments or proposed changes
5. Enforcement mechanisms and penalties
6. Interaction with other statutory schemes
7. Constitutional considerations if applicable

Organize your response clearly with headings and subheadings.
"""

        response = await self.llm_provider.generate_response(prompt)
        self.add_reasoning_step("statute_analysis", "Completed statutory analysis")
        
        return response

    async def _analyze_precedents(self, query: str, context: Dict[str, Any]) -> str:
        """Analyze legal precedents and their application."""
        self.add_reasoning_step("precedent_analysis", f"Analyzing precedents for: {query}")
        
        # Get previous research results from context
        case_law_result = context.get("search_case_law_result", "")
        statute_result = context.get("search_statutes_result", "")
        
        prompt = f"""
Based on the following legal research, analyze the precedential value and application:

Query: {query}

Case Law Research:
{case_law_result}

Statutory Research:
{statute_result}

Provide a detailed precedent analysis including:
1. Hierarchy of authority (binding vs. persuasive)
2. Precedential strength of key cases
3. Trend analysis - how the law has evolved
4. Circuit splits or conflicting authority
5. Predictive analysis for future application
6. Strategic considerations for litigation
7. Potential arguments for distinguishing adverse precedent

Focus on practical application and strategic legal advice.
"""

        response = await self.llm_provider.generate_response(prompt)
        self.add_reasoning_step("precedent_synthesis", "Completed precedent analysis")
        
        return response

    async def _verify_citations(self, citations: str, context: Dict[str, Any]) -> str:
        """Verify legal citations for accuracy and format."""
        self.add_reasoning_step("citation_verification", f"Verifying citations: {citations}")
        
        # Extract individual citations using regex
        citation_pattern = r'\\b\\d+\\s+[A-Za-z.]+\\s+\\d+.*?\\([^)]+\\)|\\b[A-Za-z.]+\\s+§\\s*\\d+[\\w.-]*'
        found_citations = re.findall(citation_pattern, citations)
        
        prompt = f"""
Verify the following legal citations for accuracy, proper format, and validity:

Citations to verify:
{citations}

Extracted citations:
{chr(10).join(found_citations)}

For each citation, provide:
1. Format verification (Blue Book/ALWD compliance)
2. Accuracy check - does the citation exist?
3. Currency - is this the current version?
4. Parallel citations if available
5. Corrections needed if any
6. Alternative citation formats

Also check for:
- Proper use of "see" and "see also" signals
- Appropriate parenthetical information
- Proper ordering of citations
- Missing pincites where needed
"""

        response = await self.llm_provider.generate_response(prompt)
        self.add_reasoning_step("citation_checked", "Completed citation verification")
        
        return response

    async def _research_legal_standards(self, query: str, context: Dict[str, Any]) -> str:
        """Research applicable legal standards and tests."""
        self.add_reasoning_step("standards_research", f"Researching legal standards for: {query}")
        
        contract_text = context.get("contract_text", "")
        analysis_type = context.get("analysis_type", "general")
        
        prompt = f"""
Research the applicable legal standards and tests for: "{query}"
Analysis Type: {analysis_type}

{f"Contract/Document Context: {contract_text[:1000]}..." if contract_text else ""}

Identify and explain:
1. Governing legal standard or test (strict scrutiny, rational basis, etc.)
2. Elements that must be proven
3. Burden of proof and which party bears it
4. Relevant factors courts consider
5. Common defenses or exceptions
6. Remedies available
7. Procedural requirements and deadlines
8. Jurisdictional variations in approach

Provide specific citations to cases establishing these standards.
"""

        response = await self.llm_provider.generate_response(prompt)
        self.add_reasoning_step("standards_identified", "Completed legal standards research")
        
        return response

    async def _conduct_general_research(self, query: str, context: Dict[str, Any]) -> str:
        """Conduct general legal research on a topic."""
        self.add_reasoning_step("general_research", f"Conducting general research: {query}")
        
        # Perform comprehensive search across all document types
        search_results = await self.document_retriever.search_documents(
            query=query,
            limit=15
        )
        
        prompt = f"""
Conduct comprehensive legal research on: "{query}"

Context: {context}

Available Research Materials:
{self._format_search_results(search_results)}

Provide a thorough legal research report including:

EXECUTIVE SUMMARY:
- Key legal issues identified
- Primary authorities governing this area
- Critical factual and legal distinctions

SUBSTANTIVE ANALYSIS:
1. Governing Law and Jurisdiction
2. Primary Authority (statutes, regulations, case law)
3. Secondary Authority (treatises, law review articles)
4. Recent Developments and Trends
5. Practical Considerations and Strategy

RESEARCH CONCLUSIONS:
- Strength of legal position
- Areas requiring additional research
- Potential challenges or obstacles
- Recommended next steps

Include proper legal citations throughout.
"""

        response = await self.llm_provider.generate_response(prompt)
        self.add_reasoning_step("research_completed", "Completed general legal research")
        
        return response

    def _format_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results for LLM consumption."""
        if not search_results:
            return "No relevant documents found in the database."
        
        formatted_results = []
        for i, result in enumerate(search_results[:10], 1):  # Limit to top 10
            formatted_results.append(f"""
Document {i}:
Title: {result.get('title', 'Unknown')}
Type: {result.get('doc_type', 'Unknown')}
Relevance Score: {result.get('score', 'N/A')}
Content Excerpt: {result.get('content', '')[:500]}...
Citation: {result.get('citation', 'No citation available')}
---
""")
        
        return "\\n".join(formatted_results)

    async def research_by_topic(self, topic: str, jurisdiction: str = "federal") -> Dict[str, Any]:
        """Convenience method for topic-based research."""
        message = Message(
            sender="user",
            receiver=self.agent_id,
            type=MessageType.TASK,
            content=topic,
            metadata={
                "action": "general_research",
                "context": {"jurisdiction": jurisdiction}
            }
        )
        
        response = await self.process_message(message)
        
        return {
            "topic": topic,
            "jurisdiction": jurisdiction,
            "research_findings": response.content,
            "reasoning_chain": self.get_reasoning_chain(),
            "agent_id": self.agent_id
        }

    async def quick_cite_check(self, citations: List[str]) -> Dict[str, Any]:
        """Quick citation verification method."""
        citations_text = "\\n".join(citations)
        
        message = Message(
            sender="user",
            receiver=self.agent_id,
            type=MessageType.TASK,
            content=citations_text,
            metadata={"action": "verify_citations"}
        )
        
        response = await self.process_message(message)
        
        return {
            "original_citations": citations,
            "verification_results": response.content,
            "agent_id": self.agent_id
        }
'''

with open("legal_agent_orchestrator/agents/research_agent.py", "w") as f:
    f.write(research_agent_content)

print("Research agent created!")