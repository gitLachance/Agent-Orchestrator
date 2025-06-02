# Create the reasoning agent
reasoning_agent_content = '''"""
Legal Reasoning Agent specialized in logical analysis, argument construction, 
and chain-of-thought reasoning for legal problems.
"""

from typing import Dict, List, Any, Optional, Tuple
import asyncio
import json

from .base_agent import BaseAgent, Message, MessageType, AgentRole
from ..llm_providers.base_provider import BaseLLMProvider
from ..reasoning.chain_of_thought import ChainOfThoughtReasoner
from ..reasoning.legal_reasoning import LegalReasoningEngine
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ReasoningAgent(BaseAgent):
    """
    Specialized agent for legal reasoning tasks including:
    - Chain-of-thought legal analysis
    - Argument construction and evaluation
    - Precedent matching and analogical reasoning
    - Risk assessment and strategic analysis
    - Logical consistency checking
    - Multi-step problem solving
    """

    def __init__(
        self,
        agent_id: str,
        llm_provider: BaseLLMProvider,
        tools: Optional[List[Any]] = None,
        memory_limit: int = 200
    ):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.REASONING,
            llm_provider=llm_provider,
            tools=tools,
            memory_limit=memory_limit
        )
        
        # Initialize reasoning engines
        self.cot_reasoner = ChainOfThoughtReasoner(llm_provider)
        self.legal_reasoner = LegalReasoningEngine(llm_provider)
        
        # Reasoning capabilities
        self.reasoning_capabilities = [
            "chain_of_thought_analysis",
            "argument_construction", 
            "precedent_matching",
            "risk_assessment",
            "logical_consistency_check",
            "analogical_reasoning",
            "strategic_analysis",
            "synthesis_and_conclusion"
        ]
        
        logger.info(f"Reasoning Agent {agent_id} initialized with capabilities: {self.reasoning_capabilities}")

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the reasoning agent."""
        return """
You are a specialized Legal Reasoning Agent with expertise in:

CORE COMPETENCIES:
- Advanced legal reasoning and logical analysis
- Chain-of-thought problem decomposition
- Argument construction and evaluation
- Analogical reasoning with legal precedents
- Risk assessment and strategic planning
- Synthesis of complex legal issues

REASONING METHODOLOGY:
1. Break down complex legal problems into components
2. Identify all relevant legal principles and rules
3. Apply logical reasoning frameworks (deductive, inductive, analogical)
4. Consider multiple perspectives and counterarguments
5. Evaluate strength of arguments and evidence
6. Synthesize conclusions with clear reasoning chains

ANALYTICAL APPROACH:
- Use structured reasoning patterns (IRAC, CRAC, etc.)
- Consider both procedural and substantive legal issues
- Evaluate factual and legal sufficiency
- Identify assumptions and test their validity
- Consider alternative interpretations and outcomes
- Assess risks, benefits, and strategic implications

QUALITY STANDARDS:
- Maintain logical consistency throughout analysis
- Provide clear reasoning chains that can be followed
- Distinguish between strong and weak arguments
- Acknowledge uncertainties and limitations
- Consider ethical implications and professional responsibilities
- Ensure conclusions are well-supported by evidence and reasoning

Always strive for clarity, thoroughness, and intellectual rigor in your analysis.
"""

    async def _handle_task(self, message: Message) -> Message:
        """Handle reasoning-specific tasks."""
        try:
            metadata = message.metadata or {}
            action = metadata.get("action", "general_reasoning")
            context = metadata.get("context", {})
            
            self.add_reasoning_step("task_received", f"Processing reasoning task: {action}")
            
            if action == "analyze_precedents":
                response_content = await self._analyze_precedents(message.content, context)
            elif action == "construct_arguments":
                response_content = await self._construct_arguments(message.content, context)
            elif action == "assess_risks":
                response_content = await self._assess_risks(message.content, context)
            elif action == "synthesize_findings":
                response_content = await self._synthesize_findings(message.content, context)
            elif action == "recommend_modifications":
                response_content = await self._recommend_modifications(message.content, context)
            elif action == "analyze_legal_issues":
                response_content = await self._analyze_legal_issues(message.content, context)
            else:
                response_content = await self._conduct_general_reasoning(message.content, context)
            
            return Message(
                sender=self.agent_id,
                receiver=message.sender,
                type=MessageType.RESPONSE,
                content=response_content,
                metadata={
                    "original_message_id": message.id,
                    "reasoning_type": action,
                    "reasoning_chain": self.get_reasoning_chain()[-5:]
                }
            )
            
        except Exception as e:
            logger.error(f"Error in reasoning task handling: {e}")
            return self._create_error_message(f"Reasoning task failed: {e}")

    async def _analyze_precedents(self, query: str, context: Dict[str, Any]) -> str:
        """Analyze precedents and their application to current situation."""
        self.add_reasoning_step("precedent_analysis_start", f"Starting precedent analysis for: {query}")
        
        # Get research results from context
        case_law_results = context.get("search_case_law_result", "")
        statute_results = context.get("search_statutes_result", "")
        
        # Use chain-of-thought reasoning for precedent analysis
        reasoning_prompt = f"""
Analyze the legal precedents for the following situation: {query}

Available Case Law Research:
{case_law_results}

Available Statutory Research:
{statute_results}

Use chain-of-thought reasoning to analyze these precedents systematically.
"""
        
        cot_analysis = await self.cot_reasoner.reason_step_by_step(reasoning_prompt)
        self.add_reasoning_step("cot_precedent_analysis", "Completed chain-of-thought precedent analysis")
        
        # Perform legal reasoning analysis
        legal_analysis = await self.legal_reasoner.analyze_precedents(
            query=query,
            precedents=case_law_results,
            statutes=statute_results
        )
        self.add_reasoning_step("legal_precedent_analysis", "Completed formal legal precedent analysis")
        
        # Synthesize both analyses
        synthesis_prompt = f"""
Synthesize the following precedent analyses into a comprehensive legal analysis:

Chain-of-Thought Analysis:
{cot_analysis}

Formal Legal Analysis:
{legal_analysis}

Provide a structured analysis including:
1. PRECEDENT HIERARCHY: Binding vs. persuasive authority
2. FACTUAL ANALOGIES: How the precedents apply to current facts
3. LEGAL DISTINCTIONS: Material differences that affect application
4. STRENGTH ASSESSMENT: How strongly precedents support our position
5. COUNTERARGUMENT ANALYSIS: Potential challenges to precedent application
6. STRATEGIC RECOMMENDATIONS: How to use or distinguish precedents

Format as a professional legal memorandum section.
"""
        
        final_analysis = await self.llm_provider.generate_response(synthesis_prompt)
        self.add_reasoning_step("precedent_synthesis", "Completed precedent analysis synthesis")
        
        return final_analysis

    async def _construct_arguments(self, query: str, context: Dict[str, Any]) -> str:
        """Construct legal arguments based on research and analysis."""
        self.add_reasoning_step("argument_construction_start", f"Constructing arguments for: {query}")
        
        research_basis = context.get("research_legal_basis_result", "")
        case_facts = context.get("case_facts", "")
        legal_issue = context.get("legal_issue", "")
        
        # Use legal reasoning engine to construct arguments
        arguments = await self.legal_reasoner.construct_arguments(
            legal_issue=legal_issue or query,
            facts=case_facts,
            legal_research=research_basis
        )
        self.add_reasoning_step("initial_arguments", "Constructed initial arguments")
        
        # Enhance with chain-of-thought reasoning
        enhancement_prompt = f"""
Enhance and refine the following legal arguments using systematic reasoning:

Legal Issue: {legal_issue or query}
Facts: {case_facts}

Initial Arguments:
{arguments}

Enhance these arguments by:
1. Strengthening logical connections
2. Adding supporting citations and evidence
3. Anticipating and addressing counterarguments
4. Improving persuasive structure and flow
5. Ensuring completeness and thoroughness

Use clear reasoning chains to show how facts support legal conclusions.
"""
        
        enhanced_arguments = await self.cot_reasoner.reason_step_by_step(enhancement_prompt)
        self.add_reasoning_step("enhanced_arguments", "Enhanced arguments with chain-of-thought reasoning")
        
        # Final argument structuring
        final_prompt = f"""
Structure the following enhanced arguments into a compelling legal argument:

{enhanced_arguments}

Organize using the following structure:
I. EXECUTIVE SUMMARY
   - Core legal argument in 2-3 sentences
   
II. FACTUAL FOUNDATION
   - Key facts supporting our position
   - Factual disputes and their resolution
   
III. LEGAL ARGUMENT
   A. Primary Legal Standard
   B. Application to Facts
   C. Supporting Precedents
   D. Distinguishing Adverse Authority
   
IV. POLICY CONSIDERATIONS
   - Why our interpretation serves legal policy
   
V. CONCLUSION
   - Clear statement of relief sought

Ensure each section builds logically on the previous one.
"""
        
        structured_arguments = await self.llm_provider.generate_response(final_prompt)
        self.add_reasoning_step("structured_arguments", "Completed argument structuring")
        
        return structured_arguments

    async def _assess_risks(self, query: str, context: Dict[str, Any]) -> str:
        """Assess legal and strategic risks."""
        self.add_reasoning_step("risk_assessment_start", f"Assessing risks for: {query}")
        
        contract_text = context.get("contract_text", "")
        analysis_type = context.get("analysis_type", "")
        legal_standards = context.get("research_legal_standards_result", "")
        
        # Comprehensive risk assessment prompt
        risk_prompt = f"""
Conduct a comprehensive risk assessment for: {query}

Context:
- Analysis Type: {analysis_type}
- Legal Standards: {legal_standards}
- Contract/Document: {contract_text[:1000] if contract_text else "N/A"}

Assess the following risk categories:

LEGAL RISKS:
1. Liability exposure and potential damages
2. Regulatory compliance issues
3. Contractual breach risks
4. Litigation probability and costs
5. Enforcement challenges

BUSINESS RISKS:
1. Financial impact and cost implications
2. Operational disruptions
3. Reputational damage
4. Relationship impacts (client, partner, etc.)
5. Market/competitive disadvantages

STRATEGIC RISKS:
1. Precedent-setting implications
2. Future constraint on business operations
3. Regulatory attention or scrutiny
4. Public relations concerns

For each risk:
- Assess probability (Low/Medium/High)
- Evaluate potential impact (Minor/Moderate/Severe)
- Suggest mitigation strategies
- Identify early warning indicators

Provide risk matrix and prioritized action items.
"""
        
        risk_analysis = await self.cot_reasoner.reason_step_by_step(risk_prompt)
        self.add_reasoning_step("risk_analysis_complete", "Completed comprehensive risk assessment")
        
        return risk_analysis

    async def _synthesize_findings(self, query: str, context: Dict[str, Any]) -> str:
        """Synthesize research findings into coherent conclusions."""
        self.add_reasoning_step("synthesis_start", f"Synthesizing findings for: {query}")
        
        # Gather all available context
        available_results = {}
        for key, value in context.items():
            if key.endswith("_result") and value:
                available_results[key] = value
        
        synthesis_prompt = f"""
Synthesize the following research findings into a coherent legal analysis:

Query: {query}

Research Findings:
{self._format_context_results(available_results)}

Provide a comprehensive synthesis that:

1. EXECUTIVE SUMMARY
   - Key legal conclusions
   - Primary recommendations
   - Critical decision points

2. LEGAL FRAMEWORK
   - Governing law and jurisdiction
   - Applicable legal standards
   - Controlling precedents

3. ANALYSIS AND APPLICATION
   - How law applies to specific facts
   - Strength of legal position
   - Potential challenges or weaknesses

4. STRATEGIC CONSIDERATIONS
   - Risk assessment summary
   - Alternative approaches
   - Timing and procedural considerations

5. RECOMMENDATIONS
   - Immediate action items
   - Long-term strategy
   - Areas requiring additional research

6. CONCLUSION
   - Bottom-line assessment
   - Confidence level in conclusions
   - Key uncertainties or contingencies

Ensure all conclusions are well-supported by the research findings.
"""
        
        synthesis = await self.llm_provider.generate_response(synthesis_prompt)
        self.add_reasoning_step("synthesis_complete", "Completed findings synthesis")
        
        return synthesis

    async def _recommend_modifications(self, query: str, context: Dict[str, Any]) -> str:
        """Recommend modifications based on risk assessment."""
        self.add_reasoning_step("modifications_start", f"Recommending modifications for: {query}")
        
        risk_assessment = context.get("assess_risks_result", "")
        contract_text = context.get("contract_text", "")
        
        modification_prompt = f"""
Based on the following risk assessment, recommend specific modifications:

Query: {query}
Risk Assessment:
{risk_assessment}

Contract/Document Text:
{contract_text[:2000] if contract_text else "N/A"}

Provide detailed modification recommendations including:

1. PRIORITY MODIFICATIONS (High Risk Items)
   - Specific language changes
   - New clauses to add
   - Sections to delete or revise
   - Rationale for each change

2. RECOMMENDED MODIFICATIONS (Medium Risk Items)
   - Suggested improvements
   - Alternative language options
   - Enhanced protections

3. OPTIONAL MODIFICATIONS (Low Risk Items)
   - Nice-to-have improvements
   - Future-proofing considerations
   - Best practice enhancements

For each modification:
- Provide specific draft language
- Explain the legal and business rationale
- Note any trade-offs or negotiation considerations
- Assess likelihood of acceptance by other party

Include implementation strategy and negotiation approach.
"""
        
        modifications = await self.cot_reasoner.reason_step_by_step(modification_prompt)
        self.add_reasoning_step("modifications_complete", "Completed modification recommendations")
        
        return modifications

    async def _analyze_legal_issues(self, query: str, context: Dict[str, Any]) -> str:
        """Analyze legal issues in depth."""
        self.add_reasoning_step("legal_analysis_start", f"Analyzing legal issues: {query}")
        
        relevant_law = context.get("gather_relevant_law_result", "")
        
        analysis_prompt = f"""
Conduct an in-depth analysis of the legal issues presented:

Legal Question: {query}

Relevant Law and Research:
{relevant_law}

Provide a comprehensive legal analysis using the IRAC method:

I. ISSUE
   - Precise statement of legal issues
   - Sub-issues and related questions
   - Jurisdictional and procedural considerations

R. RULE
   - Controlling statutes and regulations
   - Binding case law and precedents
   - Legal standards and tests
   - Elements that must be proven

A. APPLICATION
   - How the law applies to specific facts
   - Analogies to and distinctions from precedents
   - Policy arguments and considerations
   - Strengths and weaknesses of different positions

C. CONCLUSION
   - Likely outcome based on analysis
   - Confidence level and reasoning
   - Alternative scenarios and contingencies
   - Recommended course of action

Include detailed reasoning chains showing how you reached each conclusion.
"""
        
        legal_analysis = await self.legal_reasoner.analyze_legal_issues(
            issues=query,
            relevant_law=relevant_law
        )
        self.add_reasoning_step("formal_legal_analysis", "Completed formal legal analysis")
        
        # Enhance with chain-of-thought
        enhanced_analysis = await self.cot_reasoner.reason_step_by_step(analysis_prompt)
        self.add_reasoning_step("enhanced_legal_analysis", "Enhanced with chain-of-thought reasoning")
        
        # Combine both analyses
        final_prompt = f"""
Combine and synthesize these two legal analyses into a comprehensive final analysis:

Formal Legal Analysis:
{legal_analysis}

Enhanced Chain-of-Thought Analysis:
{enhanced_analysis}

Create a unified, comprehensive legal analysis that incorporates the best insights from both approaches.
"""
        
        final_analysis = await self.llm_provider.generate_response(final_prompt)
        self.add_reasoning_step("final_legal_analysis", "Completed comprehensive legal analysis")
        
        return final_analysis

    async def _conduct_general_reasoning(self, query: str, context: Dict[str, Any]) -> str:
        """Conduct general legal reasoning on a problem."""
        self.add_reasoning_step("general_reasoning_start", f"General reasoning for: {query}")
        
        # Use chain-of-thought for general reasoning
        reasoning_result = await self.cot_reasoner.reason_step_by_step(
            f"Provide thorough legal reasoning for: {query}\\n\\nContext: {context}"
        )
        
        self.add_reasoning_step("general_reasoning_complete", "Completed general reasoning")
        return reasoning_result

    def _format_context_results(self, results: Dict[str, Any]) -> str:
        """Format context results for synthesis."""
        formatted = []
        for key, value in results.items():
            formatted.append(f"\\n{key.replace('_result', '').replace('_', ' ').title()}:\\n{value}\\n{'='*50}")
        return "\\n".join(formatted)

    async def reason_about_case(
        self,
        case_facts: str,
        legal_questions: List[str],
        jurisdiction: str = "federal"
    ) -> Dict[str, Any]:
        """Convenience method for comprehensive case reasoning."""
        reasoning_results = {}
        
        for i, question in enumerate(legal_questions):
            message = Message(
                sender="user",
                receiver=self.agent_id,
                type=MessageType.TASK,
                content=question,
                metadata={
                    "action": "analyze_legal_issues",
                    "context": {
                        "case_facts": case_facts,
                        "jurisdiction": jurisdiction,
                        "question_number": i + 1
                    }
                }
            )
            
            response = await self.process_message(message)
            reasoning_results[f"question_{i+1}"] = {
                "question": question,
                "analysis": response.content,
                "reasoning_chain": self.get_reasoning_chain()[-10:]
            }
        
        return {
            "case_facts": case_facts,
            "jurisdiction": jurisdiction,
            "legal_questions": legal_questions,
            "reasoning_results": reasoning_results,
            "agent_id": self.agent_id
        }
'''

with open("legal_agent_orchestrator/agents/reasoning_agent.py", "w") as f:
    f.write(reasoning_agent_content)

print("Reasoning agent created!")