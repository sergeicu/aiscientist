# Multi-Agent System Design Guide

## Overview

The multi-agent system is the intelligence layer of the AI Scientist platform. It coordinates specialized agents to perform complex research analysis tasks.

## Agent Ecosystem

```
                            ┌─────────────────────┐
                            │  Coordinator Agent  │
                            │                     │
                            │  - Task routing     │
                            │  - Workflow mgmt    │
                            │  - Result synthesis │
                            └──────────┬──────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    v                  v                  v
         ┌──────────────────┐ ┌──────────────┐ ┌────────────────┐
         │  Investment      │ │  Prior Art   │ │  Hypothesis    │
         │  Evaluator       │ │  Researcher  │ │  Generator     │
         └──────────────────┘ └──────────────┘ └────────────────┘
                    │                  │                  │
                    └──────────────────┼──────────────────┘
                                       │
                              ┌────────┴─────────┐
                              │                  │
                              v                  v
                    ┌──────────────┐   ┌──────────────────┐
                    │  Shared      │   │  Report          │
                    │  Memory      │   │  Generator       │
                    └──────────────┘   └──────────────────┘
```

---

## Agent Specifications

### 1. Coordinator Agent

**Role**: Orchestrate workflows, route tasks, synthesize results

**Capabilities**:
- Parse user queries and identify required agents
- Create execution plans
- Manage agent communication
- Synthesize findings from multiple agents
- Generate final reports

**Prompt Template**:

```yaml
# prompts/coordinator.yaml

system_prompt: |
  You are the Coordinator Agent, responsible for orchestrating multi-agent
  research workflows. You manage specialized agents (Investment Evaluator,
  Prior Art Researcher, Hypothesis Generator) to complete complex analysis tasks.

  Your responsibilities:
  1. Understand user requests
  2. Break down tasks into subtasks
  3. Assign subtasks to appropriate agents
  4. Collect and synthesize agent findings
  5. Generate comprehensive reports

  Available agents:
  - Investment Evaluator: Assess commercial potential, market analysis
  - Prior Art Researcher: Literature review, competitive landscape
  - Hypothesis Generator: Identify research gaps, propose directions

  When coordinating, always:
  - Provide clear instructions to agents
  - Validate agent outputs
  - Ask for clarification if needed
  - Synthesize findings coherently

task_routing_template: |
  User Request: {user_query}

  Analyze this request and create an execution plan.

  Output format:
  ```json
  {
    "task_type": "investment_evaluation | research_discovery | comprehensive_analysis",
    "workflow": [
      {
        "step": 1,
        "agent": "agent_name",
        "task": "specific task description",
        "inputs": {...},
        "expected_output": "what this step should produce"
      }
    ],
    "synthesis_plan": "how to combine agent outputs"
  }
  ```

synthesis_template: |
  You have received findings from multiple agents:

  Investment Evaluator:
  {investment_findings}

  Prior Art Researcher:
  {prior_art_findings}

  Hypothesis Generator:
  {hypothesis_findings}

  Synthesize these findings into a coherent report that:
  1. Summarizes key insights from each agent
  2. Identifies agreement and disagreement between agents
  3. Provides overall recommendation
  4. Highlights areas needing further investigation

  Report structure:
  ## Executive Summary
  ## Key Findings
  ### Commercial Assessment
  ### Competitive Landscape
  ### Research Opportunities
  ## Recommendations
  ## Next Steps
```

**Implementation**:

```python
# src/agents/coordinator.py

from typing import Dict, List, Optional
from loguru import logger
from .base_agent import BaseAgent, AgentMessage
from .investment_evaluator import InvestmentEvaluatorAgent
from .prior_art_researcher import PriorArtResearcherAgent
from .hypothesis_generator import HypothesisGeneratorAgent

class CoordinatorAgent(BaseAgent):
    """Orchestrate multi-agent workflows."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize specialized agents
        self.agents = {
            "investment_evaluator": InvestmentEvaluatorAgent(
                name="investment_evaluator",
                llm_client=self.llm,
                memory=self.memory
            ),
            "prior_art_researcher": PriorArtResearcherAgent(
                name="prior_art_researcher",
                llm_client=self.llm,
                memory=self.memory
            ),
            "hypothesis_generator": HypothesisGeneratorAgent(
                name="hypothesis_generator",
                llm_client=self.llm,
                memory=self.memory
            )
        }

    async def process(self, task: Dict) -> Dict:
        """
        Main entry point.

        Args:
            task: {
                "type": "investment_evaluation | research_discovery",
                "input": {...}
            }
        """
        task_type = task["type"]

        if task_type == "investment_evaluation":
            return await self.run_investment_workflow(task["input"])
        elif task_type == "research_discovery":
            return await self.run_discovery_workflow(task["input"])
        elif task_type == "comprehensive_analysis":
            return await self.run_comprehensive_workflow(task["input"])
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def run_investment_workflow(
        self,
        trial: Dict
    ) -> Dict:
        """
        Investment evaluation workflow.

        Steps:
        1. Investment Evaluator: Initial assessment
        2. Prior Art Researcher: Competitive landscape
        3. Investment Evaluator: Final recommendation
        4. Synthesis
        """
        logger.info(f"Starting investment workflow for {trial.get('nct_id')}")

        # Step 1: Initial assessment
        logger.info("Step 1: Initial assessment")
        initial_assessment = await self.agents["investment_evaluator"].process({
            "task": "initial_assessment",
            "trial": trial
        })

        # Extract research questions
        research_questions = initial_assessment.get("research_questions", [])

        # Step 2: Prior art research
        logger.info("Step 2: Prior art research")
        prior_art = await self.agents["prior_art_researcher"].process({
            "task": "competitive_analysis",
            "indication": trial.get("indication"),
            "intervention": trial.get("intervention"),
            "questions": research_questions
        })

        # Step 3: Final recommendation
        logger.info("Step 3: Final recommendation")
        final_recommendation = await self.agents["investment_evaluator"].process({
            "task": "final_recommendation",
            "trial": trial,
            "initial_assessment": initial_assessment,
            "prior_art": prior_art
        })

        # Synthesize
        logger.info("Synthesizing findings")
        report = await self._synthesize_investment_report(
            trial, initial_assessment, prior_art, final_recommendation
        )

        return report

    async def run_discovery_workflow(
        self,
        research_area: Dict
    ) -> Dict:
        """
        Research discovery workflow.

        Steps:
        1. Collect relevant papers (semantic search)
        2. Hypothesis Generator: Identify gaps
        3. Prior Art Researcher: Validate gaps
        4. Hypothesis Generator: Refine hypotheses
        5. Synthesis
        """
        logger.info(f"Starting discovery workflow for {research_area.get('topic')}")

        topic = research_area["topic"]

        # Step 1: Collect papers
        logger.info("Step 1: Collecting relevant papers")
        papers = await self.memory.semantic_search(
            query=topic,
            n_results=100
        )

        # Step 2: Identify gaps
        logger.info("Step 2: Identifying research gaps")
        gaps = await self.agents["hypothesis_generator"].process({
            "task": "identify_gaps",
            "papers": papers
        })

        # Step 3: Validate gaps (prior art search)
        logger.info("Step 3: Validating gaps")
        validation = await self.agents["prior_art_researcher"].process({
            "task": "validate_gaps",
            "gaps": gaps["gaps"]
        })

        # Step 4: Refine hypotheses
        logger.info("Step 4: Refining hypotheses")
        hypotheses = await self.agents["hypothesis_generator"].process({
            "task": "generate_hypotheses",
            "gaps": gaps["gaps"],
            "validation": validation
        })

        # Synthesize
        logger.info("Synthesizing research proposal")
        report = await self._synthesize_research_proposal(
            topic, papers, gaps, validation, hypotheses
        )

        return report

    async def _synthesize_investment_report(
        self,
        trial: Dict,
        initial: Dict,
        prior_art: Dict,
        final: Dict
    ) -> Dict:
        """Generate investment memo."""

        prompt = f"""
        Synthesize the following findings into an investment evaluation memo.

        Trial: {trial['title']}
        NCT ID: {trial.get('nct_id')}

        Initial Assessment:
        {initial}

        Competitive Landscape:
        {prior_art}

        Final Recommendation:
        {final}

        Generate a comprehensive investment memo with:
        1. Executive Summary (3-5 sentences)
        2. Market Opportunity
        3. Competitive Position
        4. Risks
        5. Opportunities
        6. Recommendation (Go/No-Go/More Data)
        7. Next Steps

        Format as markdown.
        """

        synthesis = await self.llm.generate(prompt)

        return {
            "type": "investment_memo",
            "trial": trial,
            "memo": synthesis,
            "recommendation": final.get("recommendation"),
            "confidence": final.get("confidence"),
            "components": {
                "initial_assessment": initial,
                "prior_art": prior_art,
                "final_recommendation": final
            }
        }

    async def _synthesize_research_proposal(
        self,
        topic: str,
        papers: List[Dict],
        gaps: Dict,
        validation: Dict,
        hypotheses: Dict
    ) -> Dict:
        """Generate research proposal."""

        prompt = f"""
        Synthesize the following research discovery into a proposal.

        Topic: {topic}

        Papers Analyzed: {len(papers)}

        Research Gaps Identified:
        {gaps}

        Gap Validation:
        {validation}

        Proposed Hypotheses:
        {hypotheses}

        Generate a research proposal with:
        1. Background (current state of field)
        2. Research Gaps
        3. Proposed Hypotheses (3-5 testable hypotheses)
        4. Experimental Designs
        5. Expected Outcomes
        6. Feasibility Assessment
        7. Next Steps

        Format as markdown.
        """

        synthesis = await self.llm.generate(prompt)

        return {
            "type": "research_proposal",
            "topic": topic,
            "proposal": synthesis,
            "hypotheses": hypotheses.get("hypotheses", []),
            "components": {
                "papers": papers,
                "gaps": gaps,
                "validation": validation,
                "hypotheses": hypotheses
            }
        }
```

---

### 2. Investment Evaluator Agent

**Role**: Assess clinical trials for commercial potential

**Evaluation Framework**:

```
┌──────────────────────────────────────────────────────┐
│         INVESTMENT EVALUATION FRAMEWORK               │
├──────────────────────────────────────────────────────┤
│                                                       │
│  1. MARKET ANALYSIS (Weight: 30%)                    │
│     ├─ Indication & patient population               │
│     ├─ Market size ($B)                              │
│     ├─ Growth rate                                   │
│     └─ Unmet medical need (severity)                 │
│                                                       │
│  2. COMPETITIVE POSITION (Weight: 25%)               │
│     ├─ Current standard of care                      │
│     ├─ Competing therapies in development            │
│     ├─ Differentiation (MOA, efficacy, safety)       │
│     └─ Time to market advantage                      │
│                                                       │
│  3. SCIENTIFIC ASSESSMENT (Weight: 20%)              │
│     ├─ Novelty of approach                           │
│     ├─ Biological rationale                          │
│     ├─ Preclinical/clinical data quality             │
│     └─ Endpoints (clinical relevance)                │
│                                                       │
│  4. REGULATORY/IP (Weight: 15%)                      │
│     ├─ Regulatory pathway clarity                    │
│     ├─ Patent landscape                              │
│     ├─ Freedom to operate                            │
│     └─ Orphan designation potential                  │
│                                                       │
│  5. COMMERCIAL VIABILITY (Weight: 10%)               │
│     ├─ Reimbursement likelihood                      │
│     ├─ Pricing potential                             │
│     ├─ Manufacturing feasibility                     │
│     └─ Partnership opportunities                     │
│                                                       │
│  OVERALL SCORE: Weighted average (0-100)             │
│                                                       │
│  RECOMMENDATION:                                      │
│  - High (>75): Strong investment case                │
│  - Medium (50-75): Conditional/more data needed      │
│  - Low (<50): Pass                                   │
└──────────────────────────────────────────────────────┘
```

**Prompt Template**:

```yaml
# prompts/investment_evaluator.yaml

system_prompt: |
  You are an expert technology transfer analyst and venture investor
  specializing in biomedical innovations. You evaluate clinical trials
  and research for commercial potential.

  Your expertise includes:
  - Biotech/pharma market analysis
  - Competitive landscape assessment
  - Regulatory pathways (FDA, EMA)
  - IP strategy
  - Commercial viability

  You provide:
  - Data-driven assessments
  - Clear, structured reasoning
  - Specific, actionable recommendations
  - Risk/opportunity balance

  Evaluation framework:
  1. Market Analysis (30%) - Size, growth, unmet need
  2. Competitive Position (25%) - Differentiation, timing
  3. Scientific Assessment (20%) - Novelty, data quality
  4. Regulatory/IP (15%) - Pathway, patents
  5. Commercial Viability (10%) - Pricing, partnerships

  Always cite specific data points and explain reasoning.

initial_assessment_template: |
  Evaluate this clinical trial for initial commercial assessment:

  Title: {title}
  Phase: {phase}
  Indication: {indication}
  Intervention: {intervention}
  Sample Size: {sample_size}
  Primary Outcome: {primary_outcome}
  Secondary Outcomes: {secondary_outcomes}
  Sponsor: {sponsor}

  Provide initial assessment:

  ## Market Context
  - Patient population estimate
  - Current standard of care
  - Market size ($B, provide estimate with reasoning)

  ## Preliminary Analysis
  - Novelty/differentiation
  - Key questions for deeper research
  - Initial concerns/red flags

  ## Research Questions
  List specific questions for prior art researcher:
  1. ...
  2. ...

  Output as JSON:
  ```json
  {
    "market_size_usd_billions": 0.0,
    "patient_population": 0,
    "current_standard_of_care": "",
    "preliminary_score": 0,
    "research_questions": [],
    "concerns": []
  }
  ```

final_recommendation_template: |
  Provide final investment recommendation.

  Trial: {trial}

  Initial Assessment: {initial_assessment}

  Competitive Landscape (from prior art research): {prior_art}

  Conduct comprehensive evaluation using framework:

  ## 1. Market Analysis (Score /30)
  - Market size: $X.XB
  - Patient population: XXX,XXX
  - Growth rate: X%
  - Unmet need severity: High/Medium/Low
  - Score: XX/30
  - Reasoning: ...

  ## 2. Competitive Position (Score /25)
  - Current SoC: ...
  - Competing approaches: ...
  - Differentiation: ...
  - Time advantage: ...
  - Score: XX/25
  - Reasoning: ...

  ## 3. Scientific Assessment (Score /20)
  - Novelty: ...
  - Rationale: ...
  - Data quality: ...
  - Endpoints: ...
  - Score: XX/20
  - Reasoning: ...

  ## 4. Regulatory/IP (Score /15)
  - Pathway: ...
  - Patents: ...
  - FTO: ...
  - Score: XX/15
  - Reasoning: ...

  ## 5. Commercial Viability (Score /10)
  - Reimbursement: ...
  - Pricing: ...
  - Manufacturing: ...
  - Score: XX/10
  - Reasoning: ...

  ## Overall Score: XX/100

  ## Recommendation
  - Rating: High/Medium/Low
  - Confidence: High/Medium/Low
  - Rationale: (2-3 sentences)

  ## Key Risks
  1. ...
  2. ...

  ## Key Opportunities
  1. ...
  2. ...

  ## Next Steps
  1. ...
  2. ...

  Output as JSON:
  ```json
  {
    "overall_score": 0,
    "recommendation": "High|Medium|Low",
    "confidence": "High|Medium|Low",
    "scores": {
      "market_analysis": 0,
      "competitive_position": 0,
      "scientific_assessment": 0,
      "regulatory_ip": 0,
      "commercial_viability": 0
    },
    "market_size_usd_billions": 0.0,
    "patient_population": 0,
    "key_risks": [],
    "key_opportunities": [],
    "next_steps": [],
    "rationale": ""
  }
  ```
```

**Implementation**:

```python
# src/agents/investment_evaluator.py

from typing import Dict
from loguru import logger
from .base_agent import BaseAgent

class InvestmentEvaluatorAgent(BaseAgent):
    """Evaluate clinical trials for commercial potential."""

    async def process(self, task: Dict) -> Dict:
        """
        Process evaluation task.

        Task types:
        - initial_assessment: Quick first pass
        - final_recommendation: Comprehensive evaluation
        """
        task_type = task["task"]

        if task_type == "initial_assessment":
            return await self._initial_assessment(task["trial"])
        elif task_type == "final_recommendation":
            return await self._final_recommendation(task)
        else:
            raise ValueError(f"Unknown task: {task_type}")

    async def _initial_assessment(self, trial: Dict) -> Dict:
        """Initial quick assessment."""

        prompt = self._build_prompt(
            "initial_assessment_template",
            **trial
        )

        response = await self.llm.generate_structured(
            prompt,
            schema={
                "market_size_usd_billions": "number",
                "patient_population": "integer",
                "current_standard_of_care": "string",
                "preliminary_score": "integer",
                "research_questions": ["string"],
                "concerns": ["string"]
            }
        )

        return response

    async def _final_recommendation(self, task: Dict) -> Dict:
        """Comprehensive evaluation with prior art context."""

        trial = task["trial"]
        initial = task["initial_assessment"]
        prior_art = task["prior_art"]

        prompt = self._build_prompt(
            "final_recommendation_template",
            trial=trial,
            initial_assessment=initial,
            prior_art=prior_art
        )

        response = await self.llm.generate_structured(
            prompt,
            schema={
                "overall_score": "integer",
                "recommendation": "string",
                "confidence": "string",
                "scores": {
                    "market_analysis": "integer",
                    "competitive_position": "integer",
                    "scientific_assessment": "integer",
                    "regulatory_ip": "integer",
                    "commercial_viability": "integer"
                },
                "market_size_usd_billions": "number",
                "patient_population": "integer",
                "key_risks": ["string"],
                "key_opportunities": ["string"],
                "next_steps": ["string"],
                "rationale": "string"
            }
        )

        return response

    def _build_prompt(self, template_name: str, **kwargs) -> str:
        """Load and format prompt template."""
        # Load from prompts/investment_evaluator.yaml
        template = self.prompt_loader.get_template(template_name)
        return template.format(**kwargs)
```

---

### 3. Prior Art Researcher Agent

**Role**: Comprehensive literature and patent research

**Research Strategy**:

```
┌────────────────────────────────────────────────────┐
│         PRIOR ART RESEARCH STRATEGY                 │
├────────────────────────────────────────────────────┤
│                                                     │
│  DEPTH LEVELS:                                     │
│                                                     │
│  1. QUICK (5-10 minutes)                           │
│     ├─ Top 20 PubMed papers                        │
│     ├─ Active clinical trials                      │
│     └─ Key patents (if obvious)                    │
│                                                     │
│  2. STANDARD (20-30 minutes)                       │
│     ├─ 50-100 papers (semantic search)             │
│     ├─ Citation analysis                           │
│     ├─ Clinical trials analysis                    │
│     ├─ Patent landscape search                     │
│     └─ Key opinion leaders                         │
│                                                     │
│  3. COMPREHENSIVE (1-2 hours)                      │
│     ├─ 200+ papers                                 │
│     ├─ Full citation network                       │
│     ├─ All relevant trials                         │
│     ├─ Patent family analysis                      │
│     ├─ FDA submissions                             │
│     ├─ Conference abstracts                        │
│     └─ Competitive intelligence                    │
│                                                     │
│  SOURCES:                                          │
│  - PubMed/MEDLINE (primary literature)             │
│  - ClinicalTrials.gov (trials)                     │
│  - USPTO/EPO/WIPO (patents)                        │
│  - FDA databases (approvals, submissions)          │
│  - Conference databases (ASH, ASCO, etc.)          │
│                                                     │
└────────────────────────────────────────────────────┘
```

**Prompt Template**:

```yaml
# prompts/prior_art_researcher.yaml

system_prompt: |
  You are an expert research analyst specializing in biomedical
  literature and patent analysis. You conduct comprehensive prior
  art searches to map competitive landscapes.

  Your capabilities:
  - Literature search and synthesis
  - Citation network analysis
  - Clinical trial landscape mapping
  - Patent analysis
  - Identification of key researchers and institutions

  Your outputs are:
  - Comprehensive and well-organized
  - Evidence-based with citations
  - Focused on competitive insights
  - Structured for decision-making

competitive_analysis_template: |
  Research the competitive landscape for:

  Indication: {indication}
  Intervention Type: {intervention}
  Specific Questions: {questions}

  Conduct research and provide:

  ## Published Literature
  ### Mechanism of Action
  - Papers describing similar approaches
  - Key findings
  - Limitations

  ### Clinical Evidence
  - Published trials
  - Efficacy data
  - Safety data

  ### Key Researchers
  - Leading groups in this area
  - Institutions
  - Recent publications

  ## Clinical Trials
  ### Active Trials
  - NCT ID, title, phase
  - Sponsor
  - Timeline
  - How they compare

  ### Completed Trials
  - Results if available
  - Lessons learned

  ## Patent Landscape
  ### Relevant Patents
  - Patent numbers
  - Assignees
  - Claims relevant to this approach
  - Expiration dates

  ### Freedom to Operate
  - Potential blocking patents
  - White space opportunities

  ## Competitive Summary
  - Who are the main competitors?
  - What differentiates this approach?
  - Timing advantages/disadvantages?

  Output as structured JSON with citations.
```

---

### 4. Hypothesis Generator Agent

**Role**: Identify research gaps and propose new directions

**Discovery Process**:

```
┌─────────────────────────────────────────────────────┐
│        RESEARCH DISCOVERY METHODOLOGY                │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. CLUSTER ANALYSIS                                │
│     ├─ Group related papers                         │
│     ├─ Extract common themes                        │
│     ├─ Identify methodological patterns             │
│     └─ Note population characteristics              │
│                                                      │
│  2. GAP IDENTIFICATION                              │
│     ├─ Unstudied populations (age, sex, ethnicity)  │
│     ├─ Alternative mechanisms                       │
│     ├─ Combination therapies                        │
│     ├─ Different endpoints                          │
│     ├─ Contradictory findings                       │
│     └─ Mentioned limitations                        │
│                                                      │
│  3. HYPOTHESIS GENERATION                           │
│     For each gap:                                   │
│     ├─ Formulate testable hypothesis                │
│     ├─ Provide biological rationale                 │
│     ├─ Cite supporting evidence                     │
│     ├─ Propose methodology                          │
│     └─ Assess feasibility                           │
│                                                      │
│  4. PRIORITIZATION                                  │
│     Rank hypotheses by:                             │
│     ├─ Scientific impact                            │
│     ├─ Feasibility                                  │
│     ├─ Resource requirements                        │
│     ├─ Timeline                                     │
│     └─ Competitive landscape                        │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Shared Memory System

All agents share access to:

```python
# src/agents/memory_manager.py

from typing import List, Dict, Optional
import chromadb
from datetime import datetime

class MemoryManager:
    """Shared memory for multi-agent system."""

    def __init__(self, chroma_client: chromadb.Client):
        self.client = chroma_client

        # Collections
        self.articles = client.get_collection("articles")
        self.conversations = client.get_or_create_collection("conversations")
        self.findings = client.get_or_create_collection("findings")

    async def semantic_search(
        self,
        query: str,
        collection: str = "articles",
        n_results: int = 10,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """Semantic search across knowledge base."""
        coll = getattr(self, collection)

        results = coll.query(
            query_texts=[query],
            n_results=n_results,
            where=filter
        )

        return results

    async def store_finding(
        self,
        agent_name: str,
        finding: str,
        metadata: Dict
    ):
        """Store agent finding in long-term memory."""
        self.findings.add(
            documents=[finding],
            metadatas=[{
                "agent": agent_name,
                "timestamp": datetime.now().isoformat(),
                **metadata
            }],
            ids=[f"{agent_name}_{datetime.now().timestamp()}"]
        )

    async def retrieve_context(
        self,
        query: str,
        max_items: int = 5
    ) -> List[Dict]:
        """Retrieve relevant context for query."""
        # Search across all collections
        article_results = await self.semantic_search(
            query, "articles", max_items
        )
        finding_results = await self.semantic_search(
            query, "findings", max_items
        )

        return {
            "articles": article_results,
            "findings": finding_results
        }
```

---

## Example Workflows

### Workflow 1: Investment Evaluation

```python
# Example usage

coordinator = CoordinatorAgent(...)

# User request
task = {
    "type": "investment_evaluation",
    "input": {
        "nct_id": "NCT04567890",
        "title": "Phase 2 Study of XYZ in Pediatric Asthma",
        "phase": "Phase II",
        "indication": "Asthma, pediatric",
        "intervention": "Monoclonal antibody XYZ",
        "sample_size": 120,
        "primary_outcome": "FEV1 improvement at 12 weeks",
        "sponsor": "Boston Children's Hospital"
    }
}

# Execute workflow
report = await coordinator.process(task)

# Report contains:
# - Executive summary
# - Market analysis
# - Competitive position
# - Recommendation (Go/No-Go/More Data)
# - Detailed memo
```

**Output Example**:

```markdown
# Investment Evaluation Memo
## NCT04567890: Phase 2 Study of XYZ in Pediatric Asthma

### Executive Summary
XYZ is a novel IL-5 targeting monoclonal antibody for severe pediatric asthma.
The market opportunity is substantial ($2.3B pediatric severe asthma), with
moderate competition from existing biologics. Recommendation: **MEDIUM** -
conditional on Phase 2 results showing superiority to current SoC.

### Market Analysis (Score: 24/30)
- **Market Size**: $2.3B (severe pediatric asthma in US/EU)
- **Patient Population**: ~450,000 children with severe asthma
- **Growth**: 8% CAGR driven by increasing diagnosis
- **Unmet Need**: HIGH - current biologics have limitations in pediatrics

### Competitive Position (Score: 18/25)
- **Current SoC**: Omalizumab, mepolizumab (limited ped data)
- **Differentiators**:
  - Novel IL-5Rα binding (vs IL-5)
  - Pediatric-optimized formulation
  - Q8W dosing vs Q4W
- **Competitors**:
  - Dupilumab (approved, strong efficacy)
  - Tezepelumab (Phase 3, broad mechanism)
- **Time Advantage**: 2-3 years behind Dupilumab

### Recommendation: MEDIUM (Conditional)
**Confidence**: Medium

**Rationale**: Significant market opportunity with differentiated approach, but
late entry vs established competitor (Dupilumab). Success depends on:
1. Superior efficacy or safety in Phase 2
2. Clear differentiation from Dupilumab
3. Favorable dosing/administration

**Key Risks**:
1. Dupilumab entrenchment by launch
2. No clear superiority in MOA
3. Reimbursement hurdles (4th biologic)

**Key Opportunities**:
1. Pediatric-specific optimization
2. Subgroup targeting (high eosinophils)
3. Combination potential

**Next Steps**:
1. Monitor Phase 2 interim data (Q3 2024)
2. Analyze Dupilumab real-world data in peds
3. Assess patent landscape for combination claims
```

### Workflow 2: Research Discovery

```python
task = {
    "type": "research_discovery",
    "input": {
        "topic": "CAR-T therapy resistance mechanisms in B-cell lymphoma",
        "context": "Recent papers show heterogeneous response rates"
    }
}

report = await coordinator.process(task)
```

**Output Example**:

```markdown
# Research Proposal
## Understanding and Overcoming CAR-T Resistance in B-Cell Lymphoma

### Background
CAR-T therapy has revolutionized B-cell lymphoma treatment, but 40-60% of
patients don't achieve durable responses. Analysis of 127 papers reveals
multiple resistance mechanisms but gaps in understanding.

### Research Gaps Identified

1. **Tumor Microenvironment Impact**
   - Gap: Most studies focus on intrinsic tumor factors
   - Evidence: Only 12/127 papers examined TME in detail
   - Opportunity: TME modulation may overcome resistance

2. **Combination Strategies**
   - Gap: Limited data on CAR-T + checkpoint inhibitors
   - Evidence: 3 small pilot studies, no randomized trials
   - Opportunity: Synergistic efficacy

3. **Biomarker Development**
   - Gap: No validated predictive biomarkers
   - Evidence: Multiple candidate biomarkers, none validated
   - Opportunity: Precision patient selection

### Proposed Hypotheses

#### Hypothesis 1: TME Remodeling Enhances CAR-T Efficacy
**Statement**: Targeting CAR-T resistance-associated macrophages (CRAMs) in
the TME will improve CAR-T efficacy in resistant lymphomas.

**Rationale**:
- CRAMs identified in 3 recent studies (PMID: 12345, 67890, 11111)
- Correlate with poor CAR-T response
- Depletion in mouse models restores sensitivity

**Proposed Design**:
- Phase 1b trial: CAR-T + CSF1R inhibitor
- Patient selection: Prior CAR-T failure, CRAM-high by biopsy
- Endpoints: ORR, CAR-T expansion, TME analysis

**Feasibility**: HIGH
- CSF1R inhibitors available
- CAR-T manufacturing established
- Biomarker assay validated

... [additional hypotheses] ...

### Recommended Next Steps
1. Validate CRAM biomarker in retrospective cohort
2. Pilot combination study in mouse models
3. Engage with FDA on trial design
4. Secure funding (est. $5M for Phase 1b)
```

---

## Implementation Timeline

**Week 1-2**: Base agent framework + Coordinator
**Week 3-4**: Investment Evaluator
**Week 5**: Prior Art Researcher
**Week 6**: Hypothesis Generator
**Week 7**: Testing & refinement
**Week 8**: Production deployment

---

## Testing Strategy

```python
# tests/test_multi_agent.py

@pytest.mark.asyncio
async def test_investment_workflow():
    """Test end-to-end investment evaluation."""

    coordinator = CoordinatorAgent(...)

    task = {
        "type": "investment_evaluation",
        "input": SAMPLE_TRIAL
    }

    report = await coordinator.process(task)

    # Assertions
    assert "recommendation" in report
    assert report["recommendation"] in ["High", "Medium", "Low"]
    assert "overall_score" in report
    assert 0 <= report["overall_score"] <= 100
    assert "key_risks" in report
    assert len(report["key_risks"]) > 0
```

---

## Monitoring & Observability

```python
# src/agents/monitoring.py

from prometheus_client import Counter, Histogram

# Metrics
agent_tasks_total = Counter(
    'agent_tasks_total',
    'Total tasks processed by agent',
    ['agent_name', 'task_type']
)

agent_task_duration = Histogram(
    'agent_task_duration_seconds',
    'Agent task duration',
    ['agent_name']
)

agent_errors_total = Counter(
    'agent_errors_total',
    'Agent errors',
    ['agent_name', 'error_type']
)

# Usage in agents
async def process(self, task):
    start = time.time()

    try:
        result = await self._process(task)
        agent_tasks_total.labels(
            agent_name=self.name,
            task_type=task['task']
        ).inc()
        return result

    except Exception as e:
        agent_errors_total.labels(
            agent_name=self.name,
            error_type=type(e).__name__
        ).inc()
        raise

    finally:
        agent_task_duration.labels(
            agent_name=self.name
        ).observe(time.time() - start)
```

---

## Next Steps

1. Review agent specifications
2. Refine evaluation frameworks
3. Customize prompt templates
4. Begin implementation (start with Coordinator + Investment Evaluator)
5. Test on sample clinical trials
6. Iterate based on results

See IMPLEMENTATION_GUIDE.md for development setup.
