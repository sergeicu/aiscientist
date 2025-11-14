"""Coordinator agent for orchestrating multi-agent workflows."""

from typing import Dict, List, Optional
from loguru import logger
from .base_agent import BaseAgent
from .investment_evaluator import InvestmentEvaluatorAgent
import json


class CoordinatorAgent(BaseAgent):
    """
    Orchestrate multi-agent workflows.

    The coordinator is responsible for:
    1. Understanding user requests
    2. Breaking down complex tasks
    3. Routing tasks to appropriate agents
    4. Collecting and synthesizing results
    5. Generating final reports
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize specialized agents
        self.agents = {}
        self._initialize_agents()

        logger.info(f"Coordinator initialized with {len(self.agents)} specialized agents")

    def _initialize_agents(self):
        """Initialize all specialized agents."""
        # Investment Evaluator
        self.agents["investment_evaluator"] = InvestmentEvaluatorAgent(
            name="investment_evaluator",
            llm_client=self.llm,
            memory_manager=self.memory,
            prompt_dir=self.prompt_dir
        )

        # Prior Art Researcher (to be implemented)
        # self.agents["prior_art_researcher"] = PriorArtResearcherAgent(...)

        # Hypothesis Generator (to be implemented)
        # self.agents["hypothesis_generator"] = HypothesisGeneratorAgent(...)

    async def process(self, task: Dict) -> Dict:
        """
        Main entry point for coordinator.

        Args:
            task: {
                "type": "investment_evaluation | research_discovery | comprehensive_analysis",
                "input": {...}
            }

        Returns:
            Complete workflow result
        """
        task_type = task.get("type")

        logger.info(f"Coordinator processing task: {task_type}")

        if task_type == "investment_evaluation":
            return await self.run_investment_workflow(task["input"])
        elif task_type == "research_discovery":
            return await self.run_discovery_workflow(task["input"])
        elif task_type == "comprehensive_analysis":
            return await self.run_comprehensive_workflow(task["input"])
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def run_investment_workflow(self, trial: Dict) -> Dict:
        """
        Investment evaluation workflow.

        Workflow steps:
        1. Investment Evaluator: Initial assessment
        2. Prior Art Researcher: Competitive landscape (placeholder for now)
        3. Investment Evaluator: Final recommendation
        4. Synthesis: Generate investment memo

        Args:
            trial: Clinical trial information

        Returns:
            Complete investment evaluation report
        """
        nct_id = trial.get("nct_id", "Unknown")
        logger.info(f"Starting investment workflow for {nct_id}")

        results = {
            "workflow_type": "investment_evaluation",
            "trial": trial,
            "steps": []
        }

        # Step 1: Initial assessment
        logger.info("Step 1/3: Initial assessment")
        try:
            initial_assessment = await self.agents["investment_evaluator"].process({
                "task": "initial_assessment",
                "trial": trial
            })

            results["steps"].append({
                "step": 1,
                "agent": "investment_evaluator",
                "task": "initial_assessment",
                "success": True,
                "output": initial_assessment
            })

            logger.info(
                f"Initial assessment complete - "
                f"Score: {initial_assessment.get('preliminary_score')}"
            )

        except Exception as e:
            logger.error(f"Initial assessment failed: {e}")
            results["steps"].append({
                "step": 1,
                "agent": "investment_evaluator",
                "task": "initial_assessment",
                "success": False,
                "error": str(e)
            })
            results["status"] = "failed"
            results["error"] = f"Initial assessment failed: {e}"
            return results

        # Step 2: Prior art research (placeholder - will implement later)
        logger.info("Step 2/3: Prior art research (placeholder)")

        # For now, create a simple placeholder response
        prior_art = {
            "status": "placeholder",
            "message": "Prior art researcher not yet implemented",
            "competitive_trials": [],
            "relevant_papers": []
        }

        results["steps"].append({
            "step": 2,
            "agent": "prior_art_researcher",
            "task": "competitive_analysis",
            "success": True,
            "output": prior_art,
            "note": "Placeholder - agent not yet implemented"
        })

        # Step 3: Final recommendation
        logger.info("Step 3/3: Final recommendation")
        try:
            final_recommendation = await self.agents["investment_evaluator"].process({
                "task": "final_recommendation",
                "trial": trial,
                "initial_assessment": initial_assessment,
                "prior_art": prior_art
            })

            results["steps"].append({
                "step": 3,
                "agent": "investment_evaluator",
                "task": "final_recommendation",
                "success": True,
                "output": final_recommendation
            })

            logger.info(
                f"Final recommendation: {final_recommendation.get('recommendation')} "
                f"(Score: {final_recommendation.get('overall_score')})"
            )

        except Exception as e:
            logger.error(f"Final recommendation failed: {e}")
            results["steps"].append({
                "step": 3,
                "agent": "investment_evaluator",
                "task": "final_recommendation",
                "success": False,
                "error": str(e)
            })
            results["status"] = "partial"
            results["error"] = f"Final recommendation failed: {e}"
            return results

        # Step 4: Synthesis
        logger.info("Synthesizing investment memo")
        try:
            memo = await self._synthesize_investment_memo(
                trial=trial,
                initial=initial_assessment,
                prior_art=prior_art,
                final=final_recommendation
            )

            results["memo"] = memo
            results["status"] = "success"

            # Store finding in memory
            await self.store_finding(
                finding=memo,
                metadata={
                    "type": "investment_memo",
                    "nct_id": nct_id,
                    "recommendation": final_recommendation.get("recommendation"),
                    "score": final_recommendation.get("overall_score")
                }
            )

        except Exception as e:
            logger.error(f"Memo synthesis failed: {e}")
            results["status"] = "partial"
            results["synthesis_error"] = str(e)

        logger.info("Investment workflow complete")
        return results

    async def run_discovery_workflow(self, research_area: Dict) -> Dict:
        """
        Research discovery workflow.

        Workflow steps:
        1. Collect relevant papers (semantic search)
        2. Hypothesis Generator: Identify gaps
        3. Prior Art Researcher: Validate gaps
        4. Hypothesis Generator: Refine hypotheses
        5. Synthesis: Generate research proposal

        Args:
            research_area: {
                "topic": "research topic",
                "context": "additional context"
            }

        Returns:
            Complete research discovery report
        """
        topic = research_area.get("topic", "Unknown")
        logger.info(f"Starting discovery workflow for: {topic}")

        results = {
            "workflow_type": "research_discovery",
            "topic": topic,
            "steps": [],
            "status": "not_implemented"
        }

        # Placeholder for now
        logger.warning("Research discovery workflow not yet implemented")

        return results

    async def run_comprehensive_workflow(self, input_data: Dict) -> Dict:
        """
        Comprehensive analysis workflow.

        Combines investment evaluation and research discovery.

        Args:
            input_data: Mixed input data

        Returns:
            Comprehensive analysis report
        """
        logger.info("Starting comprehensive workflow")

        results = {
            "workflow_type": "comprehensive_analysis",
            "steps": [],
            "status": "not_implemented"
        }

        # Placeholder for now
        logger.warning("Comprehensive workflow not yet implemented")

        return results

    async def _synthesize_investment_memo(
        self,
        trial: Dict,
        initial: Dict,
        prior_art: Dict,
        final: Dict
    ) -> str:
        """
        Generate investment memo from agent findings.

        Args:
            trial: Trial information
            initial: Initial assessment
            prior_art: Competitive landscape
            final: Final recommendation

        Returns:
            Markdown-formatted memo
        """
        # Get synthesis prompt
        prompt = self.get_prompt(
            "investment_synthesis",
            trial_info=json.dumps(trial, indent=2),
            initial_assessment=json.dumps(initial, indent=2),
            prior_art=json.dumps(prior_art, indent=2),
            final_recommendation=json.dumps(final, indent=2)
        )

        # Generate memo using LLM
        try:
            memo = await self.llm.generate(prompt)
            return memo
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback: create structured memo from data
            return self._create_fallback_memo(trial, initial, prior_art, final)

    def _create_fallback_memo(
        self,
        trial: Dict,
        initial: Dict,
        prior_art: Dict,
        final: Dict
    ) -> str:
        """Create fallback memo if LLM generation fails."""
        memo = f"""# Investment Evaluation Memo
## {trial.get('title', 'Clinical Trial')}

**NCT ID**: {trial.get('nct_id', 'N/A')}
**Phase**: {trial.get('phase', 'N/A')}
**Indication**: {trial.get('indication', 'N/A')}

### Executive Summary
Overall Score: {final.get('overall_score', 'N/A')}/100
Recommendation: **{final.get('recommendation', 'N/A')}**
Confidence: {final.get('confidence', 'N/A')}

{final.get('rationale', '')}

### Market Opportunity
- Market Size: ${final.get('market_size_usd_billions', 'N/A')}B
- Patient Population: {final.get('patient_population', 'N/A'):,}
- Current Standard of Care: {initial.get('current_standard_of_care', 'N/A')}

### Scoring Breakdown
- Market Analysis: {final.get('scores', {}).get('market_analysis', 0)}/30
- Competitive Position: {final.get('scores', {}).get('competitive_position', 0)}/25
- Scientific Assessment: {final.get('scores', {}).get('scientific_assessment', 0)}/20
- Regulatory/IP: {final.get('scores', {}).get('regulatory_ip', 0)}/15
- Commercial Viability: {final.get('scores', {}).get('commercial_viability', 0)}/10

### Key Risks
"""
        for risk in final.get('key_risks', []):
            if isinstance(risk, dict):
                memo += f"- {risk.get('risk', '')} (Impact: {risk.get('impact', 'N/A')})\n"
            else:
                memo += f"- {risk}\n"

        memo += "\n### Key Opportunities\n"
        for opp in final.get('key_opportunities', []):
            if isinstance(opp, dict):
                memo += f"- {opp.get('opportunity', '')} (Value: {opp.get('value_potential', 'N/A')})\n"
            else:
                memo += f"- {opp}\n"

        memo += "\n### Next Steps\n"
        for step in final.get('next_steps', []):
            memo += f"1. {step}\n"

        return memo
