"""Investment Evaluator Agent for assessing commercial potential of clinical trials."""

from typing import Dict, List, Optional
from loguru import logger
from .base_agent import BaseAgent
import json


class InvestmentEvaluatorAgent(BaseAgent):
    """
    Evaluate clinical trials for commercial potential.

    This agent assesses trials using a 5-dimensional framework:
    1. Market Analysis (30%) - Size, growth, unmet need
    2. Competitive Position (25%) - Differentiation, timing
    3. Scientific Assessment (20%) - Novelty, data quality
    4. Regulatory/IP (15%) - Pathway, patents
    5. Commercial Viability (10%) - Reimbursement, partnerships

    Scoring:
    - 75-100: High potential
    - 50-74: Medium potential
    - 0-49: Low potential
    """

    async def process(self, task: Dict) -> Dict:
        """
        Process evaluation task.

        Args:
            task: {
                "task": "initial_assessment | final_recommendation",
                "trial": {...},
                "initial_assessment": {...},  # For final_recommendation
                "prior_art": {...}  # For final_recommendation
            }

        Returns:
            Task result with structured evaluation
        """
        task_type = task.get("task")

        logger.info(f"Investment Evaluator processing: {task_type}")

        if task_type == "initial_assessment":
            return await self._initial_assessment(task["trial"])
        elif task_type == "final_recommendation":
            return await self._final_recommendation(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def _initial_assessment(self, trial: Dict) -> Dict:
        """
        Perform initial quick assessment of trial.

        Args:
            trial: Clinical trial information

        Returns:
            Initial assessment with market context and research questions
        """
        logger.info(f"Initial assessment for: {trial.get('title', 'Unknown')}")

        # Build prompt
        prompt = self.get_prompt(
            "initial_assessment",
            title=trial.get("title", ""),
            nct_id=trial.get("nct_id", ""),
            phase=trial.get("phase", ""),
            indication=trial.get("indication", ""),
            intervention=trial.get("intervention", ""),
            sample_size=trial.get("sample_size", ""),
            primary_outcome=trial.get("primary_outcome", ""),
            secondary_outcomes=trial.get("secondary_outcomes", ""),
            sponsor=trial.get("sponsor", ""),
            status=trial.get("status", "")
        )

        # Add system prompt
        system_prompt = self.prompts.get("system_prompt", "")
        full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            # Generate with structured output
            result = await self.llm.generate_structured(
                prompt=full_prompt,
                schema={
                    "market_size_usd_billions": "number",
                    "market_size_reasoning": "string",
                    "patient_population": "integer",
                    "patient_population_reasoning": "string",
                    "current_standard_of_care": "string",
                    "novel_aspects": ["string"],
                    "strengths": ["string"],
                    "concerns": ["string"],
                    "research_questions": ["string"],
                    "preliminary_score": "integer",
                    "confidence": "string",
                    "reasoning": "string"
                }
            )

            logger.info(
                f"Initial assessment complete - "
                f"Score: {result.get('preliminary_score')}, "
                f"Confidence: {result.get('confidence')}"
            )

            return result

        except Exception as e:
            logger.error(f"Initial assessment generation failed: {e}")
            # Return fallback assessment
            return self._create_fallback_initial_assessment(trial, str(e))

    async def _final_recommendation(self, task: Dict) -> Dict:
        """
        Generate comprehensive final recommendation.

        Args:
            task: {
                "trial": trial info,
                "initial_assessment": initial assessment results,
                "prior_art": competitive landscape
            }

        Returns:
            Final recommendation with detailed scoring and analysis
        """
        trial = task["trial"]
        initial = task["initial_assessment"]
        prior_art = task["prior_art"]

        logger.info(f"Final recommendation for: {trial.get('title', 'Unknown')}")

        # Build prompt
        prompt = self.get_prompt(
            "final_recommendation",
            trial_summary=json.dumps(trial, indent=2),
            initial_assessment=json.dumps(initial, indent=2),
            prior_art_findings=json.dumps(prior_art, indent=2)
        )

        # Add system prompt
        system_prompt = self.prompts.get("system_prompt", "")
        full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            # Generate with structured output
            result = await self.llm.generate_structured(
                prompt=full_prompt,
                schema={
                    "overall_score": "integer",
                    "recommendation": "string",
                    "confidence": "string",
                    "rationale": "string",
                    "scores": {
                        "market_analysis": "integer",
                        "competitive_position": "integer",
                        "scientific_assessment": "integer",
                        "regulatory_ip": "integer",
                        "commercial_viability": "integer"
                    },
                    "market_size_usd_billions": "number",
                    "patient_population": "integer",
                    "key_risks": [
                        {
                            "risk": "string",
                            "impact": "string"
                        }
                    ],
                    "key_opportunities": [
                        {
                            "opportunity": "string",
                            "value_potential": "string"
                        }
                    ],
                    "next_steps": ["string"]
                }
            )

            # Validate recommendation
            rec = result.get("recommendation", "").upper()
            if rec not in ["HIGH", "MEDIUM", "LOW"]:
                logger.warning(f"Invalid recommendation: {rec}, defaulting to MEDIUM")
                result["recommendation"] = "MEDIUM"

            # Validate confidence
            conf = result.get("confidence", "").upper()
            if conf not in ["HIGH", "MEDIUM", "LOW"]:
                logger.warning(f"Invalid confidence: {conf}, defaulting to MEDIUM")
                result["confidence"] = "MEDIUM"

            logger.info(
                f"Final recommendation: {result.get('recommendation')} "
                f"(Score: {result.get('overall_score')}/100, "
                f"Confidence: {result.get('confidence')})"
            )

            return result

        except Exception as e:
            logger.error(f"Final recommendation generation failed: {e}")
            # Return fallback recommendation
            return self._create_fallback_final_recommendation(trial, initial, str(e))

    def _create_fallback_initial_assessment(
        self,
        trial: Dict,
        error: str
    ) -> Dict:
        """Create fallback initial assessment if generation fails."""
        logger.warning("Using fallback initial assessment")

        return {
            "market_size_usd_billions": 1.0,
            "market_size_reasoning": "Estimate unavailable due to generation error",
            "patient_population": 100000,
            "patient_population_reasoning": "Estimate unavailable",
            "current_standard_of_care": "Unknown",
            "novel_aspects": ["Analysis incomplete due to error"],
            "strengths": ["Analysis incomplete"],
            "concerns": [f"Generation failed: {error}"],
            "research_questions": [
                "What is the competitive landscape?",
                "What is the patent situation?"
            ],
            "preliminary_score": 50,
            "confidence": "Low",
            "reasoning": f"Fallback assessment due to error: {error}",
            "error": error,
            "is_fallback": True
        }

    def _create_fallback_final_recommendation(
        self,
        trial: Dict,
        initial: Dict,
        error: str
    ) -> Dict:
        """Create fallback final recommendation if generation fails."""
        logger.warning("Using fallback final recommendation")

        # Use initial assessment scores if available
        prelim_score = initial.get("preliminary_score", 50)

        return {
            "overall_score": prelim_score,
            "recommendation": "MEDIUM" if prelim_score >= 50 else "LOW",
            "confidence": "Low",
            "rationale": f"Fallback recommendation due to error: {error}",
            "scores": {
                "market_analysis": int(prelim_score * 0.3),
                "competitive_position": int(prelim_score * 0.25),
                "scientific_assessment": int(prelim_score * 0.2),
                "regulatory_ip": int(prelim_score * 0.15),
                "commercial_viability": int(prelim_score * 0.1)
            },
            "market_size_usd_billions": initial.get("market_size_usd_billions", 1.0),
            "patient_population": initial.get("patient_population", 100000),
            "key_risks": [
                {
                    "risk": "Analysis incomplete due to generation error",
                    "impact": "High"
                },
                {
                    "risk": error,
                    "impact": "High"
                }
            ],
            "key_opportunities": [
                {
                    "opportunity": "Analysis incomplete",
                    "value_potential": "Unknown"
                }
            ],
            "next_steps": [
                "Retry analysis with different model or parameters",
                "Manual review required"
            ],
            "error": error,
            "is_fallback": True
        }
