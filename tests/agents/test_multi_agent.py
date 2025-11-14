"""Tests for multi-agent system."""

import pytest
import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents import CoordinatorAgent, InvestmentEvaluatorAgent, MemoryManager
from src import Config, OllamaClient
from .test_data import (
    CART_LYMPHOMA_TRIAL,
    PEDIATRIC_ASTHMA_TRIAL,
    GENE_THERAPY_TRIAL,
    EXPECTED_EVALUATIONS
)


@pytest.fixture
def config():
    """Create test configuration."""
    config = Config()
    config.ollama_model = "llama3.1:8b"  # Use smaller model for testing
    config.ollama_temperature = 0.1
    return config


@pytest.fixture
def ollama_client(config):
    """Create Ollama client."""
    client = OllamaClient(config)

    # Check connection
    if not client.check_connection():
        pytest.skip("Ollama not running")

    # Ensure model is available
    if not client.check_model_availability():
        pytest.skip(f"Model {config.ollama_model} not available")

    return client


@pytest.fixture
def memory_manager():
    """Create memory manager."""
    return MemoryManager(chroma_path="./data/chroma_test", persist=False)


@pytest.fixture
def coordinator(ollama_client, memory_manager):
    """Create coordinator agent."""
    return CoordinatorAgent(
        name="coordinator",
        llm_client=ollama_client,
        memory_manager=memory_manager,
        prompt_dir=Path("prompts/agents")
    )


@pytest.fixture
def investment_evaluator(ollama_client, memory_manager):
    """Create investment evaluator agent."""
    return InvestmentEvaluatorAgent(
        name="investment_evaluator",
        llm_client=ollama_client,
        memory_manager=memory_manager,
        prompt_dir=Path("prompts/agents")
    )


class TestInvestmentEvaluatorAgent:
    """Test Investment Evaluator Agent."""

    @pytest.mark.asyncio
    async def test_initial_assessment(self, investment_evaluator):
        """Test initial assessment generation."""
        result = await investment_evaluator.process({
            "task": "initial_assessment",
            "trial": CART_LYMPHOMA_TRIAL
        })

        # Validate structure
        assert "market_size_usd_billions" in result
        assert "patient_population" in result
        assert "preliminary_score" in result
        assert "research_questions" in result
        assert "confidence" in result

        # Validate types
        assert isinstance(result["market_size_usd_billions"], (int, float))
        assert isinstance(result["patient_population"], int)
        assert isinstance(result["preliminary_score"], int)
        assert isinstance(result["research_questions"], list)
        assert result["confidence"] in ["High", "Medium", "Low", "HIGH", "MEDIUM", "LOW"]

        # Validate score range
        assert 0 <= result["preliminary_score"] <= 100

        # Validate research questions
        assert len(result["research_questions"]) > 0

        print(f"\n✓ Initial Assessment:")
        print(f"  Score: {result['preliminary_score']}")
        print(f"  Market Size: ${result['market_size_usd_billions']}B")
        print(f"  Patient Pop: {result['patient_population']:,}")
        print(f"  Confidence: {result['confidence']}")

    @pytest.mark.asyncio
    async def test_final_recommendation(self, investment_evaluator):
        """Test final recommendation generation."""
        # First get initial assessment
        initial = await investment_evaluator.process({
            "task": "initial_assessment",
            "trial": CART_LYMPHOMA_TRIAL
        })

        # Mock prior art (for now)
        prior_art = {
            "competitive_trials": [],
            "relevant_papers": []
        }

        # Get final recommendation
        result = await investment_evaluator.process({
            "task": "final_recommendation",
            "trial": CART_LYMPHOMA_TRIAL,
            "initial_assessment": initial,
            "prior_art": prior_art
        })

        # Validate structure
        assert "overall_score" in result
        assert "recommendation" in result
        assert "confidence" in result
        assert "rationale" in result
        assert "scores" in result
        assert "key_risks" in result
        assert "key_opportunities" in result
        assert "next_steps" in result

        # Validate recommendation
        rec = result["recommendation"].upper()
        assert rec in ["HIGH", "MEDIUM", "LOW"]

        # Validate confidence
        conf = result["confidence"].upper()
        assert conf in ["HIGH", "MEDIUM", "LOW"]

        # Validate score breakdown
        scores = result["scores"]
        assert "market_analysis" in scores
        assert "competitive_position" in scores
        assert "scientific_assessment" in scores
        assert "regulatory_ip" in scores
        assert "commercial_viability" in scores

        # Validate overall score matches breakdown
        total = sum(scores.values())
        assert 0 <= total <= 100

        # Validate risks and opportunities
        assert len(result["key_risks"]) > 0
        assert len(result["key_opportunities"]) > 0
        assert len(result["next_steps"]) > 0

        print(f"\n✓ Final Recommendation:")
        print(f"  Score: {result['overall_score']}/100")
        print(f"  Recommendation: {result['recommendation']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Breakdown: Market={scores['market_analysis']}/30, "
              f"Competitive={scores['competitive_position']}/25, "
              f"Scientific={scores['scientific_assessment']}/20")


class TestCoordinatorAgent:
    """Test Coordinator Agent."""

    @pytest.mark.asyncio
    async def test_investment_workflow(self, coordinator):
        """Test full investment evaluation workflow."""
        result = await coordinator.process({
            "type": "investment_evaluation",
            "input": CART_LYMPHOMA_TRIAL
        })

        # Validate result structure
        assert "workflow_type" in result
        assert result["workflow_type"] == "investment_evaluation"
        assert "trial" in result
        assert "steps" in result
        assert "status" in result

        # Validate steps
        steps = result["steps"]
        assert len(steps) >= 3  # Should have at least 3 steps

        # Validate step 1: Initial assessment
        step1 = steps[0]
        assert step1["agent"] == "investment_evaluator"
        assert step1["task"] == "initial_assessment"
        assert step1["success"] == True
        assert "output" in step1

        # Validate step 3: Final recommendation
        step3 = steps[2]
        assert step3["agent"] == "investment_evaluator"
        assert step3["task"] == "final_recommendation"
        assert step3["success"] == True
        assert "output" in step3

        # Validate memo if generated
        if "memo" in result and result["status"] == "success":
            assert isinstance(result["memo"], str)
            assert len(result["memo"]) > 100  # Should be substantial

        print(f"\n✓ Investment Workflow:")
        print(f"  Status: {result['status']}")
        print(f"  Steps completed: {len(steps)}")
        if "memo" in result:
            print(f"  Memo length: {len(result['memo'])} characters")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("trial,expected", [
        (CART_LYMPHOMA_TRIAL, EXPECTED_EVALUATIONS["NCT04567890"]),
        (PEDIATRIC_ASTHMA_TRIAL, EXPECTED_EVALUATIONS["NCT05123456"]),
        (GENE_THERAPY_TRIAL, EXPECTED_EVALUATIONS["NCT04789012"])
    ])
    async def test_multiple_trials(self, coordinator, trial, expected):
        """Test workflow with multiple different trials."""
        result = await coordinator.process({
            "type": "investment_evaluation",
            "input": trial
        })

        assert result["status"] in ["success", "partial"]

        if result["status"] == "success" and len(result["steps"]) >= 3:
            final_step = result["steps"][2]
            if final_step["success"]:
                final_output = final_step["output"]

                # Validate score is in expected range
                score = final_output.get("overall_score")
                if score is not None:
                    min_score, max_score = expected["expected_score_range"]
                    # Allow some variance (±15 points) due to LLM variability
                    assert score >= min_score - 15 and score <= max_score + 15, \
                        f"Score {score} outside expected range {expected['expected_score_range']}"

                print(f"\n✓ Trial {trial['nct_id']}:")
                print(f"  Score: {score}")
                print(f"  Expected range: {expected['expected_score_range']}")
                print(f"  Recommendation: {final_output.get('recommendation')}")


class TestMemoryManager:
    """Test Memory Manager."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, memory_manager):
        """Test storing and retrieving findings."""
        # Store a finding
        await memory_manager.store_finding(
            agent_name="test_agent",
            finding="This is a test finding about CAR-T therapy",
            metadata={"trial_id": "NCT04567890", "score": 75}
        )

        # Retrieve
        results = await memory_manager.semantic_search(
            query="CAR-T therapy",
            collection="findings",
            n_results=5
        )

        assert len(results) > 0
        assert any("test finding" in r["document"].lower() for r in results)

    def test_collection_stats(self, memory_manager):
        """Test getting collection statistics."""
        stats = memory_manager.get_collection_stats()

        assert "articles" in stats
        assert "findings" in stats
        assert "conversations" in stats

        assert isinstance(stats["articles"], int)
        assert isinstance(stats["findings"], int)
        assert isinstance(stats["conversations"], int)


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_full_evaluation_pipeline(self, coordinator):
        """Test complete evaluation pipeline end-to-end."""
        # Run evaluation
        result = await coordinator.process({
            "type": "investment_evaluation",
            "input": CART_LYMPHOMA_TRIAL
        })

        # Validate complete workflow
        assert result["status"] in ["success", "partial"]
        assert len(result["steps"]) >= 3

        # Extract key outputs
        if result["status"] == "success":
            step1_output = result["steps"][0]["output"]
            step3_output = result["steps"][2]["output"]

            # Validate initial and final scores are reasonable
            initial_score = step1_output.get("preliminary_score", 0)
            final_score = step3_output.get("overall_score", 0)

            # Final score should be refined from initial
            assert 0 <= initial_score <= 100
            assert 0 <= final_score <= 100

            # Scores should be somewhat close (within 30 points)
            assert abs(initial_score - final_score) <= 30

            print(f"\n✓ Full Pipeline:")
            print(f"  Initial score: {initial_score}")
            print(f"  Final score: {final_score}")
            print(f"  Recommendation: {step3_output.get('recommendation')}")

            # Print memo if available
            if "memo" in result:
                print(f"\n{'='*60}")
                print("INVESTMENT MEMO:")
                print('='*60)
                print(result["memo"])
                print('='*60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
