#!/usr/bin/env python3
"""
Test agents with mock LLM (no Ollama required).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import InvestmentEvaluatorAgent, CoordinatorAgent, MemoryManager


class MockLLMClient:
    """Mock LLM for testing without Ollama."""

    def __init__(self):
        self.calls = []

    async def generate_structured(self, prompt, schema, **kwargs):
        """Return mock structured response."""
        self.calls.append(("generate_structured", prompt, schema))

        # Check for final recommendation FIRST (more specific)
        if any(keyword in prompt.lower() for keyword in ["final recommendation", "comprehensive evaluation", "overall assessment"]):
            return {
                "overall_score": 65,
                "recommendation": "MEDIUM",
                "confidence": "Medium",
                "rationale": "Strong scientific innovation in dual-targeting approach addresses a real limitation of current CAR-T therapies. However, competitive market with established players and timing concerns moderate the investment case. Recommended to monitor Phase 2 data closely and reassess competitive landscape.",
                "scores": {
                    "market_analysis": 22,
                    "competitive_position": 18,
                    "scientific_assessment": 16,
                    "regulatory_ip": 11,
                    "commercial_viability": 8
                },
                "market_size_usd_billions": 3.2,
                "patient_population": 85000,
                "key_risks": [
                    {
                        "risk": "Competitive pressure from established CAR-T products (Kymriah, Yescarta)",
                        "impact": "High"
                    },
                    {
                        "risk": "Manufacturing complexity may impact scalability and COGS",
                        "impact": "Medium"
                    },
                    {
                        "risk": "Small Phase 2 sample size may not demonstrate clear superiority",
                        "impact": "Medium"
                    }
                ],
                "key_opportunities": [
                    {
                        "opportunity": "First-mover advantage in dual-targeting CAR-T space",
                        "value_potential": "High"
                    },
                    {
                        "opportunity": "Potential for breakthrough therapy designation if data is strong",
                        "value_potential": "Medium"
                    },
                    {
                        "opportunity": "Partnership opportunities with major pharma for commercialization",
                        "value_potential": "High"
                    }
                ],
                "next_steps": [
                    "Monitor Phase 2 interim data readout (expected Q2 2024)",
                    "Conduct detailed patent landscape analysis",
                    "Assess competitive programs in development",
                    "Evaluate potential pharma partners for commercial partnership"
                ]
            }

        # Check for initial assessment
        elif any(keyword in prompt.lower() for keyword in ["initial assessment", "preliminary score"]):
            return {
                "market_size_usd_billions": 3.2,
                "market_size_reasoning": "Based on R/R B-cell lymphoma market analysis",
                "patient_population": 85000,
                "patient_population_reasoning": "Approximately 85k patients with R/R B-cell lymphoma in US",
                "current_standard_of_care": "Single-target CAR-T (tisagenlecleucel, axicabtagene ciloleucel)",
                "novel_aspects": [
                    "Dual CD19/CD22 targeting to prevent antigen escape",
                    "Novel manufacturing process with improved expansion"
                ],
                "strengths": [
                    "Addresses major limitation of current CAR-T (antigen escape)",
                    "Strong preclinical data showing improved persistence",
                    "Experienced sponsor with CAR-T expertise"
                ],
                "concerns": [
                    "Competitive market with established CAR-T products",
                    "Manufacturing complexity with dual targeting",
                    "Small sample size (n=60) for Phase 2"
                ],
                "research_questions": [
                    "What is the competitive timeline for other dual-targeting CAR-T programs?",
                    "What is the patent landscape for dual-targeting approaches?",
                    "Are there any published results from similar dual-targeting studies?"
                ],
                "preliminary_score": 68,
                "confidence": "Medium",
                "reasoning": "Innovative approach in competitive market. Strong science but execution risk and market timing concerns."
            }

        # Fallback (shouldn't happen in normal testing)
        return {"status": "mock_response"}

    async def generate(self, prompt, **kwargs):
        """Return mock text response."""
        self.calls.append(("generate", prompt))

        # Mock memo generation
        return """# Investment Evaluation Memo
## Phase 2 Study of Novel CD19/CD22 Dual-Targeting CAR-T

**NCT ID**: NCT04567890
**Overall Score**: 65/100
**Recommendation**: MEDIUM
**Confidence**: Medium

### Executive Summary
This Phase 2 trial evaluates a novel dual-targeting CAR-T therapy for relapsed/refractory B-cell lymphoma. The dual CD19/CD22 targeting addresses a major limitation of current single-target CAR-T therapies (antigen escape). While the scientific innovation is strong, the competitive market dynamics and execution risks result in a moderate investment recommendation.

### Market Opportunity
- **Market Size**: $3.2B (R/R B-cell lymphoma)
- **Patient Population**: ~85,000 patients in US
- **Unmet Need**: High - current CAR-T therapies face 40-60% relapse rates due to antigen escape

### Competitive Position
Current standard of care includes established CAR-T products (Kymriah, Yescarta) with strong clinical data and commercial traction. This dual-targeting approach offers differentiation but faces timing challenges as a late entrant.

### Key Risks
1. **Competitive pressure** from established CAR-T products (Impact: High)
2. **Manufacturing complexity** affecting scalability (Impact: Medium)
3. **Small sample size** in Phase 2 (Impact: Medium)

### Key Opportunities
1. **First-mover advantage** in dual-targeting space (Value: High)
2. **Breakthrough designation** potential (Value: Medium)
3. **Partnership opportunities** with major pharma (Value: High)

### Next Steps
1. Monitor Phase 2 interim data readout (Q2 2024)
2. Conduct patent landscape analysis
3. Assess competitive programs
4. Evaluate partnership opportunities
"""


class MockMemoryManager:
    """Mock memory for testing."""

    def __init__(self):
        self.findings = []

    async def semantic_search(self, query, collection="articles", n_results=10, filter=None):
        return []

    async def store_finding(self, agent_name, finding, metadata):
        self.findings.append((agent_name, finding, metadata))

    def get_collection_stats(self):
        return {"articles": 0, "findings": len(self.findings), "conversations": 0}


async def test_investment_evaluator():
    """Test Investment Evaluator Agent."""
    print("\n" + "="*60)
    print("TEST 1: Investment Evaluator Agent")
    print("="*60)

    mock_llm = MockLLMClient()
    mock_memory = MockMemoryManager()

    evaluator = InvestmentEvaluatorAgent(
        name="investment_evaluator",
        llm_client=mock_llm,
        memory_manager=mock_memory,
        prompt_dir=Path("prompts/agents")
    )

    # Test data
    trial = {
        "nct_id": "NCT04567890",
        "title": "Phase 2 Study of Novel CD19/CD22 Dual-Targeting CAR-T",
        "phase": "Phase II",
        "indication": "Relapsed/refractory B-cell lymphoma",
        "intervention": "CD19/CD22 dual-targeting CAR-T",
        "sample_size": 60,
        "primary_outcome": "Overall response rate at 3 months",
        "secondary_outcomes": "Duration of response, PFS, safety",
        "sponsor": "Boston Children's Hospital",
        "status": "Recruiting"
    }

    # Test initial assessment
    print("\n→ Testing initial assessment...")
    initial = await evaluator.process({
        "task": "initial_assessment",
        "trial": trial
    })

    print(f"✓ Initial assessment complete")
    print(f"  - Preliminary score: {initial['preliminary_score']}")
    print(f"  - Market size: ${initial['market_size_usd_billions']}B")
    print(f"  - Patient population: {initial['patient_population']:,}")
    print(f"  - Confidence: {initial['confidence']}")
    print(f"  - Research questions: {len(initial['research_questions'])}")

    # Validate structure
    assert "preliminary_score" in initial
    assert 0 <= initial["preliminary_score"] <= 100
    assert initial["market_size_usd_billions"] > 0
    assert len(initial["research_questions"]) > 0
    print("✓ Initial assessment validation passed")

    # Test final recommendation
    print("\n→ Testing final recommendation...")
    prior_art = {"competitive_trials": [], "relevant_papers": []}

    final = await evaluator.process({
        "task": "final_recommendation",
        "trial": trial,
        "initial_assessment": initial,
        "prior_art": prior_art
    })

    print(f"✓ Final recommendation complete")
    print(f"  - Overall score: {final['overall_score']}/100")
    print(f"  - Recommendation: {final['recommendation']}")
    print(f"  - Confidence: {final['confidence']}")
    print(f"  - Breakdown: Market={final['scores']['market_analysis']}/30, "
          f"Competitive={final['scores']['competitive_position']}/25, "
          f"Scientific={final['scores']['scientific_assessment']}/20")

    # Validate structure
    assert "overall_score" in final
    assert final["recommendation"] in ["HIGH", "MEDIUM", "LOW"]
    assert "key_risks" in final
    assert "key_opportunities" in final
    assert len(final["key_risks"]) > 0
    assert len(final["next_steps"]) > 0
    print("✓ Final recommendation validation passed")

    print("\n✓ Investment Evaluator Agent: ALL TESTS PASSED")


async def test_coordinator():
    """Test Coordinator Agent."""
    print("\n" + "="*60)
    print("TEST 2: Coordinator Agent - Investment Workflow")
    print("="*60)

    mock_llm = MockLLMClient()
    mock_memory = MockMemoryManager()

    coordinator = CoordinatorAgent(
        name="coordinator",
        llm_client=mock_llm,
        memory_manager=mock_memory,
        prompt_dir=Path("prompts/agents")
    )

    trial = {
        "nct_id": "NCT04567890",
        "title": "Phase 2 Study of Novel CD19/CD22 Dual-Targeting CAR-T",
        "phase": "Phase II",
        "indication": "Relapsed/refractory B-cell lymphoma",
        "intervention": "CD19/CD22 dual-targeting CAR-T",
        "sample_size": 60,
        "primary_outcome": "Overall response rate at 3 months",
        "secondary_outcomes": "Duration of response, PFS, safety",
        "sponsor": "Boston Children's Hospital",
        "status": "Recruiting"
    }

    print("\n→ Running investment evaluation workflow...")
    result = await coordinator.process({
        "type": "investment_evaluation",
        "input": trial
    })

    print(f"\n✓ Workflow complete")
    print(f"  - Status: {result['status']}")
    print(f"  - Steps completed: {len(result['steps'])}")

    # Show step results
    for i, step in enumerate(result["steps"], 1):
        status = "✓" if step["success"] else "✗"
        print(f"  {status} Step {step['step']}: {step['agent']} - {step['task']}")

    # Validate workflow
    assert result["workflow_type"] == "investment_evaluation"
    assert "steps" in result
    assert len(result["steps"]) >= 3

    # Validate steps
    assert result["steps"][0]["agent"] == "investment_evaluator"
    assert result["steps"][0]["task"] == "initial_assessment"
    assert result["steps"][0]["success"] == True

    assert result["steps"][2]["agent"] == "investment_evaluator"
    assert result["steps"][2]["task"] == "final_recommendation"
    assert result["steps"][2]["success"] == True

    # Check memo
    if "memo" in result:
        print(f"  - Memo generated: {len(result['memo'])} characters")
        assert len(result["memo"]) > 100

    print("\n✓ Coordinator Agent: ALL TESTS PASSED")

    # Display memo
    if "memo" in result:
        print("\n" + "="*60)
        print("GENERATED INVESTMENT MEMO")
        print("="*60)
        print(result["memo"])
        print("="*60)


async def test_memory_manager():
    """Test Memory Manager."""
    print("\n" + "="*60)
    print("TEST 3: Memory Manager")
    print("="*60)

    memory = MockMemoryManager()

    # Test storing findings
    print("\n→ Testing finding storage...")
    await memory.store_finding(
        agent_name="test_agent",
        finding="This is a test finding about CAR-T therapy",
        metadata={"trial_id": "NCT04567890", "score": 65}
    )

    stats = memory.get_collection_stats()
    print(f"✓ Finding stored")
    print(f"  - Findings count: {stats['findings']}")

    assert stats["findings"] == 1
    print("\n✓ Memory Manager: ALL TESTS PASSED")


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("MULTI-AGENT SYSTEM - MOCK TESTS")
    print("Testing agent logic without Ollama")
    print("="*60)

    try:
        await test_investment_evaluator()
        await test_coordinator()
        await test_memory_manager()

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nThe multi-agent system is working correctly!")
        print("To test with real LLM:")
        print("  1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("  2. Pull model: ollama pull llama3.1:8b")
        print("  3. Run: python examples/multi_agent_demo.py")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
