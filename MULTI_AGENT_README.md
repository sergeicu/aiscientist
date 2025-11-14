# Multi-Agent System - Quick Start

## Overview

The multi-agent system provides intelligent analysis of clinical trials and research through specialized AI agents that collaborate to produce comprehensive evaluations.

### Available Agents

1. **Coordinator Agent** - Orchestrates workflows and synthesizes findings
2. **Investment Evaluator Agent** - Assesses commercial potential of clinical trials
3. **Prior Art Researcher Agent** - Literature and patent research (coming soon)
4. **Hypothesis Generator Agent** - Research gap analysis (coming soon)

---

## Installation

### Prerequisites

1. **Ollama** - Local LLM inference
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh

   # Pull model
   ollama pull llama3.1:8b
   ```

2. **Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **ChromaDB** (for memory)
   - Automatically initialized on first run

---

## Quick Start

### 1. Run the Demo

```bash
python examples/multi_agent_demo.py
```

This will:
- Initialize the multi-agent system
- Evaluate a sample clinical trial
- Generate an investment memo
- Display results in the terminal

### 2. Run Tests

```bash
# Run all agent tests
pytest tests/agents/ -v

# Run specific test
pytest tests/agents/test_multi_agent.py::TestInvestmentEvaluatorAgent::test_initial_assessment -v

# Run with output
pytest tests/agents/ -v -s
```

### 3. Programmatic Usage

```python
import asyncio
from pathlib import Path
from src.agents import CoordinatorAgent, MemoryManager
from src import Config, OllamaClient

async def evaluate_trial(trial_data):
    # Initialize
    config = Config()
    ollama = OllamaClient(config)
    memory = MemoryManager()

    coordinator = CoordinatorAgent(
        name="coordinator",
        llm_client=ollama,
        memory_manager=memory,
        prompt_dir=Path("prompts/agents")
    )

    # Run evaluation
    result = await coordinator.process({
        "type": "investment_evaluation",
        "input": trial_data
    })

    return result

# Run
trial = {
    "nct_id": "NCT04567890",
    "title": "Phase 2 CAR-T Study",
    "phase": "Phase II",
    "indication": "B-cell lymphoma",
    "intervention": "CAR-T therapy",
    "sample_size": 60,
    "primary_outcome": "ORR at 3 months",
    "secondary_outcomes": "DOR, PFS, safety",
    "sponsor": "Boston Children's Hospital",
    "status": "Recruiting"
}

result = asyncio.run(evaluate_trial(trial))
print(result["memo"])
```

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│           Coordinator Agent                     │
│  - Workflow orchestration                       │
│  - Result synthesis                             │
└─────────────┬───────────────────────────────────┘
              │
     ┌────────┴────────┐
     │                 │
     v                 v
┌──────────────┐  ┌──────────────┐
│ Investment   │  │ Prior Art    │
│ Evaluator    │  │ Researcher   │
└──────────────┘  └──────────────┘
     │                 │
     └────────┬────────┘
              │
              v
     ┌────────────────┐
     │ Shared Memory  │
     │  (ChromaDB)    │
     └────────────────┘
```

---

## Workflows

### Investment Evaluation

Evaluates clinical trials for commercial potential.

**Steps:**
1. Initial assessment (quick market analysis)
2. Prior art research (competitive landscape)
3. Final recommendation (comprehensive scoring)
4. Synthesis (investment memo generation)

**Input:**
```python
{
    "type": "investment_evaluation",
    "input": {
        "nct_id": "...",
        "title": "...",
        "phase": "...",
        "indication": "...",
        "intervention": "...",
        "sample_size": 0,
        "primary_outcome": "...",
        "secondary_outcomes": "...",
        "sponsor": "...",
        "status": "..."
    }
}
```

**Output:**
```python
{
    "workflow_type": "investment_evaluation",
    "status": "success",
    "steps": [
        {
            "step": 1,
            "agent": "investment_evaluator",
            "task": "initial_assessment",
            "success": True,
            "output": {...}
        },
        ...
    ],
    "memo": "# Investment Evaluation Memo\n..."
}
```

**Scoring Framework:**

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Market Analysis | 30% | Market size, growth, unmet need |
| Competitive Position | 25% | Differentiation, timing advantage |
| Scientific Assessment | 20% | Novelty, data quality, endpoints |
| Regulatory/IP | 15% | Pathway clarity, patent landscape |
| Commercial Viability | 10% | Reimbursement, partnerships |

**Interpretation:**
- **75-100**: High potential (strong investment case)
- **50-74**: Medium potential (conditional/more data needed)
- **0-49**: Low potential (pass)

---

## Customization

### Modify Evaluation Criteria

Edit `prompts/agents/investment_evaluator.yaml`:

```yaml
system_prompt: |
  You are an expert technology transfer analyst...

  # Modify evaluation framework
  Evaluation Framework:
  1. Market Analysis (35%) - <-- Changed weight
  2. Competitive Position (25%)
  ...

final_recommendation: |
  # Add custom scoring criteria
  ## 1. Market Analysis (Score /35)  <-- Updated max score
  ...
```

### Add New Agent

1. Create agent class:
```python
# src/agents/my_agent.py
from .base_agent import BaseAgent

class MyAgent(BaseAgent):
    async def process(self, task: Dict) -> Dict:
        # Implementation
        pass
```

2. Create prompts:
```yaml
# prompts/agents/my_agent.yaml
system_prompt: |
  You are...

task_template: |
  ...
```

3. Register with coordinator:
```python
# src/agents/coordinator.py
def _initialize_agents(self):
    self.agents["my_agent"] = MyAgent(...)
```

---

## Configuration

### Environment Variables

```bash
# .env
PUBMED_OLLAMA_MODEL=llama3.1:8b
PUBMED_OLLAMA_TEMPERATURE=0.1
PUBMED_OLLAMA_NUM_CTX=4096
```

### Model Selection

**Recommended models:**
- `llama3.1:8b` - Good balance of speed/quality
- `llama3.1:70b` - Best quality, slower
- `gemma2:9b` - Fast alternative

**For testing:**
- `gemma2:2b` - Fastest, less accurate

### Memory Configuration

```python
# Persistent memory (saved to disk)
memory = MemoryManager(
    chroma_path="./data/chroma",
    persist=True
)

# In-memory (for testing)
memory = MemoryManager(persist=False)
```

---

## Troubleshooting

### "Ollama not running"

```bash
# Start Ollama
ollama serve

# Verify
curl http://localhost:11434
```

### "Model not available"

```bash
# Pull model
ollama pull llama3.1:8b

# List available models
ollama list
```

### "Generation failed"

1. Check model is running: `ollama list`
2. Reduce temperature: `PUBMED_OLLAMA_TEMPERATURE=0.05`
3. Try different model: `ollama pull gemma2:9b`
4. Check logs in `logs/` directory

### "JSON parsing errors"

The system uses `json-repair` for automatic fixing. If errors persist:
- Lower temperature (more deterministic)
- Use larger model (better instruction following)
- Check prompt templates for clarity

---

## Performance

### Typical Execution Times

| Workflow | Steps | Time (8B model) | Time (70B model) |
|----------|-------|-----------------|------------------|
| Investment Evaluation | 3 | 1-2 min | 5-8 min |
| Research Discovery | 4 | 2-3 min | 8-12 min |

### Optimization Tips

1. **Use GPU**: Dramatically faster (5-10x)
   ```bash
   # Docker with GPU
   docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 ollama/ollama
   ```

2. **Smaller model for testing**: `gemma2:2b` is 5x faster
3. **Batch evaluations**: Evaluate multiple trials in parallel
4. **Persistent memory**: Avoid re-initializing ChromaDB

---

## Examples

See `examples/` directory:
- `multi_agent_demo.py` - Full investment evaluation demo
- `test_data.py` - Sample clinical trials for testing

See `tests/agents/` for more usage examples.

---

## Roadmap

### Implemented ✅
- [x] Base agent framework
- [x] Coordinator agent
- [x] Investment evaluator agent
- [x] Shared memory system
- [x] Investment evaluation workflow

### Coming Soon 🚧
- [ ] Prior Art Researcher agent
- [ ] Hypothesis Generator agent
- [ ] Research discovery workflow
- [ ] Multi-agent consensus mechanism
- [ ] Report export (PDF, DOCX)
- [ ] Web UI dashboard

---

## Support

- **Documentation**: See `MULTI_AGENT_DESIGN.md` for detailed architecture
- **Examples**: `examples/multi_agent_demo.py`
- **Tests**: `tests/agents/test_multi_agent.py`
- **Issues**: GitHub Issues

---

## License

Part of the AI Scientist project.
