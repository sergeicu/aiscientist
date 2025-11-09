# PubMed Clinical Trial Classifier

A robust Python tool for classifying PubMed articles as clinical trials using local Ollama language models with guaranteed structured JSON output.

## Features

- **Flexible CSV Parsing**: Automatic column detection with support for various CSV formats
- **Local LLM**: Uses Ollama for complete privacy and control
- **Structured Output**: Hybrid approach combining Outlines (constrained generation) and json-repair for reliable JSON extraction from small models
- **Resume Capability**: Checkpoint-based processing allows resuming interrupted runs
- **Comprehensive Extraction**: Classifies trials and extracts metadata (phase, intervention type, sample size, etc.)
- **Rich CLI**: Beautiful command-line interface with progress tracking
- **Prompt Engineering**: YAML-based prompt templates for easy iteration and customization

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Orchestrator                        │
│                    (processor.py)                            │
└──────────────┬──────────────────────────────────────────────┘
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
┌─────────────┐  ┌──────────────┐
│ CSV Parser  │  │ Ollama Client│
│             │  │              │
└─────────────┘  └──────────────┘
       │               │
       │               ▼
       │         ┌──────────────────┐
       │         │ Prompt Loader    │
       │         │ (YAML templates) │
       │         └──────────────────┘
       │               │
       └───────┬───────┘
               │
               ▼
    ┌────────────────────────┐
    │ Structured Output      │
    │ Handler                │
    │                        │
    │ 1. Outlines (primary)  │
    │ 2. json-repair         │
    │ 3. Regex extraction    │
    └────────────────────────┘
               │
               ▼
    ┌────────────────────────┐
    │ Pydantic Models        │
    │ (validation & types)   │
    └────────────────────────┘
```

## Installation

### Prerequisites

1. **Python 3.9+**
2. **Ollama** installed and running
   ```bash
   # Install Ollama (see https://ollama.ai)
   curl -fsSL https://ollama.ai/install.sh | sh

   # Pull a model (recommended: llama3.1:8b or gemma2:9b)
   ollama pull llama3.1:8b
   ```

### Setup

```bash
# Clone the repository
cd /path/to/aiscientist

# Install dependencies
pip install -r requirements.txt

# (Optional) Copy and customize environment variables
cp .env.example .env
# Edit .env with your preferred settings

# Verify installation
python main.py check
```

## Usage

### Quick Start

```bash
# Test on first 5 articles
python main.py test

# Process all articles in the default CSV
python main.py process

# Process with custom model
python main.py process --model gemma2:9b

# Process specific CSV file
python main.py process --input-csv data/my_articles.csv --output-dir output/my_run
```

### Command Reference

#### `check` - Verify System Setup

```bash
python main.py check [--model MODEL]
```

Checks:
- Ollama connection
- Model availability
- Test generation

#### `test` - Test on Sample Articles

```bash
python main.py test [OPTIONS]

Options:
  --input-csv PATH   Input CSV file
  --model TEXT       Ollama model to use
```

Processes first 5 articles and displays results in a table.

#### `process` - Full Processing

```bash
python main.py process [OPTIONS]

Options:
  --input-csv PATH        Input CSV file (default: data/pubmed_data_2000.csv)
  --output-dir PATH       Output directory (default: output)
  --model TEXT            Ollama model (default: llama3.1:8b)
  --batch-size INTEGER    Articles per checkpoint (default: 10)
  --resume / --no-resume  Resume from checkpoint (default: resume)
  --verbose               Enable debug logging
```

### CSV Format Requirements

The tool auto-detects columns. Your CSV should have:

**Required:**
- Abstract column (e.g., "abstract", "Abstract", "AB")
- PMID or ID column (e.g., "pmid", "PMID", "id")

**Optional:**
- Title, Authors, Journal, Year, DOI, etc.

Example CSV:
```csv
pmid,title,abstract,year,journal
12345,"Study Title","This randomized trial...",2020,"JAMA"
```

## Output Format

Results are saved in **JSONL format** (one JSON object per line):

```jsonl
{"article": {...}, "classification": {...}, "processing_time_seconds": 2.3, "success": true, "parsing_method": "direct"}
```

Each result contains:

```json
{
  "article": {
    "pmid": "12345",
    "title": "...",
    "abstract": "...",
    "year": 2020
  },
  "classification": {
    "is_clinical_trial": true,
    "confidence": 0.92,
    "reasoning": "This is a randomized controlled trial...",
    "trial_phase": "Phase III",
    "study_type": "Interventional",
    "intervention_type": "Drug",
    "sample_size": 689,
    "primary_outcome": "FEV1 improvement",
    "randomized": true,
    "blinded": true
  },
  "processing_time_seconds": 2.3,
  "model_used": "llama3.1:8b",
  "success": true,
  "parsing_method": "direct"
}
```

## Prompt Customization

Prompts are defined in `prompts/clinical_trial_classifier.yaml`:

```yaml
system_prompt: |
  You are an expert medical researcher...

examples:
  - label: "Clinical Trial - Phase III Drug Study"
    abstract: |
      Background: Asthma is a common...
    expected_output:
      is_clinical_trial: true
      confidence: 0.95
      ...
```

**To customize:**
1. Edit `prompts/clinical_trial_classifier.yaml`
2. Modify system prompt, examples, or output schema
3. Run `python main.py test` to validate changes

## Structured Output Strategies

The tool uses a **hybrid approach** for maximum reliability with small models:

### 1. Outlines (Primary)
- Constrains generation at decoding time
- Guarantees valid JSON matching Pydantic schema
- Near-zero overhead (microseconds)
- Works with models as small as 2B parameters

### 2. json-repair (Fallback)
- Repairs malformed JSON post-generation
- Fixes missing quotes, commas, brackets
- Zero dependencies, lightweight

### 3. Regex Extraction (Last Resort)
- Extracts key fields using pattern matching
- Provides partial results when JSON is completely malformed

## Configuration

Configuration via environment variables (`.env`) or CLI options:

| Variable | Default | Description |
|----------|---------|-------------|
| `PUBMED_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `PUBMED_OLLAMA_MODEL` | `llama3.1:8b` | Model to use |
| `PUBMED_OLLAMA_TEMPERATURE` | `0.1` | Sampling temperature |
| `PUBMED_BATCH_SIZE` | `10` | Checkpoint interval |
| `PUBMED_USE_OUTLINES` | `True` | Enable Outlines |
| `PUBMED_USE_JSON_REPAIR` | `True` | Enable json-repair |
| `PUBMED_MAX_RETRIES` | `3` | Retry attempts |

See `.env.example` for all options.

## Checkpoint & Resume

Processing automatically saves checkpoints every N articles (default: 10):

```bash
# Start processing
python main.py process

# Interrupt with Ctrl+C
^C

# Resume later
python main.py process --resume
```

Checkpoints stored in `output/checkpoints/checkpoint.json`.

## Performance Tips

1. **Model Selection**
   - Larger models (7B-13B) are more accurate
   - Smaller models (2B-3B) are faster but may need more prompt tuning
   - Recommended: `llama3.1:8b` or `gemma2:9b`

2. **Batch Processing**
   - Larger batches = fewer checkpoint writes
   - Smaller batches = safer (less re-processing if interrupted)

3. **Concurrent Requests**
   - Set `max_concurrent_requests` to 3-5 if you have GPU
   - Keep at 1 for CPU-only systems

4. **Prompt Optimization**
   - Start with 2 examples, add more if accuracy is low
   - Include edge cases in examples
   - Adjust temperature (lower = more consistent)

## Troubleshooting

### "Cannot connect to Ollama"
```bash
# Check if Ollama is running
ollama list

# Start Ollama
ollama serve
```

### "Model not found"
```bash
# Pull the model
ollama pull llama3.1:8b

# List available models
ollama list
```

### "Parsing failed" errors
- Try enabling both Outlines and json-repair
- Lower temperature (0.05-0.1)
- Add more examples to prompt
- Use larger model

### "Out of memory"
```bash
# Reduce context window
PUBMED_OLLAMA_NUM_CTX=2048 python main.py process

# Or use smaller model
python main.py process --model gemma2:2b
```

## Data Models

Key Pydantic models:

- `Article`: PubMed article metadata
- `ClassificationResult`: LLM output schema
- `ProcessingResult`: Complete result with metadata
- `ProcessingStats`: Aggregated statistics

See `src/models.py` for full schemas.

## Development

### Project Structure

```
aiscientist/
├── main.py                 # CLI entry point
├── requirements.txt        # Dependencies
├── .env.example           # Environment template
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── models.py          # Pydantic data models
│   ├── csv_parser.py      # CSV parsing
│   ├── ollama_client.py   # Ollama API client
│   ├── structured_output.py  # Output parsing
│   ├── prompt_loader.py   # Prompt template loader
│   └── processor.py       # Main orchestrator
├── prompts/
│   └── clinical_trial_classifier.yaml  # Prompt template
├── data/
│   └── pubmed_data_2000.csv
├── output/                # Results (auto-created)
├── logs/                  # Log files (auto-created)
└── CLASSIFIER_README.md   # This file
```

### Adding New Fields

1. Update `ClassificationResult` in `src/models.py`
2. Update prompt template in `prompts/clinical_trial_classifier.yaml`
3. Add examples showing the new field
4. Test with `python main.py test`

## License

Part of the AI Scientist project.

## Contributing

Contributions welcome! Areas for improvement:

- Additional prompt templates for different classification tasks
- Support for other LLM backends (vLLM, llama.cpp)
- Batch processing optimizations
- Additional extraction fields
- Multi-language support

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{pubmed_classifier,
  title={PubMed Clinical Trial Classifier},
  author={AI Scientist Project},
  year={2024},
  url={https://github.com/sergeicu/AI-Scientist}
}
```

## Support

- Issues: [GitHub Issues](https://github.com/sergeicu/AI-Scientist/issues)
- Ollama: [ollama.ai](https://ollama.ai)
- Outlines: [outlines-dev.github.io](https://outlines-dev.github.io/outlines/)
