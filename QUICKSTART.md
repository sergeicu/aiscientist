# Quick Start Guide

Get up and running with the PubMed Clinical Trial Classifier in 5 minutes!

## Prerequisites

✅ Python 3.9 or higher
✅ Ollama installed and running

## Step 1: Install Ollama (if not already installed)

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai
```

## Step 2: Pull a Model

```bash
# Recommended model (balanced speed/accuracy)
ollama pull llama3.1:8b

# Alternative: Smaller/faster model
ollama pull gemma2:2b

# Alternative: Larger/more accurate model
ollama pull llama3.1:70b
```

## Step 3: Install Python Dependencies

```bash
cd /path/to/aiscientist

# Install required packages
pip install -r requirements.txt
```

## Step 4: Verify Setup

```bash
# Check Ollama connection and model
python main.py check
```

You should see:
```
✓ Ollama is running
✓ Model 'llama3.1:8b' is available
✓ Generation test successful
✓ All checks passed!
```

## Step 5: Test on Sample Data

```bash
# Test on first 5 articles
python main.py test
```

You'll see a table with results:
```
┌───────┬────────────┬────────────┬──────────┐
│ PMID  │ Is Trial?  │ Confidence │ Method   │
├───────┼────────────┼────────────┼──────────┤
│ 12345 │ ✓ Yes      │ 0.92       │ direct   │
│ 12346 │ ✗ No       │ 0.88       │ direct   │
...
```

## Step 6: Process Full Dataset

```bash
# Process all articles in data/pubmed_data_2000.csv
python main.py process

# Or with custom options
python main.py process --model gemma2:9b --batch-size 20
```

## Understanding the Output

Results are saved to `output/` directory in JSONL format:

```bash
# View first few results
head -5 output/classification_results_*.jsonl | jq
```

Each line contains:
- Article metadata (PMID, title, abstract)
- Classification (is_clinical_trial, confidence, reasoning)
- Trial details (phase, intervention type, sample size, etc.)
- Processing metadata (time, method used)

## Common Issues

### "Cannot connect to Ollama"

**Solution**: Start Ollama
```bash
ollama serve
```

### "Model not found"

**Solution**: Pull the model
```bash
ollama pull llama3.1:8b
```

### "Parsing failed" errors

**Solution**: Try with larger model or lower temperature
```bash
python main.py process --model llama3.1:13b
```

Or edit `.env`:
```bash
cp .env.example .env
# Edit .env: set PUBMED_OLLAMA_TEMPERATURE=0.05
```

## Next Steps

### 1. Customize Prompts
Edit `prompts/clinical_trial_classifier.yaml` to tune classification accuracy.

See [PROMPT_TUNING_GUIDE.md](PROMPT_TUNING_GUIDE.md) for details.

### 2. Process Your Own Data
```bash
python main.py process --input-csv /path/to/your/data.csv
```

Your CSV needs:
- Abstract column (auto-detected)
- PMID/ID column (auto-detected)
- Optional: title, authors, year, journal

### 3. Programmatic Usage

```python
from src import Config, ArticleProcessor

config = Config()
config.input_csv = "data/my_data.csv"

processor = ArticleProcessor(config)
processor.initialize()
results = processor.process_batch(0, 10)
```

See `examples/basic_usage.py` for more examples.

## Performance Tips

**For Speed:**
- Use smaller model: `--model gemma2:2b`
- Reduce examples in prompt (edit YAML)
- Use GPU if available

**For Accuracy:**
- Use larger model: `--model llama3.1:70b`
- Add more examples to prompt
- Lower temperature: `PUBMED_OLLAMA_TEMPERATURE=0.05`

## File Structure

```
aiscientist/
├── main.py                    # CLI (start here!)
├── requirements.txt           # Dependencies
├── .env.example              # Configuration template
├── CLASSIFIER_README.md      # Full documentation
├── QUICKSTART.md             # This file
├── PROMPT_TUNING_GUIDE.md    # Prompt customization guide
│
├── src/                      # Core library
│   ├── models.py             # Data schemas
│   ├── csv_parser.py         # CSV handling
│   ├── ollama_client.py      # LLM client
│   ├── structured_output.py  # JSON parsing (Outlines + json-repair)
│   ├── prompt_loader.py      # Prompt templates
│   └── processor.py          # Main orchestrator
│
├── prompts/                  # Edit these!
│   └── clinical_trial_classifier.yaml
│
├── data/                     # Input CSVs
│   └── pubmed_data_2000.csv
│
├── output/                   # Results go here
└── logs/                     # Log files
```

## Get Help

- 📖 Full docs: [CLASSIFIER_README.md](CLASSIFIER_README.md)
- 🎯 Prompt tuning: [PROMPT_TUNING_GUIDE.md](PROMPT_TUNING_GUIDE.md)
- 💡 Examples: `examples/basic_usage.py`
- 🐛 Issues: [GitHub Issues](https://github.com/sergeicu/AI-Scientist/issues)

## Quick Reference

```bash
# System check
python main.py check

# Test mode (5 articles)
python main.py test

# Full processing
python main.py process

# With options
python main.py process \
  --input-csv data/my_data.csv \
  --output-dir output/my_run \
  --model llama3.1:8b \
  --batch-size 20 \
  --verbose

# Resume interrupted run
python main.py process --resume

# No resume (start fresh)
python main.py process --no-resume
```

## Example Session

```bash
$ python main.py check
✓ Ollama is running
✓ Model 'llama3.1:8b' is available
✓ All checks passed!

$ python main.py test
Processing article 1/5 (PMID: 12345)
✓ Classification: Trial (confidence: 0.92)
...
Processed 5 articles
[Shows results table]

$ python main.py process --batch-size 50
Processing article 1/2000
Processing article 2/2000
...
[Progress updates every 10 articles]
...
✓ Processing complete!

Total Articles: 2000
Clinical Trials: 453 (22.7%)
Non-Trials: 1547 (77.3%)
Results saved to: output/classification_results_20241109_143022.jsonl
```

That's it! You're ready to classify clinical trials. 🎉

For detailed documentation, see [CLASSIFIER_README.md](CLASSIFIER_README.md).
