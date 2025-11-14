# Data Pipeline Orchestrator

A unified data collection pipeline that coordinates PubMed scraping, ClinicalTrials.gov scraping, and author network extraction.

## Features

- **Unified Configuration**: Single YAML file to configure all scrapers
- **Parallel Execution**: Run multiple scrapers concurrently
- **Progress Tracking**: Real-time progress updates with Rich UI
- **Error Handling**: Graceful error handling with retry logic
- **Standardized Storage**: Consistent directory structure and file naming
- **Comprehensive Reports**: Detailed execution reports in JSON format

## Architecture

```
src/pipeline/
├── __init__.py          # Package exports
├── config.py            # Configuration models (Pydantic)
├── storage.py           # Unified storage manager
└── orchestrator.py      # Main pipeline orchestrator

data/
├── config/
│   └── pipeline_config.yaml    # Pipeline configuration
├── raw/
│   ├── pubmed/                 # Raw PubMed data
│   ├── clinicaltrials/         # Raw ClinicalTrials data
│   └── author_networks/        # Extracted networks
├── processed/
│   └── unified_dataset_*.json  # All data in unified format
└── logs/
    └── pipeline_report_*.json  # Execution reports
```

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Required packages:
# - biopython (PubMed API)
# - httpx, aiohttp (HTTP clients)
# - pydantic (configuration validation)
# - pyyaml (YAML parsing)
# - rich (terminal UI)
# - neo4j (author network - optional)
```

## Configuration

Create a configuration file (e.g., `data/config/pipeline_config.yaml`):

```yaml
pubmed:
  email: researcher@example.com
  api_key: null  # Optional: NCBI API key
  institutions:
    - Harvard Medical School
    - Mayo Clinic
  max_results_per_institution: 1000
  rate_limit_per_second: 3.0

clinicaltrials:
  sponsors:
    - Massachusetts General Hospital
    - Johns Hopkins Hospital
  max_results_per_sponsor: 500
  rate_limit_per_second: 2.0

output_dir: ./data
log_level: INFO
parallel_workers: 3
retry_failed: true
max_retries: 3
```

## Usage

### Command Line

```bash
# Set PYTHONPATH to include src directory
export PYTHONPATH=/path/to/aiscientist/src:$PYTHONPATH

# Run the pipeline
python -m pipeline.orchestrator data/config/pipeline_config.yaml

# With custom options
python -m pipeline.orchestrator data/config/pipeline_config.yaml \
    --output-dir ./my_data \
    --log-level DEBUG \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password mypassword
```

### Python API

```python
import asyncio
from pipeline.config import load_config
from pipeline.orchestrator import PipelineOrchestrator

async def main():
    # Load configuration
    config = load_config("data/config/pipeline_config.yaml")

    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        config=config,
        output_dir="./data"
    )

    # Run full pipeline
    report = await orchestrator.run_full_pipeline()

    print(f"Pipeline completed!")
    print(f"Total papers: {report['total_papers']}")
    print(f"Total trials: {report['total_trials']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Running Individual Components

```python
import asyncio
from pipeline.config import load_config
from pipeline.orchestrator import PipelineOrchestrator

async def main():
    config = load_config("data/config/pipeline_config.yaml")
    orchestrator = PipelineOrchestrator(config=config)

    # Run only PubMed scraper
    pubmed_results = await orchestrator.run_pubmed_scraper()

    # Run only ClinicalTrials scraper
    ct_results = await orchestrator.run_clinicaltrials_scraper()

    # Extract author networks (after scraping)
    network_results = await orchestrator.extract_author_networks()

asyncio.run(main())
```

## Output Structure

### Raw Data Files

**PubMed Papers** (`data/raw/pubmed/papers_<institution>_<timestamp>.json`):
```json
[
  {
    "pmid": "12345678",
    "title": "Research Paper Title",
    "abstract": "Paper abstract...",
    "authors": [...],
    "publication_date": "2024-01-15",
    "institution": "Harvard Medical School"
  }
]
```

**Clinical Trials** (`data/raw/clinicaltrials/trials_<sponsor>_<timestamp>.json`):
```json
[
  {
    "nct_id": "NCT12345678",
    "title": "Trial Title",
    "status": "Recruiting",
    "sponsor": "MGH"
  }
]
```

**Author Networks** (`data/raw/author_networks/network_<source>_<timestamp>.json`):
```json
{
  "authors": 150,
  "collaborations": 450,
  "papers": 1000,
  "timestamp": "2024-01-15T10:30:00"
}
```

### Unified Dataset

**Unified Dataset** (`data/processed/unified_dataset_<timestamp>.json`):
```json
{
  "papers": [...],
  "trials": [...],
  "created_at": "2024-01-15T10:30:00"
}
```

### Execution Report

**Pipeline Report** (`data/logs/pipeline_report_<timestamp>.json`):
```json
{
  "status": "completed",
  "start_time": "2024-01-15T10:00:00",
  "end_time": "2024-01-15T10:30:00",
  "duration_seconds": 1800,
  "pubmed": [...],
  "clinicaltrials": [...],
  "author_networks": [...],
  "total_papers": 2000,
  "total_trials": 500,
  "unified_dataset": "./data/processed/unified_dataset_20240115_103000.json"
}
```

## Testing

```bash
# Run all tests
PYTHONPATH=src pytest tests/pipeline/ -v

# Run specific test file
PYTHONPATH=src pytest tests/pipeline/test_config.py -v

# Run with coverage
PYTHONPATH=src pytest tests/pipeline/ --cov=pipeline --cov-report=html
```

## Error Handling

The pipeline includes robust error handling:

- **Rate Limiting**: Respects API rate limits with configurable delays
- **Retries**: Automatic retry on transient failures
- **Partial Failures**: Continues execution even if some institutions/sponsors fail
- **Detailed Logging**: All errors logged with context

## Performance Considerations

- **Parallel Workers**: Configure `parallel_workers` (1-10) based on API limits
- **Batch Size**: PubMed fetches in batches of 500 articles
- **Rate Limits**:
  - PubMed: 3 req/sec (10 with API key)
  - ClinicalTrials: 2 req/sec (recommended)

## Dependencies

This module integrates with:

- **scrapers.pubmed_scraper**: PubMed API scraper (Task 1)
- **scrapers.clinicaltrials_scraper**: ClinicalTrials.gov scraper (Task 2)
- **network.network_builder**: Author network builder (Task 3)

## Future Enhancements

- [ ] Support for additional data sources
- [ ] Advanced filtering options
- [ ] Real-time data streaming
- [ ] Web dashboard for monitoring
- [ ] Automated scheduling (cron/airflow)
- [ ] Data quality validation
- [ ] Export to multiple formats (CSV, Parquet, etc.)
