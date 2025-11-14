# Repository Restructuring Guide: Integrating Original Code with Tasks 1-13

This guide provides comprehensive instructions for restructuring your repository to integrate the existing clinical trial classifier with all 13 completed parallel tasks.

---

## Current State Analysis

### Existing Code (Original PubMed Classifier)
**Purpose**: Clinical trial classification using Ollama (local LLM)

**Files:**
- `main.py` - CLI for classification
- `src/*.py` - Core classification library (processor, ollama_client, structured_output, etc.)
- `prompts/clinical_trial_classifier.yaml` - Classification prompts
- `QUICKSTART.md`, `CLASSIFIER_README.md`, `PROMPT_TUNING_GUIDE.md` - Documentation
- `requirements.txt` - Dependencies
- `.env.example` - Configuration template

**Functionality**:
- Takes PubMed articles from CSV
- Classifies them as clinical trials or not using LLM
- Extracts trial details (phase, intervention, sample size)
- Outputs structured JSONL results

### Tasks 1-13 (Completed Separately)
**Purpose**: Complete research intelligence platform

**Modules:**
- **Data Acquisition** (Tasks 1-3): PubMed scraper, ClinicalTrials.gov scraper, Author networks
- **Embeddings & Clustering** (Tasks 8-10): Sentence embeddings, UMAP+HDBSCAN clustering, Topic labeling
- **Graph Database** (Tasks 4-5): Neo4j setup, Graph analytics
- **Visualization** (Tasks 6-7): UMAP viz, Network & geographic viz
- **Integration** (Tasks 11-13): Pipeline orchestrator, Graph integration, Streamlit dashboard

### Future Phase 4 (Mentioned but Not Yet Implemented)
**Purpose**: Multi-agent system

**Planned:**
- `src/agents/` - Multi-agent coordination
- `prompts/agents/` - Agent prompts

---

## The Challenge

**Key Issue**: The original code and Tasks 1-13 serve different but complementary purposes:

1. **Original Code** = **Phase 2.5: Classification**
   - Processes already-collected PubMed articles
   - Uses LLM to classify and extract trial details
   - NOT covered in Tasks 1-13

2. **Tasks 1-13** = **Data Collection → Analytics → Dashboard**
   - Task 1 scrapes PubMed (data collection)
   - Tasks 2-13 analyze and visualize
   - Does NOT include classification

**The Solution**: Integrate both as complementary modules in a unified architecture.

---

## Proposed Final Structure

```
aiscientist/
├── README.md                          # Main project README
├── requirements.txt                   # Consolidated dependencies
├── setup.py                          # Package configuration
├── .env.example                      # Environment template
├── .gitignore                        # Git ignore rules
│
├── docs/                             # 📚 All documentation
│   ├── README.md                     # Documentation index
│   ├── quickstart/
│   │   ├── classifier_quickstart.md  # (from QUICKSTART.md)
│   │   ├── pipeline_quickstart.md    # New: Pipeline quick start
│   │   └── dashboard_quickstart.md   # New: Dashboard quick start
│   ├── guides/
│   │   ├── classifier_guide.md       # (from CLASSIFIER_README.md)
│   │   ├── prompt_tuning.md          # (from PROMPT_TUNING_GUIDE.md)
│   │   ├── integration_guide.md      # (from tasks/INTEGRATION_GUIDE.md)
│   │   ├── deployment_guide.md       # (from tasks/DEPLOYMENT_GUIDE.md)
│   │   └── quality_validation.md     # (from tasks/QUALITY_VALIDATION_CHECKLIST.md)
│   └── architecture/
│       ├── system_architecture.md    # New: Complete system overview
│       ├── data_flow.md              # New: Data flow diagrams
│       └── module_dependencies.md    # New: Module dependency map
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_acquisition/             # ✅ Tasks 1, 2, 3
│   │   ├── __init__.py
│   │   ├── pubmed_scraper/           # Task 1
│   │   ├── clinicaltrials_scraper/   # Task 2
│   │   └── author_network/           # Task 3
│   │
│   ├── classification/               # 🔥 NEW: Original code goes here
│   │   ├── __init__.py
│   │   ├── classifier.py             # (from src/processor.py - renamed)
│   │   ├── models.py                 # (from src/models.py)
│   │   ├── ollama_client.py          # (from src/ollama_client.py)
│   │   ├── structured_output.py      # (from src/structured_output.py)
│   │   ├── prompt_loader.py          # (from src/prompt_loader.py)
│   │   ├── csv_parser.py             # (from src/csv_parser.py)
│   │   └── config.py                 # (from src/config.py)
│   │
│   ├── embeddings/                   # ✅ Tasks 8, 9, 10
│   │   ├── __init__.py
│   │   ├── embeddings_vectorstore/   # Task 8
│   │   ├── clustering_pipeline/      # Task 9
│   │   └── cluster_labeling/         # Task 10
│   │
│   ├── graph_database/               # ✅ Tasks 4, 5
│   │   ├── __init__.py
│   │   ├── neo4j_setup/              # Task 4
│   │   └── graph_analytics/          # Task 5
│   │
│   ├── visualization/                # ✅ Tasks 6, 7
│   │   ├── __init__.py
│   │   ├── umap_viz/                 # Task 6
│   │   └── network_geo_viz/          # Task 7
│   │
│   ├── pipeline/                     # ✅ Task 11
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── config.py
│   │   └── storage.py
│   │
│   ├── graph_integration/            # ✅ Task 12
│   │   ├── __init__.py
│   │   └── loader.py
│   │
│   ├── dashboard/                    # ✅ Task 13
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── pages/
│   │   ├── components/
│   │   └── utils/
│   │
│   ├── agents/                       # 🚀 FUTURE: Phase 4 multi-agent
│   │   ├── __init__.py
│   │   ├── coordinator.py
│   │   ├── research_agent.py
│   │   └── analysis_agent.py
│   │
│   └── common/                       # 🔥 NEW: Shared utilities
│       ├── __init__.py
│       ├── utils.py                  # Common utilities
│       ├── logging_config.py         # Shared logging
│       └── constants.py              # Shared constants
│
├── prompts/                          # 📝 All prompt templates
│   ├── classification/
│   │   └── clinical_trial_classifier.yaml  # (from prompts/)
│   ├── agents/                       # 🚀 FUTURE: Phase 4
│   │   ├── coordinator.yaml
│   │   └── researcher.yaml
│   └── README.md                     # Prompt documentation
│
├── cli/                              # 🔥 NEW: Command-line interfaces
│   ├── __init__.py
│   ├── main.py                       # (from root main.py - adapted)
│   ├── classifier_cli.py             # Classification commands
│   ├── pipeline_cli.py               # Pipeline commands
│   └── dashboard_cli.py              # Dashboard commands
│
├── tests/                            # ✅ All tests
│   ├── data_acquisition/
│   ├── classification/               # Tests for classifier
│   ├── embeddings/
│   ├── graph_database/
│   ├── visualization/
│   ├── pipeline/
│   ├── graph_integration/
│   └── dashboard/
│
├── data/                             # 📊 Data storage
│   ├── config/
│   │   └── pipeline_config.yaml
│   ├── raw/
│   │   ├── pubmed/
│   │   ├── clinicaltrials/
│   │   └── csv_imports/              # For classifier CSV inputs
│   ├── processed/
│   └── embeddings/
│
├── output/                           # 📤 Generated outputs
│   ├── classifications/              # Classifier results
│   ├── reports/
│   └── exports/
│
├── logs/                             # 📝 Log files
├── examples/                         # 💡 Usage examples
│   ├── basic_usage.py
│   ├── classifier_example.py         # How to use classifier
│   ├── pipeline_example.py           # How to use pipeline
│   └── end_to_end_workflow.py        # Complete workflow example
│
├── scripts/                          # 🛠️ Utility scripts
│   ├── migrate_old_code.sh           # Helper for this restructuring
│   ├── setup_dev_env.sh              # Development setup
│   └── run_integration_test.sh       # Integration testing
│
├── .streamlit/                       # Streamlit config
│   └── config.toml
│
└── tasks/                            # 📋 Task specifications (keep for reference)
    ├── README.md
    ├── task_01_pubmed_scraper/
    ├── ... (all task specs)
    └── (merge instructions, guides, etc.)
```

---

## Step-by-Step Restructuring Instructions

### Phase 1: Preparation (Before Merge)

#### Step 1.1: Create New Branch

```bash
# Create restructuring branch
git checkout claude/design-pubmed-tool-instructions-01PFJhtfpgb1Ws87qi26s9cc
git checkout -b claude/restructure-integration-<session-id>
```

#### Step 1.2: Backup Current State

```bash
# Create backup branch
git branch backup/pre-restructure-$(date +%Y%m%d)

# Or create backup directory
mkdir -p ../aiscientist_backup
cp -r . ../aiscientist_backup/
```

#### Step 1.3: Create New Directory Structure

```bash
# Create new directories
mkdir -p docs/{quickstart,guides,architecture}
mkdir -p src/{classification,common,agents}
mkdir -p prompts/{classification,agents}
mkdir -p cli
mkdir -p output/classifications
mkdir -p data/raw/csv_imports
mkdir -p scripts
mkdir -p examples
```

---

### Phase 2: Move Original Code (Classification Module)

#### Step 2.1: Move Source Files

```bash
# Move classification source files
mv src/processor.py src/classification/classifier.py
mv src/models.py src/classification/models.py
mv src/ollama_client.py src/classification/ollama_client.py
mv src/structured_output.py src/classification/structured_output.py
mv src/prompt_loader.py src/classification/prompt_loader.py
mv src/csv_parser.py src/classification/csv_parser.py
mv src/config.py src/classification/config.py

# Keep src/__init__.py for top-level package
# Create classification __init__.py
cat > src/classification/__init__.py << 'EOF'
"""Clinical Trial Classification Module.

This module provides functionality for classifying PubMed articles as
clinical trials using local LLM models (Ollama).
"""

from .classifier import ArticleProcessor
from .models import Article, ClassificationResult, TrialDetails
from .config import Config
from .ollama_client import OllamaClient

__all__ = [
    'ArticleProcessor',
    'Article',
    'ClassificationResult',
    'TrialDetails',
    'Config',
    'OllamaClient',
]
EOF
```

#### Step 2.2: Update Imports in Classifier Module

Edit `src/classification/classifier.py` (formerly processor.py):

```python
# OLD imports:
from src.models import Article, ClassificationResult
from src.ollama_client import OllamaClient
from src.structured_output import extract_structured_output

# NEW imports:
from .models import Article, ClassificationResult
from .ollama_client import OllamaClient
from .structured_output import extract_structured_output
```

Apply similar changes to all files in `src/classification/`.

#### Step 2.3: Move Prompts

```bash
# Move prompt file
mv prompts/clinical_trial_classifier.yaml prompts/classification/

# Create prompts __init__.py
cat > prompts/README.md << 'EOF'
# Prompt Templates

## Classification Prompts
- `classification/clinical_trial_classifier.yaml` - LLM prompts for classifying clinical trials

## Agent Prompts (Future - Phase 4)
- `agents/` - Multi-agent system prompts
EOF
```

#### Step 2.4: Move Documentation

```bash
# Move documentation
mv QUICKSTART.md docs/quickstart/classifier_quickstart.md
mv CLASSIFIER_README.md docs/guides/classifier_guide.md
mv PROMPT_TUNING_GUIDE.md docs/guides/prompt_tuning.md

# Create docs index
cat > docs/README.md << 'EOF'
# Documentation Index

## Quick Start Guides
- [Classifier Quick Start](quickstart/classifier_quickstart.md) - Using the clinical trial classifier
- [Pipeline Quick Start](quickstart/pipeline_quickstart.md) - Using the data pipeline
- [Dashboard Quick Start](quickstart/dashboard_quickstart.md) - Using the Streamlit dashboard

## Detailed Guides
- [Classifier Guide](guides/classifier_guide.md) - Complete classifier documentation
- [Prompt Tuning](guides/prompt_tuning.md) - Customizing LLM prompts
- [Integration Guide](guides/integration_guide.md) - Integrating all modules
- [Deployment Guide](guides/deployment_guide.md) - Production deployment
- [Quality Validation](guides/quality_validation.md) - Testing and validation

## Architecture
- [System Architecture](architecture/system_architecture.md) - Complete system overview
- [Data Flow](architecture/data_flow.md) - How data flows through the system
- [Module Dependencies](architecture/module_dependencies.md) - Module relationships
EOF
```

#### Step 2.5: Create New CLI

```bash
# Move and adapt main.py
mv main.py cli/main.py

# Create classifier-specific CLI
cat > cli/classifier_cli.py << 'EOF'
"""Clinical Trial Classifier CLI commands."""

import click
from pathlib import Path
from src.classification import ArticleProcessor, Config

@click.group()
def classifier():
    """Clinical trial classification commands."""
    pass

@classifier.command()
@click.option('--input-csv', type=click.Path(exists=True), required=True)
@click.option('--output-dir', type=click.Path(), default='output/classifications')
@click.option('--model', default='llama3.1:8b')
def classify(input_csv, output_dir, model):
    """Classify articles in CSV file as clinical trials."""
    config = Config()
    config.input_csv = Path(input_csv)
    config.output_dir = Path(output_dir)
    config.ollama_model = model

    processor = ArticleProcessor(config)
    processor.initialize()

    # Process all articles
    results = processor.process_all()

    click.echo(f"✓ Classified {len(results)} articles")
    click.echo(f"Results saved to: {output_dir}")

# ... (keep other commands from original main.py)
EOF
```

Update `cli/main.py` to include both classifier and new pipeline commands:

```python
#!/usr/bin/env python3
"""AI Scientist - Unified CLI."""

import click
from cli.classifier_cli import classifier
from cli.pipeline_cli import pipeline
from cli.dashboard_cli import dashboard

@click.group()
@click.version_option(version="2.0.0")
def cli():
    """AI Scientist - Research Intelligence Platform

    Unified CLI for:
    - Clinical trial classification
    - Data collection pipeline
    - Analytics and visualization dashboard
    """
    pass

# Register command groups
cli.add_command(classifier)
cli.add_command(pipeline)
cli.add_command(dashboard)

if __name__ == '__main__':
    cli()
```

---

### Phase 3: Merge Tasks 1-13

Follow the complete merge process from `MERGE_INSTRUCTIONS_ALL_TASKS.md`:

```bash
# 1. Fetch all task branches
git fetch --all

# 2. Merge each task into appropriate directory
# This will create src/data_acquisition/, src/embeddings/, etc.

# Follow wave-based merge from MERGE_INSTRUCTIONS_ALL_TASKS.md
# ... (complete merge process)
```

After merge, your `src/` will have:
- `src/classification/` (original code)
- `src/data_acquisition/` (Tasks 1-3)
- `src/embeddings/` (Tasks 8-10)
- `src/graph_database/` (Tasks 4-5)
- `src/visualization/` (Tasks 6-7)
- `src/pipeline/` (Task 11)
- `src/graph_integration/` (Task 12)
- `src/dashboard/` (Task 13)

---

### Phase 4: Update Imports and Dependencies

#### Step 4.1: Update Top-Level `__init__.py`

Edit `src/__init__.py`:

```python
"""AI Scientist - Research Intelligence Platform.

A comprehensive platform for research data collection, classification,
analysis, and visualization.
"""

# Classification module (original code)
from src.classification import (
    ArticleProcessor,
    Config as ClassifierConfig,
    Article,
    ClassificationResult,
)

# Data Acquisition (Tasks 1-3)
from src.data_acquisition.pubmed_scraper import PubMedScraper
from src.data_acquisition.clinicaltrials_scraper import ClinicalTrialsScraper
from src.data_acquisition.author_network import AuthorNetworkExtractor

# Pipeline (Task 11)
from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.config import PipelineConfig

# Dashboard (Task 13)
# (Imported separately as it's a Streamlit app)

__version__ = '2.0.0'

__all__ = [
    # Classification
    'ArticleProcessor',
    'ClassifierConfig',
    'Article',
    'ClassificationResult',

    # Data Acquisition
    'PubMedScraper',
    'ClinicalTrialsScraper',
    'AuthorNetworkExtractor',

    # Pipeline
    'PipelineOrchestrator',
    'PipelineConfig',
]
```

#### Step 4.2: Consolidate Requirements

Create unified `requirements.txt`:

```txt
# ============================================
# Core Dependencies
# ============================================
python-dotenv>=1.0.0
pydantic>=2.0.0
pyyaml>=6.0
click>=8.1.0
loguru>=0.7.0
rich>=13.0.0

# ============================================
# Classification Module (Original Code)
# ============================================
ollama>=0.1.0
outlines>=0.0.31
json-repair>=0.7.0
tenacity>=8.2.0

# ============================================
# Data Acquisition (Tasks 1, 2, 3)
# ============================================
requests>=2.31.0
aiohttp>=3.8.0
httpx>=0.24.0
beautifulsoup4>=4.12.0
lxml>=4.9.0

# ============================================
# Graph Database (Tasks 4, 5, 12)
# ============================================
neo4j>=5.12.0
networkx>=3.1

# ============================================
# Embeddings & Clustering (Tasks 8, 9, 10)
# ============================================
sentence-transformers>=2.2.0
chromadb>=0.4.0
umap-learn>=0.5.0
hdbscan>=0.8.0
scikit-learn>=1.3.0

# ============================================
# Visualization (Tasks 6, 7)
# ============================================
plotly>=5.17.0
folium>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0

# ============================================
# Dashboard (Task 13)
# ============================================
streamlit>=1.28.0
streamlit-folium>=0.15.0

# ============================================
# Common
# ============================================
numpy>=1.24.0
pandas>=2.0.0
asyncio-throttle>=1.0.0

# ============================================
# Testing
# ============================================
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-mock>=3.11.0
pytest-cov>=4.1.0

# ============================================
# Development
# ============================================
black>=23.0.0
ruff>=0.0.280
mypy>=1.4.0
```

---

### Phase 5: Create Integration Examples

#### Step 5.1: End-to-End Workflow Example

Create `examples/end_to_end_workflow.py`:

```python
"""End-to-end workflow demonstrating all modules working together."""

import asyncio
from pathlib import Path

# Phase 1: Data Collection
from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.config import load_config

# Phase 2: Classification
from src.classification import ArticleProcessor, ClassifierConfig

# Phase 3: Embeddings & Clustering
from src.embeddings.embeddings_vectorstore import ArticleEmbedder, VectorStore
from src.embeddings.clustering_pipeline import ClusteringPipeline

# Phase 4: Graph Integration
from src.graph_integration.loader import GraphLoader
from src.graph_database.neo4j_setup import Neo4jConnection

async def main():
    """Run complete workflow."""

    print("=" * 60)
    print("AI Scientist - Complete Workflow Demo")
    print("=" * 60)

    # Step 1: Collect data from PubMed and ClinicalTrials.gov
    print("\n[Step 1] Collecting data...")
    config = load_config("data/config/pipeline_config.yaml")
    orchestrator = PipelineOrchestrator(config, output_dir="./data")
    pipeline_report = await orchestrator.run_full_pipeline()
    print(f"✓ Collected {pipeline_report['total_papers']} papers")

    # Step 2: Classify papers as clinical trials
    print("\n[Step 2] Classifying papers...")
    classifier_config = ClassifierConfig()
    classifier_config.input_csv = Path("data/processed/unified_dataset.csv")
    classifier_config.output_dir = Path("output/classifications")

    processor = ArticleProcessor(classifier_config)
    processor.initialize()
    classifications = processor.process_all()
    print(f"✓ Classified {len(classifications)} papers")

    # Step 3: Generate embeddings
    print("\n[Step 3] Generating embeddings...")
    embedder = ArticleEmbedder(model_name='all-MiniLM-L6-v2')
    papers = [...]  # Load papers
    embeddings = embedder.embed_batch(papers)
    print(f"✓ Generated embeddings for {len(embeddings)} papers")

    # Step 4: Cluster papers
    print("\n[Step 4] Clustering papers...")
    pipeline = ClusteringPipeline()
    labels, reduced = pipeline.fit_predict(embeddings)
    print(f"✓ Found {len(set(labels))} clusters")

    # Step 5: Load into Neo4j
    print("\n[Step 5] Loading into Neo4j...")
    neo4j_conn = Neo4jConnection(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )
    loader = GraphLoader(neo4j_conn)
    network_data = {...}  # Load network
    loader.load_full_network(network_data)
    print("✓ Loaded into Neo4j")

    # Step 6: Launch dashboard
    print("\n[Step 6] Dashboard ready!")
    print("Run: streamlit run src/dashboard/app.py")
    print("\n" + "=" * 60)
    print("✓ Workflow complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

#### Step 5.2: Classifier-Only Example

Create `examples/classifier_example.py`:

```python
"""Using just the classifier module."""

from pathlib import Path
from src.classification import ArticleProcessor, Config

def main():
    """Classify articles in CSV file."""

    # Configure
    config = Config()
    config.input_csv = Path("data/raw/csv_imports/pubmed_articles.csv")
    config.output_dir = Path("output/classifications")
    config.ollama_model = "llama3.1:8b"

    # Initialize
    processor = ArticleProcessor(config)
    processor.initialize()

    # Process
    results = processor.process_all()

    # Results
    trials = [r for r in results if r.is_clinical_trial]
    print(f"Found {len(trials)} clinical trials out of {len(results)} articles")

if __name__ == "__main__":
    main()
```

---

### Phase 6: Update Documentation

#### Step 6.1: Create System Architecture Document

Create `docs/architecture/system_architecture.md`:

```markdown
# System Architecture

## Overview

The AI Scientist platform consists of 5 main modules:

### 1. Data Acquisition Module (Tasks 1-3)
- **PubMed Scraper**: Collects research papers by institution
- **ClinicalTrials.gov Scraper**: Collects trials by sponsor
- **Author Network Extractor**: Builds collaboration networks

### 2. Classification Module (Original Code)
- **Clinical Trial Classifier**: Uses LLM to classify papers
- **Structured Output Parser**: Extracts trial details
- **Ollama Integration**: Local LLM inference

### 3. Embeddings & Clustering Module (Tasks 8-10)
- **Embeddings Generator**: Sentence-BERT embeddings
- **Clustering Pipeline**: UMAP + HDBSCAN
- **Cluster Labeling**: Automatic topic extraction

### 4. Graph Database Module (Tasks 4-5, 12)
- **Neo4j Setup**: Graph schema and operations
- **Graph Analytics**: Centrality, communities
- **Graph Integration**: Load networks into Neo4j

### 5. Visualization & Dashboard Module (Tasks 6-7, 13)
- **UMAP Visualization**: Interactive cluster plots
- **Network Visualization**: Collaboration graphs
- **Streamlit Dashboard**: User interface

## Data Flow

```
[Data Sources]
    ↓
[Pipeline Orchestrator] ← Task 11
    ↓
    ├→ [Raw Data Storage]
    │       ↓
    │   [Classifier Module] ← Original Code
    │       ↓
    │   [Classified Papers]
    ↓
[Embeddings & Clustering] ← Tasks 8-10
    ↓
[Graph Database] ← Tasks 4-5, 12
    ↓
[Dashboard] ← Task 13
    ↓
[End User]
```

## Module Dependencies

- **Data Acquisition** → No dependencies
- **Classification** → No dependencies (operates on collected data)
- **Embeddings** → Needs collected data
- **Clustering** → Needs embeddings
- **Graph Database** → Needs author networks
- **Dashboard** → Needs all modules for complete functionality

## Integration Points

1. **Pipeline → Classifier**: Pipeline collects papers, classifier processes them
2. **Classifier → Embeddings**: Classified papers used for embeddings
3. **Embeddings → Dashboard**: Clusters displayed in dashboard
4. **Graph DB → Dashboard**: Network data visualized in dashboard
```

#### Step 6.2: Update Root README

Create comprehensive `README.md`:

```markdown
# AI Scientist - Research Intelligence Platform

A comprehensive platform for research data collection, classification, analysis, and visualization.

## Features

🔍 **Data Collection**
- Automated PubMed scraping by institution
- ClinicalTrials.gov data collection
- Author collaboration network extraction

🤖 **AI-Powered Classification**
- Local LLM-based clinical trial classification
- Structured data extraction
- High accuracy with customizable prompts

📊 **Analytics & Insights**
- Semantic embeddings with Sentence-BERT
- UMAP + HDBSCAN clustering
- Automatic topic labeling

🕸️ **Network Analysis**
- Neo4j graph database
- Collaboration network analytics
- Community detection

📈 **Interactive Dashboard**
- Streamlit web interface
- Search, filter, visualize
- Export reports

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### 1. Collect Data

```bash
python cli/main.py pipeline run --config data/config/pipeline_config.yaml
```

### 2. Classify Papers

```bash
python cli/main.py classifier classify \
  --input-csv data/processed/papers.csv \
  --model llama3.1:8b
```

### 3. Launch Dashboard

```bash
streamlit run src/dashboard/app.py
```

## Documentation

📖 [Complete Documentation](docs/README.md)

- [Classifier Quick Start](docs/quickstart/classifier_quickstart.md)
- [Pipeline Quick Start](docs/quickstart/pipeline_quickstart.md)
- [System Architecture](docs/architecture/system_architecture.md)
- [Integration Guide](docs/guides/integration_guide.md)
- [Deployment Guide](docs/guides/deployment_guide.md)

## CLI Commands

### Classification

```bash
# Classify papers
python cli/main.py classifier classify --input-csv data/papers.csv

# Test classifier
python cli/main.py classifier test

# Check Ollama connection
python cli/main.py classifier check
```

### Pipeline

```bash
# Run data collection
python cli/main.py pipeline run --config pipeline_config.yaml

# Check status
python cli/main.py pipeline status
```

### Dashboard

```bash
# Launch dashboard
python cli/main.py dashboard start

# Or directly:
streamlit run src/dashboard/app.py
```

## Project Structure

```
aiscientist/
├── src/                    # Source code
│   ├── classification/     # LLM-based classifier
│   ├── data_acquisition/   # Scrapers
│   ├── embeddings/         # Embeddings & clustering
│   ├── graph_database/     # Neo4j integration
│   ├── visualization/      # Viz components
│   ├── pipeline/           # Orchestration
│   └── dashboard/          # Streamlit app
├── cli/                    # Command-line interfaces
├── docs/                   # Documentation
├── tests/                  # Tests
├── examples/               # Usage examples
└── prompts/                # LLM prompts
```

## Development

```bash
# Run tests
pytest tests/ -v --cov=src

# Format code
black src/ tests/

# Lint
ruff src/ tests/
```

## Deployment

See [Deployment Guide](docs/guides/deployment_guide.md) for:
- Streamlit Cloud deployment
- AWS/GCP deployment
- Docker deployment

## License

MIT License

## Contributing

See contributing guidelines in docs/
```

---

### Phase 7: Final Steps

#### Step 7.1: Create Migration Helper Script

Create `scripts/migrate_old_code.sh`:

```bash
#!/bin/bash
# Helper script to automate the restructuring migration

set -e

echo "🚀 Starting code restructuring migration..."

# Create new directories
echo "📁 Creating new directory structure..."
mkdir -p docs/{quickstart,guides,architecture}
mkdir -p src/{classification,common,agents}
mkdir -p prompts/{classification,agents}
mkdir -p cli output/classifications data/raw/csv_imports scripts examples

# Move classification code
echo "📦 Moving classification module..."
mv src/processor.py src/classification/classifier.py
mv src/models.py src/classification/models.py
mv src/ollama_client.py src/classification/ollama_client.py
mv src/structured_output.py src/classification/structured_output.py
mv src/prompt_loader.py src/classification/prompt_loader.py
mv src/csv_parser.py src/classification/csv_parser.py
mv src/config.py src/classification/config.py

# Move prompts
echo "📝 Moving prompts..."
mv prompts/clinical_trial_classifier.yaml prompts/classification/

# Move documentation
echo "📚 Moving documentation..."
mv QUICKSTART.md docs/quickstart/classifier_quickstart.md 2>/dev/null || true
mv CLASSIFIER_README.md docs/guides/classifier_guide.md 2>/dev/null || true
mv PROMPT_TUNING_GUIDE.md docs/guides/prompt_tuning.md 2>/dev/null || true

# Move CLI
echo "⚙️  Adapting CLI..."
mv main.py cli/main.py

echo "✅ Migration complete!"
echo "Next steps:"
echo "1. Update imports in src/classification/ files"
echo "2. Merge Tasks 1-13 (follow MERGE_INSTRUCTIONS_ALL_TASKS.md)"
echo "3. Run tests: pytest tests/ -v"
echo "4. Update documentation"
```

#### Step 7.2: Update `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.pytest_cache/

# Virtual environments
venv/
env/
ENV/

# Data
data/
!data/config/
*.csv
*.json
*.jsonl
*.npy

# Output
output/
logs/

# Neo4j
neo4j/

# Streamlit
.streamlit/secrets.toml

# Environment
.env
.env.local

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Backup
backup/
*.backup

# Distribution
dist/
build/
*.egg-info/
```

#### Step 7.3: Create Setup Script

Create `setup.py`:

```python
"""Setup configuration for AI Scientist platform."""

from setuptools import setup, find_packages

setup(
    name="aiscientist",
    version="2.0.0",
    description="Research Intelligence Platform with classification, analytics, and visualization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="AI Scientist Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=open("requirements.txt").read().splitlines(),
    entry_points={
        "console_scripts": [
            "aiscientist=cli.main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
)
```

---

## Execution Checklist

Complete this checklist in order:

### ✅ Phase 1: Preparation
- [ ] Create restructuring branch
- [ ] Create backup
- [ ] Create new directory structure

### ✅ Phase 2: Move Original Code
- [ ] Move src/ files to src/classification/
- [ ] Update imports in classification module
- [ ] Move prompts to prompts/classification/
- [ ] Move docs to docs/
- [ ] Adapt CLI to cli/main.py

### ✅ Phase 3: Merge Tasks 1-13
- [ ] Follow `MERGE_INSTRUCTIONS_ALL_TASKS.md`
- [ ] Merge in 4 waves respecting dependencies
- [ ] Resolve conflicts (requirements.txt, etc.)
- [ ] Verify directory structure

### ✅ Phase 4: Integration
- [ ] Update src/__init__.py
- [ ] Consolidate requirements.txt
- [ ] Update all import paths
- [ ] Create common utilities module

### ✅ Phase 5: Examples & Documentation
- [ ] Create end-to-end workflow example
- [ ] Create module-specific examples
- [ ] Write system architecture doc
- [ ] Update root README
- [ ] Create docs index

### ✅ Phase 6: Testing
- [ ] Run all tests
- [ ] Verify test coverage ≥80%
- [ ] Run integration smoke test
- [ ] Test CLI commands

### ✅ Phase 7: Finalization
- [ ] Update .gitignore
- [ ] Create setup.py
- [ ] Update .env.example
- [ ] Commit all changes
- [ ] Push to remote

---

## Expected Result

After completing this restructuring:

✅ **Unified Codebase**: All modules integrated cleanly
✅ **Clear Organization**: Logical directory structure
✅ **No Duplication**: Original code and tasks work together
✅ **Full Functionality**: Everything works end-to-end
✅ **Comprehensive Docs**: Clear documentation for all modules
✅ **Easy CLI**: Simple commands for all operations
✅ **Ready for Phase 4**: Structure supports future multi-agent system
✅ **Production Ready**: Can deploy immediately

---

**End of Restructuring Guide**
