# Task 11: Data Pipeline Orchestrator

**Status:** Integration Task
**Dependencies:** Tasks 1, 2, 3 must be completed
**Estimated Time:** 4-6 hours
**Difficulty:** Medium

---

## Objective

Create a unified data pipeline orchestrator that coordinates data collection from PubMed and ClinicalTrials.gov, processes the data, and stores it in a standardized format for downstream use.

---

## Background

You have three independent scrapers:
1. PubMed scraper (institution-based)
2. ClinicalTrials.gov scraper (sponsor-based)
3. Author network extractor

This task integrates them into a single coordinated pipeline that:
- Runs scrapers based on a configuration file
- Handles errors gracefully
- Provides progress tracking
- Stores all data in a unified structure

---

## Requirements

### Functional Requirements

1. **Unified Pipeline**
   - Single command to run all scrapers
   - Configuration-driven (YAML/JSON)
   - Parallel execution where possible
   - Progress tracking and logging

2. **Error Handling**
   - Retry failed scrapes
   - Continue on partial failures
   - Log all errors
   - Generate summary report

3. **Data Storage**
   - Standardized directory structure
   - Consistent file naming
   - Metadata tracking (timestamps, sources)

4. **Monitoring**
   - Real-time progress updates
   - Estimated time remaining
   - Success/failure counts

### Technical Requirements

- Python 3.9+
- Async/await for parallelization
- Pydantic for configuration validation
- Rich for terminal UI
- Structured logging (JSON logs)

---

## Architecture

```
src/pipeline/
├── __init__.py
├── orchestrator.py          # Main pipeline orchestrator
├── config.py                # Configuration models
├── storage.py               # Unified storage manager
├── models.py                # Data models
└── utils.py                 # Utilities

data/
├── config/
│   └── pipeline_config.yaml # Pipeline configuration
├── raw/
│   ├── pubmed/              # Raw PubMed data
│   ├── clinicaltrials/      # Raw ClinicalTrials data
│   └── author_networks/     # Extracted networks
├── processed/
│   └── unified_papers.json  # All papers in unified format
└── logs/
    └── pipeline_YYYYMMDD_HHMMSS.log

tests/
├── __init__.py
├── test_orchestrator.py
├── test_config.py
└── test_storage.py
```

---

## Implementation Guide (TDD)

### Step 1: Configuration Models

**Test First** (`tests/test_config.py`):

```python
import pytest
from pathlib import Path
from pydantic import ValidationError
from pipeline.config import (
    PubMedConfig,
    ClinicalTrialsConfig,
    PipelineConfig,
    load_config
)


class TestPubMedConfig:
    """Test PubMed configuration model."""

    def test_valid_config(self):
        """Test valid PubMed configuration."""
        config = PubMedConfig(
            email="test@example.com",
            api_key="optional_key",
            institutions=["Harvard Medical School", "Mayo Clinic"],
            max_results_per_institution=100,
            rate_limit_per_second=3.0
        )

        assert config.email == "test@example.com"
        assert len(config.institutions) == 2
        assert config.max_results_per_institution == 100

    def test_missing_email_raises_error(self):
        """Test that missing email raises validation error."""
        with pytest.raises(ValidationError):
            PubMedConfig(institutions=["Harvard"])

    def test_default_values(self):
        """Test default configuration values."""
        config = PubMedConfig(
            email="test@example.com",
            institutions=["Harvard"]
        )

        assert config.api_key is None
        assert config.max_results_per_institution == 1000
        assert config.rate_limit_per_second == 3.0


class TestClinicalTrialsConfig:
    """Test ClinicalTrials configuration model."""

    def test_valid_config(self):
        """Test valid ClinicalTrials configuration."""
        config = ClinicalTrialsConfig(
            sponsors=["Massachusetts General Hospital", "Johns Hopkins"],
            max_results_per_sponsor=50,
            rate_limit_per_second=2.0
        )

        assert len(config.sponsors) == 2
        assert config.max_results_per_sponsor == 50


class TestPipelineConfig:
    """Test overall pipeline configuration."""

    def test_full_config(self):
        """Test complete pipeline configuration."""
        config = PipelineConfig(
            pubmed=PubMedConfig(
                email="test@example.com",
                institutions=["Harvard"]
            ),
            clinicaltrials=ClinicalTrialsConfig(
                sponsors=["MGH"]
            ),
            output_dir="./data",
            log_level="INFO",
            parallel_workers=3
        )

        assert config.pubmed.email == "test@example.com"
        assert config.clinicaltrials.sponsors == ["MGH"]
        assert config.parallel_workers == 3

    def test_load_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
pubmed:
  email: test@example.com
  institutions:
    - Harvard Medical School
    - Mayo Clinic
  max_results_per_institution: 100

clinicaltrials:
  sponsors:
    - Massachusetts General Hospital
  max_results_per_sponsor: 50

output_dir: ./data
log_level: INFO
parallel_workers: 2
        """)

        config = load_config(config_file)

        assert config.pubmed.email == "test@example.com"
        assert len(config.pubmed.institutions) == 2
        assert config.parallel_workers == 2
```

**Implementation** (`src/pipeline/config.py`):

```python
"""Configuration models for the data pipeline."""

from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel, Field, EmailStr, validator
import yaml


class PubMedConfig(BaseModel):
    """Configuration for PubMed scraper."""

    email: EmailStr
    api_key: Optional[str] = None
    institutions: List[str] = Field(..., min_items=1)
    max_results_per_institution: int = Field(default=1000, gt=0)
    rate_limit_per_second: float = Field(default=3.0, gt=0)

    @validator('institutions')
    def validate_institutions(cls, v):
        """Ensure institutions list is not empty."""
        if not v:
            raise ValueError("At least one institution required")
        return v


class ClinicalTrialsConfig(BaseModel):
    """Configuration for ClinicalTrials.gov scraper."""

    sponsors: List[str] = Field(..., min_items=1)
    max_results_per_sponsor: int = Field(default=500, gt=0)
    rate_limit_per_second: float = Field(default=2.0, gt=0)


class PipelineConfig(BaseModel):
    """Overall pipeline configuration."""

    pubmed: PubMedConfig
    clinicaltrials: ClinicalTrialsConfig
    output_dir: str = "./data"
    log_level: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR)$")
    parallel_workers: int = Field(default=3, ge=1, le=10)
    retry_failed: bool = True
    max_retries: int = Field(default=3, ge=0)

    @validator('output_dir')
    def validate_output_dir(cls, v):
        """Ensure output directory exists or can be created."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())


def load_config(config_path: Path) -> PipelineConfig:
    """Load pipeline configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return PipelineConfig(**config_dict)
```

---

### Step 2: Storage Manager

**Test First** (`tests/test_storage.py`):

```python
import pytest
import json
from pathlib import Path
from datetime import datetime
from pipeline.storage import StorageManager
from pipeline.models import UnifiedPaper


class TestStorageManager:
    """Test unified storage manager."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create storage manager with temp directory."""
        return StorageManager(base_dir=tmp_path)

    def test_directory_creation(self, storage):
        """Test that all required directories are created."""
        assert (storage.base_dir / "raw" / "pubmed").exists()
        assert (storage.base_dir / "raw" / "clinicaltrials").exists()
        assert (storage.base_dir / "raw" / "author_networks").exists()
        assert (storage.base_dir / "processed").exists()
        assert (storage.base_dir / "logs").exists()

    def test_save_pubmed_papers(self, storage):
        """Test saving PubMed papers."""
        papers = [
            {
                "pmid": "12345",
                "title": "Test Paper",
                "abstract": "Test abstract",
                "authors": [{"name": "John Doe"}],
                "publication_date": "2024-01-15",
                "institution": "Harvard"
            }
        ]

        filepath = storage.save_pubmed_papers(
            papers=papers,
            institution="Harvard Medical School"
        )

        assert filepath.exists()
        assert "harvard_medical_school" in str(filepath)

        # Verify content
        with open(filepath) as f:
            saved = json.load(f)

        assert len(saved) == 1
        assert saved[0]["pmid"] == "12345"

    def test_save_clinicaltrials(self, storage):
        """Test saving clinical trials."""
        trials = [
            {
                "nct_id": "NCT12345",
                "title": "Test Trial",
                "status": "Recruiting",
                "sponsor": "MGH"
            }
        ]

        filepath = storage.save_clinicaltrials(
            trials=trials,
            sponsor="Massachusetts General Hospital"
        )

        assert filepath.exists()
        assert "massachusetts_general_hospital" in str(filepath)

    def test_save_author_network(self, storage):
        """Test saving author network."""
        network = {
            "authors": [{"author_id": "1", "name": "John Doe"}],
            "collaborations": [{"author_1": "1", "author_2": "2"}]
        }

        filepath = storage.save_author_network(
            network=network,
            source="Harvard"
        )

        assert filepath.exists()

    def test_load_all_pubmed_papers(self, storage):
        """Test loading all PubMed papers."""
        # Save some papers
        storage.save_pubmed_papers(
            papers=[{"pmid": "1", "title": "Paper 1"}],
            institution="Harvard"
        )
        storage.save_pubmed_papers(
            papers=[{"pmid": "2", "title": "Paper 2"}],
            institution="Mayo"
        )

        # Load all
        all_papers = storage.load_all_pubmed_papers()

        assert len(all_papers) == 2
        assert any(p["pmid"] == "1" for p in all_papers)
        assert any(p["pmid"] == "2" for p in all_papers)

    def test_create_unified_papers(self, storage):
        """Test creating unified paper format."""
        # Save PubMed and CT data
        storage.save_pubmed_papers(
            papers=[{"pmid": "1", "title": "Paper 1"}],
            institution="Harvard"
        )
        storage.save_clinicaltrials(
            trials=[{"nct_id": "NCT1", "title": "Trial 1"}],
            sponsor="MGH"
        )

        # Create unified
        unified_path = storage.create_unified_dataset()

        assert unified_path.exists()

        with open(unified_path) as f:
            unified = json.load(f)

        assert "papers" in unified
        assert "trials" in unified
        assert len(unified["papers"]) == 1
        assert len(unified["trials"]) == 1
```

**Implementation** (`src/pipeline/storage.py`):

```python
"""Unified storage manager for pipeline data."""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages data storage for the pipeline."""

    def __init__(self, base_dir: str = "./data"):
        """Initialize storage manager."""
        self.base_dir = Path(base_dir)
        self._create_directories()

    def _create_directories(self):
        """Create required directory structure."""
        dirs = [
            self.base_dir / "raw" / "pubmed",
            self.base_dir / "raw" / "clinicaltrials",
            self.base_dir / "raw" / "author_networks",
            self.base_dir / "processed",
            self.base_dir / "logs",
            self.base_dir / "config"
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created directory structure in {self.base_dir}")

    def _slugify(self, text: str) -> str:
        """Convert text to slug format."""
        return text.lower().replace(" ", "_").replace(",", "").replace(".", "")

    def save_pubmed_papers(
        self,
        papers: List[Dict[str, Any]],
        institution: str
    ) -> Path:
        """Save PubMed papers for an institution."""
        slug = self._slugify(institution)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"papers_{slug}_{timestamp}.json"
        filepath = self.base_dir / "raw" / "pubmed" / filename

        with open(filepath, 'w') as f:
            json.dump(papers, f, indent=2)

        logger.info(f"Saved {len(papers)} PubMed papers to {filepath}")
        return filepath

    def save_clinicaltrials(
        self,
        trials: List[Dict[str, Any]],
        sponsor: str
    ) -> Path:
        """Save clinical trials for a sponsor."""
        slug = self._slugify(sponsor)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trials_{slug}_{timestamp}.json"
        filepath = self.base_dir / "raw" / "clinicaltrials" / filename

        with open(filepath, 'w') as f:
            json.dump(trials, f, indent=2)

        logger.info(f"Saved {len(trials)} clinical trials to {filepath}")
        return filepath

    def save_author_network(
        self,
        network: Dict[str, Any],
        source: str
    ) -> Path:
        """Save author network."""
        slug = self._slugify(source)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"network_{slug}_{timestamp}.json"
        filepath = self.base_dir / "raw" / "author_networks" / filename

        with open(filepath, 'w') as f:
            json.dump(network, f, indent=2)

        logger.info(f"Saved author network to {filepath}")
        return filepath

    def load_all_pubmed_papers(self) -> List[Dict[str, Any]]:
        """Load all PubMed papers from all institutions."""
        pubmed_dir = self.base_dir / "raw" / "pubmed"
        all_papers = []

        for filepath in pubmed_dir.glob("papers_*.json"):
            with open(filepath) as f:
                papers = json.load(f)
                all_papers.extend(papers)

        logger.info(f"Loaded {len(all_papers)} total PubMed papers")
        return all_papers

    def load_all_clinicaltrials(self) -> List[Dict[str, Any]]:
        """Load all clinical trials from all sponsors."""
        ct_dir = self.base_dir / "raw" / "clinicaltrials"
        all_trials = []

        for filepath in ct_dir.glob("trials_*.json"):
            with open(filepath) as f:
                trials = json.load(f)
                all_trials.extend(trials)

        logger.info(f"Loaded {len(all_trials)} total clinical trials")
        return all_trials

    def create_unified_dataset(self) -> Path:
        """Create unified dataset from all sources."""
        unified = {
            "papers": self.load_all_pubmed_papers(),
            "trials": self.load_all_clinicaltrials(),
            "created_at": datetime.now().isoformat()
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.base_dir / "processed" / f"unified_dataset_{timestamp}.json"

        with open(filepath, 'w') as f:
            json.dump(unified, f, indent=2)

        logger.info(f"Created unified dataset at {filepath}")
        return filepath
```

---

### Step 3: Pipeline Orchestrator

**Test First** (`tests/test_orchestrator.py`):

```python
import pytest
from unittest.mock import Mock, AsyncMock, patch
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.config import PipelineConfig, PubMedConfig, ClinicalTrialsConfig


@pytest.fixture
def mock_config():
    """Create mock pipeline configuration."""
    return PipelineConfig(
        pubmed=PubMedConfig(
            email="test@example.com",
            institutions=["Harvard", "Mayo Clinic"]
        ),
        clinicaltrials=ClinicalTrialsConfig(
            sponsors=["MGH", "Johns Hopkins"]
        ),
        parallel_workers=2
    )


@pytest.mark.asyncio
class TestPipelineOrchestrator:
    """Test pipeline orchestrator."""

    async def test_initialization(self, mock_config, tmp_path):
        """Test orchestrator initialization."""
        orchestrator = PipelineOrchestrator(
            config=mock_config,
            output_dir=tmp_path
        )

        assert orchestrator.config == mock_config
        assert orchestrator.storage is not None

    @patch('pipeline.orchestrator.PubMedScraper')
    async def test_run_pubmed_scraper(self, mock_scraper_class, mock_config, tmp_path):
        """Test running PubMed scraper for institutions."""
        # Setup mock
        mock_scraper = Mock()
        mock_scraper.fetch_institution_papers = AsyncMock(
            return_value=[{"pmid": "123", "title": "Test"}]
        )
        mock_scraper_class.return_value = mock_scraper

        orchestrator = PipelineOrchestrator(mock_config, tmp_path)

        # Run scraper
        results = await orchestrator.run_pubmed_scraper()

        assert len(results) == 2  # Two institutions
        assert all(r["status"] == "success" for r in results)
        assert mock_scraper.fetch_institution_papers.call_count == 2

    @patch('pipeline.orchestrator.ClinicalTrialsScraper')
    async def test_run_clinicaltrials_scraper(
        self,
        mock_scraper_class,
        mock_config,
        tmp_path
    ):
        """Test running ClinicalTrials scraper."""
        mock_scraper = Mock()
        mock_scraper.fetch_by_sponsor = AsyncMock(
            return_value=[{"nct_id": "NCT123"}]
        )
        mock_scraper_class.return_value = mock_scraper

        orchestrator = PipelineOrchestrator(mock_config, tmp_path)

        results = await orchestrator.run_clinicaltrials_scraper()

        assert len(results) == 2  # Two sponsors
        assert all(r["status"] == "success" for r in results)

    @patch('pipeline.orchestrator.PubMedScraper')
    @patch('pipeline.orchestrator.ClinicalTrialsScraper')
    @patch('pipeline.orchestrator.AuthorNetworkExtractor')
    async def test_run_full_pipeline(
        self,
        mock_network,
        mock_ct,
        mock_pubmed,
        mock_config,
        tmp_path
    ):
        """Test running full pipeline."""
        # Setup mocks
        mock_pubmed.return_value.fetch_institution_papers = AsyncMock(
            return_value=[{"pmid": "123"}]
        )
        mock_ct.return_value.fetch_by_sponsor = AsyncMock(
            return_value=[{"nct_id": "NCT123"}]
        )
        mock_network.return_value.build_network = Mock(
            return_value={"authors": [], "collaborations": []}
        )

        orchestrator = PipelineOrchestrator(mock_config, tmp_path)

        # Run full pipeline
        report = await orchestrator.run_full_pipeline()

        assert report["status"] == "completed"
        assert "pubmed" in report
        assert "clinicaltrials" in report
        assert "author_networks" in report
        assert report["total_papers"] > 0

    async def test_error_handling(self, mock_config, tmp_path):
        """Test error handling in pipeline."""
        orchestrator = PipelineOrchestrator(mock_config, tmp_path)

        with patch('pipeline.orchestrator.PubMedScraper') as mock:
            mock.return_value.fetch_institution_papers = AsyncMock(
                side_effect=Exception("API Error")
            )

            results = await orchestrator.run_pubmed_scraper()

            # Should catch errors and continue
            assert any(r["status"] == "failed" for r in results)
            assert any("error" in r for r in results)
```

**Implementation** (`src/pipeline/orchestrator.py`):

```python
"""Pipeline orchestrator for data collection."""

import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.console import Console

from pipeline.config import PipelineConfig
from pipeline.storage import StorageManager

# Import scrapers (these will be from completed tasks)
from data_acquisition.pubmed_scraper import PubMedScraper
from data_acquisition.clinicaltrials_scraper import ClinicalTrialsScraper
from data_acquisition.author_network import AuthorNetworkExtractor

logger = logging.getLogger(__name__)
console = Console()


class PipelineOrchestrator:
    """Orchestrates the full data collection pipeline."""

    def __init__(self, config: PipelineConfig, output_dir: str = "./data"):
        """Initialize orchestrator."""
        self.config = config
        self.storage = StorageManager(base_dir=output_dir)
        self.pubmed_scraper = None
        self.ct_scraper = None
        self.network_extractor = None

    def _init_scrapers(self):
        """Initialize all scrapers."""
        self.pubmed_scraper = PubMedScraper(
            email=self.config.pubmed.email,
            api_key=self.config.pubmed.api_key
        )
        self.ct_scraper = ClinicalTrialsScraper()
        self.network_extractor = AuthorNetworkExtractor()

        logger.info("Initialized all scrapers")

    async def run_pubmed_scraper(self) -> List[Dict[str, Any]]:
        """Run PubMed scraper for all institutions."""
        if not self.pubmed_scraper:
            self._init_scrapers()

        results = []

        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                "[cyan]Scraping PubMed...",
                total=len(self.config.pubmed.institutions)
            )

            for institution in self.config.pubmed.institutions:
                try:
                    logger.info(f"Fetching papers for {institution}")

                    papers = await self.pubmed_scraper.fetch_institution_papers(
                        institution=institution,
                        max_results=self.config.pubmed.max_results_per_institution
                    )

                    # Save to storage
                    filepath = self.storage.save_pubmed_papers(
                        papers=papers,
                        institution=institution
                    )

                    results.append({
                        "institution": institution,
                        "status": "success",
                        "papers_count": len(papers),
                        "filepath": str(filepath)
                    })

                    console.print(
                        f"✓ {institution}: {len(papers)} papers",
                        style="green"
                    )

                except Exception as e:
                    logger.error(f"Failed to scrape {institution}: {e}")
                    results.append({
                        "institution": institution,
                        "status": "failed",
                        "error": str(e)
                    })
                    console.print(f"✗ {institution}: {str(e)}", style="red")

                finally:
                    progress.update(task, advance=1)

        return results

    async def run_clinicaltrials_scraper(self) -> List[Dict[str, Any]]:
        """Run ClinicalTrials scraper for all sponsors."""
        if not self.ct_scraper:
            self._init_scrapers()

        results = []

        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                "[cyan]Scraping ClinicalTrials.gov...",
                total=len(self.config.clinicaltrials.sponsors)
            )

            for sponsor in self.config.clinicaltrials.sponsors:
                try:
                    logger.info(f"Fetching trials for {sponsor}")

                    trials = await self.ct_scraper.fetch_by_sponsor(
                        sponsor=sponsor,
                        max_results=self.config.clinicaltrials.max_results_per_sponsor
                    )

                    filepath = self.storage.save_clinicaltrials(
                        trials=trials,
                        sponsor=sponsor
                    )

                    results.append({
                        "sponsor": sponsor,
                        "status": "success",
                        "trials_count": len(trials),
                        "filepath": str(filepath)
                    })

                    console.print(
                        f"✓ {sponsor}: {len(trials)} trials",
                        style="green"
                    )

                except Exception as e:
                    logger.error(f"Failed to scrape {sponsor}: {e}")
                    results.append({
                        "sponsor": sponsor,
                        "status": "failed",
                        "error": str(e)
                    })
                    console.print(f"✗ {sponsor}: {str(e)}", style="red")

                finally:
                    progress.update(task, advance=1)

        return results

    async def extract_author_networks(self) -> List[Dict[str, Any]]:
        """Extract author networks from all papers."""
        if not self.network_extractor:
            self._init_scrapers()

        results = []

        console.print("[cyan]Extracting author networks...[/cyan]")

        # Load all PubMed papers
        all_papers = self.storage.load_all_pubmed_papers()

        if not all_papers:
            console.print("[yellow]No papers found to extract networks[/yellow]")
            return results

        try:
            # Build network
            network = self.network_extractor.build_network(all_papers)

            # Save network
            filepath = self.storage.save_author_network(
                network=network,
                source="all_institutions"
            )

            results.append({
                "source": "all_institutions",
                "status": "success",
                "authors_count": len(network["authors"]),
                "collaborations_count": len(network["collaborations"]),
                "filepath": str(filepath)
            })

            console.print(
                f"✓ Extracted {len(network['authors'])} authors, "
                f"{len(network['collaborations'])} collaborations",
                style="green"
            )

        except Exception as e:
            logger.error(f"Failed to extract networks: {e}")
            results.append({
                "source": "all_institutions",
                "status": "failed",
                "error": str(e)
            })
            console.print(f"✗ Network extraction failed: {str(e)}", style="red")

        return results

    async def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline."""
        console.print("\n[bold cyan]Starting Data Collection Pipeline[/bold cyan]\n")

        start_time = datetime.now()

        # Run scrapers in parallel
        pubmed_task = asyncio.create_task(self.run_pubmed_scraper())
        ct_task = asyncio.create_task(self.run_clinicaltrials_scraper())

        pubmed_results, ct_results = await asyncio.gather(
            pubmed_task,
            ct_task,
            return_exceptions=True
        )

        # Extract author networks
        network_results = await self.extract_author_networks()

        # Create unified dataset
        console.print("\n[cyan]Creating unified dataset...[/cyan]")
        unified_path = self.storage.create_unified_dataset()
        console.print(f"✓ Unified dataset: {unified_path}", style="green")

        # Generate report
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Count totals
        total_papers = sum(
            r.get("papers_count", 0) for r in pubmed_results
            if isinstance(r, dict) and r.get("status") == "success"
        )
        total_trials = sum(
            r.get("trials_count", 0) for r in ct_results
            if isinstance(r, dict) and r.get("status") == "success"
        )

        report = {
            "status": "completed",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "pubmed": pubmed_results,
            "clinicaltrials": ct_results,
            "author_networks": network_results,
            "total_papers": total_papers,
            "total_trials": total_trials,
            "unified_dataset": str(unified_path)
        }

        # Save report
        report_path = self.storage.base_dir / "logs" / f"pipeline_report_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            import json
            json.dump(report, f, indent=2)

        # Print summary
        console.print("\n[bold green]Pipeline Completed![/bold green]")
        console.print(f"Duration: {duration:.1f}s")
        console.print(f"Total Papers: {total_papers}")
        console.print(f"Total Trials: {total_trials}")
        console.print(f"Report: {report_path}\n")

        return report


# CLI entry point
async def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Run data collection pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default="./data/config/pipeline_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data",
        help="Output directory"
    )

    args = parser.parse_args()

    # Load config
    from pipeline.config import load_config
    config = load_config(args.config)

    # Run pipeline
    orchestrator = PipelineOrchestrator(config, args.output)
    await orchestrator.run_full_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Usage

### 1. Create Configuration File

```yaml
# data/config/pipeline_config.yaml

pubmed:
  email: your.email@example.com
  api_key: null  # Optional
  institutions:
    - Harvard Medical School
    - Mayo Clinic
    - Johns Hopkins University
  max_results_per_institution: 500
  rate_limit_per_second: 3.0

clinicaltrials:
  sponsors:
    - Massachusetts General Hospital
    - Johns Hopkins Hospital
    - Mayo Clinic
  max_results_per_sponsor: 300
  rate_limit_per_second: 2.0

output_dir: ./data
log_level: INFO
parallel_workers: 3
retry_failed: true
max_retries: 3
```

### 2. Run Pipeline

```bash
# Install dependencies
pip install pyyaml pydantic rich

# Run pipeline
python -m pipeline.orchestrator --config data/config/pipeline_config.yaml
```

### 3. Programmatic Usage

```python
import asyncio
from pipeline.config import load_config
from pipeline.orchestrator import PipelineOrchestrator

async def run():
    # Load config
    config = load_config("data/config/pipeline_config.yaml")

    # Create orchestrator
    orchestrator = PipelineOrchestrator(config, output_dir="./data")

    # Run full pipeline
    report = await orchestrator.run_full_pipeline()

    print(f"Collected {report['total_papers']} papers")
    print(f"Collected {report['total_trials']} trials")

asyncio.run(run())
```

---

## Dependencies

```txt
# requirements.txt
pydantic>=2.0.0
pyyaml>=6.0
rich>=13.0.0
asyncio
```

---

## Success Criteria

- [ ] Pipeline runs all scrapers based on configuration
- [ ] Handles errors gracefully (continues on failures)
- [ ] Provides real-time progress updates
- [ ] Saves all data in standardized format
- [ ] Generates comprehensive report
- [ ] All tests pass
- [ ] Test coverage ≥ 80%

---

## Deliverables

1. Working pipeline orchestrator
2. Configuration model with validation
3. Unified storage manager
4. Comprehensive test suite
5. CLI interface
6. Example configuration file
7. Usage documentation

---

**End of Task 11 Specification**
