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
