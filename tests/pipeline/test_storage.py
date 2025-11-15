"""Tests for pipeline storage manager."""

import pytest
import json
from pathlib import Path
from datetime import datetime
from src.pipeline.storage import StorageManager


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
