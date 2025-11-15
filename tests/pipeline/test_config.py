"""Tests for pipeline configuration models."""

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
