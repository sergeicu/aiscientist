"""Configuration models for the data pipeline."""

from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel, Field, EmailStr, field_validator
import yaml


class PubMedConfig(BaseModel):
    """Configuration for PubMed scraper."""

    email: EmailStr
    api_key: Optional[str] = None
    institutions: List[str] = Field(..., min_length=1)
    max_results_per_institution: int = Field(default=1000, gt=0)
    rate_limit_per_second: float = Field(default=3.0, gt=0)

    @field_validator('institutions')
    @classmethod
    def validate_institutions(cls, v):
        """Ensure institutions list is not empty."""
        if not v:
            raise ValueError("At least one institution required")
        return v


class ClinicalTrialsConfig(BaseModel):
    """Configuration for ClinicalTrials.gov scraper."""

    sponsors: List[str] = Field(..., min_length=1)
    max_results_per_sponsor: int = Field(default=500, gt=0)
    rate_limit_per_second: float = Field(default=2.0, gt=0)


class PipelineConfig(BaseModel):
    """Overall pipeline configuration."""

    pubmed: PubMedConfig
    clinicaltrials: ClinicalTrialsConfig
    output_dir: str = "./data"
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")
    parallel_workers: int = Field(default=3, ge=1, le=10)
    retry_failed: bool = True
    max_retries: int = Field(default=3, ge=0)

    @field_validator('output_dir')
    @classmethod
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
