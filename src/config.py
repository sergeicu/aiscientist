"""
Configuration management for PubMed Clinical Trial Classifier.

Handles application settings, defaults, and environment variables.
"""

from typing import Optional, List
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Config(BaseSettings):
    """Application configuration with defaults and environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PUBMED_",
        case_sensitive=False
    )

    # Ollama settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama API"
    )

    ollama_model: str = Field(
        default="llama3.1:8b",
        description="Ollama model to use (e.g., 'llama3.1:8b', 'gemma2:9b', 'mistral:7b')"
    )

    ollama_timeout: int = Field(
        default=120,
        ge=10,
        description="Timeout for Ollama API calls in seconds"
    )

    ollama_num_ctx: int = Field(
        default=4096,
        ge=512,
        description="Context window size for the model"
    )

    ollama_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (lower = more deterministic)"
    )

    # Processing settings
    batch_size: int = Field(
        default=10,
        ge=1,
        description="Number of articles to process before saving checkpoint"
    )

    max_concurrent_requests: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum number of concurrent requests to Ollama"
    )

    resume_from_checkpoint: bool = Field(
        default=True,
        description="Whether to resume from last checkpoint if available"
    )

    # File paths
    input_csv: Path = Field(
        default=Path("data/pubmed_data_2000.csv"),
        description="Path to input CSV file"
    )

    output_dir: Path = Field(
        default=Path("output"),
        description="Directory for output files"
    )

    checkpoint_dir: Path = Field(
        default=Path("output/checkpoints"),
        description="Directory for checkpoint files"
    )

    log_dir: Path = Field(
        default=Path("logs"),
        description="Directory for log files"
    )

    prompt_template: Path = Field(
        default=Path("prompts/clinical_trial_classifier.yaml"),
        description="Path to prompt template YAML file"
    )

    # CSV column mapping
    abstract_column_names: List[str] = Field(
        default=["abstract", "Abstract", "ABSTRACT", "AB", "abstractText"],
        description="Possible names for the abstract column (in priority order)"
    )

    pmid_column_names: List[str] = Field(
        default=["pmid", "PMID", "PubMed ID", "pubmed_id", "id", "ID"],
        description="Possible names for the PMID column"
    )

    title_column_names: List[str] = Field(
        default=["title", "Title", "TITLE", "TI", "ArticleTitle"],
        description="Possible names for the title column"
    )

    # Structured output settings
    use_outlines: bool = Field(
        default=True,
        description="Whether to use Outlines for constrained generation"
    )

    use_json_repair: bool = Field(
        default=True,
        description="Whether to use json-repair as fallback for malformed JSON"
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries for failed requests"
    )

    retry_delay_seconds: float = Field(
        default=2.0,
        ge=0.0,
        description="Delay between retries in seconds"
    )

    # Logging settings
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    verbose: bool = Field(
        default=False,
        description="Enable verbose output"
    )

    @field_validator('input_csv', 'prompt_template')
    @classmethod
    def validate_file_exists(cls, v: Path, info) -> Path:
        """Validate that required files exist."""
        # Only validate at runtime, not during model creation
        return v

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_output_filename(self, prefix: str = "results") -> Path:
        """Generate output filename with timestamp."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"{prefix}_{timestamp}.jsonl"

    def get_checkpoint_filename(self) -> Path:
        """Get checkpoint filename."""
        return self.checkpoint_dir / "checkpoint.json"


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = Config()
        _config.ensure_directories()
    return _config


def set_config(config: Config) -> None:
    """Set global config instance."""
    global _config
    _config = config
    _config.ensure_directories()
