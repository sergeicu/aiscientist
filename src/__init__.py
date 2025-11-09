"""
PubMed Clinical Trial Classifier

A tool for classifying PubMed articles as clinical trials using local Ollama models.
"""

__version__ = "0.1.0"

from .config import Config, get_config, set_config
from .models import (
    Article,
    ClassificationResult,
    ProcessingResult,
    ProcessingStats,
    TrialPhase,
    StudyType,
    InterventionType,
)
from .csv_parser import CSVParser
from .ollama_client import OllamaClient
from .structured_output import StructuredOutputHandler
from .prompt_loader import PromptLoader
from .processor import ArticleProcessor

__all__ = [
    "Config",
    "get_config",
    "set_config",
    "Article",
    "ClassificationResult",
    "ProcessingResult",
    "ProcessingStats",
    "TrialPhase",
    "StudyType",
    "InterventionType",
    "CSVParser",
    "OllamaClient",
    "StructuredOutputHandler",
    "PromptLoader",
    "ArticleProcessor",
]
