"""
Data models for PubMed Clinical Trial Classifier.

This module defines Pydantic models for:
- Classification results from LLM
- Article metadata
- Processing state and checkpoints
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict


class TrialPhase(str, Enum):
    """Clinical trial phase enumeration."""
    PHASE_0 = "Phase 0"
    PHASE_I = "Phase I"
    PHASE_II = "Phase II"
    PHASE_III = "Phase III"
    PHASE_IV = "Phase IV"
    PHASE_I_II = "Phase I/II"
    PHASE_II_III = "Phase II/III"
    NOT_APPLICABLE = "Not Applicable"
    UNKNOWN = "Unknown"


class StudyType(str, Enum):
    """Type of clinical study."""
    INTERVENTIONAL = "Interventional"
    OBSERVATIONAL = "Observational"
    EXPANDED_ACCESS = "Expanded Access"
    META_ANALYSIS = "Meta-Analysis"
    SYSTEMATIC_REVIEW = "Systematic Review"
    CASE_REPORT = "Case Report"
    CASE_SERIES = "Case Series"
    OTHER = "Other"
    UNKNOWN = "Unknown"


class InterventionType(str, Enum):
    """Type of intervention in clinical trial."""
    DRUG = "Drug"
    BIOLOGICAL = "Biological"
    DEVICE = "Device"
    PROCEDURE = "Procedure"
    BEHAVIORAL = "Behavioral"
    DIETARY_SUPPLEMENT = "Dietary Supplement"
    RADIATION = "Radiation"
    COMBINATION = "Combination"
    OTHER = "Other"
    NOT_APPLICABLE = "Not Applicable"


class ClassificationResult(BaseModel):
    """
    Structured output schema for clinical trial classification.

    This model represents the expected JSON output from the LLM.
    It will be used by Outlines to constrain generation.
    """
    model_config = ConfigDict(use_enum_values=True)

    # Primary classification
    is_clinical_trial: bool = Field(
        description="Whether the article describes a clinical trial"
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the classification (0.0-1.0)"
    )

    reasoning: str = Field(
        description="Brief explanation for the classification decision"
    )

    # Trial characteristics (if applicable)
    trial_phase: Optional[TrialPhase] = Field(
        default=None,
        description="Phase of the clinical trial (if applicable)"
    )

    study_type: Optional[StudyType] = Field(
        default=None,
        description="Type of study conducted"
    )

    intervention_type: Optional[InterventionType] = Field(
        default=None,
        description="Type of intervention being tested (if applicable)"
    )

    sample_size: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of participants in the study"
    )

    primary_outcome: Optional[str] = Field(
        default=None,
        description="Primary outcome measure of the study"
    )

    study_population: Optional[str] = Field(
        default=None,
        description="Description of the study population (e.g., 'pediatric patients with asthma')"
    )

    randomized: Optional[bool] = Field(
        default=None,
        description="Whether the study was randomized"
    )

    blinded: Optional[bool] = Field(
        default=None,
        description="Whether the study was blinded"
    )

    multi_center: Optional[bool] = Field(
        default=None,
        description="Whether the study was conducted at multiple centers"
    )

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v


class Article(BaseModel):
    """Represents a PubMed article."""

    pmid: str = Field(description="PubMed ID")
    title: Optional[str] = Field(default=None, description="Article title")
    abstract: str = Field(description="Article abstract")
    authors: Optional[str] = Field(default=None, description="Authors")
    journal: Optional[str] = Field(default=None, description="Journal name")
    year: Optional[int] = Field(default=None, description="Publication year")

    # Additional metadata fields
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
    keywords: Optional[List[str]] = Field(default=None, description="Article keywords")
    mesh_terms: Optional[List[str]] = Field(default=None, description="MeSH terms")

    @field_validator('pmid')
    @classmethod
    def validate_pmid(cls, v: str) -> str:
        """Ensure PMID is not empty."""
        if not v or not v.strip():
            raise ValueError('PMID cannot be empty')
        return v.strip()

    @field_validator('abstract')
    @classmethod
    def validate_abstract(cls, v: str) -> str:
        """Ensure abstract is not empty."""
        if not v or not v.strip():
            raise ValueError('Abstract cannot be empty')
        return v.strip()


class ProcessingResult(BaseModel):
    """Complete result for a processed article."""

    article: Article
    classification: Optional[ClassificationResult] = None

    # Processing metadata
    processing_time_seconds: float = Field(default=0.0)
    model_used: Optional[str] = None
    success: bool = Field(default=False)
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    # Output parsing metadata
    raw_output: Optional[str] = None
    parsing_method: Optional[str] = Field(
        default=None,
        description="Method used to parse output: 'outlines', 'json_repair', 'regex', 'failed'"
    )


class ProcessingCheckpoint(BaseModel):
    """Checkpoint for resuming processing."""

    last_processed_index: int = Field(
        ge=0,
        description="Index of the last successfully processed article"
    )
    total_processed: int = Field(
        ge=0,
        description="Total number of articles processed"
    )
    successful: int = Field(
        ge=0,
        description="Number of successfully classified articles"
    )
    failed: int = Field(
        ge=0,
        description="Number of failed classifications"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    output_file: str = Field(description="Path to output file")


class ProcessingStats(BaseModel):
    """Statistics for a processing run."""

    total_articles: int
    processed: int
    successful: int
    failed: int

    # Classification breakdown
    clinical_trials: int = 0
    non_trials: int = 0

    # Timing
    total_time_seconds: float = 0.0
    average_time_per_article: float = 0.0

    # Trial statistics (for clinical trials only)
    trial_phases: Dict[str, int] = Field(default_factory=dict)
    study_types: Dict[str, int] = Field(default_factory=dict)
    intervention_types: Dict[str, int] = Field(default_factory=dict)

    # Parsing method statistics
    parsing_methods: Dict[str, int] = Field(default_factory=dict)

    def update_from_result(self, result: ProcessingResult) -> None:
        """Update statistics from a processing result."""
        self.processed += 1

        if result.success and result.classification:
            self.successful += 1

            # Update classification counts
            if result.classification.is_clinical_trial:
                self.clinical_trials += 1

                # Update trial-specific stats
                if result.classification.trial_phase:
                    phase = str(result.classification.trial_phase)
                    self.trial_phases[phase] = self.trial_phases.get(phase, 0) + 1

                if result.classification.study_type:
                    study_type = str(result.classification.study_type)
                    self.study_types[study_type] = self.study_types.get(study_type, 0) + 1

                if result.classification.intervention_type:
                    intervention = str(result.classification.intervention_type)
                    self.intervention_types[intervention] = self.intervention_types.get(intervention, 0) + 1
            else:
                self.non_trials += 1

            # Track parsing method
            if result.parsing_method:
                self.parsing_methods[result.parsing_method] = \
                    self.parsing_methods.get(result.parsing_method, 0) + 1
        else:
            self.failed += 1

        # Update timing
        self.total_time_seconds += result.processing_time_seconds
        if self.processed > 0:
            self.average_time_per_article = self.total_time_seconds / self.processed
