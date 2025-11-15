# src/graph/models.py

from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class Author(BaseModel):
    """
    Author node model.

    Represents a researcher/author in the knowledge graph.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "author_id": "smith_j_bch",
                "full_name": "John Smith",
                "last_name": "Smith",
                "first_name": "John",
                "initials": "JS"
            }
        }
    )

    author_id: str = Field(..., description="Unique author identifier")
    full_name: str = Field(..., description="Full name")
    last_name: Optional[str] = Field(None, description="Last name")
    first_name: Optional[str] = Field(None, description="First name")
    initials: Optional[str] = Field(None, description="Initials")
    orcid: Optional[str] = Field(None, description="ORCID identifier")
    h_index: Optional[int] = Field(None, description="H-index")


class Paper(BaseModel):
    """
    Paper node model.

    Represents a publication in the knowledge graph.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pmid": "12345678",
                "title": "Novel CAR-T Therapy",
                "year": "2023",
                "journal": "Nature Medicine"
            }
        }
    )

    pmid: str = Field(..., description="PubMed ID")
    title: str = Field(..., description="Paper title")
    abstract: Optional[str] = Field(None, description="Abstract")
    year: Optional[str] = Field(None, description="Publication year")
    journal: Optional[str] = Field(None, description="Journal name")
    doi: Optional[str] = Field(None, description="DOI")


class Institution(BaseModel):
    """
    Institution node model.

    Represents a research institution/organization.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Boston Children's Hospital",
                "city": "Boston",
                "state": "MA",
                "country": "USA"
            }
        }
    )

    name: str = Field(..., description="Institution name")
    city: Optional[str] = Field(None, description="City")
    state: Optional[str] = Field(None, description="State/Province")
    country: Optional[str] = Field(None, description="Country")
    latitude: Optional[float] = Field(None, description="Latitude")
    longitude: Optional[float] = Field(None, description="Longitude")


class Topic(BaseModel):
    """
    Topic node model.

    Represents a research topic/cluster.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "topic_id": "topic_001",
                "label": "Immunotherapy",
                "description": "Cancer immunotherapy research"
            }
        }
    )

    topic_id: str = Field(..., description="Topic identifier")
    label: str = Field(..., description="Topic name/label")
    description: Optional[str] = Field(None, description="Topic description")
    size: Optional[int] = Field(None, description="Number of papers")


class Relationship(BaseModel):
    """
    Base relationship model.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "COLLABORATED_WITH",
                "properties": {"count": 5, "years": ["2020", "2021"]}
            }
        }
    )

    type: str = Field(..., description="Relationship type")
    properties: Optional[dict] = Field(default_factory=dict, description="Relationship properties")
