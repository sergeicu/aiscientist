# tests/test_models.py

import pytest
from pydantic import ValidationError
from src.graph.models import Author, Paper, Institution, Topic


def test_author_model_valid():
    """Should create valid Author model."""
    author = Author(
        author_id="abc123",
        full_name="John Smith",
        last_name="Smith",
        first_name="John",
        initials="JS"
    )

    assert author.author_id == "abc123"
    assert author.full_name == "John Smith"


def test_author_model_missing_required_field():
    """Should raise error if required field missing."""
    with pytest.raises(ValidationError):
        Author(
            full_name="John Smith"
            # Missing author_id
        )


def test_paper_model_valid():
    """Should create valid Paper model."""
    paper = Paper(
        pmid="12345678",
        title="Test Paper",
        year="2023",
        journal="Nature"
    )

    assert paper.pmid == "12345678"
    assert paper.year == "2023"


def test_paper_model_optional_fields():
    """Should handle optional fields."""
    paper = Paper(
        pmid="12345678",
        title="Test Paper"
        # year and journal are optional
    )

    assert paper.year is None
    assert paper.journal is None


def test_institution_model():
    """Should create valid Institution model."""
    institution = Institution(
        name="Boston Children's Hospital",
        city="Boston",
        state="MA",
        country="USA"
    )

    assert institution.name == "Boston Children's Hospital"
    assert institution.country == "USA"


def test_topic_model():
    """Should create valid Topic model."""
    topic = Topic(
        topic_id="topic_001",
        label="Immunotherapy",
        description="Research on immune-based treatments"
    )

    assert topic.topic_id == "topic_001"
    assert topic.label == "Immunotherapy"


def test_model_to_dict():
    """Should convert model to dictionary."""
    author = Author(
        author_id="abc123",
        full_name="John Smith",
        last_name="Smith",
        first_name="John"
    )

    data = author.model_dump()

    assert isinstance(data, dict)
    assert data['author_id'] == "abc123"


def test_model_json_serialization():
    """Should serialize to JSON."""
    paper = Paper(
        pmid="12345678",
        title="Test Paper",
        year="2023"
    )

    json_str = paper.model_dump_json()

    assert isinstance(json_str, str)
    assert "12345678" in json_str
