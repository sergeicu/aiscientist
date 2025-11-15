# tests/fixtures/sample_data.py

"""
Sample data fixtures for testing the graph database operations.
"""

from src.graph.models import Author, Paper, Institution, Topic


# Sample authors
SAMPLE_AUTHORS = [
    Author(
        author_id="smith_j_001",
        full_name="John Smith",
        last_name="Smith",
        first_name="John",
        initials="JS",
        h_index=25
    ),
    Author(
        author_id="doe_j_002",
        full_name="Jane Doe",
        last_name="Doe",
        first_name="Jane",
        initials="JD",
        h_index=30
    ),
    Author(
        author_id="brown_a_003",
        full_name="Alice Brown",
        last_name="Brown",
        first_name="Alice",
        initials="AB",
        h_index=15
    ),
]

# Sample papers
SAMPLE_PAPERS = [
    Paper(
        pmid="12345678",
        title="Novel CAR-T Therapy for Pediatric Leukemia",
        abstract="This study investigates...",
        year="2023",
        journal="Nature Medicine",
        doi="10.1038/nm.12345"
    ),
    Paper(
        pmid="87654321",
        title="Immunotherapy Advances in Cancer Treatment",
        abstract="Recent advances in...",
        year="2024",
        journal="Science",
        doi="10.1126/science.87654"
    ),
    Paper(
        pmid="11111111",
        title="Genetic Markers in Rare Diseases",
        abstract="We identified novel...",
        year="2023",
        journal="Cell",
        doi="10.1016/j.cell.11111"
    ),
]

# Sample institutions
SAMPLE_INSTITUTIONS = [
    Institution(
        name="Boston Children's Hospital",
        city="Boston",
        state="MA",
        country="USA",
        latitude=42.3378,
        longitude=-71.1041
    ),
    Institution(
        name="Harvard Medical School",
        city="Boston",
        state="MA",
        country="USA",
        latitude=42.3355,
        longitude=-71.1033
    ),
]

# Sample topics
SAMPLE_TOPICS = [
    Topic(
        topic_id="topic_001",
        label="Immunotherapy",
        description="Cancer immunotherapy research",
        size=150
    ),
    Topic(
        topic_id="topic_002",
        label="Pediatric Oncology",
        description="Childhood cancer research",
        size=200
    ),
]
