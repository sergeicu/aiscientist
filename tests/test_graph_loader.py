"""Tests for Graph Database Loader."""

import pytest
from src.graph_integration.loader import GraphLoader
from src.graph.connection import Neo4jConnection


@pytest.fixture
def neo4j_conn():
    """Create test Neo4j connection."""
    conn = Neo4jConnection(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )
    yield conn
    # Cleanup test data
    with conn.get_session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    conn.close()


@pytest.fixture
def sample_network():
    """Sample author network data."""
    return {
        "authors": [
            {
                "author_id": "a1",
                "name": "John Doe",
                "affiliations": ["Harvard Medical School"],
                "papers": ["pmid1", "pmid2"]
            },
            {
                "author_id": "a2",
                "name": "Jane Smith",
                "affiliations": ["Mayo Clinic"],
                "papers": ["pmid1"]
            }
        ],
        "collaborations": [
            {
                "author_1": "a1",
                "author_2": "a2",
                "papers": ["pmid1"],
                "weight": 1
            }
        ],
        "papers": [
            {
                "pmid": "pmid1",
                "title": "Cancer Research",
                "publication_date": "2024-01-15"
            },
            {
                "pmid": "pmid2",
                "title": "Heart Disease",
                "publication_date": "2024-02-20"
            }
        ]
    }


class TestGraphLoader:
    """Test graph database loader."""

    def test_load_authors(self, neo4j_conn, sample_network):
        """Test loading authors into Neo4j."""
        loader = GraphLoader(neo4j_conn)

        stats = loader.load_authors(sample_network["authors"])

        assert stats["created"] == 2
        assert stats["updated"] == 0

        # Verify in database
        with neo4j_conn.get_session() as session:
            result = session.run("MATCH (a:Author) RETURN count(a) as count")
            count = result.single()["count"]
            assert count == 2

    def test_load_institutions(self, neo4j_conn, sample_network):
        """Test loading institutions."""
        loader = GraphLoader(neo4j_conn)

        institutions = ["Harvard Medical School", "Mayo Clinic"]
        stats = loader.load_institutions(institutions)

        assert stats["created"] == 2

        # Verify
        with neo4j_conn.get_session() as session:
            result = session.run("MATCH (i:Institution) RETURN count(i) as count")
            assert result.single()["count"] == 2

    def test_load_papers(self, neo4j_conn, sample_network):
        """Test loading papers."""
        loader = GraphLoader(neo4j_conn)

        stats = loader.load_papers(sample_network["papers"])

        assert stats["created"] == 2

    def test_create_authorship_relationships(self, neo4j_conn, sample_network):
        """Test creating AUTHORED relationships."""
        loader = GraphLoader(neo4j_conn)

        # First load authors and papers
        loader.load_authors(sample_network["authors"])
        loader.load_papers(sample_network["papers"])

        # Create relationships
        stats = loader.create_authorship_relationships(sample_network["authors"])

        assert stats["created"] == 3  # a1->pmid1, a1->pmid2, a2->pmid1

        # Verify
        with neo4j_conn.get_session() as session:
            result = session.run(
                "MATCH (:Author)-[r:AUTHORED]->(:Paper) RETURN count(r) as count"
            )
            assert result.single()["count"] == 3

    def test_create_collaboration_relationships(self, neo4j_conn, sample_network):
        """Test creating COLLABORATED_WITH relationships."""
        loader = GraphLoader(neo4j_conn)

        # Load authors first
        loader.load_authors(sample_network["authors"])

        # Create collaborations
        stats = loader.create_collaboration_relationships(
            sample_network["collaborations"]
        )

        assert stats["created"] == 1

    def test_load_full_network(self, neo4j_conn, sample_network):
        """Test loading complete network."""
        loader = GraphLoader(neo4j_conn)

        report = loader.load_full_network(sample_network)

        assert report["authors"]["created"] == 2
        assert report["papers"]["created"] == 2
        assert report["collaborations"]["created"] == 1
        assert report["status"] == "success"

    def test_merge_duplicate_authors(self, neo4j_conn, sample_network):
        """Test that duplicate authors are merged, not created."""
        loader = GraphLoader(neo4j_conn)

        # Load twice
        loader.load_authors(sample_network["authors"])
        stats = loader.load_authors(sample_network["authors"])

        # Should update, not create new
        assert stats["created"] == 0
        assert stats["updated"] == 2

        # Verify only 2 authors exist
        with neo4j_conn.get_session() as session:
            result = session.run("MATCH (a:Author) RETURN count(a) as count")
            assert result.single()["count"] == 2

    def test_incremental_load(self, neo4j_conn, sample_network):
        """Test incremental data loading."""
        loader = GraphLoader(neo4j_conn)

        # Initial load
        loader.load_full_network(sample_network)

        # Add new author
        new_network = {
            "authors": [
                {
                    "author_id": "a3",
                    "name": "Bob Johnson",
                    "affiliations": ["Stanford"],
                    "papers": ["pmid3"]
                }
            ],
            "papers": [
                {
                    "pmid": "pmid3",
                    "title": "AI Research",
                    "publication_date": "2024-03-01"
                }
            ],
            "collaborations": []
        }

        report = loader.load_full_network(new_network)

        assert report["authors"]["created"] == 1
        assert report["papers"]["created"] == 1

        # Total should be 3 authors, 3 papers
        with neo4j_conn.get_session() as session:
            result = session.run("MATCH (a:Author) RETURN count(a) as count")
            assert result.single()["count"] == 3
