# Task 12: Graph Database Integration

**Status:** Integration Task
**Dependencies:** Tasks 3 (Author Network), 4 (Neo4j Setup), 11 (Pipeline)
**Estimated Time:** 3-4 hours
**Difficulty:** Medium

---

## Objective

Create a data loader that ingests author networks from the pipeline into Neo4j graph database, providing a unified interface for graph queries and updates.

---

## Background

After the pipeline collects data (Task 11) and extracts author networks (Task 3), we need to load this into Neo4j (Task 4) for graph analytics and visualization.

This task creates:
- Bulk data loader for Neo4j
- Update/merge strategies for incremental loads
- Validation and quality checks
- Simple query interface

---

## Requirements

### Functional Requirements

1. **Data Loading**
   - Load authors, papers, institutions into Neo4j
   - Create relationships (AUTHORED, COLLABORATED_WITH, AFFILIATED_WITH)
   - Handle duplicates (merge not create)
   - Bulk insert for performance

2. **Update Strategy**
   - Incremental updates (new data only)
   - Update existing nodes/edges
   - Track data freshness

3. **Validation**
   - Verify data integrity
   - Check constraint violations
   - Count nodes/edges

4. **Query Interface**
   - Simple queries (find author, get collaborators)
   - Export graph data for visualizations

---

## Implementation Guide (TDD)

### Test First (`tests/test_graph_loader.py`):

```python
import pytest
from graph_integration.loader import GraphLoader
from neo4j_setup import Neo4jConnection


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
```

**Implementation** (`src/graph_integration/loader.py`):

```python
"""Graph database loader for author networks."""

import logging
from typing import List, Dict, Any
from datetime import datetime
from neo4j import Session

logger = logging.getLogger(__name__)


class GraphLoader:
    """Loads author network data into Neo4j."""

    def __init__(self, neo4j_connection):
        """Initialize loader with Neo4j connection."""
        self.conn = neo4j_connection

    def load_authors(self, authors: List[Dict[str, Any]]) -> Dict[str, int]:
        """Load authors into Neo4j using MERGE (upsert)."""
        created = 0
        updated = 0

        with self.conn.get_session() as session:
            for author in authors:
                result = session.run(
                    """
                    MERGE (a:Author {author_id: $author_id})
                    ON CREATE SET
                        a.name = $name,
                        a.affiliations = $affiliations,
                        a.papers = $papers,
                        a.created_at = datetime(),
                        a.updated_at = datetime()
                    ON MATCH SET
                        a.name = $name,
                        a.affiliations = $affiliations,
                        a.papers = $papers,
                        a.updated_at = datetime()
                    RETURN
                        a,
                        CASE WHEN a.created_at = a.updated_at THEN 'created' ELSE 'updated' END as action
                    """,
                    author_id=author["author_id"],
                    name=author["name"],
                    affiliations=author.get("affiliations", []),
                    papers=author.get("papers", [])
                )

                record = result.single()
                if record["action"] == "created":
                    created += 1
                else:
                    updated += 1

        logger.info(f"Loaded authors: {created} created, {updated} updated")

        return {"created": created, "updated": updated}

    def load_institutions(self, institutions: List[str]) -> Dict[str, int]:
        """Load institutions into Neo4j."""
        created = 0

        with self.conn.get_session() as session:
            for institution in institutions:
                result = session.run(
                    """
                    MERGE (i:Institution {name: $name})
                    ON CREATE SET
                        i.created_at = datetime()
                    RETURN
                        CASE WHEN i.created_at >= datetime() - duration('PT1S')
                        THEN 'created' ELSE 'exists' END as action
                    """,
                    name=institution
                )

                if result.single()["action"] == "created":
                    created += 1

        logger.info(f"Loaded institutions: {created} created")

        return {"created": created}

    def load_papers(self, papers: List[Dict[str, Any]]) -> Dict[str, int]:
        """Load papers into Neo4j."""
        created = 0
        updated = 0

        with self.conn.get_session() as session:
            for paper in papers:
                result = session.run(
                    """
                    MERGE (p:Paper {pmid: $pmid})
                    ON CREATE SET
                        p.title = $title,
                        p.publication_date = date($pub_date),
                        p.created_at = datetime()
                    ON MATCH SET
                        p.title = $title,
                        p.publication_date = date($pub_date)
                    RETURN
                        CASE WHEN p.created_at >= datetime() - duration('PT1S')
                        THEN 'created' ELSE 'updated' END as action
                    """,
                    pmid=paper["pmid"],
                    title=paper.get("title", ""),
                    pub_date=paper.get("publication_date", "2024-01-01")
                )

                if result.single()["action"] == "created":
                    created += 1
                else:
                    updated += 1

        logger.info(f"Loaded papers: {created} created, {updated} updated")

        return {"created": created, "updated": updated}

    def create_authorship_relationships(
        self,
        authors: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Create AUTHORED relationships between authors and papers."""
        created = 0

        with self.conn.get_session() as session:
            for author in authors:
                for pmid in author.get("papers", []):
                    result = session.run(
                        """
                        MATCH (a:Author {author_id: $author_id})
                        MATCH (p:Paper {pmid: $pmid})
                        MERGE (a)-[r:AUTHORED]->(p)
                        ON CREATE SET r.created_at = datetime()
                        RETURN
                            CASE WHEN r.created_at >= datetime() - duration('PT1S')
                            THEN 'created' ELSE 'exists' END as action
                        """,
                        author_id=author["author_id"],
                        pmid=pmid
                    )

                    if result.single()["action"] == "created":
                        created += 1

        logger.info(f"Created {created} authorship relationships")

        return {"created": created}

    def create_affiliation_relationships(
        self,
        authors: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Create AFFILIATED_WITH relationships."""
        created = 0

        with self.conn.get_session() as session:
            for author in authors:
                for institution in author.get("affiliations", []):
                    # First ensure institution exists
                    session.run(
                        "MERGE (i:Institution {name: $name})",
                        name=institution
                    )

                    # Create relationship
                    result = session.run(
                        """
                        MATCH (a:Author {author_id: $author_id})
                        MATCH (i:Institution {name: $institution})
                        MERGE (a)-[r:AFFILIATED_WITH]->(i)
                        ON CREATE SET r.created_at = datetime()
                        RETURN
                            CASE WHEN r.created_at >= datetime() - duration('PT1S')
                            THEN 'created' ELSE 'exists' END as action
                        """,
                        author_id=author["author_id"],
                        institution=institution
                    )

                    if result.single()["action"] == "created":
                        created += 1

        logger.info(f"Created {created} affiliation relationships")

        return {"created": created}

    def create_collaboration_relationships(
        self,
        collaborations: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Create COLLABORATED_WITH relationships."""
        created = 0

        with self.conn.get_session() as session:
            for collab in collaborations:
                result = session.run(
                    """
                    MATCH (a1:Author {author_id: $author_1})
                    MATCH (a2:Author {author_id: $author_2})
                    MERGE (a1)-[r:COLLABORATED_WITH]-(a2)
                    ON CREATE SET
                        r.papers = $papers,
                        r.weight = $weight,
                        r.created_at = datetime()
                    ON MATCH SET
                        r.papers = r.papers + $papers,
                        r.weight = r.weight + $weight
                    RETURN
                        CASE WHEN r.created_at >= datetime() - duration('PT1S')
                        THEN 'created' ELSE 'updated' END as action
                    """,
                    author_1=collab["author_1"],
                    author_2=collab["author_2"],
                    papers=collab.get("papers", []),
                    weight=collab.get("weight", 1)
                )

                if result.single()["action"] == "created":
                    created += 1

        logger.info(f"Created {created} collaboration relationships")

        return {"created": created}

    def load_full_network(self, network: Dict[str, Any]) -> Dict[str, Any]:
        """Load complete network into Neo4j."""
        logger.info("Starting full network load")

        try:
            # Load nodes
            authors_stats = self.load_authors(network.get("authors", []))
            papers_stats = self.load_papers(network.get("papers", []))

            # Extract institutions from authors
            institutions = set()
            for author in network.get("authors", []):
                institutions.update(author.get("affiliations", []))

            institutions_stats = self.load_institutions(list(institutions))

            # Create relationships
            authorship_stats = self.create_authorship_relationships(
                network.get("authors", [])
            )
            affiliation_stats = self.create_affiliation_relationships(
                network.get("authors", [])
            )
            collaboration_stats = self.create_collaboration_relationships(
                network.get("collaborations", [])
            )

            report = {
                "status": "success",
                "authors": authors_stats,
                "papers": papers_stats,
                "institutions": institutions_stats,
                "authorship": authorship_stats,
                "affiliations": affiliation_stats,
                "collaborations": collaboration_stats,
                "timestamp": datetime.now().isoformat()
            }

            logger.info("Full network load completed successfully")

            return report

        except Exception as e:
            logger.error(f"Failed to load network: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def get_database_stats(self) -> Dict[str, int]:
        """Get current database statistics."""
        with self.conn.get_session() as session:
            # Count nodes
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
            """)

            node_counts = {record["label"]: record["count"] for record in result}

            # Count relationships
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
            """)

            relationship_counts = {record["type"]: record["count"] for record in result}

            return {
                "nodes": node_counts,
                "relationships": relationship_counts
            }


# CLI entry point
def main():
    """Load network data from pipeline into Neo4j."""
    import argparse
    import json
    from pathlib import Path
    from neo4j_setup import Neo4jConnection

    parser = argparse.ArgumentParser(description="Load data into Neo4j")
    parser.add_argument(
        "--network-file",
        type=Path,
        required=True,
        help="Path to author network JSON file"
    )
    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="Neo4j URI"
    )
    parser.add_argument(
        "--neo4j-user",
        default="neo4j",
        help="Neo4j username"
    )
    parser.add_argument(
        "--neo4j-password",
        required=True,
        help="Neo4j password"
    )

    args = parser.parse_args()

    # Load network data
    with open(args.network_file) as f:
        network = json.load(f)

    # Connect to Neo4j
    conn = Neo4jConnection(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password
    )

    # Load data
    loader = GraphLoader(conn)
    report = loader.load_full_network(network)

    # Print report
    print(json.dumps(report, indent=2))

    # Print stats
    stats = loader.get_database_stats()
    print("\nDatabase Statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
```

---

## Usage

```bash
# Load author network into Neo4j
python -m graph_integration.loader \
    --network-file data/raw/author_networks/network_all_institutions_*.json \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password yourpassword
```

---

## Success Criteria

- [ ] Loads authors, papers, institutions into Neo4j
- [ ] Creates all relationships correctly
- [ ] Handles duplicates (merge strategy)
- [ ] Supports incremental updates
- [ ] All tests pass
- [ ] Test coverage ≥ 80%

---

**End of Task 12 Specification**
