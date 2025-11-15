"""Graph database loader for author networks."""

import logging
from typing import List, Dict, Any
from datetime import datetime

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
    from src.graph.connection import Neo4jConnection

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
