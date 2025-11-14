# src/graph/operations.py

from typing import Dict, List, Optional, Any
from loguru import logger
from .models import Author, Paper, Institution, Topic


class GraphOperations:
    """
    Core CRUD operations for graph database.

    Provides methods to create, read, update, delete nodes and relationships.

    Args:
        connection: Neo4jConnection instance

    Example:
        >>> ops = GraphOperations(connection)
        >>> ops.create_author(author_model)
    """

    def __init__(self, connection):
        self.connection = connection

    # ========== Author Operations ==========

    def create_author(self, author: Author) -> Dict:
        """
        Create Author node.

        Args:
            author: Author model

        Returns:
            Created author properties
        """
        with self.connection.get_session() as session:
            query = """
            MERGE (a:Author {author_id: $author_id})
            SET a.full_name = $full_name,
                a.last_name = $last_name,
                a.first_name = $first_name,
                a.initials = $initials,
                a.orcid = $orcid,
                a.h_index = $h_index
            RETURN a
            """

            result = session.run(query, **author.model_dump())
            record = result.single()

            if record:
                return dict(record['a'])

            return {}

    def get_author(self, author_id: str) -> Optional[Dict]:
        """
        Get author by ID.

        Args:
            author_id: Author identifier

        Returns:
            Author properties or None
        """
        with self.connection.get_session() as session:
            query = """
            MATCH (a:Author {author_id: $author_id})
            RETURN a
            """

            result = session.run(query, author_id=author_id)
            record = result.single()

            if record:
                return dict(record['a'])

            return None

    def update_author(self, author_id: str, properties: Dict) -> Dict:
        """
        Update author properties.

        Args:
            author_id: Author identifier
            properties: Properties to update

        Returns:
            Updated author properties
        """
        with self.connection.get_session() as session:
            # Build SET clause
            set_clause = ", ".join([f"a.{key} = ${key}" for key in properties.keys()])

            query = f"""
            MATCH (a:Author {{author_id: $author_id}})
            SET {set_clause}
            RETURN a
            """

            result = session.run(query, author_id=author_id, **properties)
            record = result.single()

            if record:
                return dict(record['a'])

            return {}

    def delete_author(self, author_id: str) -> bool:
        """
        Delete author node.

        Args:
            author_id: Author identifier

        Returns:
            True if deleted
        """
        with self.connection.get_session() as session:
            query = """
            MATCH (a:Author {author_id: $author_id})
            DETACH DELETE a
            """

            session.run(query, author_id=author_id)
            return True

    def batch_create_authors(self, authors: List[Author]) -> None:
        """
        Create multiple authors in batch.

        Args:
            authors: List of Author models
        """
        with self.connection.get_session() as session:
            query = """
            UNWIND $authors AS author
            MERGE (a:Author {author_id: author.author_id})
            SET a.full_name = author.full_name,
                a.last_name = author.last_name,
                a.first_name = author.first_name,
                a.initials = author.initials
            """

            author_dicts = [a.model_dump() for a in authors]
            session.run(query, authors=author_dicts)

    # ========== Paper Operations ==========

    def create_paper(self, paper: Paper) -> Dict:
        """Create Paper node."""
        with self.connection.get_session() as session:
            query = """
            MERGE (p:Paper {pmid: $pmid})
            SET p.title = $title,
                p.abstract = $abstract,
                p.year = $year,
                p.journal = $journal,
                p.doi = $doi
            RETURN p
            """

            result = session.run(query, **paper.model_dump())
            record = result.single()

            if record:
                return dict(record['p'])

            return {}

    def get_paper(self, pmid: str) -> Optional[Dict]:
        """Get paper by PMID."""
        with self.connection.get_session() as session:
            query = """
            MATCH (p:Paper {pmid: $pmid})
            RETURN p
            """

            result = session.run(query, pmid=pmid)
            record = result.single()

            if record:
                return dict(record['p'])

            return None

    # ========== Institution Operations ==========

    def create_institution(self, institution: Institution) -> Dict:
        """Create Institution node."""
        with self.connection.get_session() as session:
            query = """
            MERGE (i:Institution {name: $name})
            SET i.city = $city,
                i.state = $state,
                i.country = $country,
                i.latitude = $latitude,
                i.longitude = $longitude
            RETURN i
            """

            result = session.run(query, **institution.model_dump())
            record = result.single()

            if record:
                return dict(record['i'])

            return {}

    # ========== Relationship Operations ==========

    def create_authored_relationship(
        self,
        author_id: str,
        pmid: str,
        position: int = 0,
        is_first_author: bool = False,
        is_last_author: bool = False
    ) -> None:
        """
        Create AUTHORED relationship.

        Args:
            author_id: Author identifier
            pmid: Paper PMID
            position: Author position in author list
            is_first_author: Is first author
            is_last_author: Is last author
        """
        with self.connection.get_session() as session:
            query = """
            MATCH (a:Author {author_id: $author_id})
            MATCH (p:Paper {pmid: $pmid})
            MERGE (a)-[r:AUTHORED]->(p)
            SET r.position = $position,
                r.is_first_author = $is_first_author,
                r.is_last_author = $is_last_author
            """

            session.run(
                query,
                author_id=author_id,
                pmid=pmid,
                position=position,
                is_first_author=is_first_author,
                is_last_author=is_last_author
            )

    def create_collaboration(
        self,
        author1_id: str,
        author2_id: str,
        pmid: str
    ) -> None:
        """
        Create COLLABORATED_WITH relationship.

        Args:
            author1_id: First author ID
            author2_id: Second author ID
            pmid: Paper they collaborated on
        """
        with self.connection.get_session() as session:
            query = """
            MATCH (a1:Author {author_id: $author1_id})
            MATCH (a2:Author {author_id: $author2_id})
            MERGE (a1)-[r:COLLABORATED_WITH]-(a2)
            ON CREATE SET r.count = 1, r.pmids = [$pmid]
            ON MATCH SET r.count = r.count + 1,
                        r.pmids = r.pmids + $pmid
            """

            session.run(
                query,
                author1_id=author1_id,
                author2_id=author2_id,
                pmid=pmid
            )

    # ========== Query Operations ==========

    def get_author_papers(self, author_id: str) -> List[Dict]:
        """
        Get all papers authored by author.

        Args:
            author_id: Author identifier

        Returns:
            List of paper dictionaries
        """
        with self.connection.get_session() as session:
            query = """
            MATCH (a:Author {author_id: $author_id})-[:AUTHORED]->(p:Paper)
            RETURN p
            """

            result = session.run(query, author_id=author_id)

            return [dict(record['p']) for record in result]

    def get_author_collaborators(self, author_id: str) -> List[Dict]:
        """
        Get all collaborators of author.

        Args:
            author_id: Author identifier

        Returns:
            List of collaborator dictionaries
        """
        with self.connection.get_session() as session:
            query = """
            MATCH (a:Author {author_id: $author_id})-[r:COLLABORATED_WITH]-(other:Author)
            RETURN other, r.count as collaboration_count
            """

            result = session.run(query, author_id=author_id)

            return [
                {
                    **dict(record['other']),
                    'collaboration_count': record['collaboration_count']
                }
                for record in result
            ]

    # ========== Utility Operations ==========

    def count_nodes(self, label: str) -> int:
        """
        Count nodes by label.

        Args:
            label: Node label (e.g., 'Author', 'Paper')

        Returns:
            Node count
        """
        with self.connection.get_session() as session:
            query = f"MATCH (n:{label}) RETURN count(n) as count"

            result = session.run(query)
            record = result.single()

            return record['count'] if record else 0

    def clear_all(self) -> None:
        """Clear all nodes and relationships from database."""
        logger.warning("Clearing entire database...")

        with self.connection.get_session() as session:
            query = "MATCH (n) DETACH DELETE n"
            session.run(query)

        logger.warning("Database cleared")
