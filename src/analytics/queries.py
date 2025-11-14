# src/analytics/queries.py

from typing import List, Dict, Optional
from loguru import logger


class AdvancedQueries:
    """
    Advanced Cypher queries for research network analysis.

    Provides high-level query methods for common use cases.

    Args:
        connection: Neo4jConnection instance

    Example:
        >>> queries = AdvancedQueries(connection)
        >>> path = queries.find_collaboration_path(author1, author2)
    """

    def __init__(self, connection):
        self.connection = connection

    def find_collaboration_path(
        self,
        author1_id: str,
        author2_id: str,
        max_depth: int = 5
    ) -> Optional[Dict]:
        """
        Find shortest collaboration path between two authors.

        Args:
            author1_id: First author
            author2_id: Second author
            max_depth: Maximum path length

        Returns:
            Path information or None
        """
        with self.connection.get_session() as session:
            query = """
            MATCH path = shortestPath(
                (a1:Author {author_id: $author1_id})-[:COLLABORATED_WITH*..%d]-(a2:Author {author_id: $author2_id})
            )
            RETURN path, length(path) as length
            """ % max_depth

            result = session.run(
                query,
                author1_id=author1_id,
                author2_id=author2_id
            )

            record = result.single()

            if record:
                return {
                    'length': record['length'],
                    'nodes': [dict(node) for node in record['path'].nodes]
                }

            return None

    def find_common_collaborators(
        self,
        author1_id: str,
        author2_id: str
    ) -> List[Dict]:
        """
        Find authors who collaborated with both authors.

        Args:
            author1_id: First author
            author2_id: Second author

        Returns:
            List of common collaborators
        """
        with self.connection.get_session() as session:
            query = """
            MATCH (a1:Author {author_id: $author1_id})-[:COLLABORATED_WITH]-(common:Author)
            MATCH (a2:Author {author_id: $author2_id})-[:COLLABORATED_WITH]-(common)
            WHERE common.author_id <> $author1_id
              AND common.author_id <> $author2_id
            RETURN DISTINCT common
            """

            result = session.run(
                query,
                author1_id=author1_id,
                author2_id=author2_id
            )

            return [dict(record['common']) for record in result]

    def find_prolific_authors(
        self,
        limit: int = 20,
        min_year: Optional[str] = None
    ) -> List[Dict]:
        """
        Find authors with most publications.

        Args:
            limit: Number of results
            min_year: Minimum publication year

        Returns:
            List of authors with paper counts
        """
        with self.connection.get_session() as session:
            query = """
            MATCH (a:Author)-[:AUTHORED]->(p:Paper)
            """

            if min_year:
                query += f"WHERE p.year >= '{min_year}' "

            query += """
            WITH a, count(p) as paper_count
            RETURN a, paper_count
            ORDER BY paper_count DESC
            LIMIT $limit
            """

            result = session.run(query, limit=limit)

            return [
                {
                    **dict(record['a']),
                    'paper_count': record['paper_count']
                }
                for record in result
            ]

    def find_research_trends(
        self,
        start_year: str,
        end_year: str
    ) -> List[Dict]:
        """
        Find research trends over time period.

        Args:
            start_year: Start year
            end_year: End year

        Returns:
            Publication trends by year
        """
        with self.connection.get_session() as session:
            query = """
            MATCH (p:Paper)
            WHERE p.year >= $start_year AND p.year <= $end_year
            WITH p.year as year, count(p) as publications
            RETURN year, publications
            ORDER BY year
            """

            result = session.run(
                query,
                start_year=start_year,
                end_year=end_year
            )

            return [dict(record) for record in result]

    def recommend_collaborators(
        self,
        author_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Recommend potential collaborators.

        Based on common collaborators and research areas.

        Args:
            author_id: Author to find recommendations for
            limit: Number of recommendations

        Returns:
            List of recommended authors with scores
        """
        with self.connection.get_session() as session:
            # Find authors who collaborated with my collaborators
            # but not with me
            query = """
            MATCH (me:Author {author_id: $author_id})-[:COLLABORATED_WITH]-(colleague:Author)
            MATCH (colleague)-[:COLLABORATED_WITH]-(candidate:Author)
            WHERE candidate.author_id <> $author_id
              AND NOT (me)-[:COLLABORATED_WITH]-(candidate)
            WITH candidate, count(DISTINCT colleague) as common_colleagues
            RETURN candidate, common_colleagues as score
            ORDER BY score DESC
            LIMIT $limit
            """

            result = session.run(query, author_id=author_id, limit=limit)

            return [
                {
                    **dict(record['candidate']),
                    'score': record['score']
                }
                for record in result
            ]

    def find_institutional_leaders(
        self,
        institution: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Find leading researchers at institution.

        Args:
            institution: Institution name
            limit: Number of results

        Returns:
            List of top authors
        """
        with self.connection.get_session() as session:
            query = """
            MATCH (a:Author)-[:AFFILIATED_WITH]->(i:Institution {name: $institution})
            MATCH (a)-[:AUTHORED]->(p:Paper)
            WITH a, count(p) as paper_count
            RETURN a, paper_count
            ORDER BY paper_count DESC
            LIMIT $limit
            """

            result = session.run(query, institution=institution, limit=limit)

            return [
                {
                    **dict(record['a']),
                    'paper_count': record['paper_count']
                }
                for record in result
            ]
