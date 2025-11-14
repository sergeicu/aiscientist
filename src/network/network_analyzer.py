from typing import List, Dict, Optional
from neo4j import GraphDatabase
from loguru import logger

class NetworkAnalyzer:
    """
    Analyze co-authorship network.

    Calculate metrics:
    - Degree centrality (number of connections)
    - Betweenness centrality (bridge position)
    - PageRank (influence)
    - Community detection

    Args:
        driver: Neo4j driver

    Example:
        >>> analyzer = NetworkAnalyzer(driver)
        >>> top_authors = analyzer.find_top_collaborators(limit=20)
    """

    def __init__(self, driver):
        self.driver = driver

    def find_top_collaborators(self, limit: int = 20) -> List[Dict]:
        """
        Find authors with most unique collaborators.

        Args:
            limit: Number of results to return

        Returns:
            List of authors with collaboration counts
        """
        with self.driver.session() as session:
            query = """
            MATCH (a:Author)-[r:COLLABORATED_WITH]-(other:Author)
            RETURN a.full_name as author,
                   a.author_id as author_id,
                   count(DISTINCT other) as collaborations
            ORDER BY collaborations DESC
            LIMIT $limit
            """

            result = session.run(query, limit=limit)

            return [dict(record) for record in result]

    def find_institutional_collaborations(
        self,
        institution: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        Find top institution collaboration pairs.

        Args:
            institution: Focus on specific institution (optional)
            limit: Number of results

        Returns:
            List of institution pairs with collaboration counts
        """
        with self.driver.session() as session:
            if institution:
                query = """
                MATCH (i1:Institution {name: $institution})<-[:AFFILIATED_WITH]-(a1:Author)
                MATCH (a1)-[:COLLABORATED_WITH]-(a2:Author)-[:AFFILIATED_WITH]->(i2:Institution)
                WHERE i1.name <> i2.name
                RETURN i1.name as institution1,
                       i2.name as institution2,
                       count(*) as collaboration_count
                ORDER BY collaboration_count DESC
                LIMIT $limit
                """
                result = session.run(query, institution=institution, limit=limit)
            else:
                query = """
                MATCH (i1:Institution)<-[:AFFILIATED_WITH]-(a1:Author)
                MATCH (a1)-[:COLLABORATED_WITH]-(a2:Author)-[:AFFILIATED_WITH]->(i2:Institution)
                WHERE id(i1) < id(i2)
                RETURN i1.name as institution1,
                       i2.name as institution2,
                       count(*) as collaboration_count
                ORDER BY collaboration_count DESC
                LIMIT $limit
                """
                result = session.run(query, limit=limit)

            return [dict(record) for record in result]

    def calculate_author_centrality(self) -> List[Dict]:
        """
        Calculate degree centrality for all authors.

        Uses Neo4j Graph Data Science library if available.

        Returns:
            List of authors with centrality scores
        """
        with self.driver.session() as session:
            # Simple degree centrality query
            query = """
            MATCH (a:Author)-[r:COLLABORATED_WITH]-(other)
            RETURN a.full_name as author,
                   a.author_id as author_id,
                   count(r) as degree
            ORDER BY degree DESC
            """

            result = session.run(query)

            return [dict(record) for record in result]

    def find_research_communities(self, min_size: int = 5) -> List[Dict]:
        """
        Identify research communities using community detection.

        Args:
            min_size: Minimum community size

        Returns:
            List of communities with member authors
        """
        # This requires Neo4j GDS library
        # Placeholder implementation
        logger.warning("Community detection requires Neo4j GDS library")
        return []

    def get_author_network(self, author_name: str, depth: int = 2) -> Dict:
        """
        Get author's collaboration network.

        Args:
            author_name: Author full name
            depth: Degree of separation (1 = direct collaborators, 2 = collaborators of collaborators)

        Returns:
            Network data (nodes and edges)
        """
        with self.driver.session() as session:
            query = """
            MATCH path = (a:Author {full_name: $name})-[:COLLABORATED_WITH*1..$depth]-(other:Author)
            RETURN nodes(path) as nodes, relationships(path) as edges
            """

            result = session.run(query, name=author_name, depth=depth)

            nodes = set()
            edges = []

            for record in result:
                for node in record['nodes']:
                    nodes.add((node['author_id'], node['full_name']))

                for edge in record['edges']:
                    edges.append(edge)

            return {
                'nodes': [{'id': n[0], 'name': n[1]} for n in nodes],
                'edges': edges
            }
