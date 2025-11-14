# src/analytics/community.py

from typing import List, Dict, Optional
from loguru import logger
import random


class CommunityDetector:
    """
    Detect research communities in collaboration network.

    Identifies groups of closely connected authors.

    Args:
        connection: Neo4jConnection instance

    Example:
        >>> detector = CommunityDetector(connection)
        >>> communities = detector.detect_communities()
    """

    def __init__(self, connection):
        self.connection = connection

    def detect_communities(
        self,
        algorithm: str = 'louvain',
        node_label: str = 'Author',
        relationship_type: str = 'COLLABORATED_WITH'
    ) -> List[Dict]:
        """
        Detect communities using specified algorithm.

        Args:
            algorithm: 'louvain' or 'label_propagation'
            node_label: Node label to analyze
            relationship_type: Relationship to follow

        Returns:
            List of communities with members
        """
        logger.info(f"Detecting communities using {algorithm}...")

        if algorithm == 'louvain':
            return self._louvain(node_label, relationship_type)
        elif algorithm == 'label_propagation':
            return self._label_propagation(node_label, relationship_type)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _louvain(self, node_label: str, relationship_type: str) -> List[Dict]:
        """
        Louvain community detection (simplified).

        Note: Full implementation requires GDS library.
        This is a simplified version for testing.
        """
        with self.connection.get_session() as session:
            # Simplified: Use connected components as communities
            query = f"""
            MATCH (n:{node_label})
            SET n.community_id = id(n)
            """
            session.run(query)

            # Iterate to propagate communities
            for _ in range(10):
                update_query = f"""
                MATCH (n:{node_label})-[:{relationship_type}]-(neighbor:{node_label})
                WITH n, neighbor
                WHERE neighbor.community_id < n.community_id
                SET n.community_id = neighbor.community_id
                """
                session.run(update_query)

            # Get communities
            result_query = f"""
            MATCH (n:{node_label})
            WITH n.community_id as community_id, collect(n) as members
            RETURN community_id, members
            ORDER BY size(members) DESC
            """

            result = session.run(result_query)

            return [
                {
                    'community_id': record['community_id'],
                    'members': [
                        dict(member) for member in record['members']
                    ],
                    'size': len(record['members'])
                }
                for record in result
            ]

    def _label_propagation(
        self,
        node_label: str,
        relationship_type: str
    ) -> List[Dict]:
        """Label propagation algorithm."""
        with self.connection.get_session() as session:
            # Initialize with random labels
            init_query = f"""
            MATCH (n:{node_label})
            SET n.label = id(n)
            """
            session.run(init_query)

            # Propagate labels
            for _ in range(20):
                propagate_query = f"""
                MATCH (n:{node_label})-[:{relationship_type}]-(neighbor:{node_label})
                WITH n, neighbor.label as label, count(*) as weight
                ORDER BY weight DESC
                WITH n, collect(label)[0] as most_common_label
                SET n.label = most_common_label
                """
                session.run(propagate_query)

            # Get communities
            result_query = f"""
            MATCH (n:{node_label})
            WITH n.label as community_id, collect(n) as members
            RETURN community_id, members
            ORDER BY size(members) DESC
            """

            result = session.run(result_query)

            return [
                {
                    'community_id': record['community_id'],
                    'members': [dict(m) for m in record['members']],
                    'size': len(record['members'])
                }
                for record in result
            ]

    def get_author_community(self, author_id: str) -> Optional[Dict]:
        """
        Get community for specific author.

        Args:
            author_id: Author identifier

        Returns:
            Community information
        """
        with self.connection.get_session() as session:
            query = """
            MATCH (a:Author {author_id: $author_id})
            RETURN a.community_id as community_id
            """

            result = session.run(query, author_id=author_id)
            record = result.single()

            if record:
                return {
                    'author_id': author_id,
                    'community_id': record['community_id']
                }

            return None

    def get_community_stats(self) -> Dict:
        """
        Calculate community statistics.

        Returns:
            Statistics about communities
        """
        with self.connection.get_session() as session:
            query = """
            MATCH (n:Author)
            WITH n.community_id as community, count(n) as size
            RETURN count(DISTINCT community) as total_communities,
                   avg(size) as average_size,
                   max(size) as largest_community_size,
                   min(size) as smallest_community_size
            """

            result = session.run(query)
            record = result.single()

            return dict(record) if record else {}
