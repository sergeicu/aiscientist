# src/analytics/centrality.py

from typing import List, Dict, Optional
from loguru import logger


class CentralityAnalyzer:
    """
    Calculate centrality metrics for graph nodes.

    Centrality measures identify important/influential nodes.

    Args:
        connection: Neo4jConnection instance

    Example:
        >>> analyzer = CentralityAnalyzer(connection)
        >>> top_authors = analyzer.degree_centrality(node_label='Author')
    """

    def __init__(self, connection):
        self.connection = connection

    def degree_centrality(
        self,
        node_label: str = 'Author',
        normalize: bool = False
    ) -> List[Dict]:
        """
        Calculate degree centrality (number of connections).

        Args:
            node_label: Node label to analyze
            normalize: Normalize scores to 0-1

        Returns:
            List of nodes with centrality scores
        """
        logger.info(f"Calculating degree centrality for {node_label}...")

        with self.connection.get_session() as session:
            query = f"""
            MATCH (n:{node_label})-[r:COLLABORATED_WITH]-(other)
            WITH n, count(DISTINCT other) as degree
            """

            if normalize:
                # Get total nodes for normalization
                count_query = f"MATCH (n:{node_label}) RETURN count(n) as total"
                total = session.run(count_query).single()['total']

                query += f"""
                RETURN n, degree, toFloat(degree) / {total - 1} as centrality
                ORDER BY centrality DESC
                """
            else:
                query += """
                RETURN n, degree
                ORDER BY degree DESC
                """

            result = session.run(query)

            return [
                {
                    **dict(record['n']),
                    'degree': record['degree'],
                    'centrality': record.get('centrality', record['degree'])
                }
                for record in result
            ]

    def betweenness_centrality(
        self,
        node_label: str = 'Author'
    ) -> List[Dict]:
        """
        Calculate betweenness centrality (bridge position).

        Identifies nodes that lie on many shortest paths.

        Args:
            node_label: Node label to analyze

        Returns:
            List of nodes with betweenness scores
        """
        logger.info(f"Calculating betweenness centrality for {node_label}...")

        # Note: This requires GDS library
        # Simplified version without GDS:
        with self.connection.get_session() as session:
            query = f"""
            MATCH (n:{node_label})
            WITH n
            MATCH (a:{node_label}), (b:{node_label})
            WHERE a <> b AND a <> n AND b <> n
            MATCH path = allShortestPaths((a)-[:COLLABORATED_WITH*]-(b))
            WHERE n IN nodes(path)
            WITH n, count(DISTINCT path) as paths_through
            RETURN n, paths_through as betweenness
            ORDER BY betweenness DESC
            """

            result = session.run(query)

            return [
                {
                    **dict(record['n']),
                    'betweenness': record['betweenness']
                }
                for record in result
            ]

    def pagerank(
        self,
        node_label: str = 'Author',
        relationship_type: str = 'COLLABORATED_WITH',
        max_iterations: int = 20,
        damping_factor: float = 0.85
    ) -> List[Dict]:
        """
        Calculate PageRank scores.

        Measures influence based on network connections.

        Args:
            node_label: Node label to analyze
            relationship_type: Relationship type to follow
            max_iterations: Maximum iterations
            damping_factor: Damping factor (0.85 standard)

        Returns:
            List of nodes with PageRank scores
        """
        logger.info(f"Calculating PageRank for {node_label}...")

        # Simplified PageRank without GDS
        with self.connection.get_session() as session:
            # Initialize scores
            init_query = f"""
            MATCH (n:{node_label})
            SET n.pagerank = 1.0
            RETURN count(n) as total
            """

            total = session.run(init_query).single()['total']

            # Iterate
            for iteration in range(max_iterations):
                update_query = f"""
                MATCH (n:{node_label})-[:{relationship_type}]-(neighbor:{node_label})
                WITH n, collect(neighbor) as neighbors
                WITH n, neighbors, size(neighbors) as degree
                UNWIND neighbors as neighbor
                MATCH (neighbor)-[:{relationship_type}]-(other)
                WITH n, neighbor, toFloat(count(DISTINCT other)) as neighbor_degree
                WITH n, sum(neighbor.pagerank / neighbor_degree) as rank_sum
                SET n.pagerank = {(1 - damping_factor) / total} + {damping_factor} * rank_sum
                """

                session.run(update_query)

            # Get results
            result_query = f"""
            MATCH (n:{node_label})
            RETURN n, n.pagerank as pagerank
            ORDER BY pagerank DESC
            """

            result = session.run(result_query)

            return [
                {
                    **dict(record['n']),
                    'pagerank': record['pagerank']
                }
                for record in result
            ]

    def find_top_influencers(
        self,
        limit: int = 20,
        metric: str = 'combined'
    ) -> List[Dict]:
        """
        Find most influential authors.

        Combines multiple centrality metrics.

        Args:
            limit: Number of results
            metric: 'degree', 'betweenness', 'pagerank', or 'combined'

        Returns:
            List of influential authors
        """
        logger.info(f"Finding top {limit} influencers by {metric}...")

        if metric == 'combined':
            # Calculate multiple metrics and combine
            degree_scores = self.degree_centrality(normalize=True)
            pagerank_scores = self.pagerank()

            # Create combined score
            degree_map = {
                d['author_id']: d['centrality']
                for d in degree_scores
            }

            pagerank_map = {
                p['author_id']: p['pagerank']
                for p in pagerank_scores
            }

            # Combine (simple average)
            combined = []
            for author_id in degree_map.keys():
                influence = (
                    degree_map.get(author_id, 0) +
                    pagerank_map.get(author_id, 0)
                ) / 2

                combined.append({
                    'author_id': author_id,
                    'influence_score': influence,
                    'degree_centrality': degree_map.get(author_id, 0),
                    'pagerank': pagerank_map.get(author_id, 0)
                })

            # Sort and limit
            combined.sort(key=lambda x: x['influence_score'], reverse=True)
            return combined[:limit]

        elif metric == 'degree':
            return self.degree_centrality()[:limit]

        elif metric == 'pagerank':
            return self.pagerank()[:limit]

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def compare_centrality_metrics(self, author_id: str) -> Dict:
        """
        Compare all centrality metrics for specific author.

        Args:
            author_id: Author identifier

        Returns:
            Dictionary with all centrality scores
        """
        degree = self.degree_centrality()
        pagerank = self.pagerank()

        degree_score = next(
            (d for d in degree if d['author_id'] == author_id),
            {}
        )

        pagerank_score = next(
            (p for p in pagerank if p['author_id'] == author_id),
            {}
        )

        return {
            'author_id': author_id,
            'degree_centrality': degree_score.get('centrality', 0),
            'pagerank': pagerank_score.get('pagerank', 0)
        }
