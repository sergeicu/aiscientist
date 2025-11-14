# Task 5: Graph Analytics & Advanced Queries - TDD Implementation

## Executive Summary

Implement production-ready graph analytics and advanced query capabilities for the Neo4j knowledge graph. This includes network metrics (centrality, PageRank), community detection, path finding, and complex Cypher queries for research network analysis.

**Key Requirements:**
- Calculate network metrics (degree centrality, betweenness, PageRank)
- Identify research communities using Louvain algorithm
- Find collaboration paths between authors
- Detect key opinion leaders and research hubs
- Implement similarity algorithms for author/paper recommendations
- Create advanced Cypher query templates
- Follow Test-Driven Development (TDD) principles

**Dependencies**: Requires Task 4 (Neo4j Graph Setup) to be completed first.

## Background & Context

### Graph Analytics Use Cases

**For Research Intelligence:**
- **Author Discovery**: Find experts in specific research areas
- **Collaboration Opportunities**: Identify potential collaborators
- **Research Trends**: Detect emerging communities
- **Impact Analysis**: Measure author/institution influence
- **Knowledge Flow**: Track how ideas spread through networks

### Neo4j Graph Data Science (GDS)

Neo4j GDS library provides production-grade algorithms:
- **Centrality**: Degree, Betweenness, PageRank, Eigenvector
- **Community Detection**: Louvain, Label Propagation, Weakly Connected Components
- **Similarity**: Node Similarity, Jaccard, Cosine
- **Path Finding**: Shortest Path, All Paths, Dijkstra
- **Link Prediction**: Adamic Adar, Common Neighbors

### Why This is a Separate Task

This task focuses on **analytics and queries**, making it independent from:
- Task 4 (Graph Setup) - requires it as foundation
- Task 6-7 (Visualization) - provides data to these
- Data collection tasks - can use any graph data

## Technical Architecture

### Module Structure

```
src/analytics/
├── __init__.py
├── centrality.py         # Centrality metrics
├── community.py          # Community detection
├── pathfinding.py        # Path algorithms
├── similarity.py         # Similarity algorithms
├── queries.py            # Advanced Cypher queries
└── metrics.py            # Network metrics

tests/
├── __init__.py
├── test_centrality.py
├── test_community.py
├── test_pathfinding.py
├── test_similarity.py
├── test_queries.py
└── fixtures/
    └── test_graph_data.py
```

### Dependencies

```python
# Required packages
neo4j>=5.14            # Neo4j driver
networkx>=3.2          # For validation/comparison
pydantic>=2.0          # Data validation
loguru>=0.7            # Logging
pytest>=7.4            # Testing
pytest-mock>=3.12      # Mocking
numpy>=1.24            # Numerical operations
pandas>=2.0            # Data analysis
```

## TDD Implementation Plan

### Phase 1: Centrality Metrics (TDD)

#### Test Cases

```python
# tests/test_centrality.py

import pytest
from src.analytics.centrality import CentralityAnalyzer
from src.graph.connection import Neo4jConnection
from src.graph.operations import GraphOperations
from src.graph.models import Author, Paper

@pytest.fixture
def test_graph(neo4j_connection):
    """Create test graph with known structure."""
    ops = GraphOperations(neo4j_connection)

    # Create authors
    authors = [
        Author(author_id=f"author_{i}", full_name=f"Author {i}")
        for i in range(5)
    ]

    for author in authors:
        ops.create_author(author)

    # Create collaboration network (star topology)
    # Author 0 collaborates with all others
    for i in range(1, 5):
        ops.create_collaboration("author_0", f"author_{i}", pmid="12345")

    # Author 1 and 2 also collaborate
    ops.create_collaboration("author_1", "author_2", pmid="67890")

    return neo4j_connection

@pytest.fixture
def analyzer(test_graph):
    return CentralityAnalyzer(test_graph)

def test_calculate_degree_centrality(analyzer):
    """Should calculate degree centrality for all authors."""
    results = analyzer.degree_centrality(node_label='Author')

    # Author 0 should have highest degree (connected to 4 others)
    top_author = results[0]
    assert top_author['author_id'] == 'author_0'
    assert top_author['degree'] == 4

def test_degree_centrality_normalized(analyzer):
    """Should normalize centrality scores."""
    results = analyzer.degree_centrality(
        node_label='Author',
        normalize=True
    )

    # Normalized scores should be 0-1
    for result in results:
        assert 0 <= result['centrality'] <= 1

def test_calculate_betweenness_centrality(analyzer):
    """Should calculate betweenness centrality."""
    results = analyzer.betweenness_centrality(node_label='Author')

    # Author 0 should have high betweenness (bridges network)
    top_author = next(r for r in results if r['author_id'] == 'author_0')
    assert top_author['betweenness'] > 0

def test_calculate_pagerank(analyzer):
    """Should calculate PageRank scores."""
    results = analyzer.pagerank(
        node_label='Author',
        relationship_type='COLLABORATED_WITH'
    )

    assert len(results) == 5

    # All should have PageRank scores
    for result in results:
        assert 'pagerank' in result
        assert result['pagerank'] > 0

def test_identify_top_influencers(analyzer):
    """Should identify most influential authors."""
    influencers = analyzer.find_top_influencers(limit=3)

    assert len(influencers) <= 3

    # Should be sorted by influence score
    scores = [i['influence_score'] for i in influencers]
    assert scores == sorted(scores, reverse=True)

def test_centrality_comparison(analyzer):
    """Should compare multiple centrality metrics."""
    comparison = analyzer.compare_centrality_metrics('author_0')

    assert 'degree_centrality' in comparison
    assert 'betweenness_centrality' in comparison
    assert 'pagerank' in comparison
```

#### Implementation

```python
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
```

### Phase 2: Community Detection (TDD)

#### Test Cases

```python
# tests/test_community.py

import pytest
from src.analytics.community import CommunityDetector

@pytest.fixture
def detector(test_graph):
    return CommunityDetector(test_graph)

def test_detect_communities_louvain(detector):
    """Should detect communities using Louvain."""
    communities = detector.detect_communities(algorithm='louvain')

    # Should have at least 1 community
    assert len(communities) >= 1

    # Each community should have members
    for community in communities:
        assert 'community_id' in community
        assert 'members' in community
        assert len(community['members']) > 0

def test_label_propagation(detector):
    """Should detect communities using label propagation."""
    communities = detector.detect_communities(algorithm='label_propagation')

    assert len(communities) >= 1

def test_find_author_community(detector):
    """Should find which community an author belongs to."""
    # First detect communities
    detector.detect_communities(algorithm='louvain')

    # Find community for specific author
    community = detector.get_author_community('author_0')

    assert community is not None
    assert 'community_id' in community

def test_community_statistics(detector):
    """Should calculate community statistics."""
    communities = detector.detect_communities(algorithm='louvain')

    stats = detector.get_community_stats()

    assert 'total_communities' in stats
    assert 'average_size' in stats
    assert 'largest_community_size' in stats
```

#### Implementation

```python
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
```

### Phase 3: Advanced Queries (TDD)

#### Test Cases

```python
# tests/test_queries.py

import pytest
from src.analytics.queries import AdvancedQueries

@pytest.fixture
def queries(test_graph):
    return AdvancedQueries(test_graph)

def test_find_collaboration_path(queries):
    """Should find path between two authors."""
    path = queries.find_collaboration_path('author_0', 'author_2')

    assert path is not None
    assert 'length' in path
    assert 'nodes' in path

def test_find_common_collaborators(queries):
    """Should find common collaborators between two authors."""
    common = queries.find_common_collaborators('author_1', 'author_2')

    # author_0 collaborates with both
    assert len(common) >= 1

def test_find_prolific_authors(queries):
    """Should find authors with most publications."""
    prolific = queries.find_prolific_authors(limit=10)

    assert len(prolific) > 0

    # Should be sorted by paper count
    if len(prolific) > 1:
        assert prolific[0]['paper_count'] >= prolific[1]['paper_count']

def test_find_research_trends(queries):
    """Should find research trends over time."""
    trends = queries.find_research_trends(
        start_year='2020',
        end_year='2024'
    )

    assert isinstance(trends, list)

def test_recommend_collaborators(queries):
    """Should recommend potential collaborators."""
    recommendations = queries.recommend_collaborators(
        author_id='author_0',
        limit=5
    )

    assert len(recommendations) <= 5

    # Should have recommendation scores
    for rec in recommendations:
        assert 'author_id' in rec
        assert 'score' in rec
```

#### Implementation

```python
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
```

## Running Tests

```bash
# Start Neo4j
docker run -d \
  --name neo4j-analytics \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Install dependencies
pip install neo4j networkx pydantic loguru pytest numpy pandas

# Run all tests
pytest tests/ -v --cov=src/analytics

# Run specific tests
pytest tests/test_centrality.py -v
```

## Usage Examples

### Centrality Analysis

```python
from src.analytics.centrality import CentralityAnalyzer

# Find influential authors
analyzer = CentralityAnalyzer(connection)

# Top influencers
influencers = analyzer.find_top_influencers(limit=20)

for author in influencers:
    print(f"{author['full_name']}: {author['influence_score']:.3f}")
```

### Community Detection

```python
from src.analytics.community import CommunityDetector

detector = CommunityDetector(connection)

# Detect communities
communities = detector.detect_communities(algorithm='louvain')

print(f"Found {len(communities)} communities")

for community in communities[:5]:
    print(f"Community {community['community_id']}: {community['size']} members")
```

### Advanced Queries

```python
from src.analytics.queries import AdvancedQueries

queries = AdvancedQueries(connection)

# Find collaboration path
path = queries.find_collaboration_path('author_123', 'author_456')
print(f"Collaboration distance: {path['length']}")

# Recommend collaborators
recommendations = queries.recommend_collaborators('author_123', limit=10)

for rec in recommendations:
    print(f"{rec['full_name']}: score {rec['score']}")
```

## Success Criteria

✅ **Must Have:**
1. All unit tests passing (>90% coverage)
2. Degree centrality calculation
3. PageRank implementation
4. Community detection (at least one algorithm)
5. Path finding queries
6. Collaboration recommendations

✅ **Should Have:**
7. Betweenness centrality
8. Multiple community detection algorithms
9. Research trend analysis
10. Performance benchmarks

✅ **Nice to Have:**
11. Neo4j GDS library integration
12. Advanced similarity algorithms
13. Temporal analysis

## Deliverables

1. **Source code** (>90% test coverage)
2. **Tests** (unit + integration)
3. **Query templates** (Cypher)
4. **Documentation** and examples
5. **Performance benchmarks**

## Environment Setup

```bash
# Start Neo4j
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password neo4j:latest

# Install dependencies
pip install neo4j networkx pydantic loguru pytest numpy pandas

# Run tests
pytest tests/ -v --cov=src/analytics
```

## Cypher Query Reference

```cypher
// Degree centrality
MATCH (a:Author)-[:COLLABORATED_WITH]-(other)
RETURN a.full_name, count(other) as degree
ORDER BY degree DESC

// Find communities (connected components)
CALL gds.wcc.stream('author-network')
YIELD nodeId, componentId
RETURN gds.util.asNode(nodeId).full_name, componentId
ORDER BY componentId

// Shortest path
MATCH path = shortestPath(
  (a1:Author {author_id: 'id1'})-[:COLLABORATED_WITH*]-(a2:Author {author_id: 'id2'})
)
RETURN path, length(path)
```

---

**Task completion**: When all tests pass, centrality metrics calculate correctly, communities are detected, and advanced queries return meaningful results.
