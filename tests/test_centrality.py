# tests/test_centrality.py

import pytest
from unittest.mock import Mock, MagicMock
from src.analytics.centrality import CentralityAnalyzer
from src.graph.operations import GraphOperations
from src.graph.models import Author


@pytest.fixture
def mock_connection():
    """Create mock connection."""
    conn = Mock()
    mock_session = Mock()
    mock_session.__enter__ = Mock(return_value=mock_session)
    mock_session.__exit__ = Mock(return_value=False)
    mock_session.run = Mock()
    conn.get_session = Mock(return_value=mock_session)
    return conn


@pytest.fixture
def analyzer(mock_connection):
    """Create centrality analyzer."""
    return CentralityAnalyzer(mock_connection)


def test_calculate_degree_centrality(analyzer, mock_connection):
    """Should calculate degree centrality for all authors."""
    # Mock the result
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_result.__iter__ = Mock(return_value=iter([
        {'n': {'author_id': 'author_0', 'full_name': 'Author 0'}, 'degree': 4, 'centrality': 4},
        {'n': {'author_id': 'author_1', 'full_name': 'Author 1'}, 'degree': 2, 'centrality': 2},
        {'n': {'author_id': 'author_2', 'full_name': 'Author 2'}, 'degree': 2, 'centrality': 2},
    ]))
    mock_session.run.return_value = mock_result

    results = analyzer.degree_centrality(node_label='Author')

    # Author 0 should have highest degree (connected to 4 others)
    top_author = results[0]
    assert top_author['author_id'] == 'author_0'
    assert top_author['degree'] == 4


def test_degree_centrality_normalized(analyzer, mock_connection):
    """Should normalize centrality scores."""
    # Mock count query for normalization
    mock_session = mock_connection.get_session.return_value.__enter__.return_value

    # Mock total count
    mock_count_result = Mock()
    mock_count_result.single.return_value = {'total': 5}

    # Mock main query result
    mock_result = Mock()
    mock_result.__iter__ = Mock(return_value=iter([
        {'n': {'author_id': 'author_0', 'full_name': 'Author 0'}, 'degree': 4, 'centrality': 1.0},
        {'n': {'author_id': 'author_1', 'full_name': 'Author 1'}, 'degree': 2, 'centrality': 0.5},
        {'n': {'author_id': 'author_2', 'full_name': 'Author 2'}, 'degree': 2, 'centrality': 0.5},
    ]))

    mock_session.run.side_effect = [mock_count_result, mock_result]

    results = analyzer.degree_centrality(
        node_label='Author',
        normalize=True
    )

    # Normalized scores should be 0-1
    for result in results:
        assert 0 <= result['centrality'] <= 1


def test_calculate_betweenness_centrality(analyzer, mock_connection):
    """Should calculate betweenness centrality."""
    # Mock the result
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_result.__iter__ = Mock(return_value=iter([
        {'n': {'author_id': 'author_0', 'full_name': 'Author 0'}, 'betweenness': 6},
        {'n': {'author_id': 'author_1', 'full_name': 'Author 1'}, 'betweenness': 1},
        {'n': {'author_id': 'author_2', 'full_name': 'Author 2'}, 'betweenness': 1},
    ]))
    mock_session.run.return_value = mock_result

    results = analyzer.betweenness_centrality(node_label='Author')

    # Author 0 should have high betweenness (bridges network)
    top_author = next(r for r in results if r['author_id'] == 'author_0')
    assert top_author['betweenness'] > 0


def test_calculate_pagerank(analyzer, mock_connection):
    """Should calculate PageRank scores."""
    # Mock the initialization and iteration
    mock_session = mock_connection.get_session.return_value.__enter__.return_value

    # Mock init query
    mock_init_result = Mock()
    mock_init_result.single.return_value = {'total': 5}

    # Mock iteration queries (will run 20 times)
    mock_update = Mock()

    # Mock final result query
    mock_final_result = Mock()
    mock_final_result.__iter__ = Mock(return_value=iter([
        {'n': {'author_id': 'author_0', 'full_name': 'Author 0', 'pagerank': 0.35}, 'pagerank': 0.35},
        {'n': {'author_id': 'author_1', 'full_name': 'Author 1', 'pagerank': 0.20}, 'pagerank': 0.20},
        {'n': {'author_id': 'author_2', 'full_name': 'Author 2', 'pagerank': 0.20}, 'pagerank': 0.20},
        {'n': {'author_id': 'author_3', 'full_name': 'Author 3', 'pagerank': 0.15}, 'pagerank': 0.15},
        {'n': {'author_id': 'author_4', 'full_name': 'Author 4', 'pagerank': 0.10}, 'pagerank': 0.10},
    ]))

    # Set up side effects for sequential calls (1 init + 20 updates + 1 final)
    mock_session.run.side_effect = [mock_init_result] + [mock_update] * 20 + [mock_final_result]

    results = analyzer.pagerank(
        node_label='Author',
        relationship_type='COLLABORATED_WITH'
    )

    assert len(results) == 5

    # All should have PageRank scores
    for result in results:
        assert 'pagerank' in result
        assert result['pagerank'] > 0


def test_identify_top_influencers(analyzer, mock_connection):
    """Should identify most influential authors."""
    # Mock degree_centrality call
    mock_session = mock_connection.get_session.return_value.__enter__.return_value

    # First call: count for degree centrality
    mock_count_result = Mock()
    mock_count_result.single.return_value = {'total': 5}

    # Second call: degree centrality results
    mock_degree_result = Mock()
    mock_degree_result.__iter__ = Mock(return_value=iter([
        {'n': {'author_id': 'author_0', 'full_name': 'Author 0'}, 'degree': 4, 'centrality': 1.0},
        {'n': {'author_id': 'author_1', 'full_name': 'Author 1'}, 'degree': 2, 'centrality': 0.5},
        {'n': {'author_id': 'author_2', 'full_name': 'Author 2'}, 'degree': 2, 'centrality': 0.5},
    ]))

    # Calls for pagerank (init + 20 updates + result)
    mock_pr_init = Mock()
    mock_pr_init.single.return_value = {'total': 3}

    mock_pr_update = Mock()

    mock_pr_result = Mock()
    mock_pr_result.__iter__ = Mock(return_value=iter([
        {'n': {'author_id': 'author_0', 'full_name': 'Author 0', 'pagerank': 0.5}, 'pagerank': 0.5},
        {'n': {'author_id': 'author_1', 'full_name': 'Author 1', 'pagerank': 0.3}, 'pagerank': 0.3},
        {'n': {'author_id': 'author_2', 'full_name': 'Author 2', 'pagerank': 0.2}, 'pagerank': 0.2},
    ]))

    mock_session.run.side_effect = [
        mock_count_result, mock_degree_result,  # degree_centrality calls
        mock_pr_init] + [mock_pr_update] * 20 + [mock_pr_result  # pagerank calls
    ]

    influencers = analyzer.find_top_influencers(limit=3)

    assert len(influencers) <= 3

    # Should be sorted by influence score
    scores = [i['influence_score'] for i in influencers]
    assert scores == sorted(scores, reverse=True)


def test_centrality_comparison(analyzer, mock_connection):
    """Should compare multiple centrality metrics."""
    # Mock degree_centrality call
    mock_session = mock_connection.get_session.return_value.__enter__.return_value

    # Degree centrality result
    mock_degree_result = Mock()
    mock_degree_result.__iter__ = Mock(return_value=iter([
        {'n': {'author_id': 'author_0', 'full_name': 'Author 0'}, 'degree': 4, 'centrality': 4},
        {'n': {'author_id': 'author_1', 'full_name': 'Author 1'}, 'degree': 2, 'centrality': 2},
    ]))

    # PageRank calls
    mock_pr_init = Mock()
    mock_pr_init.single.return_value = {'total': 2}

    mock_pr_update = Mock()

    mock_pr_result = Mock()
    mock_pr_result.__iter__ = Mock(return_value=iter([
        {'n': {'author_id': 'author_0', 'full_name': 'Author 0', 'pagerank': 0.6}, 'pagerank': 0.6},
        {'n': {'author_id': 'author_1', 'full_name': 'Author 1', 'pagerank': 0.4}, 'pagerank': 0.4},
    ]))

    mock_session.run.side_effect = [
        mock_degree_result,  # degree_centrality
        mock_pr_init] + [mock_pr_update] * 20 + [mock_pr_result  # pagerank
    ]

    comparison = analyzer.compare_centrality_metrics('author_0')

    assert 'degree_centrality' in comparison
    assert 'pagerank' in comparison
    assert comparison['author_id'] == 'author_0'


def test_find_top_influencers_by_metric(analyzer, mock_connection):
    """Should find top influencers by specific metric."""
    # Mock degree centrality
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_result.__iter__ = Mock(return_value=iter([
        {'n': {'author_id': 'author_0', 'full_name': 'Author 0'}, 'degree': 4, 'centrality': 4},
        {'n': {'author_id': 'author_1', 'full_name': 'Author 1'}, 'degree': 2, 'centrality': 2},
    ]))
    mock_session.run.return_value = mock_result

    influencers = analyzer.find_top_influencers(limit=2, metric='degree')

    assert len(influencers) <= 2
