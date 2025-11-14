# tests/test_queries.py

import pytest
from unittest.mock import Mock, MagicMock
from src.analytics.queries import AdvancedQueries


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
def queries(mock_connection):
    """Create advanced queries."""
    return AdvancedQueries(mock_connection)


def test_find_collaboration_path(queries, mock_connection):
    """Should find path between two authors."""
    # Mock the query result
    mock_session = mock_connection.get_session.return_value.__enter__.return_value

    # Create mock path object
    mock_path = Mock()
    mock_path.nodes = [
        {'author_id': 'author_0', 'full_name': 'Author 0'},
        {'author_id': 'author_1', 'full_name': 'Author 1'},
        {'author_id': 'author_2', 'full_name': 'Author 2'}
    ]

    mock_result = Mock()
    mock_result.single.return_value = {
        'path': mock_path,
        'length': 2
    }
    mock_session.run.return_value = mock_result

    path = queries.find_collaboration_path('author_0', 'author_2')

    assert path is not None
    assert 'length' in path
    assert 'nodes' in path
    assert path['length'] == 2


def test_find_collaboration_path_no_path(queries, mock_connection):
    """Should return None if no path exists."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_result.single.return_value = None
    mock_session.run.return_value = mock_result

    path = queries.find_collaboration_path('author_0', 'author_99')

    assert path is None


def test_find_common_collaborators(queries, mock_connection):
    """Should find common collaborators between two authors."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_result.__iter__ = Mock(return_value=iter([
        {'common': {'author_id': 'author_0', 'full_name': 'Author 0'}},
        {'common': {'author_id': 'author_3', 'full_name': 'Author 3'}},
    ]))
    mock_session.run.return_value = mock_result

    common = queries.find_common_collaborators('author_1', 'author_2')

    # author_0 collaborates with both
    assert len(common) >= 1


def test_find_prolific_authors(queries, mock_connection):
    """Should find authors with most publications."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_result.__iter__ = Mock(return_value=iter([
        {'a': {'author_id': 'author_0', 'full_name': 'Author 0'}, 'paper_count': 10},
        {'a': {'author_id': 'author_1', 'full_name': 'Author 1'}, 'paper_count': 8},
        {'a': {'author_id': 'author_2', 'full_name': 'Author 2'}, 'paper_count': 5},
    ]))
    mock_session.run.return_value = mock_result

    prolific = queries.find_prolific_authors(limit=10)

    assert len(prolific) > 0

    # Should be sorted by paper count
    if len(prolific) > 1:
        assert prolific[0]['paper_count'] >= prolific[1]['paper_count']


def test_find_research_trends(queries, mock_connection):
    """Should find research trends over time."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_result.__iter__ = Mock(return_value=iter([
        {'year': '2020', 'publications': 100},
        {'year': '2021', 'publications': 120},
        {'year': '2022', 'publications': 150},
        {'year': '2023', 'publications': 180},
        {'year': '2024', 'publications': 200},
    ]))
    mock_session.run.return_value = mock_result

    trends = queries.find_research_trends(
        start_year='2020',
        end_year='2024'
    )

    assert isinstance(trends, list)
    assert len(trends) == 5


def test_recommend_collaborators(queries, mock_connection):
    """Should recommend potential collaborators."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_result.__iter__ = Mock(return_value=iter([
        {'candidate': {'author_id': 'author_5', 'full_name': 'Author 5'}, 'score': 3},
        {'candidate': {'author_id': 'author_6', 'full_name': 'Author 6'}, 'score': 2},
        {'candidate': {'author_id': 'author_7', 'full_name': 'Author 7'}, 'score': 1},
    ]))
    mock_session.run.return_value = mock_result

    recommendations = queries.recommend_collaborators(
        author_id='author_0',
        limit=5
    )

    assert len(recommendations) <= 5

    # Should have recommendation scores
    for rec in recommendations:
        assert 'author_id' in rec
        assert 'score' in rec


def test_find_institutional_leaders(queries, mock_connection):
    """Should find leading researchers at institution."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_result.__iter__ = Mock(return_value=iter([
        {'a': {'author_id': 'author_0', 'full_name': 'Author 0'}, 'paper_count': 25},
        {'a': {'author_id': 'author_1', 'full_name': 'Author 1'}, 'paper_count': 20},
    ]))
    mock_session.run.return_value = mock_result

    leaders = queries.find_institutional_leaders(
        institution="Boston Children's Hospital",
        limit=10
    )

    assert len(leaders) > 0
    assert leaders[0]['paper_count'] >= leaders[1]['paper_count']


def test_prolific_authors_with_min_year(queries, mock_connection):
    """Should filter prolific authors by minimum year."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_result.__iter__ = Mock(return_value=iter([
        {'a': {'author_id': 'author_0', 'full_name': 'Author 0'}, 'paper_count': 5},
    ]))
    mock_session.run.return_value = mock_result

    prolific = queries.find_prolific_authors(limit=10, min_year='2020')

    assert len(prolific) > 0


def test_recommend_collaborators_sorted_by_score(queries, mock_connection):
    """Should sort recommendations by score."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_result.__iter__ = Mock(return_value=iter([
        {'candidate': {'author_id': 'author_5', 'full_name': 'Author 5'}, 'score': 10},
        {'candidate': {'author_id': 'author_6', 'full_name': 'Author 6'}, 'score': 8},
        {'candidate': {'author_id': 'author_7', 'full_name': 'Author 7'}, 'score': 5},
    ]))
    mock_session.run.return_value = mock_result

    recommendations = queries.recommend_collaborators('author_0', limit=3)

    # Should be sorted by score (descending)
    scores = [rec['score'] for rec in recommendations]
    assert scores == sorted(scores, reverse=True)


def test_empty_results(queries, mock_connection):
    """Should handle empty query results."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_result.__iter__ = Mock(return_value=iter([]))
    mock_session.run.return_value = mock_result

    common = queries.find_common_collaborators('author_1', 'author_2')
    assert len(common) == 0
