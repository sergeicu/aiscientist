# tests/test_community.py

import pytest
from unittest.mock import Mock, MagicMock
from src.analytics.community import CommunityDetector


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
def detector(mock_connection):
    """Create community detector."""
    return CommunityDetector(mock_connection)


def test_detect_communities_louvain(detector, mock_connection):
    """Should detect communities using Louvain."""
    # Mock the queries
    mock_session = mock_connection.get_session.return_value.__enter__.return_value

    # Mock init query
    mock_init = Mock()

    # Mock update queries (10 iterations)
    mock_update = Mock()

    # Mock result query
    mock_result = Mock()
    mock_result.__iter__ = Mock(return_value=iter([
        {
            'community_id': 1,
            'members': [
                {'author_id': 'author_0', 'full_name': 'Author 0', 'community_id': 1},
                {'author_id': 'author_1', 'full_name': 'Author 1', 'community_id': 1},
                {'author_id': 'author_2', 'full_name': 'Author 2', 'community_id': 1},
            ]
        },
        {
            'community_id': 2,
            'members': [
                {'author_id': 'author_3', 'full_name': 'Author 3', 'community_id': 2},
                {'author_id': 'author_4', 'full_name': 'Author 4', 'community_id': 2},
            ]
        }
    ]))

    mock_session.run.side_effect = [mock_init] + [mock_update] * 10 + [mock_result]

    communities = detector.detect_communities(algorithm='louvain')

    # Should have at least 1 community
    assert len(communities) >= 1

    # Each community should have members
    for community in communities:
        assert 'community_id' in community
        assert 'members' in community
        assert len(community['members']) > 0


def test_label_propagation(detector, mock_connection):
    """Should detect communities using label propagation."""
    # Mock the queries
    mock_session = mock_connection.get_session.return_value.__enter__.return_value

    # Mock init query
    mock_init = Mock()

    # Mock propagation queries (20 iterations)
    mock_propagate = Mock()

    # Mock result query
    mock_result = Mock()
    mock_result.__iter__ = Mock(return_value=iter([
        {
            'community_id': 100,
            'members': [
                {'author_id': 'author_0', 'full_name': 'Author 0', 'label': 100},
                {'author_id': 'author_1', 'full_name': 'Author 1', 'label': 100},
            ]
        }
    ]))

    mock_session.run.side_effect = [mock_init] + [mock_propagate] * 20 + [mock_result]

    communities = detector.detect_communities(algorithm='label_propagation')

    assert len(communities) >= 1


def test_find_author_community(detector, mock_connection):
    """Should find which community an author belongs to."""
    # Mock the query
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_result.single.return_value = {'community_id': 1}
    mock_session.run.return_value = mock_result

    # Find community for specific author
    community = detector.get_author_community('author_0')

    assert community is not None
    assert 'community_id' in community
    assert community['community_id'] == 1


def test_author_not_found(detector, mock_connection):
    """Should return None for non-existent author."""
    # Mock the query
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_result.single.return_value = None
    mock_session.run.return_value = mock_result

    community = detector.get_author_community('nonexistent_author')

    assert community is None


def test_community_statistics(detector, mock_connection):
    """Should calculate community statistics."""
    # Mock the query
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_result.single.return_value = {
        'total_communities': 3,
        'average_size': 4.5,
        'largest_community_size': 10,
        'smallest_community_size': 2
    }
    mock_session.run.return_value = mock_result

    stats = detector.get_community_stats()

    assert 'total_communities' in stats
    assert 'average_size' in stats
    assert 'largest_community_size' in stats
    assert stats['total_communities'] == 3


def test_unknown_algorithm_raises_error(detector):
    """Should raise error for unknown algorithm."""
    with pytest.raises(ValueError, match="Unknown algorithm"):
        detector.detect_communities(algorithm='unknown_algo')


def test_community_has_size_field(detector, mock_connection):
    """Should include size field in community results."""
    # Mock the queries
    mock_session = mock_connection.get_session.return_value.__enter__.return_value

    mock_init = Mock()
    mock_update = Mock()

    mock_result = Mock()
    mock_result.__iter__ = Mock(return_value=iter([
        {
            'community_id': 1,
            'members': [
                {'author_id': 'author_0', 'full_name': 'Author 0', 'community_id': 1},
                {'author_id': 'author_1', 'full_name': 'Author 1', 'community_id': 1},
            ]
        }
    ]))

    mock_session.run.side_effect = [mock_init] + [mock_update] * 10 + [mock_result]

    communities = detector.detect_communities(algorithm='louvain')

    for community in communities:
        assert 'size' in community
        assert community['size'] == len(community['members'])
