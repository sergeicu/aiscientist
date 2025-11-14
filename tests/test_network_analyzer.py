import pytest
from unittest.mock import Mock
from src.network.network_analyzer import NetworkAnalyzer

@pytest.fixture
def mock_driver(mocker):
    driver = Mock()
    session = Mock()
    driver.session.return_value.__enter__ = Mock(return_value=session)
    driver.session.return_value.__exit__ = Mock(return_value=None)
    return driver

@pytest.fixture
def analyzer(mock_driver):
    return NetworkAnalyzer(driver=mock_driver)

def test_find_top_collaborators(analyzer, mock_driver):
    """Should find authors with most collaborations."""
    session = mock_driver.session.return_value.__enter__.return_value

    # Mock result - each record should be dict-like
    mock_record1 = {'author': 'John Smith', 'author_id': 'abc123', 'collaborations': 50}
    mock_record2 = {'author': 'Jane Doe', 'author_id': 'def456', 'collaborations': 45}
    session.run.return_value = [mock_record1, mock_record2]

    top_collab = analyzer.find_top_collaborators(limit=10)

    session.run.assert_called_once()
    assert len(top_collab) == 2

def test_find_institutional_collaborations(analyzer, mock_driver):
    """Should find top institution pairs."""
    session = mock_driver.session.return_value.__enter__.return_value

    mock_result = [
        {
            'institution1': 'BCH',
            'institution2': 'HMS',
            'collaboration_count': 100
        }
    ]
    session.run.return_value = mock_result

    collabs = analyzer.find_institutional_collaborations(limit=10)

    assert len(collabs) == 1

def test_calculate_author_centrality(analyzer, mock_driver):
    """Should calculate degree centrality for authors."""
    session = mock_driver.session.return_value.__enter__.return_value

    mock_result = [
        {'author': 'John Smith', 'author_id': 'abc123', 'degree': 50}
    ]
    session.run.return_value = mock_result

    centrality = analyzer.calculate_author_centrality()

    session.run.assert_called()
