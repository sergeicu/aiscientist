import pytest
from unittest.mock import Mock, MagicMock
from src.network.network_builder import NetworkBuilder

@pytest.fixture
def mock_neo4j_driver(mocker):
    """Mock Neo4j driver."""
    driver = Mock()
    session = Mock()
    driver.session.return_value.__enter__ = Mock(return_value=session)
    driver.session.return_value.__exit__ = Mock(return_value=None)
    return driver

@pytest.fixture
def builder(mock_neo4j_driver):
    b = NetworkBuilder(driver=mock_neo4j_driver)
    # Reset mock after schema creation
    session = mock_neo4j_driver.session.return_value.__enter__.return_value
    session.run.reset_mock()
    return b

@pytest.fixture
def sample_authors():
    return [
        {
            'full_name': 'John Smith',
            'author_id': 'abc123',
            'last_name': 'Smith',
            'first_name': 'John',
            'initials': 'J',
            'position': 0,
            'is_first_author': True,
            'is_last_author': False,
            'pmid': '12345678',
            'year': '2023',
            'affiliations': [
                {'institution': "Boston Children's Hospital"}
            ]
        },
        {
            'full_name': 'Jane Doe',
            'author_id': 'def456',
            'last_name': 'Doe',
            'first_name': 'Jane',
            'initials': 'J',
            'position': 1,
            'is_first_author': False,
            'is_last_author': True,
            'pmid': '12345678',
            'year': '2023',
            'affiliations': [
                {'institution': 'Harvard Medical School'}
            ]
        }
    ]

def test_create_author_nodes(builder, mock_neo4j_driver, sample_authors):
    """Should create Author nodes in Neo4j."""
    session = mock_neo4j_driver.session.return_value.__enter__.return_value

    builder.create_author_nodes(sample_authors)

    # Should have called session.run for each author
    assert session.run.call_count == len(sample_authors)

def test_create_paper_node(builder, mock_neo4j_driver):
    """Should create Paper node."""
    session = mock_neo4j_driver.session.return_value.__enter__.return_value

    paper = {
        'pmid': '12345678',
        'title': 'Test Paper',
        'year': '2023'
    }

    builder.create_paper_node(paper)

    session.run.assert_called_once()
    call_args = session.run.call_args[0][0]
    assert 'CREATE' in call_args or 'MERGE' in call_args
    assert 'Paper' in call_args

def test_create_authorship_relationships(builder, mock_neo4j_driver, sample_authors):
    """Should create AUTHORED relationships."""
    session = mock_neo4j_driver.session.return_value.__enter__.return_value

    paper_pmid = '12345678'

    builder.create_authorship_relationships(sample_authors, paper_pmid)

    # Should create relationship for each author
    assert session.run.call_count == len(sample_authors)

def test_create_collaboration_relationships(builder, mock_neo4j_driver):
    """Should create COLLABORATED_WITH relationships."""
    session = mock_neo4j_driver.session.return_value.__enter__.return_value

    co_authorships = [
        {
            'author1_id': 'abc123',
            'author2_id': 'def456',
            'pmid': '12345678'
        }
    ]

    builder.create_collaboration_relationships(co_authorships)

    session.run.assert_called()

def test_build_network_from_articles(builder, mocker):
    """Should build complete network from articles."""
    articles = [
        {
            'pmid': '123',
            'title': 'Paper 1',
            'authors': [
                {'last_name': 'Smith', 'first_name': 'John', 'affiliations': []}
            ]
        }
    ]

    mock_create_paper = mocker.patch.object(builder, 'create_paper_node')
    mock_create_authors = mocker.patch.object(builder, 'create_author_nodes')
    mock_create_authorship = mocker.patch.object(builder, 'create_authorship_relationships')

    builder.build_network(articles)

    # Should have called all creation methods
    mock_create_paper.assert_called()
    mock_create_authors.assert_called()
    mock_create_authorship.assert_called()

def test_incremental_update(builder, mocker):
    """Should add new articles without rebuilding."""
    new_articles = [{'pmid': '999', 'title': 'New Paper', 'authors': []}]

    mock_create = mocker.patch.object(builder, 'create_paper_node')

    builder.incremental_update(new_articles)

    # Should process new articles
    assert mock_create.call_count == len(new_articles)
