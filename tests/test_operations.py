# tests/test_operations.py

import pytest
from unittest.mock import Mock, MagicMock
from src.graph.operations import GraphOperations
from src.graph.models import Author, Paper, Institution


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
def ops(mock_connection):
    """Create operations manager."""
    return GraphOperations(mock_connection)


def test_create_author_node(ops, mock_connection):
    """Should create Author node."""
    author = Author(
        author_id="test_001",
        full_name="John Smith",
        last_name="Smith",
        first_name="John"
    )

    # Mock the result
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_record = {'a': {'author_id': 'test_001', 'full_name': 'John Smith', 'last_name': 'Smith', 'first_name': 'John'}}
    mock_result.single.return_value = mock_record
    mock_session.run.return_value = mock_result

    result = ops.create_author(author)

    assert result['author_id'] == "test_001"
    mock_session.run.assert_called_once()


def test_create_duplicate_author_ignores(ops, mock_connection):
    """Should handle duplicate author creation."""
    author = Author(
        author_id="test_001",
        full_name="John Smith",
        last_name="Smith",
        first_name="John"
    )

    # Mock the result - MERGE will return the existing node
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_record = {'a': {'author_id': 'test_001', 'full_name': 'John Smith'}}
    mock_result.single.return_value = mock_record
    mock_session.run.return_value = mock_result

    # Create twice
    result1 = ops.create_author(author)
    result2 = ops.create_author(author)

    # Should not error, should return existing
    assert result2['author_id'] == "test_001"


def test_get_author_by_id(ops, mock_connection):
    """Should retrieve author by ID."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_record = {'a': {'author_id': 'test_002', 'full_name': 'Jane Doe', 'last_name': 'Doe', 'first_name': 'Jane'}}
    mock_result.single.return_value = mock_record
    mock_session.run.return_value = mock_result

    retrieved = ops.get_author("test_002")

    assert retrieved is not None
    assert retrieved['full_name'] == "Jane Doe"


def test_get_nonexistent_author(ops, mock_connection):
    """Should return None for nonexistent author."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_result.single.return_value = None
    mock_session.run.return_value = mock_result

    result = ops.get_author("nonexistent_id")

    assert result is None


def test_update_author(ops, mock_connection):
    """Should update author properties."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_record = {'a': {'author_id': 'test_003', 'full_name': 'Alice Brown', 'h_index': 25}}
    mock_result.single.return_value = mock_record
    mock_session.run.return_value = mock_result

    # Update h_index
    updated = ops.update_author("test_003", {'h_index': 25})
    assert updated['h_index'] == 25


def test_delete_author(ops, mock_connection):
    """Should delete author node."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value

    # Delete
    result = ops.delete_author("test_004")

    # Should return True
    assert result is True
    mock_session.run.assert_called_once()


def test_create_paper_node(ops, mock_connection):
    """Should create Paper node."""
    paper = Paper(
        pmid="12345678",
        title="Test Paper",
        year="2023"
    )

    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_record = {'p': {'pmid': '12345678', 'title': 'Test Paper', 'year': '2023'}}
    mock_result.single.return_value = mock_record
    mock_session.run.return_value = mock_result

    result = ops.create_paper(paper)

    assert result['pmid'] == "12345678"


def test_create_institution_node(ops, mock_connection):
    """Should create Institution node."""
    institution = Institution(
        name="Test University",
        city="Boston",
        country="USA"
    )

    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_record = {'i': {'name': 'Test University', 'city': 'Boston', 'country': 'USA'}}
    mock_result.single.return_value = mock_record
    mock_session.run.return_value = mock_result

    result = ops.create_institution(institution)

    assert result['name'] == "Test University"


def test_create_authored_relationship(ops, mock_connection):
    """Should create AUTHORED relationship."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value

    # Create relationship
    ops.create_authored_relationship(
        author_id="test_005",
        pmid="99999",
        position=0,
        is_first_author=True
    )

    # Verify session.run was called
    mock_session.run.assert_called_once()


def test_create_collaboration_relationship(ops, mock_connection):
    """Should create COLLABORATED_WITH relationship."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value

    # Create collaboration
    ops.create_collaboration(
        author1_id="test_006",
        author2_id="test_007",
        pmid="88888"
    )

    # Verify
    mock_session.run.assert_called_once()


def test_batch_create_authors(ops, mock_connection):
    """Should create multiple authors in batch."""
    authors = [
        Author(author_id=f"batch_{i}", full_name=f"Author {i}")
        for i in range(10)
    ]

    mock_session = mock_connection.get_session.return_value.__enter__.return_value

    ops.batch_create_authors(authors)

    # Verify session.run was called
    mock_session.run.assert_called_once()


def test_count_nodes(ops, mock_connection):
    """Should count nodes by label."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_result.single.return_value = {'count': 5}
    mock_session.run.return_value = mock_result

    count = ops.count_nodes('Author')
    assert count == 5


def test_clear_database(ops, mock_connection):
    """Should clear all nodes and relationships."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value

    # Clear
    ops.clear_all()

    # Verify session.run was called
    mock_session.run.assert_called_once()


def test_get_author_papers(ops, mock_connection):
    """Should get all papers authored by author."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_result.__iter__ = Mock(return_value=iter([
        {'p': {'pmid': '99999', 'title': 'Paper Title'}},
    ]))
    mock_session.run.return_value = mock_result

    result = ops.get_author_papers("test_005")
    assert len(result) == 1
    assert result[0]['pmid'] == "99999"


def test_get_author_collaborators(ops, mock_connection):
    """Should get all collaborators of author."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_result.__iter__ = Mock(return_value=iter([
        {'other': {'author_id': 'test_007', 'full_name': 'Author Seven'}, 'collaboration_count': 3},
    ]))
    mock_session.run.return_value = mock_result

    collabs = ops.get_author_collaborators("test_006")
    assert len(collabs) == 1
    assert collabs[0]['collaboration_count'] == 3


def test_get_paper(ops, mock_connection):
    """Should get paper by PMID."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_record = {'p': {'pmid': '12345678', 'title': 'Test Paper'}}
    mock_result.single.return_value = mock_record
    mock_session.run.return_value = mock_result

    result = ops.get_paper("12345678")

    assert result is not None
    assert result['pmid'] == "12345678"


def test_get_nonexistent_paper(ops, mock_connection):
    """Should return None for nonexistent paper."""
    mock_session = mock_connection.get_session.return_value.__enter__.return_value
    mock_result = Mock()
    mock_result.single.return_value = None
    mock_session.run.return_value = mock_result

    result = ops.get_paper("nonexistent_pmid")

    assert result is None
