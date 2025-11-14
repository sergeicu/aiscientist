# tests/test_schema.py

import pytest
from unittest.mock import Mock, MagicMock
from src.graph.schema import GraphSchema


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
def schema(mock_connection):
    """Create schema manager."""
    return GraphSchema(mock_connection)


def test_create_constraints(schema, mock_connection):
    """Should create uniqueness constraints."""
    schema.create_constraints()

    # Verify session was used
    mock_connection.get_session.assert_called()


def test_create_indexes(schema, mock_connection):
    """Should create indexes for performance."""
    schema.create_indexes()

    # Verify session was used
    mock_connection.get_session.assert_called()


def test_idempotent_schema_creation(schema):
    """Should handle multiple schema creation calls."""
    # Create schema twice - should not raise errors
    schema.create_all()
    schema.create_all()


def test_drop_all_constraints(schema, mock_connection):
    """Should drop all constraints."""
    schema.drop_all_constraints()

    # Verify session was used
    mock_connection.get_session.assert_called()


def test_get_schema_info(schema, mock_connection):
    """Should return schema information."""
    # Mock the results
    mock_session = mock_connection.get_session.return_value.__enter__.return_value

    # Mock constraints query
    constraints_result = Mock()
    constraints_result.__iter__ = Mock(return_value=iter([
        {'name': 'author_id_unique', 'type': 'UNIQUE'},
        {'name': 'pmid_unique', 'type': 'UNIQUE'}
    ]))

    # Mock indexes query
    indexes_result = Mock()
    indexes_result.__iter__ = Mock(return_value=iter([
        {'name': 'author_name_idx', 'state': 'ONLINE'},
        {'name': 'paper_year_idx', 'state': 'ONLINE'}
    ]))

    # Mock labels query
    labels_result = Mock()
    labels_result.__iter__ = Mock(return_value=iter([
        {'label': 'Author'},
        {'label': 'Paper'}
    ]))

    # Mock relationship types query
    rel_types_result = Mock()
    rel_types_result.__iter__ = Mock(return_value=iter([
        {'relationshipType': 'AUTHORED'},
        {'relationshipType': 'COLLABORATED_WITH'}
    ]))

    # Set up the side effects for sequential calls
    mock_session.run.side_effect = [
        constraints_result,
        indexes_result,
        labels_result,
        rel_types_result
    ]

    info = schema.get_schema_info()

    assert 'constraints' in info
    assert 'indexes' in info
    assert 'node_labels' in info
    assert 'relationship_types' in info
    assert len(info['constraints']) == 2
    assert len(info['indexes']) == 2


def test_validate_schema(schema, mock_connection):
    """Should validate schema is properly set up."""
    # Mock the get_schema_info to return all required constraints and indexes
    mock_session = mock_connection.get_session.return_value.__enter__.return_value

    # Mock constraints query
    constraints_result = Mock()
    constraints_result.__iter__ = Mock(return_value=iter([
        {'name': 'author_id_unique', 'type': 'UNIQUE'},
        {'name': 'pmid_unique', 'type': 'UNIQUE'},
        {'name': 'institution_name_unique', 'type': 'UNIQUE'},
        {'name': 'topic_id_unique', 'type': 'UNIQUE'}
    ]))

    # Mock indexes query
    indexes_result = Mock()
    indexes_result.__iter__ = Mock(return_value=iter([
        {'name': 'author_name_idx', 'state': 'ONLINE'},
        {'name': 'author_last_name_idx', 'state': 'ONLINE'},
        {'name': 'paper_year_idx', 'state': 'ONLINE'},
        {'name': 'paper_title_idx', 'state': 'ONLINE'},
        {'name': 'institution_name_idx', 'state': 'ONLINE'}
    ]))

    # Mock labels query
    labels_result = Mock()
    labels_result.__iter__ = Mock(return_value=iter([]))

    # Mock relationship types query
    rel_types_result = Mock()
    rel_types_result.__iter__ = Mock(return_value=iter([]))

    # Set up the side effects for sequential calls
    mock_session.run.side_effect = [
        constraints_result,
        indexes_result,
        labels_result,
        rel_types_result
    ]

    is_valid = schema.validate_schema()

    assert is_valid is True


def test_validate_schema_missing_constraints(schema, mock_connection):
    """Should return False when constraints are missing."""
    # Mock the get_schema_info to return incomplete constraints
    mock_session = mock_connection.get_session.return_value.__enter__.return_value

    # Mock constraints query - missing some
    constraints_result = Mock()
    constraints_result.__iter__ = Mock(return_value=iter([
        {'name': 'author_id_unique', 'type': 'UNIQUE'}
        # Missing other required constraints
    ]))

    # Mock indexes query
    indexes_result = Mock()
    indexes_result.__iter__ = Mock(return_value=iter([
        {'name': 'author_name_idx', 'state': 'ONLINE'},
        {'name': 'author_last_name_idx', 'state': 'ONLINE'},
        {'name': 'paper_year_idx', 'state': 'ONLINE'},
        {'name': 'paper_title_idx', 'state': 'ONLINE'},
        {'name': 'institution_name_idx', 'state': 'ONLINE'}
    ]))

    # Mock labels query
    labels_result = Mock()
    labels_result.__iter__ = Mock(return_value=iter([]))

    # Mock relationship types query
    rel_types_result = Mock()
    rel_types_result.__iter__ = Mock(return_value=iter([]))

    mock_session.run.side_effect = [
        constraints_result,
        indexes_result,
        labels_result,
        rel_types_result
    ]

    is_valid = schema.validate_schema()

    assert is_valid is False


def test_schema_constants(schema):
    """Should have correct schema constants defined."""
    assert 'Author' in GraphSchema.NODE_LABELS
    assert 'Paper' in GraphSchema.NODE_LABELS
    assert 'Institution' in GraphSchema.NODE_LABELS
    assert 'Topic' in GraphSchema.NODE_LABELS

    assert 'AUTHORED' in GraphSchema.RELATIONSHIP_TYPES
    assert 'AFFILIATED_WITH' in GraphSchema.RELATIONSHIP_TYPES
    assert 'COLLABORATED_WITH' in GraphSchema.RELATIONSHIP_TYPES
    assert 'CITES' in GraphSchema.RELATIONSHIP_TYPES
    assert 'BELONGS_TO' in GraphSchema.RELATIONSHIP_TYPES


def test_create_all_calls_both_methods(schema, mock_connection):
    """Should call both create_constraints and create_indexes."""
    schema.create_all()

    # Verify get_session was called multiple times (for constraints and indexes)
    assert mock_connection.get_session.call_count >= 2
