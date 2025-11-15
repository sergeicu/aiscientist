# tests/test_connection.py

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.graph.connection import Neo4jConnection
from neo4j.exceptions import ServiceUnavailable, AuthError


@pytest.fixture
def mock_driver():
    """Create a mock Neo4j driver."""
    driver = Mock()
    driver.verify_connectivity = Mock()
    driver._closed = False
    driver.close = Mock(side_effect=lambda: setattr(driver, '_closed', True))
    return driver


@pytest.fixture
def mock_session():
    """Create a mock Neo4j session."""
    session = Mock()
    session.close = Mock()
    return session


def test_connection_success(mock_driver):
    """Should establish connection to Neo4j."""
    with patch('neo4j.GraphDatabase.driver', return_value=mock_driver):
        conn = Neo4jConnection(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test_password"
        )

        assert conn.driver is not None
        mock_driver.verify_connectivity.assert_called_once()
        conn.close()


def test_connection_failure_wrong_credentials():
    """Should raise error with wrong credentials."""
    with patch('neo4j.GraphDatabase.driver') as mock_gd:
        mock_gd.return_value.verify_connectivity.side_effect = AuthError("Invalid credentials")

        with pytest.raises(AuthError):
            conn = Neo4jConnection(
                uri="bolt://localhost:7687",
                user="wrong",
                password="wrong"
            )


def test_connection_pooling(mock_driver, mock_session):
    """Should support connection pooling configuration."""
    with patch('neo4j.GraphDatabase.driver', return_value=mock_driver) as mock_gd:
        mock_driver.session = Mock(return_value=mock_session)

        conn = Neo4jConnection(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
            max_connection_pool_size=10
        )

        # Verify driver was created with pool size
        mock_gd.assert_called_once()
        call_kwargs = mock_gd.call_args[1]
        assert call_kwargs['max_connection_pool_size'] == 10

        # Get multiple sessions
        sessions = [conn.get_session() for _ in range(5)]

        # Should be able to get sessions
        assert len(sessions) == 5

        for session in sessions:
            session.close()

        conn.close()


def test_context_manager(mock_driver):
    """Should work as context manager."""
    with patch('neo4j.GraphDatabase.driver', return_value=mock_driver):
        mock_driver.session = Mock()

        with Neo4jConnection(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        ) as conn:
            session = conn.get_session()
            assert session is not None

        # Connection should be closed
        assert mock_driver._closed is True


def test_verify_connectivity(mock_driver):
    """Should verify database connectivity."""
    with patch('neo4j.GraphDatabase.driver', return_value=mock_driver):
        conn = Neo4jConnection(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )

        # Should successfully verify
        result = conn.verify_connectivity()
        assert result is True

        conn.close()


def test_retry_on_transient_errors():
    """Should retry connection on transient errors."""
    mock_driver = Mock()

    # Fail twice, then succeed
    mock_driver.verify_connectivity.side_effect = [
        ServiceUnavailable("Connection failed"),
        ServiceUnavailable("Connection failed"),
        None  # Success on third try
    ]

    with patch('neo4j.GraphDatabase.driver', return_value=mock_driver):
        conn = Neo4jConnection(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )

        # Should eventually succeed (verify_connectivity is called in __init__)
        assert conn.driver is not None

        # Verify it was called 3 times (retried twice)
        assert mock_driver.verify_connectivity.call_count == 3


def test_get_database_info(mock_driver, mock_session):
    """Should retrieve database information."""
    with patch('neo4j.GraphDatabase.driver', return_value=mock_driver):
        mock_driver.session = Mock(return_value=mock_session)

        # Mock the database info query result
        mock_result = Mock()
        mock_record = {
            'name': 'neo4j',
            'versions': ['5.14.0'],
            'edition': 'community'
        }
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        conn = Neo4jConnection(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )

        info = conn.get_database_info()

        assert 'version' in info
        assert 'edition' in info
        assert info['version'] == '5.14.0'
        assert info['edition'] == 'community'

        conn.close()


def test_get_session(mock_driver, mock_session):
    """Should get session from pool."""
    with patch('neo4j.GraphDatabase.driver', return_value=mock_driver):
        mock_driver.session = Mock(return_value=mock_session)

        conn = Neo4jConnection(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )

        session = conn.get_session()
        assert session is not None
        mock_driver.session.assert_called_once_with(database="neo4j")

        conn.close()


def test_get_session_custom_database(mock_driver, mock_session):
    """Should support custom database name."""
    with patch('neo4j.GraphDatabase.driver', return_value=mock_driver):
        mock_driver.session = Mock(return_value=mock_session)

        conn = Neo4jConnection(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )

        session = conn.get_session(database="custom")
        assert session is not None
        mock_driver.session.assert_called_once_with(database="custom")

        conn.close()
