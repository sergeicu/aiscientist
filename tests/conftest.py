# tests/conftest.py

"""
Pytest configuration and shared fixtures.
"""

import pytest
import os


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (requires running Neo4j)"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (use mocks, no external dependencies)"
    )


@pytest.fixture(scope="session")
def neo4j_uri():
    """Get Neo4j URI from environment or use default."""
    return os.getenv("NEO4J_URI", "bolt://localhost:7687")


@pytest.fixture(scope="session")
def neo4j_user():
    """Get Neo4j username from environment or use default."""
    return os.getenv("NEO4J_USER", "neo4j")


@pytest.fixture(scope="session")
def neo4j_password():
    """Get Neo4j password from environment or use default."""
    return os.getenv("NEO4J_PASSWORD", "test_password")
