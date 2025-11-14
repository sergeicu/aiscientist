# src/graph/connection.py

from typing import Optional, Dict, Any
from neo4j import GraphDatabase, Session
from neo4j.exceptions import ServiceUnavailable, AuthError
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os


class Neo4jConnection:
    """
    Manage Neo4j database connections with pooling and retry logic.

    Features:
    - Connection pooling
    - Automatic retries on transient failures
    - Context manager support
    - Health checks

    Args:
        uri: Neo4j connection URI (bolt://host:port)
        user: Database username
        password: Database password
        max_connection_pool_size: Maximum connections in pool
        connection_timeout: Connection timeout in seconds

    Example:
        >>> with Neo4jConnection(uri="bolt://localhost:7687") as conn:
        ...     session = conn.get_session()
        ...     result = session.run("MATCH (n) RETURN count(n)")
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: Optional[str] = None,
        max_connection_pool_size: int = 50,
        connection_timeout: float = 30.0
    ):
        if password is None:
            password = os.getenv("NEO4J_PASSWORD", "password")

        self.uri = uri
        self.user = user
        self._password = password

        logger.info(f"Connecting to Neo4j at {uri}")

        try:
            self.driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                max_connection_pool_size=max_connection_pool_size,
                connection_timeout=connection_timeout
            )

            # Verify connectivity
            self.verify_connectivity()
            logger.info("✓ Neo4j connection established")

        except AuthError as e:
            logger.error(f"Authentication failed: {e}")
            raise
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            raise
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(ServiceUnavailable)
    )
    def verify_connectivity(self) -> bool:
        """
        Verify database connectivity.

        Returns:
            True if connection successful

        Raises:
            ServiceUnavailable: If cannot connect after retries
            AuthError: If authentication fails
        """
        try:
            self.driver.verify_connectivity()
            return True
        except AuthError:
            # Don't retry on auth errors
            raise
        except ServiceUnavailable as e:
            logger.warning(f"Connectivity check failed: {e}")
            raise
        except Exception as e:
            logger.warning(f"Connectivity check failed: {e}")
            raise

    def get_session(self, database: str = "neo4j") -> Session:
        """
        Get database session from pool.

        Args:
            database: Database name (default: "neo4j")

        Returns:
            Neo4j session
        """
        return self.driver.session(database=database)

    def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information.

        Returns:
            Dictionary with version, edition, etc.
        """
        with self.get_session() as session:
            result = session.run(
                "CALL dbms.components() YIELD name, versions, edition"
            )

            record = result.single()

            return {
                'name': record['name'],
                'version': record['versions'][0],
                'edition': record['edition']
            }

    def close(self):
        """Close database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
