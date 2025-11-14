# Task 4: Neo4j Graph Database Setup & Core Operations - TDD Implementation

## Executive Summary

Implement a production-ready Neo4j graph database layer with proper schema, constraints, indexes, and core CRUD operations for the AI Scientist platform's knowledge graph. This forms the foundation for storing and querying research networks (authors, papers, institutions, collaborations).

**Key Requirements:**
- Set up Neo4j database with Docker
- Define graph schema (node types and relationship types)
- Create constraints and indexes for performance
- Implement core graph operations (create, read, update, delete)
- Build database connection manager with connection pooling
- Implement transaction management
- Create data validation layer
- Follow Test-Driven Development (TDD) principles

## Background & Context

### Why Neo4j?

Neo4j is a native graph database optimized for relationship queries:
- **Performance**: Constant-time relationship traversal (vs. exponential in SQL joins)
- **Cypher Query Language**: Intuitive pattern matching
- **ACID Compliance**: Full transaction support
- **Scalability**: Billions of nodes and relationships
- **Visualization**: Built-in browser for graph exploration

### Graph Model

Our knowledge graph models research networks:

**Nodes (Entities):**
- `Author`: Researchers
- `Paper`: Publications
- `Institution`: Organizations
- `Topic`: Research clusters/themes

**Relationships (Edges):**
- `AUTHORED`: Author → Paper
- `AFFILIATED_WITH`: Author → Institution
- `COLLABORATED_WITH`: Author ↔ Author
- `CITES`: Paper → Paper
- `BELONGS_TO`: Paper → Topic

### Why This is a Separate Task

This task focuses on **infrastructure and core operations**, making it independent from:
- Task 3 (Author Network Extractor) - uses this but doesn't need to wait
- Task 5 (Graph Analytics) - builds on top of this foundation
- Data collection tasks - can run in parallel

## Technical Architecture

### Module Structure

```
src/graph/
├── __init__.py
├── connection.py        # Neo4j connection manager
├── schema.py           # Schema definition and creation
├── models.py           # Pydantic models for graph entities
├── operations.py       # Core CRUD operations
├── transactions.py     # Transaction management
└── validators.py       # Data validation

tests/
├── __init__.py
├── test_connection.py
├── test_schema.py
├── test_models.py
├── test_operations.py
├── test_transactions.py
└── fixtures/
    └── sample_data.py
```

### Dependencies

```python
# Required packages
neo4j>=5.14            # Official Neo4j driver
pydantic>=2.0          # Data validation
loguru>=0.7            # Logging
pytest>=7.4            # Testing
pytest-asyncio>=0.21   # Async testing
pytest-mock>=3.12      # Mocking
python-dotenv>=1.0     # Environment variables
tenacity>=8.2          # Retry logic
```

## TDD Implementation Plan

### Phase 1: Connection Manager (TDD)

#### Test Cases

```python
# tests/test_connection.py

import pytest
from unittest.mock import Mock, patch
from src.graph.connection import Neo4jConnection
from neo4j.exceptions import ServiceUnavailable, AuthError

def test_connection_success():
    """Should establish connection to Neo4j."""
    conn = Neo4jConnection(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="test_password"
    )

    assert conn.driver is not None
    conn.close()

def test_connection_failure_wrong_credentials():
    """Should raise error with wrong credentials."""
    with pytest.raises(AuthError):
        conn = Neo4jConnection(
            uri="bolt://localhost:7687",
            user="wrong",
            password="wrong"
        )

def test_connection_pooling():
    """Should reuse connections from pool."""
    conn = Neo4jConnection(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        max_connection_pool_size=10
    )

    # Get multiple sessions
    sessions = [conn.get_session() for _ in range(5)]

    # Should not exceed pool size
    assert len(sessions) <= 10

    for session in sessions:
        session.close()

    conn.close()

def test_context_manager():
    """Should work as context manager."""
    with Neo4jConnection(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    ) as conn:
        session = conn.get_session()
        assert session is not None
        session.close()

    # Connection should be closed
    assert conn.driver._closed

def test_verify_connectivity():
    """Should verify database connectivity."""
    conn = Neo4jConnection(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )

    # Should successfully verify
    result = conn.verify_connectivity()
    assert result is True

    conn.close()

def test_retry_on_transient_errors(mocker):
    """Should retry connection on transient errors."""
    mock_driver = Mock()
    mock_driver.verify_connectivity.side_effect = [
        ServiceUnavailable("Connection failed"),
        ServiceUnavailable("Connection failed"),
        None  # Success on third try
    ]

    mocker.patch('neo4j.GraphDatabase.driver', return_value=mock_driver)

    conn = Neo4jConnection(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )

    # Should eventually succeed
    assert conn.verify_connectivity() is True

def test_get_database_info():
    """Should retrieve database information."""
    conn = Neo4jConnection(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )

    info = conn.get_database_info()

    assert 'version' in info
    assert 'edition' in info

    conn.close()
```

#### Implementation

```python
# src/graph/connection.py

from typing import Optional, Dict, Any
from neo4j import GraphDatabase, Session
from neo4j.exceptions import ServiceUnavailable, AuthError
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
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
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def verify_connectivity(self) -> bool:
        """
        Verify database connectivity.

        Returns:
            True if connection successful

        Raises:
            ServiceUnavailable: If cannot connect after retries
        """
        try:
            self.driver.verify_connectivity()
            return True
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
```

### Phase 2: Schema Definition (TDD)

#### Test Cases

```python
# tests/test_schema.py

import pytest
from src.graph.schema import GraphSchema
from src.graph.connection import Neo4jConnection

@pytest.fixture
def connection():
    """Create test connection."""
    conn = Neo4jConnection(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="test_password"
    )
    yield conn
    conn.close()

@pytest.fixture
def schema(connection):
    """Create schema manager."""
    return GraphSchema(connection)

def test_create_constraints(schema, connection):
    """Should create uniqueness constraints."""
    schema.create_constraints()

    # Verify constraints exist
    with connection.get_session() as session:
        result = session.run("SHOW CONSTRAINTS")
        constraints = [record['name'] for record in result]

        assert any('author_id' in c for c in constraints)
        assert any('pmid' in c for c in constraints)
        assert any('institution' in c for c in constraints)

def test_create_indexes(schema, connection):
    """Should create indexes for performance."""
    schema.create_indexes()

    # Verify indexes exist
    with connection.get_session() as session:
        result = session.run("SHOW INDEXES")
        indexes = [record['name'] for record in result]

        assert any('author_name' in i for i in indexes)
        assert any('paper_year' in i for i in indexes)

def test_idempotent_schema_creation(schema):
    """Should handle multiple schema creation calls."""
    # Create schema twice
    schema.create_all()
    schema.create_all()

    # Should not raise errors

def test_drop_all_constraints(schema, connection):
    """Should drop all constraints."""
    # Create first
    schema.create_constraints()

    # Drop
    schema.drop_all_constraints()

    # Verify dropped
    with connection.get_session() as session:
        result = session.run("SHOW CONSTRAINTS")
        constraints = list(result)

        # Should have no custom constraints
        assert len(constraints) == 0 or all(
            'author_id' not in c['name'] for c in constraints
        )

def test_get_schema_info(schema):
    """Should return schema information."""
    schema.create_all()

    info = schema.get_schema_info()

    assert 'constraints' in info
    assert 'indexes' in info
    assert 'node_labels' in info
    assert 'relationship_types' in info

def test_validate_schema(schema):
    """Should validate schema is properly set up."""
    schema.create_all()

    is_valid = schema.validate_schema()

    assert is_valid is True
```

#### Implementation

```python
# src/graph/schema.py

from typing import Dict, List, Any
from loguru import logger

class GraphSchema:
    """
    Define and manage Neo4j graph schema.

    Creates constraints, indexes, and validates schema.

    Args:
        connection: Neo4jConnection instance

    Example:
        >>> schema = GraphSchema(connection)
        >>> schema.create_all()
    """

    # Node labels
    NODE_LABELS = ['Author', 'Paper', 'Institution', 'Topic']

    # Relationship types
    RELATIONSHIP_TYPES = [
        'AUTHORED',
        'AFFILIATED_WITH',
        'COLLABORATED_WITH',
        'CITES',
        'BELONGS_TO'
    ]

    # Constraint definitions
    CONSTRAINTS = [
        {
            'name': 'author_id_unique',
            'label': 'Author',
            'property': 'author_id',
            'type': 'UNIQUE'
        },
        {
            'name': 'pmid_unique',
            'label': 'Paper',
            'property': 'pmid',
            'type': 'UNIQUE'
        },
        {
            'name': 'institution_name_unique',
            'label': 'Institution',
            'property': 'name',
            'type': 'UNIQUE'
        },
        {
            'name': 'topic_id_unique',
            'label': 'Topic',
            'property': 'topic_id',
            'type': 'UNIQUE'
        }
    ]

    # Index definitions
    INDEXES = [
        {
            'name': 'author_name_idx',
            'label': 'Author',
            'property': 'full_name'
        },
        {
            'name': 'author_last_name_idx',
            'label': 'Author',
            'property': 'last_name'
        },
        {
            'name': 'paper_year_idx',
            'label': 'Paper',
            'property': 'year'
        },
        {
            'name': 'paper_title_idx',
            'label': 'Paper',
            'property': 'title'
        },
        {
            'name': 'institution_name_idx',
            'label': 'Institution',
            'property': 'name'
        }
    ]

    def __init__(self, connection):
        self.connection = connection

    def create_constraints(self):
        """Create uniqueness constraints."""
        logger.info("Creating constraints...")

        with self.connection.get_session() as session:
            for constraint in self.CONSTRAINTS:
                query = f"""
                CREATE CONSTRAINT {constraint['name']} IF NOT EXISTS
                FOR (n:{constraint['label']})
                REQUIRE n.{constraint['property']} IS UNIQUE
                """

                try:
                    session.run(query)
                    logger.debug(f"Created constraint: {constraint['name']}")
                except Exception as e:
                    logger.warning(f"Constraint {constraint['name']} may already exist: {e}")

        logger.info("✓ Constraints created")

    def create_indexes(self):
        """Create indexes for performance."""
        logger.info("Creating indexes...")

        with self.connection.get_session() as session:
            for index in self.INDEXES:
                query = f"""
                CREATE INDEX {index['name']} IF NOT EXISTS
                FOR (n:{index['label']})
                ON (n.{index['property']})
                """

                try:
                    session.run(query)
                    logger.debug(f"Created index: {index['name']}")
                except Exception as e:
                    logger.warning(f"Index {index['name']} may already exist: {e}")

        logger.info("✓ Indexes created")

    def create_all(self):
        """Create complete schema (constraints + indexes)."""
        logger.info("Setting up database schema...")
        self.create_constraints()
        self.create_indexes()
        logger.info("✓ Schema setup complete")

    def drop_all_constraints(self):
        """Drop all custom constraints."""
        logger.warning("Dropping all constraints...")

        with self.connection.get_session() as session:
            for constraint in self.CONSTRAINTS:
                query = f"DROP CONSTRAINT {constraint['name']} IF EXISTS"

                try:
                    session.run(query)
                    logger.debug(f"Dropped constraint: {constraint['name']}")
                except Exception as e:
                    logger.debug(f"Could not drop {constraint['name']}: {e}")

    def drop_all_indexes(self):
        """Drop all custom indexes."""
        logger.warning("Dropping all indexes...")

        with self.connection.get_session() as session:
            for index in self.INDEXES:
                query = f"DROP INDEX {index['name']} IF EXISTS"

                try:
                    session.run(query)
                    logger.debug(f"Dropped index: {index['name']}")
                except Exception as e:
                    logger.debug(f"Could not drop {index['name']}: {e}")

    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get schema information.

        Returns:
            Dictionary with constraints, indexes, labels, relationships
        """
        info = {
            'constraints': [],
            'indexes': [],
            'node_labels': [],
            'relationship_types': []
        }

        with self.connection.get_session() as session:
            # Get constraints
            result = session.run("SHOW CONSTRAINTS")
            info['constraints'] = [
                {
                    'name': record['name'],
                    'type': record['type']
                }
                for record in result
            ]

            # Get indexes
            result = session.run("SHOW INDEXES")
            info['indexes'] = [
                {
                    'name': record['name'],
                    'state': record['state']
                }
                for record in result
            ]

            # Get node labels
            result = session.run("CALL db.labels()")
            info['node_labels'] = [record['label'] for record in result]

            # Get relationship types
            result = session.run("CALL db.relationshipTypes()")
            info['relationship_types'] = [record['relationshipType'] for record in result]

        return info

    def validate_schema(self) -> bool:
        """
        Validate schema is properly configured.

        Returns:
            True if all required constraints and indexes exist
        """
        info = self.get_schema_info()

        # Check constraints
        constraint_names = [c['name'] for c in info['constraints']]
        required_constraints = [c['name'] for c in self.CONSTRAINTS]

        missing_constraints = set(required_constraints) - set(constraint_names)
        if missing_constraints:
            logger.error(f"Missing constraints: {missing_constraints}")
            return False

        # Check indexes
        index_names = [i['name'] for i in info['indexes']]
        required_indexes = [i['name'] for i in self.INDEXES]

        missing_indexes = set(required_indexes) - set(index_names)
        if missing_indexes:
            logger.error(f"Missing indexes: {missing_indexes}")
            return False

        logger.info("✓ Schema validation passed")
        return True
```

### Phase 3: Pydantic Models (TDD)

#### Test Cases

```python
# tests/test_models.py

import pytest
from pydantic import ValidationError
from src.graph.models import Author, Paper, Institution, Topic

def test_author_model_valid():
    """Should create valid Author model."""
    author = Author(
        author_id="abc123",
        full_name="John Smith",
        last_name="Smith",
        first_name="John",
        initials="JS"
    )

    assert author.author_id == "abc123"
    assert author.full_name == "John Smith"

def test_author_model_missing_required_field():
    """Should raise error if required field missing."""
    with pytest.raises(ValidationError):
        Author(
            full_name="John Smith"
            # Missing author_id
        )

def test_paper_model_valid():
    """Should create valid Paper model."""
    paper = Paper(
        pmid="12345678",
        title="Test Paper",
        year="2023",
        journal="Nature"
    )

    assert paper.pmid == "12345678"
    assert paper.year == "2023"

def test_paper_model_optional_fields():
    """Should handle optional fields."""
    paper = Paper(
        pmid="12345678",
        title="Test Paper"
        # year and journal are optional
    )

    assert paper.year is None
    assert paper.journal is None

def test_institution_model():
    """Should create valid Institution model."""
    institution = Institution(
        name="Boston Children's Hospital",
        city="Boston",
        state="MA",
        country="USA"
    )

    assert institution.name == "Boston Children's Hospital"
    assert institution.country == "USA"

def test_topic_model():
    """Should create valid Topic model."""
    topic = Topic(
        topic_id="topic_001",
        label="Immunotherapy",
        description="Research on immune-based treatments"
    )

    assert topic.topic_id == "topic_001"
    assert topic.label == "Immunotherapy"

def test_model_to_dict():
    """Should convert model to dictionary."""
    author = Author(
        author_id="abc123",
        full_name="John Smith",
        last_name="Smith",
        first_name="John"
    )

    data = author.model_dump()

    assert isinstance(data, dict)
    assert data['author_id'] == "abc123"

def test_model_json_serialization():
    """Should serialize to JSON."""
    paper = Paper(
        pmid="12345678",
        title="Test Paper",
        year="2023"
    )

    json_str = paper.model_dump_json()

    assert isinstance(json_str, str)
    assert "12345678" in json_str
```

#### Implementation

```python
# src/graph/models.py

from typing import Optional, List
from pydantic import BaseModel, Field

class Author(BaseModel):
    """
    Author node model.

    Represents a researcher/author in the knowledge graph.
    """
    author_id: str = Field(..., description="Unique author identifier")
    full_name: str = Field(..., description="Full name")
    last_name: Optional[str] = Field(None, description="Last name")
    first_name: Optional[str] = Field(None, description="First name")
    initials: Optional[str] = Field(None, description="Initials")
    orcid: Optional[str] = Field(None, description="ORCID identifier")
    h_index: Optional[int] = Field(None, description="H-index")

    class Config:
        json_schema_extra = {
            "example": {
                "author_id": "smith_j_bch",
                "full_name": "John Smith",
                "last_name": "Smith",
                "first_name": "John",
                "initials": "JS"
            }
        }

class Paper(BaseModel):
    """
    Paper node model.

    Represents a publication in the knowledge graph.
    """
    pmid: str = Field(..., description="PubMed ID")
    title: str = Field(..., description="Paper title")
    abstract: Optional[str] = Field(None, description="Abstract")
    year: Optional[str] = Field(None, description="Publication year")
    journal: Optional[str] = Field(None, description="Journal name")
    doi: Optional[str] = Field(None, description="DOI")

    class Config:
        json_schema_extra = {
            "example": {
                "pmid": "12345678",
                "title": "Novel CAR-T Therapy",
                "year": "2023",
                "journal": "Nature Medicine"
            }
        }

class Institution(BaseModel):
    """
    Institution node model.

    Represents a research institution/organization.
    """
    name: str = Field(..., description="Institution name")
    city: Optional[str] = Field(None, description="City")
    state: Optional[str] = Field(None, description="State/Province")
    country: Optional[str] = Field(None, description="Country")
    latitude: Optional[float] = Field(None, description="Latitude")
    longitude: Optional[float] = Field(None, description="Longitude")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Boston Children's Hospital",
                "city": "Boston",
                "state": "MA",
                "country": "USA"
            }
        }

class Topic(BaseModel):
    """
    Topic node model.

    Represents a research topic/cluster.
    """
    topic_id: str = Field(..., description="Topic identifier")
    label: str = Field(..., description="Topic name/label")
    description: Optional[str] = Field(None, description="Topic description")
    size: Optional[int] = Field(None, description="Number of papers")

    class Config:
        json_schema_extra = {
            "example": {
                "topic_id": "topic_001",
                "label": "Immunotherapy",
                "description": "Cancer immunotherapy research"
            }
        }

class Relationship(BaseModel):
    """
    Base relationship model.
    """
    type: str = Field(..., description="Relationship type")
    properties: Optional[dict] = Field(default_factory=dict, description="Relationship properties")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "COLLABORATED_WITH",
                "properties": {"count": 5, "years": ["2020", "2021"]}
            }
        }
```

### Phase 4: Core CRUD Operations (TDD)

#### Test Cases

```python
# tests/test_operations.py

import pytest
from src.graph.operations import GraphOperations
from src.graph.connection import Neo4jConnection
from src.graph.schema import GraphSchema
from src.graph.models import Author, Paper, Institution

@pytest.fixture
def connection():
    conn = Neo4jConnection(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="test_password"
    )

    # Setup schema
    schema = GraphSchema(conn)
    schema.create_all()

    yield conn

    # Cleanup
    with conn.get_session() as session:
        session.run("MATCH (n) DETACH DELETE n")

    conn.close()

@pytest.fixture
def ops(connection):
    return GraphOperations(connection)

def test_create_author_node(ops):
    """Should create Author node."""
    author = Author(
        author_id="test_001",
        full_name="John Smith",
        last_name="Smith",
        first_name="John"
    )

    result = ops.create_author(author)

    assert result['author_id'] == "test_001"

def test_create_duplicate_author_ignores(ops):
    """Should handle duplicate author creation."""
    author = Author(
        author_id="test_001",
        full_name="John Smith",
        last_name="Smith",
        first_name="John"
    )

    # Create twice
    ops.create_author(author)
    result = ops.create_author(author)

    # Should not error, should return existing
    assert result['author_id'] == "test_001"

def test_get_author_by_id(ops):
    """Should retrieve author by ID."""
    author = Author(
        author_id="test_002",
        full_name="Jane Doe",
        last_name="Doe",
        first_name="Jane"
    )

    ops.create_author(author)

    retrieved = ops.get_author("test_002")

    assert retrieved is not None
    assert retrieved['full_name'] == "Jane Doe"

def test_get_nonexistent_author(ops):
    """Should return None for nonexistent author."""
    result = ops.get_author("nonexistent_id")

    assert result is None

def test_update_author(ops):
    """Should update author properties."""
    author = Author(
        author_id="test_003",
        full_name="Alice Brown",
        last_name="Brown",
        first_name="Alice"
    )

    ops.create_author(author)

    # Update h_index
    ops.update_author("test_003", {'h_index': 25})

    updated = ops.get_author("test_003")
    assert updated['h_index'] == 25

def test_delete_author(ops):
    """Should delete author node."""
    author = Author(
        author_id="test_004",
        full_name="Bob Wilson",
        last_name="Wilson",
        first_name="Bob"
    )

    ops.create_author(author)

    # Delete
    ops.delete_author("test_004")

    # Should not exist
    result = ops.get_author("test_004")
    assert result is None

def test_create_paper_node(ops):
    """Should create Paper node."""
    paper = Paper(
        pmid="12345678",
        title="Test Paper",
        year="2023"
    )

    result = ops.create_paper(paper)

    assert result['pmid'] == "12345678"

def test_create_institution_node(ops):
    """Should create Institution node."""
    institution = Institution(
        name="Test University",
        city="Boston",
        country="USA"
    )

    result = ops.create_institution(institution)

    assert result['name'] == "Test University"

def test_create_authored_relationship(ops):
    """Should create AUTHORED relationship."""
    author = Author(
        author_id="test_005",
        full_name="Author Five"
    )

    paper = Paper(
        pmid="99999",
        title="Paper Title"
    )

    ops.create_author(author)
    ops.create_paper(paper)

    # Create relationship
    ops.create_authored_relationship(
        author_id="test_005",
        pmid="99999",
        position=0,
        is_first_author=True
    )

    # Verify relationship exists
    result = ops.get_author_papers("test_005")
    assert len(result) == 1
    assert result[0]['pmid'] == "99999"

def test_create_collaboration_relationship(ops):
    """Should create COLLABORATED_WITH relationship."""
    author1 = Author(author_id="test_006", full_name="Author Six")
    author2 = Author(author_id="test_007", full_name="Author Seven")

    ops.create_author(author1)
    ops.create_author(author2)

    # Create collaboration
    ops.create_collaboration(
        author1_id="test_006",
        author2_id="test_007",
        pmid="88888"
    )

    # Verify
    collabs = ops.get_author_collaborators("test_006")
    assert len(collabs) == 1

def test_batch_create_authors(ops):
    """Should create multiple authors in batch."""
    authors = [
        Author(author_id=f"batch_{i}", full_name=f"Author {i}")
        for i in range(10)
    ]

    ops.batch_create_authors(authors)

    # Verify all created
    for i in range(10):
        author = ops.get_author(f"batch_{i}")
        assert author is not None

def test_count_nodes(ops):
    """Should count nodes by label."""
    # Create some nodes
    for i in range(5):
        ops.create_author(Author(author_id=f"count_{i}", full_name=f"Author {i}"))

    count = ops.count_nodes('Author')
    assert count >= 5

def test_clear_database(ops, connection):
    """Should clear all nodes and relationships."""
    # Create some data
    ops.create_author(Author(author_id="clear_test", full_name="Clear Test"))

    # Clear
    ops.clear_all()

    # Verify empty
    count = ops.count_nodes('Author')
    assert count == 0
```

#### Implementation

```python
# src/graph/operations.py

from typing import Dict, List, Optional, Any
from loguru import logger
from .models import Author, Paper, Institution, Topic

class GraphOperations:
    """
    Core CRUD operations for graph database.

    Provides methods to create, read, update, delete nodes and relationships.

    Args:
        connection: Neo4jConnection instance

    Example:
        >>> ops = GraphOperations(connection)
        >>> ops.create_author(author_model)
    """

    def __init__(self, connection):
        self.connection = connection

    # ========== Author Operations ==========

    def create_author(self, author: Author) -> Dict:
        """
        Create Author node.

        Args:
            author: Author model

        Returns:
            Created author properties
        """
        with self.connection.get_session() as session:
            query = """
            MERGE (a:Author {author_id: $author_id})
            SET a.full_name = $full_name,
                a.last_name = $last_name,
                a.first_name = $first_name,
                a.initials = $initials,
                a.orcid = $orcid,
                a.h_index = $h_index
            RETURN a
            """

            result = session.run(query, **author.model_dump())
            record = result.single()

            if record:
                return dict(record['a'])

            return {}

    def get_author(self, author_id: str) -> Optional[Dict]:
        """
        Get author by ID.

        Args:
            author_id: Author identifier

        Returns:
            Author properties or None
        """
        with self.connection.get_session() as session:
            query = """
            MATCH (a:Author {author_id: $author_id})
            RETURN a
            """

            result = session.run(query, author_id=author_id)
            record = result.single()

            if record:
                return dict(record['a'])

            return None

    def update_author(self, author_id: str, properties: Dict) -> Dict:
        """
        Update author properties.

        Args:
            author_id: Author identifier
            properties: Properties to update

        Returns:
            Updated author properties
        """
        with self.connection.get_session() as session:
            # Build SET clause
            set_clause = ", ".join([f"a.{key} = ${key}" for key in properties.keys()])

            query = f"""
            MATCH (a:Author {{author_id: $author_id}})
            SET {set_clause}
            RETURN a
            """

            result = session.run(query, author_id=author_id, **properties)
            record = result.single()

            if record:
                return dict(record['a'])

            return {}

    def delete_author(self, author_id: str) -> bool:
        """
        Delete author node.

        Args:
            author_id: Author identifier

        Returns:
            True if deleted
        """
        with self.connection.get_session() as session:
            query = """
            MATCH (a:Author {author_id: $author_id})
            DETACH DELETE a
            """

            session.run(query, author_id=author_id)
            return True

    def batch_create_authors(self, authors: List[Author]) -> None:
        """
        Create multiple authors in batch.

        Args:
            authors: List of Author models
        """
        with self.connection.get_session() as session:
            query = """
            UNWIND $authors AS author
            MERGE (a:Author {author_id: author.author_id})
            SET a.full_name = author.full_name,
                a.last_name = author.last_name,
                a.first_name = author.first_name,
                a.initials = author.initials
            """

            author_dicts = [a.model_dump() for a in authors]
            session.run(query, authors=author_dicts)

    # ========== Paper Operations ==========

    def create_paper(self, paper: Paper) -> Dict:
        """Create Paper node."""
        with self.connection.get_session() as session:
            query = """
            MERGE (p:Paper {pmid: $pmid})
            SET p.title = $title,
                p.abstract = $abstract,
                p.year = $year,
                p.journal = $journal,
                p.doi = $doi
            RETURN p
            """

            result = session.run(query, **paper.model_dump())
            record = result.single()

            if record:
                return dict(record['p'])

            return {}

    def get_paper(self, pmid: str) -> Optional[Dict]:
        """Get paper by PMID."""
        with self.connection.get_session() as session:
            query = """
            MATCH (p:Paper {pmid: $pmid})
            RETURN p
            """

            result = session.run(query, pmid=pmid)
            record = result.single()

            if record:
                return dict(record['p'])

            return None

    # ========== Institution Operations ==========

    def create_institution(self, institution: Institution) -> Dict:
        """Create Institution node."""
        with self.connection.get_session() as session:
            query = """
            MERGE (i:Institution {name: $name})
            SET i.city = $city,
                i.state = $state,
                i.country = $country,
                i.latitude = $latitude,
                i.longitude = $longitude
            RETURN i
            """

            result = session.run(query, **institution.model_dump())
            record = result.single()

            if record:
                return dict(record['i'])

            return {}

    # ========== Relationship Operations ==========

    def create_authored_relationship(
        self,
        author_id: str,
        pmid: str,
        position: int = 0,
        is_first_author: bool = False,
        is_last_author: bool = False
    ) -> None:
        """
        Create AUTHORED relationship.

        Args:
            author_id: Author identifier
            pmid: Paper PMID
            position: Author position in author list
            is_first_author: Is first author
            is_last_author: Is last author
        """
        with self.connection.get_session() as session:
            query = """
            MATCH (a:Author {author_id: $author_id})
            MATCH (p:Paper {pmid: $pmid})
            MERGE (a)-[r:AUTHORED]->(p)
            SET r.position = $position,
                r.is_first_author = $is_first_author,
                r.is_last_author = $is_last_author
            """

            session.run(
                query,
                author_id=author_id,
                pmid=pmid,
                position=position,
                is_first_author=is_first_author,
                is_last_author=is_last_author
            )

    def create_collaboration(
        self,
        author1_id: str,
        author2_id: str,
        pmid: str
    ) -> None:
        """
        Create COLLABORATED_WITH relationship.

        Args:
            author1_id: First author ID
            author2_id: Second author ID
            pmid: Paper they collaborated on
        """
        with self.connection.get_session() as session:
            query = """
            MATCH (a1:Author {author_id: $author1_id})
            MATCH (a2:Author {author_id: $author2_id})
            MERGE (a1)-[r:COLLABORATED_WITH]-(a2)
            ON CREATE SET r.count = 1, r.pmids = [$pmid]
            ON MATCH SET r.count = r.count + 1,
                        r.pmids = r.pmids + $pmid
            """

            session.run(
                query,
                author1_id=author1_id,
                author2_id=author2_id,
                pmid=pmid
            )

    # ========== Query Operations ==========

    def get_author_papers(self, author_id: str) -> List[Dict]:
        """
        Get all papers authored by author.

        Args:
            author_id: Author identifier

        Returns:
            List of paper dictionaries
        """
        with self.connection.get_session() as session:
            query = """
            MATCH (a:Author {author_id: $author_id})-[:AUTHORED]->(p:Paper)
            RETURN p
            """

            result = session.run(query, author_id=author_id)

            return [dict(record['p']) for record in result]

    def get_author_collaborators(self, author_id: str) -> List[Dict]:
        """
        Get all collaborators of author.

        Args:
            author_id: Author identifier

        Returns:
            List of collaborator dictionaries
        """
        with self.connection.get_session() as session:
            query = """
            MATCH (a:Author {author_id: $author_id})-[r:COLLABORATED_WITH]-(other:Author)
            RETURN other, r.count as collaboration_count
            """

            result = session.run(query, author_id=author_id)

            return [
                {
                    **dict(record['other']),
                    'collaboration_count': record['collaboration_count']
                }
                for record in result
            ]

    # ========== Utility Operations ==========

    def count_nodes(self, label: str) -> int:
        """
        Count nodes by label.

        Args:
            label: Node label (e.g., 'Author', 'Paper')

        Returns:
            Node count
        """
        with self.connection.get_session() as session:
            query = f"MATCH (n:{label}) RETURN count(n) as count"

            result = session.run(query)
            record = result.single()

            return record['count'] if record else 0

    def clear_all(self) -> None:
        """Clear all nodes and relationships from database."""
        logger.warning("Clearing entire database...")

        with self.connection.get_session() as session:
            query = "MATCH (n) DETACH DELETE n"
            session.run(query)

        logger.warning("Database cleared")
```

## Running Tests

```bash
# Start Neo4j with Docker
docker run -d \
  --name neo4j-test \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/test_password \
  neo4j:latest

# Install dependencies
pip install neo4j pydantic loguru pytest pytest-mock tenacity python-dotenv

# Run all tests
pytest tests/ -v --cov=src/graph

# Run specific test file
pytest tests/test_operations.py -v

# Run with Neo4j connection tests (requires running database)
pytest tests/ -v -m "not integration"
```

## Integration Testing

```python
# tests/integration/test_real_neo4j.py

import pytest
from src.graph.connection import Neo4jConnection
from src.graph.schema import GraphSchema
from src.graph.operations import GraphOperations
from src.graph.models import Author, Paper

@pytest.mark.integration
def test_end_to_end_workflow():
    """Test complete workflow with real Neo4j."""
    # Connect
    with Neo4jConnection(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="test_password"
    ) as conn:
        # Setup schema
        schema = GraphSchema(conn)
        schema.create_all()

        # Create operations
        ops = GraphOperations(conn)

        # Create author
        author = Author(
            author_id="integration_test",
            full_name="Integration Test",
            last_name="Test",
            first_name="Integration"
        )

        result = ops.create_author(author)
        assert result['author_id'] == "integration_test"

        # Create paper
        paper = Paper(
            pmid="99999999",
            title="Integration Test Paper",
            year="2024"
        )

        ops.create_paper(paper)

        # Create relationship
        ops.create_authored_relationship(
            author_id="integration_test",
            pmid="99999999",
            position=0,
            is_first_author=True
        )

        # Query
        papers = ops.get_author_papers("integration_test")
        assert len(papers) == 1
        assert papers[0]['title'] == "Integration Test Paper"

        # Cleanup
        ops.clear_all()
```

## Usage Examples

### Basic Setup

```python
from src.graph.connection import Neo4jConnection
from src.graph.schema import GraphSchema
from src.graph.operations import GraphOperations
from src.graph.models import Author, Paper

# Connect to Neo4j
with Neo4jConnection(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your_password"
) as conn:
    # Setup schema
    schema = GraphSchema(conn)
    schema.create_all()

    # Create operations manager
    ops = GraphOperations(conn)

    # Create author
    author = Author(
        author_id="smith_j_001",
        full_name="John Smith",
        last_name="Smith",
        first_name="John"
    )

    ops.create_author(author)

    # Create paper
    paper = Paper(
        pmid="12345678",
        title="Novel Research",
        year="2024"
    )

    ops.create_paper(paper)

    # Link author to paper
    ops.create_authored_relationship(
        author_id="smith_j_001",
        pmid="12345678",
        position=0,
        is_first_author=True
    )
```

### Batch Operations

```python
# Create multiple authors
authors = [
    Author(author_id=f"author_{i}", full_name=f"Author {i}")
    for i in range(100)
]

ops.batch_create_authors(authors)
```

## Success Criteria

✅ **Must Have:**
1. All unit tests passing (>90% coverage)
2. Neo4j connection with pooling
3. Schema creation (constraints + indexes)
4. Pydantic models for validation
5. Core CRUD operations (create, read, update, delete)
6. Relationship creation
7. Transaction support

✅ **Should Have:**
8. Integration tests with real Neo4j
9. Batch operations
10. Query helpers
11. Error handling and retries

✅ **Nice to Have:**
12. Connection health checks
13. Performance monitoring
14. Migration support

## Deliverables

1. **Source code** (with >90% test coverage)
2. **Tests** (unit + integration)
3. **Docker setup** for Neo4j
4. **Documentation** and examples
5. **Sample Cypher queries**

## Environment Setup

```bash
# Start Neo4j
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -v neo4j_data:/data \
  neo4j:5.14

# Install dependencies
pip install neo4j pydantic loguru pytest tenacity python-dotenv

# Run tests
pytest tests/ -v --cov=src/graph
```

## Cypher Query Reference

```cypher
// Count all nodes
MATCH (n) RETURN labels(n) as type, count(n) as count

// Find authors with most papers
MATCH (a:Author)-[:AUTHORED]->(p:Paper)
RETURN a.full_name, count(p) as paper_count
ORDER BY paper_count DESC
LIMIT 20

// Find collaborations
MATCH (a1:Author)-[:COLLABORATED_WITH]-(a2:Author)
RETURN a1.full_name, a2.full_name, a1.collaboration_count
```

---

**Task completion**: When all tests pass, schema is properly set up, and core CRUD operations work with real Neo4j database.
