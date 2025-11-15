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
