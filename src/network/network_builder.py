from typing import List, Dict, Optional
from neo4j import GraphDatabase
from loguru import logger
from .author_extractor import AuthorExtractor

class NetworkBuilder:
    """
    Build co-authorship network in Neo4j graph database.

    Creates nodes for authors, papers, institutions.
    Creates relationships for authorship and collaboration.

    Args:
        uri: Neo4j connection URI
        user: Neo4j username
        password: Neo4j password
        driver: Optional Neo4j driver (for testing)

    Example:
        >>> builder = NetworkBuilder(
        ...     uri="bolt://localhost:7687",
        ...     user="neo4j",
        ...     password="password"
        ... )
        >>> builder.build_network(articles)
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        driver: Optional[any] = None
    ):
        if driver:
            self.driver = driver
        else:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))

        self.author_extractor = AuthorExtractor()

        # Create constraints and indexes
        self._create_schema()

    def _create_schema(self):
        """Create Neo4j schema: constraints and indexes."""
        with self.driver.session() as session:
            # Constraints (ensure uniqueness)
            constraints = [
                "CREATE CONSTRAINT author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.author_id IS UNIQUE",
                "CREATE CONSTRAINT pmid IF NOT EXISTS FOR (p:Paper) REQUIRE p.pmid IS UNIQUE",
                "CREATE CONSTRAINT institution IF NOT EXISTS FOR (i:Institution) REQUIRE i.name IS UNIQUE"
            ]

            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint may already exist: {e}")

            # Indexes (improve query performance)
            indexes = [
                "CREATE INDEX author_name IF NOT EXISTS FOR (a:Author) ON (a.full_name)",
                "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)",
                "CREATE INDEX institution_name IF NOT EXISTS FOR (i:Institution) ON (i.name)"
            ]

            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.debug(f"Index may already exist: {e}")

    def create_author_nodes(self, authors: List[Dict]):
        """
        Create Author nodes in Neo4j.

        Args:
            authors: List of author dictionaries
        """
        with self.driver.session() as session:
            for author in authors:
                query = """
                MERGE (a:Author {author_id: $author_id})
                SET a.full_name = $full_name,
                    a.last_name = $last_name,
                    a.first_name = $first_name,
                    a.initials = $initials
                """

                session.run(
                    query,
                    author_id=author['author_id'],
                    full_name=author['full_name'],
                    last_name=author['last_name'],
                    first_name=author['first_name'],
                    initials=author['initials']
                )

    def create_paper_node(self, paper: Dict):
        """
        Create Paper node in Neo4j.

        Args:
            paper: Paper dictionary
        """
        with self.driver.session() as session:
            query = """
            MERGE (p:Paper {pmid: $pmid})
            SET p.title = $title,
                p.year = $year,
                p.journal = $journal,
                p.doi = $doi
            """

            session.run(
                query,
                pmid=paper.get('pmid'),
                title=paper.get('title', ''),
                year=paper.get('year', ''),
                journal=paper.get('journal', ''),
                doi=paper.get('doi')
            )

    def create_institution_nodes(self, institutions: List[str]):
        """
        Create Institution nodes.

        Args:
            institutions: List of institution names
        """
        with self.driver.session() as session:
            for institution in institutions:
                if not institution:
                    continue

                query = """
                MERGE (i:Institution {name: $name})
                """

                session.run(query, name=institution)

    def create_authorship_relationships(
        self,
        authors: List[Dict],
        paper_pmid: str
    ):
        """
        Create AUTHORED relationships between authors and paper.

        Args:
            authors: List of author dictionaries
            paper_pmid: PMID of paper
        """
        with self.driver.session() as session:
            for author in authors:
                query = """
                MATCH (a:Author {author_id: $author_id})
                MATCH (p:Paper {pmid: $pmid})
                MERGE (a)-[r:AUTHORED]->(p)
                SET r.position = $position,
                    r.is_first_author = $is_first,
                    r.is_last_author = $is_last
                """

                session.run(
                    query,
                    author_id=author['author_id'],
                    pmid=paper_pmid,
                    position=author['position'],
                    is_first=author['is_first_author'],
                    is_last=author['is_last_author']
                )

    def create_affiliation_relationships(self, authors: List[Dict]):
        """
        Create AFFILIATED_WITH relationships.

        Args:
            authors: List of author dictionaries with affiliations
        """
        with self.driver.session() as session:
            for author in authors:
                for affil in author['affiliations']:
                    institution = affil.get('institution')
                    if not institution:
                        continue

                    query = """
                    MATCH (a:Author {author_id: $author_id})
                    MERGE (i:Institution {name: $institution})
                    MERGE (a)-[r:AFFILIATED_WITH]->(i)
                    SET r.pmid = $pmid,
                        r.year = $year
                    """

                    session.run(
                        query,
                        author_id=author['author_id'],
                        institution=institution,
                        pmid=author['pmid'],
                        year=author['year']
                    )

    def create_collaboration_relationships(self, co_authorships: List[Dict]):
        """
        Create COLLABORATED_WITH relationships between authors.

        Args:
            co_authorships: List of co-authorship dictionaries
        """
        with self.driver.session() as session:
            for collab in co_authorships:
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
                    author1_id=collab['author1_id'],
                    author2_id=collab['author2_id'],
                    pmid=collab['pmid']
                )

    def build_network(self, articles: List[Dict]):
        """
        Build complete network from articles.

        Processes articles in batch, creating all nodes and relationships.

        Args:
            articles: List of article dictionaries
        """
        logger.info(f"Building network from {len(articles)} articles...")

        for i, article in enumerate(articles):
            if i % 100 == 0:
                logger.info(f"Processing article {i+1}/{len(articles)}")

            try:
                # Extract authors
                authors = self.author_extractor.extract_authors(article)

                # Create nodes
                self.create_paper_node(article)
                self.create_author_nodes(authors)

                # Extract institutions
                institutions = []
                for author in authors:
                    for affil in author['affiliations']:
                        inst = affil.get('institution')
                        if inst:
                            institutions.append(inst)

                self.create_institution_nodes(institutions)

                # Create relationships
                self.create_authorship_relationships(authors, article['pmid'])
                self.create_affiliation_relationships(authors)

                # Create collaborations
                co_authorships = self.author_extractor.extract_co_authorship(article)
                self.create_collaboration_relationships(co_authorships)

            except Exception as e:
                logger.error(f"Failed to process article {article.get('pmid')}: {e}")
                continue

        logger.info(f"✓ Network built: {len(articles)} articles processed")

    def incremental_update(self, new_articles: List[Dict]):
        """
        Add new articles to existing network.

        More efficient than rebuilding entire network.

        Args:
            new_articles: List of new article dictionaries
        """
        logger.info(f"Incrementally adding {len(new_articles)} new articles...")

        self.build_network(new_articles)

    def get_network_stats(self) -> Dict:
        """
        Get network statistics.

        Returns:
            Dictionary with node and relationship counts
        """
        with self.driver.session() as session:
            # Count nodes
            author_count = session.run("MATCH (a:Author) RETURN count(a) as count").single()['count']
            paper_count = session.run("MATCH (p:Paper) RETURN count(p) as count").single()['count']
            institution_count = session.run("MATCH (i:Institution) RETURN count(i) as count").single()['count']

            # Count relationships
            authorship_count = session.run(
                "MATCH ()-[r:AUTHORED]->() RETURN count(r) as count"
            ).single()['count']

            collaboration_count = session.run(
                "MATCH ()-[r:COLLABORATED_WITH]-() RETURN count(r) as count"
            ).single()['count']

            return {
                'authors': author_count,
                'papers': paper_count,
                'institutions': institution_count,
                'authorships': authorship_count,
                'collaborations': collaboration_count
            }

    def close(self):
        """Close Neo4j driver connection."""
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
