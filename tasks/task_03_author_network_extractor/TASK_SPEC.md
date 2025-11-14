# Task 3: Author Network Extractor - TDD Implementation

## Executive Summary

Implement a production-ready Author Network Extractor that builds a co-authorship knowledge graph from PubMed article metadata. This component will extract author collaborations, institutional affiliations, and research relationships, storing them in a Neo4j graph database for network analysis and visualization.

**Key Requirements:**
- Extract author information from PubMed article JSON
- Build co-authorship networks (who collaborates with whom)
- Map institutional affiliations and geographic locations
- Create knowledge graph in Neo4j with nodes and relationships
- Calculate network metrics (centrality, clustering, communities)
- Handle author disambiguation (same name, different people)
- Implement incremental updates (add new articles without rebuilding entire graph)
- Follow Test-Driven Development (TDD) principles

## Background & Context

### Author Network Analysis

Co-authorship networks reveal:
- **Collaboration patterns**: Who works together
- **Research hubs**: Central authors/institutions
- **Knowledge flow**: How ideas spread between groups
- **Communities**: Natural research clusters
- **Bridge authors**: Connect disparate groups

### Graph Database (Neo4j)

Neo4j stores data as nodes and relationships:

**Nodes** (entities):
- Author
- Institution
- Paper
- Topic/Cluster

**Relationships** (edges):
- AUTHORED (Author → Paper)
- AFFILIATED_WITH (Author → Institution)
- COLLABORATED_WITH (Author → Author)
- CITES (Paper → Paper)
- BELONGS_TO (Paper → Topic)

### Author Disambiguation Challenge

Common problem: "J. Smith" could be multiple people. Strategies:
1. **ORCID matching**: Use ORCID IDs when available
2. **Affiliation matching**: Same name + same institution = likely same person
3. **Co-author overlap**: Authors who share many co-authors are likely same
4. **Time-based clustering**: Active in same time period

## Technical Architecture

### Module Structure

```
src/network/
├── __init__.py
├── author_extractor.py          # Extract authors from articles
├── affiliation_parser.py        # Parse affiliation strings
├── author_disambiguator.py      # Resolve author identities
├── network_builder.py           # Build Neo4j graph
├── network_analyzer.py          # Calculate metrics
└── geo_locator.py               # Geographic coordinate lookup

tests/
├── __init__.py
├── test_author_extractor.py
├── test_affiliation_parser.py
├── test_author_disambiguator.py
├── test_network_builder.py
├── test_network_analyzer.py
└── fixtures/
    ├── sample_articles.json     # Sample data
    └── sample_graph.cypher      # Sample Cypher queries
```

### Dependencies

```python
# Required packages
neo4j>=5.14            # Neo4j driver
pydantic>=2.0          # Data validation
loguru>=0.7            # Logging
geopy>=2.4             # Geocoding
thefuzz>=0.20          # Fuzzy string matching
networkx>=3.2          # Network analysis (backup/testing)
pytest>=7.4            # Testing
pytest-asyncio>=0.21   # Async testing
pytest-mock>=3.12      # Mocking
```

## TDD Implementation Plan

### Phase 1: Affiliation Parser (TDD)

#### Test Cases

```python
# tests/test_affiliation_parser.py

import pytest
from src.network.affiliation_parser import AffiliationParser

@pytest.fixture
def parser():
    return AffiliationParser()

def test_parse_simple_affiliation(parser):
    """Should parse simple affiliation string."""
    affil = "Boston Children's Hospital, Boston, MA, USA"

    result = parser.parse(affil)

    assert result['institution'] == "Boston Children's Hospital"
    assert result['city'] == 'Boston'
    assert result['state'] == 'MA'
    assert result['country'] == 'USA'

def test_parse_with_department(parser):
    """Should extract department information."""
    affil = "Department of Pediatrics, Harvard Medical School, Boston, MA"

    result = parser.parse(affil)

    assert result['department'] == 'Department of Pediatrics'
    assert result['institution'] == 'Harvard Medical School'

def test_parse_multiple_institutions(parser):
    """Should handle multiple institutions in one string."""
    affil = "Dept of Medicine, Hospital A; Dept of Surgery, Hospital B"

    results = parser.parse_multiple(affil)

    assert len(results) == 2
    assert 'Hospital A' in results[0]['institution']
    assert 'Hospital B' in results[1]['institution']

def test_normalize_institution_name(parser):
    """Should normalize institution names."""
    assert parser.normalize("BCH") == "Boston Children's Hospital"
    assert parser.normalize("MGH") == "Massachusetts General Hospital"
    assert parser.normalize("Harvard Med School") == "Harvard Medical School"

def test_extract_country(parser):
    """Should extract country from various formats."""
    assert parser.extract_country("Boston, MA, USA") == "USA"
    assert parser.extract_country("London, UK") == "UK"
    assert parser.extract_country("Beijing, China") == "China"

def test_handle_malformed_affiliation(parser):
    """Should handle poorly formatted strings gracefully."""
    affil = "Some random text without clear structure"

    result = parser.parse(affil)

    # Should still extract what it can
    assert result['institution'] is not None or result['raw'] == affil
```

#### Implementation

```python
# src/network/affiliation_parser.py

import re
from typing import Dict, List, Optional
from thefuzz import fuzz
from loguru import logger

class AffiliationParser:
    """
    Parse affiliation strings to extract structured information.

    Handles various affiliation formats from PubMed.
    Extracts: institution, department, city, state, country.

    Example:
        >>> parser = AffiliationParser()
        >>> result = parser.parse(
        ...     "Dept of Pediatrics, Boston Children's Hospital, Boston, MA, USA"
        ... )
        >>> result['institution']
        "Boston Children's Hospital"
    """

    # Common institution abbreviations
    INSTITUTION_MAP = {
        'BCH': "Boston Children's Hospital",
        'MGH': 'Massachusetts General Hospital',
        'BWH': 'Brigham and Women\'s Hospital',
        'HMS': 'Harvard Medical School',
        'MIT': 'Massachusetts Institute of Technology',
        'NIH': 'National Institutes of Health',
        'CDC': 'Centers for Disease Control and Prevention'
    }

    # Country patterns
    COUNTRY_PATTERNS = [
        r',\s*(USA|United States)$',
        r',\s*(UK|United Kingdom)$',
        r',\s*China$',
        r',\s*Japan$',
        r',\s*Germany$',
        r',\s*France$',
        r',\s*Canada$'
    ]

    # Department keywords
    DEPT_KEYWORDS = [
        'Department', 'Dept', 'Division', 'Div',
        'Center', 'Centre', 'Institute', 'School'
    ]

    def parse(self, affiliation: str) -> Dict:
        """
        Parse affiliation string to structured dict.

        Args:
            affiliation: Raw affiliation string

        Returns:
            Dictionary with keys: institution, department, city, state, country, raw
        """
        if not affiliation:
            return self._empty_result(affiliation)

        # Split by common separators
        parts = re.split(r'[;,]\s*', affiliation)

        # Extract components
        institution = None
        department = None
        city = None
        state = None
        country = self.extract_country(affiliation)

        # Identify department (usually first part)
        if parts and any(kw in parts[0] for kw in self.DEPT_KEYWORDS):
            department = parts[0].strip()
            remaining_parts = parts[1:]
        else:
            remaining_parts = parts

        # Identify institution (usually has "Hospital", "University", "Institute")
        institution_keywords = ['Hospital', 'University', 'Institute', 'College', 'School']

        for part in remaining_parts:
            if any(kw in part for kw in institution_keywords):
                institution = part.strip()
                break

        # Extract city and state (usually last 2-3 parts before country)
        if len(remaining_parts) >= 2:
            # Common pattern: "City, STATE" or "City, STATE, Country"
            state_pattern = r'\b[A-Z]{2}\b'  # Two capital letters (US states)

            for i, part in enumerate(remaining_parts[-3:]):
                if re.search(state_pattern, part):
                    state = re.search(state_pattern, part).group()
                    city = remaining_parts[-(3-i)-1].strip() if i > 0 else None

        return {
            'institution': institution,
            'department': department,
            'city': city,
            'state': state,
            'country': country,
            'raw': affiliation
        }

    def parse_multiple(self, affiliation: str) -> List[Dict]:
        """
        Parse affiliation string with multiple institutions.

        Some authors list multiple affiliations separated by semicolons.

        Args:
            affiliation: Affiliation string (may contain multiple)

        Returns:
            List of parsed affiliation dicts
        """
        # Split by semicolon (common separator for multiple affiliations)
        affiliations = affiliation.split(';')

        return [self.parse(a.strip()) for a in affiliations if a.strip()]

    def normalize(self, name: str) -> str:
        """
        Normalize institution name.

        Expands abbreviations and standardizes names.

        Args:
            name: Institution name or abbreviation

        Returns:
            Normalized name
        """
        # Check if it's a known abbreviation
        if name in self.INSTITUTION_MAP:
            return self.INSTITUTION_MAP[name]

        # Remove common suffixes for matching
        normalized = name.strip()

        # Fuzzy match against known institutions
        best_match = None
        best_score = 0

        for abbr, full_name in self.INSTITUTION_MAP.items():
            score = fuzz.ratio(normalized.lower(), full_name.lower())
            if score > best_score:
                best_score = score
                best_match = full_name

        # Use fuzzy match if score > 85
        if best_score > 85:
            return best_match

        return normalized

    def extract_country(self, affiliation: str) -> Optional[str]:
        """
        Extract country from affiliation string.

        Args:
            affiliation: Affiliation string

        Returns:
            Country name or None
        """
        for pattern in self.COUNTRY_PATTERNS:
            match = re.search(pattern, affiliation)
            if match:
                return match.group(1)

        # Check last part
        parts = affiliation.split(',')
        if parts:
            last_part = parts[-1].strip()
            # If last part is short and capitalized, likely country
            if len(last_part) < 30 and last_part[0].isupper():
                return last_part

        return None

    def _empty_result(self, raw: str) -> Dict:
        """Return empty result dict."""
        return {
            'institution': None,
            'department': None,
            'city': None,
            'state': None,
            'country': None,
            'raw': raw
        }
```

### Phase 2: Author Extractor (TDD)

#### Test Cases

```python
# tests/test_author_extractor.py

import pytest
from src.network.author_extractor import AuthorExtractor

@pytest.fixture
def extractor():
    return AuthorExtractor()

@pytest.fixture
def sample_article():
    return {
        'pmid': '12345678',
        'title': 'Novel CAR-T Therapy',
        'year': '2023',
        'authors': [
            {
                'last_name': 'Smith',
                'first_name': 'John',
                'initials': 'J',
                'affiliations': ["Boston Children's Hospital, Boston, MA, USA"]
            },
            {
                'last_name': 'Doe',
                'first_name': 'Jane',
                'initials': 'J',
                'affiliations': ["Harvard Medical School, Boston, MA"]
            },
            {
                'last_name': 'Brown',
                'first_name': 'Alice',
                'initials': 'A',
                'affiliations': [
                    "Boston Children's Hospital, Boston, MA",
                    "Harvard Medical School, Boston, MA"
                ]
            }
        ]
    }

def test_extract_authors_from_article(extractor, sample_article):
    """Should extract all authors with metadata."""
    authors = extractor.extract_authors(sample_article)

    assert len(authors) == 3
    assert authors[0]['full_name'] == 'John Smith'
    assert authors[0]['position'] == 0  # First author
    assert authors[2]['position'] == 2  # Third author

def test_extract_affiliations(extractor, sample_article):
    """Should extract affiliations for each author."""
    authors = extractor.extract_authors(sample_article)

    # First author
    assert len(authors[0]['affiliations']) == 1
    assert "Boston Children's Hospital" in authors[0]['affiliations'][0]['institution']

    # Third author (multiple affiliations)
    assert len(authors[2]['affiliations']) == 2

def test_extract_co_authors(extractor, sample_article):
    """Should identify co-author relationships."""
    co_authors = extractor.extract_co_authorship(sample_article)

    # Should have 3 pairs: (Smith,Doe), (Smith,Brown), (Doe,Brown)
    assert len(co_authors) == 3

    # Check structure
    assert co_authors[0]['author1'] == 'John Smith'
    assert co_authors[0]['author2'] == 'Jane Doe'
    assert co_authors[0]['pmid'] == '12345678'
    assert co_authors[0]['year'] == '2023'

def test_identify_first_and_last_authors(extractor, sample_article):
    """Should mark first and last author positions."""
    authors = extractor.extract_authors(sample_article)

    assert authors[0]['is_first_author'] is True
    assert authors[1]['is_first_author'] is False
    assert authors[2]['is_last_author'] is True

def test_handle_missing_author_info(extractor):
    """Should handle articles with incomplete author data."""
    article = {
        'pmid': '99999',
        'authors': [
            {'last_name': 'Unknown', 'first_name': ''},
            {'last_name': '', 'first_name': 'Mystery'}
        ]
    }

    authors = extractor.extract_authors(article)

    # Should still create author entries
    assert len(authors) == 2

def test_create_author_id(extractor):
    """Should create consistent author identifiers."""
    author1 = {
        'last_name': 'Smith',
        'first_name': 'John',
        'affiliations': ["Boston Children's Hospital"]
    }

    author2 = {
        'last_name': 'Smith',
        'first_name': 'John',
        'affiliations': ["Boston Children's Hospital"]
    }

    id1 = extractor.create_author_id(author1)
    id2 = extractor.create_author_id(author2)

    # Same author should get same ID
    assert id1 == id2
```

#### Implementation

```python
# src/network/author_extractor.py

from typing import List, Dict
import hashlib
from loguru import logger
from .affiliation_parser import AffiliationParser

class AuthorExtractor:
    """
    Extract author information and co-authorship relationships from articles.

    Creates structured author records and identifies collaboration patterns.

    Example:
        >>> extractor = AuthorExtractor()
        >>> authors = extractor.extract_authors(article)
        >>> co_authors = extractor.extract_co_authorship(article)
    """

    def __init__(self):
        self.affiliation_parser = AffiliationParser()

    def extract_authors(self, article: Dict) -> List[Dict]:
        """
        Extract author information from article.

        Args:
            article: Article dictionary with 'authors' field

        Returns:
            List of author dictionaries with enriched metadata
        """
        authors = []
        author_list = article.get('authors', [])
        total_authors = len(author_list)

        for i, author in enumerate(author_list):
            # Basic info
            last_name = author.get('last_name', '')
            first_name = author.get('first_name', '')
            initials = author.get('initials', '')

            full_name = f"{first_name} {last_name}".strip()
            if not full_name:
                full_name = f"{initials} {last_name}".strip()

            # Parse affiliations
            affiliations = []
            for affil_str in author.get('affiliations', []):
                parsed = self.affiliation_parser.parse(affil_str)
                affiliations.append(parsed)

            # Position metadata
            is_first = (i == 0)
            is_last = (i == total_authors - 1)

            # Create author record
            author_record = {
                'full_name': full_name,
                'last_name': last_name,
                'first_name': first_name,
                'initials': initials,
                'position': i,
                'is_first_author': is_first,
                'is_last_author': is_last,
                'affiliations': affiliations,
                'pmid': article.get('pmid'),
                'year': article.get('year'),
                'author_id': self.create_author_id({
                    'last_name': last_name,
                    'first_name': first_name,
                    'affiliations': author.get('affiliations', [])
                })
            }

            authors.append(author_record)

        return authors

    def extract_co_authorship(self, article: Dict) -> List[Dict]:
        """
        Extract co-authorship relationships from article.

        Creates pairwise collaborations between all authors.

        Args:
            article: Article dictionary

        Returns:
            List of co-author relationship dicts
        """
        authors = self.extract_authors(article)
        co_authorships = []

        # Create pairwise combinations
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                author1 = authors[i]
                author2 = authors[j]

                co_authorship = {
                    'author1': author1['full_name'],
                    'author1_id': author1['author_id'],
                    'author2': author2['full_name'],
                    'author2_id': author2['author_id'],
                    'pmid': article.get('pmid'),
                    'year': article.get('year'),
                    'title': article.get('title', '')
                }

                co_authorships.append(co_authorship)

        return co_authorships

    def create_author_id(self, author: Dict) -> str:
        """
        Create unique author identifier.

        Uses last name + first initial + primary affiliation.
        This is a simple disambiguation strategy.

        Args:
            author: Author dictionary

        Returns:
            Author ID string
        """
        last_name = author.get('last_name', '').lower()
        first_name = author.get('first_name', '')
        first_initial = first_name[0].lower() if first_name else ''

        # Use primary affiliation if available
        affiliations = author.get('affiliations', [])
        affiliation = ''
        if affiliations:
            # Parse first affiliation
            if isinstance(affiliations[0], str):
                parsed = self.affiliation_parser.parse(affiliations[0])
                affiliation = parsed.get('institution', '') or ''

        # Create ID: lastname_fi_institution
        id_string = f"{last_name}_{first_initial}_{affiliation}"

        # Hash for consistent length
        return hashlib.md5(id_string.encode()).hexdigest()[:16]

    def extract_institutional_collaborations(
        self,
        articles: List[Dict]
    ) -> List[Dict]:
        """
        Extract institution-level collaborations across articles.

        Identifies which institutions collaborate most frequently.

        Args:
            articles: List of article dictionaries

        Returns:
            List of institution collaboration records
        """
        collaborations = {}

        for article in articles:
            authors = self.extract_authors(article)

            # Get unique institutions from article
            institutions = set()
            for author in authors:
                for affil in author['affiliations']:
                    inst = affil.get('institution')
                    if inst:
                        institutions.add(inst)

            # Create pairwise collaborations
            inst_list = list(institutions)
            for i in range(len(inst_list)):
                for j in range(i + 1, len(inst_list)):
                    inst1, inst2 = sorted([inst_list[i], inst_list[j]])
                    key = (inst1, inst2)

                    if key not in collaborations:
                        collaborations[key] = {
                            'institution1': inst1,
                            'institution2': inst2,
                            'count': 0,
                            'pmids': []
                        }

                    collaborations[key]['count'] += 1
                    collaborations[key]['pmids'].append(article['pmid'])

        return list(collaborations.values())
```

### Phase 3: Network Builder (Neo4j) (TDD)

#### Test Cases

```python
# tests/test_network_builder.py

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
    return NetworkBuilder(driver=mock_neo4j_driver)

@pytest.fixture
def sample_authors():
    return [
        {
            'full_name': 'John Smith',
            'author_id': 'abc123',
            'affiliations': [
                {'institution': "Boston Children's Hospital"}
            ]
        },
        {
            'full_name': 'Jane Doe',
            'author_id': 'def456',
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
```

#### Implementation

```python
# src/network/network_builder.py

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
```

### Phase 4: Network Analysis (TDD)

#### Test Cases

```python
# tests/test_network_analyzer.py

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

    # Mock result
    mock_result = [
        Mock(data=lambda: {'author': 'John Smith', 'collaborations': 50}),
        Mock(data=lambda: {'author': 'Jane Doe', 'collaborations': 45})
    ]
    session.run.return_value = mock_result

    top_collab = analyzer.find_top_collaborators(limit=10)

    session.run.assert_called_once()
    assert len(top_collab) == 2

def test_find_institutional_collaborations(analyzer, mock_driver):
    """Should find top institution pairs."""
    session = mock_driver.session.return_value.__enter__.return_value

    mock_result = [
        Mock(data=lambda: {
            'inst1': 'BCH',
            'inst2': 'HMS',
            'count': 100
        })
    ]
    session.run.return_value = mock_result

    collabs = analyzer.find_institutional_collaborations(limit=10)

    assert len(collabs) == 1

def test_calculate_author_centrality(analyzer, mock_driver):
    """Should calculate degree centrality for authors."""
    session = mock_driver.session.return_value.__enter__.return_value

    mock_result = [
        Mock(data=lambda: {'author': 'John Smith', 'centrality': 0.85})
    ]
    session.run.return_value = mock_result

    centrality = analyzer.calculate_author_centrality()

    session.run.assert_called()
```

#### Implementation

```python
# src/network/network_analyzer.py

from typing import List, Dict, Optional
from neo4j import GraphDatabase
from loguru import logger

class NetworkAnalyzer:
    """
    Analyze co-authorship network.

    Calculate metrics:
    - Degree centrality (number of connections)
    - Betweenness centrality (bridge position)
    - PageRank (influence)
    - Community detection

    Args:
        driver: Neo4j driver

    Example:
        >>> analyzer = NetworkAnalyzer(driver)
        >>> top_authors = analyzer.find_top_collaborators(limit=20)
    """

    def __init__(self, driver):
        self.driver = driver

    def find_top_collaborators(self, limit: int = 20) -> List[Dict]:
        """
        Find authors with most unique collaborators.

        Args:
            limit: Number of results to return

        Returns:
            List of authors with collaboration counts
        """
        with self.driver.session() as session:
            query = """
            MATCH (a:Author)-[r:COLLABORATED_WITH]-(other:Author)
            RETURN a.full_name as author,
                   a.author_id as author_id,
                   count(DISTINCT other) as collaborations
            ORDER BY collaborations DESC
            LIMIT $limit
            """

            result = session.run(query, limit=limit)

            return [dict(record) for record in result]

    def find_institutional_collaborations(
        self,
        institution: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        Find top institution collaboration pairs.

        Args:
            institution: Focus on specific institution (optional)
            limit: Number of results

        Returns:
            List of institution pairs with collaboration counts
        """
        with self.driver.session() as session:
            if institution:
                query = """
                MATCH (i1:Institution {name: $institution})<-[:AFFILIATED_WITH]-(a1:Author)
                MATCH (a1)-[:COLLABORATED_WITH]-(a2:Author)-[:AFFILIATED_WITH]->(i2:Institution)
                WHERE i1.name <> i2.name
                RETURN i1.name as institution1,
                       i2.name as institution2,
                       count(*) as collaboration_count
                ORDER BY collaboration_count DESC
                LIMIT $limit
                """
                result = session.run(query, institution=institution, limit=limit)
            else:
                query = """
                MATCH (i1:Institution)<-[:AFFILIATED_WITH]-(a1:Author)
                MATCH (a1)-[:COLLABORATED_WITH]-(a2:Author)-[:AFFILIATED_WITH]->(i2:Institution)
                WHERE id(i1) < id(i2)
                RETURN i1.name as institution1,
                       i2.name as institution2,
                       count(*) as collaboration_count
                ORDER BY collaboration_count DESC
                LIMIT $limit
                """
                result = session.run(query, limit=limit)

            return [dict(record) for record in result]

    def calculate_author_centrality(self) -> List[Dict]:
        """
        Calculate degree centrality for all authors.

        Uses Neo4j Graph Data Science library if available.

        Returns:
            List of authors with centrality scores
        """
        with self.driver.session() as session:
            # Simple degree centrality query
            query = """
            MATCH (a:Author)-[r:COLLABORATED_WITH]-(other)
            RETURN a.full_name as author,
                   a.author_id as author_id,
                   count(r) as degree
            ORDER BY degree DESC
            """

            result = session.run(query)

            return [dict(record) for record in result]

    def find_research_communities(self, min_size: int = 5) -> List[Dict]:
        """
        Identify research communities using community detection.

        Args:
            min_size: Minimum community size

        Returns:
            List of communities with member authors
        """
        # This requires Neo4j GDS library
        # Placeholder implementation
        logger.warning("Community detection requires Neo4j GDS library")
        return []

    def get_author_network(self, author_name: str, depth: int = 2) -> Dict:
        """
        Get author's collaboration network.

        Args:
            author_name: Author full name
            depth: Degree of separation (1 = direct collaborators, 2 = collaborators of collaborators)

        Returns:
            Network data (nodes and edges)
        """
        with self.driver.session() as session:
            query = """
            MATCH path = (a:Author {full_name: $name})-[:COLLABORATED_WITH*1..$depth]-(other:Author)
            RETURN nodes(path) as nodes, relationships(path) as edges
            """

            result = session.run(query, name=author_name, depth=depth)

            nodes = set()
            edges = []

            for record in result:
                for node in record['nodes']:
                    nodes.add((node['author_id'], node['full_name']))

                for edge in record['edges']:
                    edges.append(edge)

            return {
                'nodes': [{'id': n[0], 'name': n[1]} for n in nodes],
                'edges': edges
            }
```

## Usage Examples

### Build Network from Articles

```python
import asyncio
import json
from src.network.network_builder import NetworkBuilder

async def main():
    # Load articles
    with open('data/pubmed_articles.json') as f:
        articles = json.load(f)

    # Build network
    with NetworkBuilder(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    ) as builder:
        builder.build_network(articles)

        # Get stats
        stats = builder.get_network_stats()
        print(f"Network: {stats}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Analyze Network

```python
from src.network.network_analyzer import NetworkAnalyzer
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

analyzer = NetworkAnalyzer(driver)

# Top collaborators
top = analyzer.find_top_collaborators(limit=20)
for author in top:
    print(f"{author['author']}: {author['collaborations']} collaborations")

# Institutional collaborations
inst_collab = analyzer.find_institutional_collaborations(
    institution="Boston Children's Hospital",
    limit=10
)

for collab in inst_collab:
    print(f"{collab['institution1']} <-> {collab['institution2']}: {collab['collaboration_count']}")
```

## Success Criteria

✅ **Must Have:**
1. All unit tests passing (>90% coverage)
2. Parse author affiliations correctly
3. Build Neo4j graph with nodes and relationships
4. Handle co-authorship extraction
5. Calculate basic network metrics
6. Support incremental updates

✅ **Should Have:**
7. Author disambiguation (basic)
8. Institution normalization
9. Network visualization export
10. Community detection

✅ **Nice to Have:**
11. ORCID integration
12. Geographic mapping
13. Temporal analysis (evolution over time)

## Deliverables

1. **Source code**
2. **Tests** (>90% coverage)
3. **Sample network** (Neo4j database dump)
4. **Documentation** and usage examples
5. **Cypher queries** for common analyses

## Environment Setup

```bash
# Install Neo4j (via Docker)
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Install Python dependencies
pip install neo4j pydantic loguru geopy thefuzz networkx pytest

# Run tests
pytest tests/ -v --cov=src/network
```

## Neo4j Browser Queries

```cypher
// Count nodes
MATCH (n) RETURN labels(n) as type, count(n) as count

// Top collaborators
MATCH (a:Author)-[r:COLLABORATED_WITH]-()
RETURN a.full_name, count(r) as collaborations
ORDER BY collaborations DESC
LIMIT 20

// Institutional network
MATCH (i1:Institution)<-[:AFFILIATED_WITH]-(a1:Author)
MATCH (a1)-[:COLLABORATED_WITH]-(a2:Author)-[:AFFILIATED_WITH]->(i2:Institution)
WHERE i1.name <> i2.name
RETURN i1.name, i2.name, count(*) as collabs
ORDER BY collabs DESC
LIMIT 20
```

---

**Task completion**: When all tests pass, network builds successfully in Neo4j, and basic network metrics can be calculated.
