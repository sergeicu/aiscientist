# Task 1: PubMed Institutional Scraper - TDD Implementation

## Executive Summary

Implement a production-ready PubMed scraper that searches and retrieves biomedical articles by institution/affiliation using NCBI's E-utilities API. This scraper will be a core component of the AI Scientist platform's data acquisition layer.

**Key Requirements:**
- Search PubMed by affiliation (e.g., "Boston Children's Hospital[Affiliation]")
- Respect NCBI API rate limits (3 req/s without API key, 10 req/s with key)
- Handle large result sets (10,000+ articles) using history server
- Parse XML responses into structured JSON
- Extract comprehensive metadata: authors, affiliations, citations, MeSH terms
- Implement robust error handling and retry logic
- Follow Test-Driven Development (TDD) principles

## Background & Context

### PubMed E-utilities API

PubMed provides a free API called E-utilities for programmatic access:
- **Base URL**: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`
- **Key endpoints**:
  - `esearch.fcgi` - Search for PMIDs
  - `efetch.fcgi` - Fetch article details
  - `esummary.fcgi` - Get article summaries
- **Authentication**: Optional API key (recommended, increases rate limit)
- **Documentation**: https://www.ncbi.nlm.nih.gov/books/NBK25501/

### Rate Limiting Requirements

**Without API key**: Maximum 3 requests per second
**With API key**: Maximum 10 requests per second

Exceeding limits results in IP blocking. Implement token bucket rate limiter.

### Affiliation Search Syntax

PubMed affiliation searches use special syntax:
```
"Boston Children's Hospital"[Affiliation]
"Harvard Medical School"[Affiliation]
"Massachusetts General Hospital"[Affiliation] AND "2020:2024"[PDAT]
```

## Technical Architecture

### Module Structure

```
src/scrapers/
├── __init__.py
├── rate_limiter.py          # Token bucket rate limiter
├── pubmed_scraper.py        # Main scraper class
├── pubmed_parser.py         # XML parsing utilities
└── exceptions.py            # Custom exception classes

tests/
├── __init__.py
├── test_rate_limiter.py
├── test_pubmed_scraper.py
├── test_pubmed_parser.py
└── fixtures/
    ├── sample_esearch.xml   # Sample API responses
    └── sample_efetch.xml
```

### Dependencies

```python
# Required packages
biopython>=1.81      # Provides Entrez utilities
aiohttp>=3.9         # Async HTTP client
loguru>=0.7          # Logging
pydantic>=2.0        # Data validation
pytest>=7.4          # Testing
pytest-asyncio>=0.21 # Async testing
pytest-mock>=3.12    # Mocking
responses>=0.24      # HTTP mocking
```

## TDD Implementation Plan

### Phase 1: Rate Limiter (TDD)

#### Test Cases

```python
# tests/test_rate_limiter.py

import pytest
import asyncio
import time
from src.scrapers.rate_limiter import RateLimiter

@pytest.mark.asyncio
async def test_rate_limiter_allows_burst():
    """Should allow burst_size requests immediately."""
    limiter = RateLimiter(requests_per_second=3.0, burst_size=10)

    start = time.time()
    for _ in range(10):
        await limiter.acquire()
    elapsed = time.time() - start

    # Should complete almost instantly (< 0.1s)
    assert elapsed < 0.1

@pytest.mark.asyncio
async def test_rate_limiter_enforces_rate():
    """Should enforce rate limit after burst exhausted."""
    limiter = RateLimiter(requests_per_second=3.0, burst_size=3)

    # Exhaust burst
    for _ in range(3):
        await limiter.acquire()

    # Next 3 requests should take ~1 second
    start = time.time()
    for _ in range(3):
        await limiter.acquire()
    elapsed = time.time() - start

    # Should take approximately 1 second (3 req / 3 req/s)
    assert 0.9 < elapsed < 1.2

@pytest.mark.asyncio
async def test_rate_limiter_refills_tokens():
    """Tokens should refill over time."""
    limiter = RateLimiter(requests_per_second=10.0, burst_size=10)

    # Exhaust tokens
    for _ in range(10):
        await limiter.acquire()

    # Wait for refill
    await asyncio.sleep(0.5)  # Should refill 5 tokens

    # Should be able to make 5 more requests quickly
    start = time.time()
    for _ in range(5):
        await limiter.acquire()
    elapsed = time.time() - start

    assert elapsed < 0.1

@pytest.mark.asyncio
async def test_rate_limiter_concurrent_safety():
    """Should be thread-safe for concurrent requests."""
    limiter = RateLimiter(requests_per_second=10.0, burst_size=20)

    async def make_request():
        await limiter.acquire()
        return True

    # 50 concurrent requests
    tasks = [make_request() for _ in range(50)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 50
    assert all(results)
```

#### Implementation

```python
# src/scrapers/rate_limiter.py

import asyncio
import time
from typing import Optional

class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Allows burst traffic while maintaining average rate limit.
    Thread-safe for async operations.

    Args:
        requests_per_second: Average rate limit
        burst_size: Maximum burst capacity

    Example:
        >>> limiter = RateLimiter(requests_per_second=3.0, burst_size=10)
        >>> await limiter.acquire()  # Will wait if rate exceeded
    """

    def __init__(
        self,
        requests_per_second: float = 3.0,
        burst_size: int = 10
    ):
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        if burst_size <= 0:
            raise ValueError("burst_size must be positive")

        self.rate = requests_per_second
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Acquire permission to make a request.

        Blocks if rate limit would be exceeded.
        """
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update

            # Add tokens based on elapsed time
            self.tokens = min(
                self.burst_size,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now

            # Wait if insufficient tokens
            if self.tokens < 1.0:
                wait_time = (1.0 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0

    def reset(self) -> None:
        """Reset to full burst capacity."""
        self.tokens = float(self.burst_size)
        self.last_update = time.time()
```

### Phase 2: XML Parser (TDD)

#### Test Cases

```python
# tests/test_pubmed_parser.py

import pytest
from pathlib import Path
from src.scrapers.pubmed_parser import PubMedParser

@pytest.fixture
def sample_article_xml():
    """Load sample PubMed XML."""
    return '''
    <PubmedArticle>
        <MedlineCitation Status="MEDLINE" Owner="NLM">
            <PMID Version="1">12345678</PMID>
            <Article PubModel="Print">
                <ArticleTitle>A Novel CAR-T Therapy for Pediatric Leukemia</ArticleTitle>
                <Abstract>
                    <AbstractText>This study demonstrates...</AbstractText>
                </Abstract>
                <AuthorList CompleteYN="Y">
                    <Author ValidYN="Y">
                        <LastName>Smith</LastName>
                        <ForeName>John</ForeName>
                        <Initials>J</Initials>
                        <AffiliationInfo>
                            <Affiliation>Boston Children's Hospital, Boston, MA, USA</Affiliation>
                        </AffiliationInfo>
                    </Author>
                    <Author ValidYN="Y">
                        <LastName>Doe</LastName>
                        <ForeName>Jane</ForeName>
                        <Initials>J</Initials>
                        <AffiliationInfo>
                            <Affiliation>Harvard Medical School, Boston, MA, USA</Affiliation>
                        </AffiliationInfo>
                    </Author>
                </AuthorList>
                <Journal>
                    <Title>Nature Medicine</Title>
                    <JournalIssue CitedMedium="Internet">
                        <Volume>29</Volume>
                        <Issue>3</Issue>
                        <PubDate>
                            <Year>2023</Year>
                            <Month>Mar</Month>
                        </PubDate>
                    </JournalIssue>
                </Journal>
                <ELocationID EIdType="doi" ValidYN="Y">10.1038/s41591-023-12345-6</ELocationID>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
    '''

def test_parse_basic_metadata(sample_article_xml):
    """Should extract PMID, title, abstract."""
    parser = PubMedParser()
    article = parser.parse_article(sample_article_xml)

    assert article['pmid'] == '12345678'
    assert 'CAR-T Therapy' in article['title']
    assert 'demonstrates' in article['abstract']

def test_parse_authors(sample_article_xml):
    """Should extract all authors with affiliations."""
    parser = PubMedParser()
    article = parser.parse_article(sample_article_xml)

    assert len(article['authors']) == 2
    assert article['authors'][0]['last_name'] == 'Smith'
    assert article['authors'][0]['first_name'] == 'John'
    assert "Boston Children's Hospital" in article['authors'][0]['affiliations'][0]

def test_parse_journal_info(sample_article_xml):
    """Should extract journal, volume, issue, date."""
    parser = PubMedParser()
    article = parser.parse_article(sample_article_xml)

    assert article['journal'] == 'Nature Medicine'
    assert article['year'] == '2023'
    assert article['volume'] == '29'

def test_parse_doi(sample_article_xml):
    """Should extract DOI."""
    parser = PubMedParser()
    article = parser.parse_article(sample_article_xml)

    assert article['doi'] == '10.1038/s41591-023-12345-6'

def test_handle_missing_fields():
    """Should handle articles with missing fields gracefully."""
    minimal_xml = '''
    <PubmedArticle>
        <MedlineCitation>
            <PMID>99999</PMID>
            <Article>
                <ArticleTitle>Minimal Article</ArticleTitle>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
    '''

    parser = PubMedParser()
    article = parser.parse_article(minimal_xml)

    assert article['pmid'] == '99999'
    assert article['abstract'] == ''
    assert article['authors'] == []
    assert article['doi'] is None
```

#### Implementation

```python
# src/scrapers/pubmed_parser.py

from typing import Dict, List, Optional
from xml.etree import ElementTree as ET
from loguru import logger

class PubMedParser:
    """
    Parse PubMed XML responses to structured dictionaries.

    Handles both ESearch and EFetch XML formats.
    Robust to missing fields and malformed data.
    """

    @staticmethod
    def parse_article(xml_string: str) -> Dict:
        """
        Parse single PubMed article XML.

        Args:
            xml_string: XML string from efetch

        Returns:
            Dictionary with article metadata
        """
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
            raise ValueError(f"Invalid XML: {e}")

        # Navigate XML structure
        article_elem = root.find('.//Article')
        medline_elem = root.find('.//MedlineCitation')

        if article_elem is None or medline_elem is None:
            raise ValueError("Invalid PubMed XML structure")

        # Extract basic metadata
        pmid = PubMedParser._get_text(medline_elem, 'PMID', '')
        title = PubMedParser._get_text(article_elem, 'ArticleTitle', '')

        # Abstract (can be multiple AbstractText elements)
        abstract_parts = article_elem.findall('.//AbstractText')
        abstract = ' '.join(
            elem.text for elem in abstract_parts if elem.text
        )

        # Journal info
        journal_elem = article_elem.find('.//Journal')
        journal = PubMedParser._get_text(journal_elem, 'Title', '') if journal_elem else ''

        pub_date = journal_elem.find('.//PubDate') if journal_elem else None
        year = PubMedParser._get_text(pub_date, 'Year', '') if pub_date else ''

        # Volume/Issue
        journal_issue = journal_elem.find('.//JournalIssue') if journal_elem else None
        volume = PubMedParser._get_text(journal_issue, 'Volume', '') if journal_issue else ''

        # Authors
        authors = PubMedParser._parse_authors(article_elem)

        # DOI
        doi = PubMedParser._extract_doi(article_elem)

        # MeSH terms
        mesh_terms = PubMedParser._extract_mesh_terms(medline_elem)

        return {
            'pmid': pmid,
            'title': title,
            'abstract': abstract,
            'journal': journal,
            'year': year,
            'volume': volume,
            'doi': doi,
            'authors': authors,
            'mesh_terms': mesh_terms
        }

    @staticmethod
    def _parse_authors(article_elem: ET.Element) -> List[Dict]:
        """Extract author list with affiliations."""
        authors = []
        author_list = article_elem.find('.//AuthorList')

        if author_list is None:
            return []

        for author_elem in author_list.findall('Author'):
            author = {
                'last_name': PubMedParser._get_text(author_elem, 'LastName', ''),
                'first_name': PubMedParser._get_text(author_elem, 'ForeName', ''),
                'initials': PubMedParser._get_text(author_elem, 'Initials', ''),
                'affiliations': []
            }

            # Extract affiliations
            affil_list = author_elem.findall('.//AffiliationInfo')
            for affil_elem in affil_list:
                affil_text = PubMedParser._get_text(affil_elem, 'Affiliation', '')
                if affil_text:
                    author['affiliations'].append(affil_text)

            authors.append(author)

        return authors

    @staticmethod
    def _extract_doi(article_elem: ET.Element) -> Optional[str]:
        """Extract DOI from ELocationID elements."""
        for eloc in article_elem.findall('.//ELocationID'):
            if eloc.get('EIdType') == 'doi':
                return eloc.text
        return None

    @staticmethod
    def _extract_mesh_terms(medline_elem: ET.Element) -> List[str]:
        """Extract MeSH (Medical Subject Headings) terms."""
        mesh_list = medline_elem.find('.//MeshHeadingList')
        if mesh_list is None:
            return []

        terms = []
        for mesh_heading in mesh_list.findall('MeshHeading'):
            descriptor = mesh_heading.find('DescriptorName')
            if descriptor is not None and descriptor.text:
                terms.append(descriptor.text)

        return terms

    @staticmethod
    def _get_text(element: Optional[ET.Element], tag: str, default: str = '') -> str:
        """Safely get text from XML element."""
        if element is None:
            return default

        child = element.find(tag)
        if child is not None and child.text:
            return child.text.strip()

        return default
```

### Phase 3: PubMed Scraper (TDD)

#### Test Cases

```python
# tests/test_pubmed_scraper.py

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.scrapers.pubmed_scraper import PubMedScraper
from src.scrapers.exceptions import RateLimitError, PubMedAPIError

@pytest.fixture
def scraper():
    """Create scraper instance."""
    return PubMedScraper(
        email="test@example.com",
        api_key="test_key"
    )

@pytest.mark.asyncio
async def test_search_by_affiliation_returns_pmids(scraper, mocker):
    """Should return list of PMIDs for affiliation search."""
    # Mock Entrez.esearch
    mock_search = mocker.patch('Bio.Entrez.esearch')
    mock_search.return_value.__enter__ = Mock(return_value=Mock())

    mocker.patch('Bio.Entrez.read', return_value={
        'IdList': ['123', '456', '789'],
        'Count': '3',
        'WebEnv': '',
        'QueryKey': ''
    })

    pmids = await scraper.search_by_affiliation(
        "Boston Children's Hospital[Affiliation]",
        max_results=100
    )

    assert len(pmids) == 3
    assert '123' in pmids

@pytest.mark.asyncio
async def test_search_with_date_range(scraper, mocker):
    """Should construct query with date range."""
    mock_search = mocker.patch('Bio.Entrez.esearch')

    await scraper.search_by_affiliation(
        "Harvard Medical School[Affiliation]",
        date_range="2020:2024[PDAT]"
    )

    # Verify query includes date range
    call_args = mock_search.call_args
    assert "2020:2024[PDAT]" in str(call_args)

@pytest.mark.asyncio
async def test_handles_large_result_sets(scraper, mocker):
    """Should use history server for >10k results."""
    mocker.patch('Bio.Entrez.esearch', return_value=Mock())
    mocker.patch('Bio.Entrez.read', return_value={
        'IdList': [],
        'Count': '25000',  # Large result set
        'WebEnv': 'test_webenv',
        'QueryKey': '1'
    })

    mock_fetch_all = mocker.patch.object(
        scraper,
        '_fetch_all_pmids',
        return_value=['1', '2', '3']
    )

    pmids = await scraper.search_by_affiliation(
        "Test Hospital[Affiliation]",
        max_results=25000
    )

    # Should call _fetch_all_pmids for large sets
    mock_fetch_all.assert_called_once()

@pytest.mark.asyncio
async def test_fetch_article_details_batches(scraper, mocker):
    """Should fetch articles in batches."""
    pmids = [str(i) for i in range(500)]  # 500 PMIDs

    mock_fetch = mocker.patch('Bio.Entrez.efetch', return_value=Mock())
    mocker.patch('Bio.Entrez.read', return_value={
        'PubmedArticle': [
            {'MedlineCitation': {'PMID': i, 'Article': {'ArticleTitle': f'Title {i}'}}}
            for i in range(200)
        ]
    })

    articles = await scraper.fetch_article_details(pmids, batch_size=200)

    # Should make 3 calls (500/200 = 3)
    assert mock_fetch.call_count == 3

@pytest.mark.asyncio
async def test_respects_rate_limit(scraper, mocker):
    """Should use rate limiter for all API calls."""
    mock_acquire = mocker.patch.object(
        scraper.rate_limiter,
        'acquire',
        new_callable=AsyncMock
    )

    mocker.patch('Bio.Entrez.esearch', return_value=Mock())
    mocker.patch('Bio.Entrez.read', return_value={
        'IdList': ['123'],
        'Count': '1',
        'WebEnv': '',
        'QueryKey': ''
    })

    await scraper.search_by_affiliation("Test[Affiliation]")

    # Should acquire rate limit permission
    mock_acquire.assert_called()

@pytest.mark.asyncio
async def test_retry_on_network_error(scraper, mocker):
    """Should retry on transient network errors."""
    mock_search = mocker.patch('Bio.Entrez.esearch')

    # Fail twice, succeed third time
    mock_search.side_effect = [
        Exception("Network error"),
        Exception("Network error"),
        Mock()
    ]

    mocker.patch('Bio.Entrez.read', return_value={
        'IdList': ['123'],
        'Count': '1',
        'WebEnv': '',
        'QueryKey': ''
    })

    # Should eventually succeed
    pmids = await scraper.search_by_affiliation("Test[Affiliation]")
    assert len(pmids) == 1

@pytest.mark.asyncio
async def test_scrape_and_save_workflow(scraper, mocker, tmp_path):
    """Should complete full scrape-to-file workflow."""
    # Mock search
    mocker.patch.object(
        scraper,
        'search_by_affiliation',
        return_value=['123', '456']
    )

    # Mock fetch
    mocker.patch.object(
        scraper,
        'fetch_article_details',
        return_value=[
            {'pmid': '123', 'title': 'Article 1'},
            {'pmid': '456', 'title': 'Article 2'}
        ]
    )

    output_file = tmp_path / "articles.json"

    await scraper.scrape_and_save(
        affiliation="Test[Affiliation]",
        output_file=str(output_file),
        max_results=100
    )

    # Should create output file
    assert output_file.exists()

    # Should contain articles
    import json
    with open(output_file) as f:
        data = json.load(f)

    assert len(data) == 2
    assert data[0]['pmid'] == '123'
```

#### Implementation

```python
# src/scrapers/pubmed_scraper.py

from Bio import Entrez
import asyncio
import json
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger

from .rate_limiter import RateLimiter
from .pubmed_parser import PubMedParser
from .exceptions import PubMedAPIError, RateLimitError

class PubMedScraper:
    """
    Scrape PubMed articles by institutional affiliation.

    Features:
    - Affiliation-based search
    - Rate limiting (respects NCBI limits)
    - Batch processing for large result sets
    - Robust error handling and retry logic
    - History server for 10k+ results

    Args:
        email: Contact email (required by NCBI)
        api_key: NCBI API key (optional, increases rate limit)

    Example:
        >>> scraper = PubMedScraper(email="researcher@example.com")
        >>> pmids = await scraper.search_by_affiliation(
        ...     "Boston Children's Hospital[Affiliation]",
        ...     max_results=1000
        ... )
        >>> articles = await scraper.fetch_article_details(pmids)
    """

    def __init__(
        self,
        email: str,
        api_key: Optional[str] = None,
        max_retries: int = 3
    ):
        if not email:
            raise ValueError("Email is required by NCBI")

        Entrez.email = email
        self.max_retries = max_retries

        if api_key:
            Entrez.api_key = api_key
            # With API key: 10 req/s, use 9 to be safe
            self.rate_limiter = RateLimiter(
                requests_per_second=9.0,
                burst_size=20
            )
            logger.info("Using API key: 10 req/s rate limit")
        else:
            # Without API key: 3 req/s, use 2.5 to be safe
            self.rate_limiter = RateLimiter(
                requests_per_second=2.5,
                burst_size=10
            )
            logger.warning("No API key: 3 req/s rate limit. Get key at https://www.ncbi.nlm.nih.gov/account/")

        self.parser = PubMedParser()

    async def search_by_affiliation(
        self,
        affiliation: str,
        max_results: int = 10000,
        date_range: Optional[str] = None
    ) -> List[str]:
        """
        Search PubMed for articles from specific affiliation.

        Args:
            affiliation: PubMed affiliation query
                Examples:
                - "Boston Children's Hospital[Affiliation]"
                - "Harvard Medical School[Affiliation]"
            max_results: Maximum PMIDs to return
            date_range: Optional date filter
                Examples:
                - "2020:2024[PDAT]"
                - "2023[PDAT]"

        Returns:
            List of PMIDs (PubMed IDs)

        Raises:
            PubMedAPIError: If API request fails
        """
        await self.rate_limiter.acquire()

        # Build query
        query = affiliation
        if date_range:
            query = f"{affiliation} AND {date_range}"

        logger.info(f"Searching PubMed: {query}")

        try:
            # Use history server for potentially large result sets
            search_handle = await asyncio.to_thread(
                Entrez.esearch,
                db="pubmed",
                term=query,
                retmax=max_results,
                usehistory="y"
            )

            search_results = await asyncio.to_thread(Entrez.read, search_handle)
            search_handle.close()

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise PubMedAPIError(f"ESearch failed: {e}")

        pmids = search_results.get("IdList", [])
        count = int(search_results.get("Count", 0))
        webenv = search_results.get("WebEnv", "")
        query_key = search_results.get("QueryKey", "")

        logger.info(f"Found {count} articles, retrieved {len(pmids)} PMIDs")

        # If result set is large, use history server
        if count > len(pmids) and webenv and query_key:
            logger.info(f"Using history server for large result set ({count} articles)")
            return await self._fetch_all_pmids(
                webenv, query_key, count, max_results
            )

        return pmids

    async def _fetch_all_pmids(
        self,
        webenv: str,
        query_key: str,
        total: int,
        max_results: int
    ) -> List[str]:
        """
        Fetch all PMIDs from history server in batches.

        Used for large result sets (>10k articles).
        """
        batch_size = 500
        pmids = []

        for start in range(0, min(total, max_results), batch_size):
            await self.rate_limiter.acquire()

            try:
                handle = await asyncio.to_thread(
                    Entrez.esearch,
                    db="pubmed",
                    retstart=start,
                    retmax=batch_size,
                    webenv=webenv,
                    query_key=query_key
                )

                results = await asyncio.to_thread(Entrez.read, handle)
                handle.close()

                batch_pmids = results.get("IdList", [])
                pmids.extend(batch_pmids)

                logger.debug(f"Fetched PMIDs {start}-{start+len(batch_pmids)}/{total}")

            except Exception as e:
                logger.error(f"Batch fetch failed at {start}: {e}")
                # Continue with next batch
                continue

        return pmids

    async def fetch_article_details(
        self,
        pmids: List[str],
        batch_size: int = 200
    ) -> List[Dict]:
        """
        Fetch detailed metadata for list of PMIDs.

        Args:
            pmids: List of PubMed IDs
            batch_size: Articles per request (max 500)

        Returns:
            List of article dictionaries

        Example output:
            [
                {
                    'pmid': '12345678',
                    'title': 'CAR-T therapy for...',
                    'abstract': 'This study...',
                    'authors': [
                        {
                            'last_name': 'Smith',
                            'first_name': 'John',
                            'affiliations': ['BCH']
                        }
                    ],
                    'journal': 'Nature Medicine',
                    'year': '2023',
                    'doi': '10.1038/...'
                }
            ]
        """
        if batch_size > 500:
            logger.warning("Batch size > 500 may cause errors, reducing to 500")
            batch_size = 500

        articles = []
        total_batches = (len(pmids) + batch_size - 1) // batch_size

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i+batch_size]
            batch_num = i // batch_size + 1

            await self.rate_limiter.acquire()

            logger.info(
                f"Fetching batch {batch_num}/{total_batches}: "
                f"articles {i+1}-{min(i+batch_size, len(pmids))}"
            )

            try:
                # Fetch with retries
                for attempt in range(self.max_retries):
                    try:
                        handle = await asyncio.to_thread(
                            Entrez.efetch,
                            db="pubmed",
                            id=batch,
                            rettype="xml",
                            retmode="xml"
                        )

                        records = await asyncio.to_thread(Entrez.read, handle)
                        handle.close()
                        break

                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            wait = 2 ** attempt  # Exponential backoff
                            logger.warning(f"Fetch failed (attempt {attempt+1}), retrying in {wait}s: {e}")
                            await asyncio.sleep(wait)
                        else:
                            logger.error(f"Batch {batch_num} failed after {self.max_retries} attempts")
                            raise

                # Parse articles
                for record in records.get("PubmedArticle", []):
                    try:
                        article = self.parser.parse_article_from_record(record)
                        articles.append(article)
                    except Exception as e:
                        pmid = record.get("MedlineCitation", {}).get("PMID", "unknown")
                        logger.error(f"Failed to parse article {pmid}: {e}")
                        continue

            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")
                continue

        logger.info(f"Successfully fetched {len(articles)}/{len(pmids)} articles")
        return articles

    async def scrape_and_save(
        self,
        affiliation: str,
        output_file: str,
        max_results: int = 10000,
        date_range: Optional[str] = None
    ) -> None:
        """
        Complete workflow: search, fetch, save to JSON.

        Args:
            affiliation: Institution to search
            output_file: Path to save JSON results
            max_results: Maximum articles to retrieve
            date_range: Optional date filter

        Example:
            >>> await scraper.scrape_and_save(
            ...     affiliation="Boston Children's Hospital[Affiliation]",
            ...     output_file="data/bch_articles.json",
            ...     max_results=5000
            ... )
        """
        logger.info(f"Starting scrape: {affiliation}")

        # Step 1: Search
        pmids = await self.search_by_affiliation(
            affiliation, max_results, date_range
        )

        if not pmids:
            logger.warning("No articles found")
            return

        logger.info(f"Found {len(pmids)} PMIDs")

        # Step 2: Fetch details
        articles = await self.fetch_article_details(pmids)

        if not articles:
            logger.warning("No article details retrieved")
            return

        # Step 3: Save
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(articles, f, indent=2)

        logger.info(f"✓ Saved {len(articles)} articles to {output_file}")


# Update parser to handle Record objects
class PubMedParser:
    # ... existing code ...

    @staticmethod
    def parse_article_from_record(record: Dict) -> Dict:
        """Parse article from Entrez.read() record object."""
        medline = record.get("MedlineCitation", {})
        article = medline.get("Article", {})

        # PMID
        pmid = str(medline.get("PMID", ""))

        # Title
        title = article.get("ArticleTitle", "")

        # Abstract
        abstract_parts = article.get("Abstract", {}).get("AbstractText", [])
        if isinstance(abstract_parts, list):
            abstract = " ".join(str(part) for part in abstract_parts)
        else:
            abstract = str(abstract_parts) if abstract_parts else ""

        # Journal
        journal = article.get("Journal", {}).get("Title", "")

        # Date
        pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = pub_date.get("Year", "")

        # Volume
        volume = article.get("Journal", {}).get("JournalIssue", {}).get("Volume", "")

        # Authors
        authors = []
        for author in article.get("AuthorList", []):
            author_dict = {
                "last_name": author.get("LastName", ""),
                "first_name": author.get("ForeName", ""),
                "initials": author.get("Initials", ""),
                "affiliations": []
            }

            # Affiliations
            for affil in author.get("AffiliationInfo", []):
                affil_text = affil.get("Affiliation", "")
                if affil_text:
                    author_dict["affiliations"].append(affil_text)

            authors.append(author_dict)

        # DOI
        doi = None
        for eloc_id in article.get("ELocationID", []):
            if hasattr(eloc_id, 'attributes'):
                if eloc_id.attributes.get("EIdType") == "doi":
                    doi = str(eloc_id)

        # MeSH terms
        mesh_terms = []
        for mesh_heading in medline.get("MeshHeadingList", []):
            descriptor = mesh_heading.get("DescriptorName", "")
            if descriptor:
                mesh_terms.append(str(descriptor))

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "volume": volume,
            "doi": doi,
            "authors": authors,
            "mesh_terms": mesh_terms
        }
```

### Phase 4: Custom Exceptions

```python
# src/scrapers/exceptions.py

class PubMedAPIError(Exception):
    """Base exception for PubMed API errors."""
    pass

class RateLimitError(PubMedAPIError):
    """Raised when rate limit is exceeded."""
    pass

class ParseError(PubMedAPIError):
    """Raised when XML parsing fails."""
    pass
```

## Running Tests

```bash
# Install dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/scrapers --cov-report=html

# Run specific test file
pytest tests/test_pubmed_scraper.py -v

# Run tests matching pattern
pytest tests/ -k "rate_limit" -v
```

## Integration Testing

```python
# tests/integration/test_real_pubmed.py
# These tests hit the real PubMed API (skip in CI)

import pytest
import os

@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("NCBI_API_KEY"),
    reason="Requires NCBI_API_KEY"
)
@pytest.mark.asyncio
async def test_real_search_small_dataset():
    """Test real PubMed search with small dataset."""
    from src.scrapers.pubmed_scraper import PubMedScraper

    scraper = PubMedScraper(
        email=os.getenv("NCBI_EMAIL"),
        api_key=os.getenv("NCBI_API_KEY")
    )

    # Search for small set
    pmids = await scraper.search_by_affiliation(
        "Boston Children's Hospital[Affiliation]",
        max_results=10
    )

    assert len(pmids) > 0
    assert len(pmids) <= 10

    # Fetch details
    articles = await scraper.fetch_article_details(pmids[:5])

    assert len(articles) > 0
    assert articles[0]['pmid']
    assert articles[0]['title']
```

Run integration tests:
```bash
export NCBI_EMAIL="your_email@example.com"
export NCBI_API_KEY="your_api_key"

pytest tests/integration/ -v -m integration
```

## Usage Examples

### Basic Usage

```python
import asyncio
from src.scrapers.pubmed_scraper import PubMedScraper

async def main():
    # Initialize scraper
    scraper = PubMedScraper(
        email="researcher@example.com",
        api_key="your_api_key_here"  # Optional
    )

    # Search by affiliation
    pmids = await scraper.search_by_affiliation(
        affiliation="Boston Children's Hospital[Affiliation]",
        max_results=1000
    )

    print(f"Found {len(pmids)} articles")

    # Fetch details for first 100
    articles = await scraper.fetch_article_details(pmids[:100])

    # Print first article
    if articles:
        article = articles[0]
        print(f"\nTitle: {article['title']}")
        print(f"Journal: {article['journal']} ({article['year']})")
        print(f"Authors: {len(article['authors'])}")
        print(f"Abstract: {article['abstract'][:200]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

### Save to File

```python
async def scrape_institution():
    scraper = PubMedScraper(
        email="researcher@example.com",
        api_key="your_api_key"
    )

    await scraper.scrape_and_save(
        affiliation="Harvard Medical School[Affiliation]",
        output_file="data/raw/hms_articles.json",
        max_results=5000,
        date_range="2020:2024[PDAT]"
    )

asyncio.run(scrape_institution())
```

### With Date Filtering

```python
# Get publications from last year
from datetime import datetime

current_year = datetime.now().year
last_year = current_year - 1

pmids = await scraper.search_by_affiliation(
    "MIT[Affiliation]",
    date_range=f"{last_year}:{current_year}[PDAT]"
)
```

## Performance Benchmarks

Expected performance (with API key):
- Search: ~0.5-1s for any result set size
- Fetch: ~10-15 articles/second
- 1,000 articles: ~1-2 minutes
- 10,000 articles: ~10-15 minutes
- 25,000 articles: ~25-30 minutes

## Success Criteria

✅ **Must Have:**
1. All unit tests passing (>90% coverage)
2. Rate limiter prevents API blocking
3. Handles large result sets (>10k articles)
4. Robust error handling and retry logic
5. Parses all required fields: pmid, title, abstract, authors, affiliations
6. Saves results to valid JSON

✅ **Should Have:**
7. Integration tests with real API
8. Extracts MeSH terms and DOI
9. Progress logging
10. Graceful degradation on partial failures

✅ **Nice to Have:**
11. Resume from checkpoint
12. Parallel batch fetching
13. Caching mechanism

## Validation Checklist

Before considering task complete:

- [ ] All unit tests pass
- [ ] Integration test with real PubMed works
- [ ] Can scrape 100 articles successfully
- [ ] Rate limiter prevents blocking (verify in logs)
- [ ] JSON output is valid and complete
- [ ] Error handling tested (network failures, malformed XML)
- [ ] Code follows PEP 8 style
- [ ] Docstrings complete
- [ ] No API key hard-coded (use environment variables)

## Deliverables

1. **Source code**:
   - `src/scrapers/rate_limiter.py`
   - `src/scrapers/pubmed_scraper.py`
   - `src/scrapers/pubmed_parser.py`
   - `src/scrapers/exceptions.py`

2. **Tests**:
   - `tests/test_rate_limiter.py`
   - `tests/test_pubmed_scraper.py`
   - `tests/test_pubmed_parser.py`
   - `tests/integration/test_real_pubmed.py`

3. **Documentation**:
   - README with usage examples
   - API documentation (docstrings)
   - Sample output JSON

4. **Validation**:
   - Test coverage report (>90%)
   - Sample dataset (100-1000 articles)
   - Performance benchmark results

## Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install biopython aiohttp loguru pydantic pytest pytest-asyncio pytest-cov pytest-mock

# Set environment variables
export NCBI_EMAIL="your_email@example.com"
export NCBI_API_KEY="your_api_key"  # Get from https://www.ncbi.nlm.nih.gov/account/

# Run tests
pytest tests/ -v --cov=src/scrapers
```

## Getting NCBI API Key

1. Create account: https://www.ncbi.nlm.nih.gov/account/
2. Go to Settings → API Key
3. Create new API key
4. Copy key to `.env` file

## Troubleshooting

**Rate limiting errors:**
- Ensure rate limiter is being used
- Check for correct API key
- Verify requests_per_second is conservative

**XML parsing errors:**
- Log raw XML for debugging
- Check for malformed records
- Skip unparseable records gracefully

**Large result sets timing out:**
- Use history server (automatic for 10k+)
- Increase batch size for fetching
- Add progress checkpoints

## Additional Resources

- **PubMed E-utilities Guide**: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- **Biopython Tutorial**: http://biopython.org/DIST/docs/tutorial/Tutorial.html
- **PubMed Search Tips**: https://pubmed.ncbi.nlm.nih.gov/help/

---

**Task completion**: When all tests pass, integration test works with real PubMed, and you can successfully scrape 1000+ articles with proper metadata extraction.

Good luck! Remember: **Write tests first, then implement.**
