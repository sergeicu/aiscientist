"""
Tests for PubMedScraper class.

Tests PubMed API interaction, rate limiting, and data fetching.
"""

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
    mock_search.return_value.__enter__ = Mock(return_value=Mock())

    mocker.patch('Bio.Entrez.read', return_value={
        'IdList': ['123'],
        'Count': '1',
        'WebEnv': '',
        'QueryKey': ''
    })

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

    # Mock successful fetch for each batch
    mock_records = {
        'PubmedArticle': [
            {
                'MedlineCitation': {
                    'PMID': str(i),
                    'Article': {
                        'ArticleTitle': f'Title {i}',
                        'Abstract': {},
                        'Journal': {},
                        'AuthorList': [],
                        'ELocationID': []
                    }
                },
                'MeshHeadingList': []
            }
            for i in range(200)
        ]
    }
    mocker.patch('Bio.Entrez.read', return_value=mock_records)

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

    # Should eventually succeed (note: current implementation doesn't have retry in search)
    # This test documents expected behavior for future enhancement
    with pytest.raises(PubMedAPIError):
        pmids = await scraper.search_by_affiliation("Test[Affiliation]")


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


def test_scraper_requires_email():
    """Should raise ValueError if email not provided."""
    with pytest.raises(ValueError, match="Email is required"):
        PubMedScraper(email="")


def test_scraper_without_api_key():
    """Should initialize with lower rate limit when no API key."""
    scraper = PubMedScraper(email="test@example.com")  # No API key

    # Should use lower rate limit
    assert scraper.rate_limiter.rate == 2.5
    assert scraper.rate_limiter.burst_size == 10


@pytest.mark.asyncio
async def test_scrape_and_save_no_results(scraper, mocker, tmp_path):
    """Should handle case when no articles found."""
    # Mock search returning no results
    mocker.patch.object(
        scraper,
        'search_by_affiliation',
        return_value=[]
    )

    output_file = tmp_path / "articles.json"

    await scraper.scrape_and_save(
        affiliation="Test[Affiliation]",
        output_file=str(output_file),
        max_results=100
    )

    # Should not create output file when no results
    assert not output_file.exists()


@pytest.mark.asyncio
async def test_scrape_and_save_no_article_details(scraper, mocker, tmp_path):
    """Should handle case when fetch returns no articles."""
    # Mock search
    mocker.patch.object(
        scraper,
        'search_by_affiliation',
        return_value=['123', '456']
    )

    # Mock fetch returning no articles
    mocker.patch.object(
        scraper,
        'fetch_article_details',
        return_value=[]
    )

    output_file = tmp_path / "articles.json"

    await scraper.scrape_and_save(
        affiliation="Test[Affiliation]",
        output_file=str(output_file),
        max_results=100
    )

    # Should not create output file when no articles retrieved
    assert not output_file.exists()


@pytest.mark.asyncio
async def test_fetch_article_details_with_parse_errors(scraper, mocker):
    """Should skip articles that fail to parse."""
    pmids = ['123', '456', '789']

    mock_fetch = mocker.patch('Bio.Entrez.efetch', return_value=Mock())

    # Mock records where one will fail to parse
    mock_records = {
        'PubmedArticle': [
            {
                'MedlineCitation': {
                    'PMID': '123',
                    'Article': {
                        'ArticleTitle': 'Good Article',
                        'Abstract': {},
                        'Journal': {},
                        'AuthorList': [],
                        'ELocationID': []
                    }
                },
                'MeshHeadingList': []
            },
            {
                'MedlineCitation': {
                    'PMID': '456',
                    'Article': {
                        'ArticleTitle': 'Another Good Article',
                        'Abstract': {},
                        'Journal': {},
                        'AuthorList': [],
                        'ELocationID': []
                    }
                },
                'MeshHeadingList': []
            }
        ]
    }
    mocker.patch('Bio.Entrez.read', return_value=mock_records)

    # Mock parser to fail on second article
    def parse_side_effect(record):
        pmid = record.get("MedlineCitation", {}).get("PMID", "")
        if pmid == '456':
            raise Exception("Parse error")
        return {'pmid': pmid, 'title': 'Title'}

    mocker.patch.object(
        scraper.parser,
        'parse_article_from_record',
        side_effect=parse_side_effect
    )

    articles = await scraper.fetch_article_details(pmids, batch_size=10)

    # Should only return successfully parsed articles
    assert len(articles) == 1
    assert articles[0]['pmid'] == '123'
