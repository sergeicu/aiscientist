"""Tests for ClinicalTrials.gov scraper."""

import pytest
from unittest.mock import Mock, AsyncMock
from src.scrapers.clinicaltrials_scraper import ClinicalTrialsScraper


@pytest.fixture
def scraper():
    """Create scraper instance."""
    return ClinicalTrialsScraper()


@pytest.mark.asyncio
async def test_search_by_institution_returns_trials(scraper, mocker):
    """Should return trials for institution search."""
    mock_response = {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {"nctId": "NCT123"},
                    "statusModule": {"overallStatus": "RECRUITING"}
                }
            },
            {
                "protocolSection": {
                    "identificationModule": {"nctId": "NCT456"},
                    "statusModule": {"overallStatus": "ACTIVE_NOT_RECRUITING"}
                }
            }
        ],
        "nextPageToken": None
    }

    mock_client = mocker.patch.object(scraper, 'http_client')
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.close = AsyncMock()

    trials = await scraper.search_by_institution(
        "Boston Children's Hospital",
        max_results=100
    )

    assert len(trials) == 2
    assert trials[0]['nct_id'] == 'NCT123'
    await scraper.close()


@pytest.mark.asyncio
async def test_search_with_status_filter(scraper, mocker):
    """Should filter by trial status."""
    mock_client = mocker.patch.object(scraper, 'http_client')
    mock_client.get = AsyncMock(return_value={"studies": [], "nextPageToken": None})
    mock_client.close = AsyncMock()

    await scraper.search_by_institution(
        "Test Hospital",
        status=["RECRUITING", "ACTIVE_NOT_RECRUITING"]
    )

    # Verify filter was applied
    call_args = mock_client.get.call_args
    assert "RECRUITING" in str(call_args)
    await scraper.close()


@pytest.mark.asyncio
async def test_handles_pagination(scraper, mocker):
    """Should handle paginated responses."""
    # First page
    page1 = {
        "studies": [{"protocolSection": {"identificationModule": {"nctId": "NCT1"}}}],
        "nextPageToken": "token123"
    }

    # Second page
    page2 = {
        "studies": [{"protocolSection": {"identificationModule": {"nctId": "NCT2"}}}],
        "nextPageToken": None
    }

    mock_client = mocker.patch.object(scraper, 'http_client')
    mock_client.get = AsyncMock(side_effect=[page1, page2])
    mock_client.close = AsyncMock()

    trials = await scraper.search_by_institution(
        "Test Hospital",
        max_results=1000
    )

    # Should make 2 requests
    assert mock_client.get.call_count == 2
    assert len(trials) == 2
    await scraper.close()


@pytest.mark.asyncio
async def test_respects_max_results_limit(scraper, mocker):
    """Should stop at max_results."""
    # Return many pages
    mock_response = {
        "studies": [
            {"protocolSection": {"identificationModule": {"nctId": f"NCT{i}"}}}
            for i in range(100)
        ],
        "nextPageToken": "more"
    }

    mock_client = mocker.patch.object(scraper, 'http_client')
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.close = AsyncMock()

    trials = await scraper.search_by_institution(
        "Test Hospital",
        max_results=150
    )

    # Should stop after getting 150 results
    assert len(trials) <= 150
    await scraper.close()


@pytest.mark.asyncio
async def test_search_by_sponsor_with_collaborator(scraper, mocker):
    """Should search by sponsor or collaborator."""
    mock_client = mocker.patch.object(scraper, 'http_client')
    mock_client.get = AsyncMock(return_value={"studies": [], "nextPageToken": None})
    mock_client.close = AsyncMock()

    await scraper.search_by_sponsor(
        "Boston Children's Hospital",
        include_collaborator=True
    )

    call_args = mock_client.get.call_args
    # Should search both lead sponsor and collaborator
    assert call_args is not None
    await scraper.close()


@pytest.mark.asyncio
async def test_get_study_details(scraper, mocker):
    """Should fetch full study details by NCT ID."""
    mock_response = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT12345678",
                "briefTitle": "Test Study"
            }
        }
    }

    mock_client = mocker.patch.object(scraper, 'http_client')
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.close = AsyncMock()

    study = await scraper.get_study_details("NCT12345678")

    assert study['nct_id'] == 'NCT12345678'
    assert study['title'] == 'Test Study'
    await scraper.close()


@pytest.mark.asyncio
async def test_scrape_and_save_workflow(scraper, mocker, tmp_path):
    """Should complete full scrape-to-file workflow."""
    mock_trials = [
        {'nct_id': 'NCT123', 'title': 'Trial 1'},
        {'nct_id': 'NCT456', 'title': 'Trial 2'}
    ]

    mocker.patch.object(
        scraper,
        'search_by_institution',
        return_value=mock_trials
    )

    output_file = tmp_path / "trials.json"

    await scraper.scrape_and_save(
        institution="Test Hospital",
        output_file=str(output_file)
    )

    # Should create file
    assert output_file.exists()

    # Should contain trials
    import json
    with open(output_file) as f:
        data = json.load(f)

    assert len(data) == 2
    await scraper.close()
