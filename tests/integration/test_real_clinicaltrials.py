"""Integration tests with real ClinicalTrials.gov API."""

import pytest
import os


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_search_small_dataset():
    """Test real ClinicalTrials.gov search."""
    from src.scrapers.clinicaltrials_scraper import ClinicalTrialsScraper

    async with ClinicalTrialsScraper() as scraper:
        # Search for small set
        trials = await scraper.search_by_institution(
            "Boston Children's Hospital",
            status=["RECRUITING"],
            max_results=10
        )

        assert len(trials) > 0
        assert trials[0]['nct_id'].startswith('NCT')
        assert trials[0]['lead_sponsor']


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_specific_study():
    """Test fetching specific study by NCT ID."""
    from src.scrapers.clinicaltrials_scraper import ClinicalTrialsScraper

    async with ClinicalTrialsScraper() as scraper:
        # Known NCT ID
        study = await scraper.get_study_details("NCT00000102")

        assert study['nct_id'] == 'NCT00000102'
        assert study['title']
