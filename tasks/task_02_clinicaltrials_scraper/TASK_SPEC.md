# Task 2: ClinicalTrials.gov Institutional Scraper - TDD Implementation

## Executive Summary

Implement a production-ready ClinicalTrials.gov scraper that searches and retrieves clinical trial data by institution and/or sponsor. This scraper will complement the PubMed scraper as part of the AI Scientist platform's data acquisition layer.

**Key Requirements:**
- Search ClinicalTrials.gov by lead organization/sponsor
- Use official ClinicalTrials.gov API v2
- Extract comprehensive trial metadata: phase, status, interventions, outcomes, enrollment
- Handle pagination for large result sets
- Parse structured trial data into normalized JSON
- Implement robust error handling and retry logic
- Follow Test-Driven Development (TDD) principles

## Background & Context

### ClinicalTrials.gov API v2

ClinicalTrials.gov provides a REST API for programmatic access:
- **Base URL**: `https://clinicaltrials.gov/api/v2/studies`
- **Authentication**: No API key required (public API)
- **Rate Limiting**: Reasonable use policy (no hard limits documented)
- **Data Format**: JSON responses
- **Documentation**: https://clinicaltrials.gov/data-api/api

### Key Concepts

**NCT ID**: Unique identifier for each trial (e.g., NCT00000102)

**Sponsor Types**:
- **Lead Sponsor**: Primary organization responsible for trial
- **Collaborators**: Supporting organizations

**Trial Statuses**:
- Recruiting
- Active, not recruiting
- Completed
- Terminated
- Suspended
- Withdrawn
- Unknown

**Trial Phases**:
- Early Phase 1 (Phase 0)
- Phase 1
- Phase 1/Phase 2
- Phase 2
- Phase 2/Phase 3
- Phase 3
- Phase 4
- Not Applicable

### Search Capabilities

The API supports queries by:
- Organization name (lead sponsor or collaborator)
- Geographic location
- Condition/disease
- Intervention
- Study type
- Status
- Phase

## Technical Architecture

### Module Structure

```
src/scrapers/
├── __init__.py
├── clinicaltrials_scraper.py    # Main scraper class
├── clinicaltrials_parser.py     # Response parsing utilities
├── http_client.py               # HTTP client with retry logic
└── exceptions.py                # Custom exception classes

tests/
├── __init__.py
├── test_clinicaltrials_scraper.py
├── test_clinicaltrials_parser.py
├── test_http_client.py
└── fixtures/
    ├── sample_study.json         # Sample API response
    └── sample_search_results.json
```

### Dependencies

```python
# Required packages
aiohttp>=3.9           # Async HTTP client
httpx>=0.25            # Alternative HTTP client with retry
pydantic>=2.0          # Data validation
loguru>=0.7            # Logging
pytest>=7.4            # Testing
pytest-asyncio>=0.21   # Async testing
pytest-mock>=3.12      # Mocking
respx>=0.20            # HTTP mocking for httpx
tenacity>=8.2          # Retry logic
```

## TDD Implementation Plan

### Phase 1: HTTP Client with Retry Logic (TDD)

#### Test Cases

```python
# tests/test_http_client.py

import pytest
import httpx
from src.scrapers.http_client import HTTPClient
from src.scrapers.exceptions import APIError, NetworkError

@pytest.mark.asyncio
async def test_get_request_success(respx_mock):
    """Should make successful GET request."""
    respx_mock.get("https://api.example.com/test").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )

    client = HTTPClient()
    response = await client.get("https://api.example.com/test")

    assert response["status"] == "ok"

@pytest.mark.asyncio
async def test_handles_404_error(respx_mock):
    """Should raise APIError on 404."""
    respx_mock.get("https://api.example.com/notfound").mock(
        return_value=httpx.Response(404, json={"error": "Not found"})
    )

    client = HTTPClient()

    with pytest.raises(APIError) as exc_info:
        await client.get("https://api.example.com/notfound")

    assert "404" in str(exc_info.value)

@pytest.mark.asyncio
async def test_retries_on_transient_error(respx_mock):
    """Should retry on 5xx errors."""
    # Fail twice, succeed third time
    route = respx_mock.get("https://api.example.com/retry")
    route.mock(side_effect=[
        httpx.Response(503, text="Service unavailable"),
        httpx.Response(503, text="Service unavailable"),
        httpx.Response(200, json={"status": "ok"})
    ])

    client = HTTPClient(max_retries=3)
    response = await client.get("https://api.example.com/retry")

    assert response["status"] == "ok"
    assert route.call_count == 3

@pytest.mark.asyncio
async def test_exponential_backoff(respx_mock, mocker):
    """Should use exponential backoff between retries."""
    route = respx_mock.get("https://api.example.com/backoff")
    route.mock(side_effect=[
        httpx.Response(503),
        httpx.Response(503),
        httpx.Response(200, json={})
    ])

    mock_sleep = mocker.patch('asyncio.sleep')

    client = HTTPClient(max_retries=3)
    await client.get("https://api.example.com/backoff")

    # Check backoff: 1s, 2s
    calls = mock_sleep.call_args_list
    assert len(calls) == 2
    assert calls[0][0][0] == pytest.approx(1.0, rel=0.5)
    assert calls[1][0][0] == pytest.approx(2.0, rel=0.5)

@pytest.mark.asyncio
async def test_timeout_handling(respx_mock):
    """Should handle request timeouts."""
    respx_mock.get("https://api.example.com/timeout").mock(
        side_effect=httpx.TimeoutException("Timeout")
    )

    client = HTTPClient(timeout=5.0, max_retries=1)

    with pytest.raises(NetworkError):
        await client.get("https://api.example.com/timeout")

@pytest.mark.asyncio
async def test_post_request_with_params(respx_mock):
    """Should make POST request with query parameters."""
    route = respx_mock.post("https://api.example.com/search").mock(
        return_value=httpx.Response(200, json={"results": []})
    )

    client = HTTPClient()
    await client.post(
        "https://api.example.com/search",
        params={"query": "test", "pageSize": 100}
    )

    assert route.called
    assert "query=test" in str(route.calls[0].request.url)
```

#### Implementation

```python
# src/scrapers/http_client.py

import asyncio
from typing import Dict, Optional, Any
import httpx
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from .exceptions import APIError, NetworkError

class HTTPClient:
    """
    HTTP client with retry logic and error handling.

    Features:
    - Automatic retries on transient errors (5xx)
    - Exponential backoff
    - Timeout handling
    - Request/response logging

    Args:
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        base_url: Base URL for all requests (optional)

    Example:
        >>> client = HTTPClient(timeout=30.0, max_retries=3)
        >>> data = await client.get("https://api.example.com/data")
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        base_url: Optional[str] = None
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True
        )

    async def get(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> Dict:
        """
        Make GET request with retry logic.

        Args:
            url: Request URL
            params: Query parameters
            headers: Request headers

        Returns:
            Parsed JSON response

        Raises:
            APIError: On 4xx/5xx errors (after retries)
            NetworkError: On network failures
        """
        return await self._request("GET", url, params=params, headers=headers)

    async def post(
        self,
        url: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> Dict:
        """
        Make POST request with retry logic.

        Args:
            url: Request URL
            params: Query parameters
            json: JSON body
            headers: Request headers

        Returns:
            Parsed JSON response
        """
        return await self._request(
            "POST", url, params=params, json=json, headers=headers
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
        reraise=True
    )
    async def _request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> Dict:
        """Internal request method with retry decorator."""
        full_url = f"{self.base_url}{url}" if self.base_url else url

        try:
            logger.debug(f"{method} {full_url}")

            response = await self.client.request(method, full_url, **kwargs)

            # Raise on 4xx/5xx
            response.raise_for_status()

            # Parse JSON
            data = response.json()
            logger.debug(f"Response: {response.status_code}")

            return data

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")

            # Don't retry 4xx errors (except 429 Too Many Requests)
            if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                raise APIError(
                    f"API returned {e.response.status_code}: {e.response.text}"
                ) from e

            # Retry 5xx and 429
            raise

        except httpx.TimeoutException as e:
            logger.error(f"Request timeout: {e}")
            raise NetworkError(f"Request timed out after {self.timeout}s") from e

        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise NetworkError(f"Network error: {e}") from e

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise APIError(f"Unexpected error: {e}") from e

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
```

### Phase 2: Response Parser (TDD)

#### Test Cases

```python
# tests/test_clinicaltrials_parser.py

import pytest
from pathlib import Path
import json
from src.scrapers.clinicaltrials_parser import ClinicalTrialsParser

@pytest.fixture
def sample_study():
    """Sample study from ClinicalTrials.gov API."""
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT12345678",
                "briefTitle": "A Phase 2 Trial of Novel CAR-T Therapy"
            },
            "statusModule": {
                "statusVerifiedDate": "2024-01",
                "overallStatus": "RECRUITING",
                "startDateStruct": {
                    "date": "2024-01-15"
                }
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {
                    "name": "Boston Children's Hospital"
                },
                "collaborators": [
                    {"name": "Harvard Medical School"}
                ]
            },
            "designModule": {
                "phases": ["PHASE2"],
                "studyType": "INTERVENTIONAL",
                "enrollmentInfo": {
                    "count": 50
                }
            },
            "armsInterventionsModule": {
                "interventions": [
                    {
                        "type": "BIOLOGICAL",
                        "name": "CAR-T Cell Therapy"
                    }
                ]
            },
            "outcomesModule": {
                "primaryOutcomes": [
                    {
                        "measure": "Overall Response Rate",
                        "timeFrame": "6 months"
                    }
                ],
                "secondaryOutcomes": [
                    {
                        "measure": "Progression-Free Survival",
                        "timeFrame": "12 months"
                    }
                ]
            },
            "conditionsModule": {
                "conditions": ["Pediatric Leukemia", "ALL"]
            },
            "descriptionModule": {
                "briefSummary": "This study evaluates CAR-T therapy..."
            }
        }
    }

def test_parse_basic_metadata(sample_study):
    """Should extract NCT ID, title, status."""
    parser = ClinicalTrialsParser()
    trial = parser.parse_study(sample_study)

    assert trial['nct_id'] == 'NCT12345678'
    assert 'CAR-T Therapy' in trial['title']
    assert trial['status'] == 'RECRUITING'

def test_parse_sponsor_info(sample_study):
    """Should extract lead sponsor and collaborators."""
    parser = ClinicalTrialsParser()
    trial = parser.parse_study(sample_study)

    assert trial['lead_sponsor'] == "Boston Children's Hospital"
    assert len(trial['collaborators']) == 1
    assert trial['collaborators'][0] == "Harvard Medical School"

def test_parse_phase_and_type(sample_study):
    """Should extract trial phase and study type."""
    parser = ClinicalTrialsParser()
    trial = parser.parse_study(sample_study)

    assert trial['phase'] == 'Phase 2'
    assert trial['study_type'] == 'INTERVENTIONAL'
    assert trial['enrollment'] == 50

def test_parse_interventions(sample_study):
    """Should extract intervention details."""
    parser = ClinicalTrialsParser()
    trial = parser.parse_study(sample_study)

    assert len(trial['interventions']) == 1
    assert trial['interventions'][0]['type'] == 'BIOLOGICAL'
    assert trial['interventions'][0]['name'] == 'CAR-T Cell Therapy'

def test_parse_outcomes(sample_study):
    """Should extract primary and secondary outcomes."""
    parser = ClinicalTrialsParser()
    trial = parser.parse_study(sample_study)

    assert len(trial['primary_outcomes']) == 1
    assert 'Response Rate' in trial['primary_outcomes'][0]['measure']

    assert len(trial['secondary_outcomes']) == 1
    assert 'Survival' in trial['secondary_outcomes'][0]['measure']

def test_parse_conditions(sample_study):
    """Should extract conditions/diseases."""
    parser = ClinicalTrialsParser()
    trial = parser.parse_study(sample_study)

    assert 'Pediatric Leukemia' in trial['conditions']
    assert 'ALL' in trial['conditions']

def test_handle_missing_fields():
    """Should handle studies with minimal data."""
    minimal_study = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT99999999",
                "briefTitle": "Minimal Study"
            },
            "statusModule": {
                "overallStatus": "COMPLETED"
            }
        }
    }

    parser = ClinicalTrialsParser()
    trial = parser.parse_study(minimal_study)

    assert trial['nct_id'] == 'NCT99999999'
    assert trial['lead_sponsor'] is None
    assert trial['interventions'] == []
    assert trial['enrollment'] is None

def test_normalize_phase():
    """Should normalize phase names."""
    parser = ClinicalTrialsParser()

    assert parser.normalize_phase(['PHASE1']) == 'Phase 1'
    assert parser.normalize_phase(['PHASE2', 'PHASE3']) == 'Phase 2/Phase 3'
    assert parser.normalize_phase(['NA']) == 'Not Applicable'
    assert parser.normalize_phase([]) is None
```

#### Implementation

```python
# src/scrapers/clinicaltrials_parser.py

from typing import Dict, List, Optional
from loguru import logger

class ClinicalTrialsParser:
    """
    Parse ClinicalTrials.gov API v2 responses.

    Extracts structured metadata from study JSON.
    Handles missing fields gracefully.
    """

    # Phase normalization mapping
    PHASE_MAP = {
        'EARLY_PHASE1': 'Early Phase 1',
        'PHASE1': 'Phase 1',
        'PHASE2': 'Phase 2',
        'PHASE3': 'Phase 3',
        'PHASE4': 'Phase 4',
        'NA': 'Not Applicable'
    }

    @staticmethod
    def parse_study(study: Dict) -> Dict:
        """
        Parse study JSON to structured dictionary.

        Args:
            study: Study object from API response

        Returns:
            Normalized trial metadata
        """
        protocol = study.get('protocolSection', {})

        # Identification
        ident_module = protocol.get('identificationModule', {})
        nct_id = ident_module.get('nctId', '')
        title = ident_module.get('briefTitle', '')
        official_title = ident_module.get('officialTitle', '')

        # Status
        status_module = protocol.get('statusModule', {})
        status = status_module.get('overallStatus', '')
        start_date = ClinicalTrialsParser._extract_date(
            status_module.get('startDateStruct', {})
        )
        completion_date = ClinicalTrialsParser._extract_date(
            status_module.get('completionDateStruct', {})
        )

        # Sponsor
        sponsor_module = protocol.get('sponsorCollaboratorsModule', {})
        lead_sponsor = None
        if 'leadSponsor' in sponsor_module:
            lead_sponsor = sponsor_module['leadSponsor'].get('name')

        collaborators = [
            c.get('name', '') for c in sponsor_module.get('collaborators', [])
        ]

        # Design
        design_module = protocol.get('designModule', {})
        phases = design_module.get('phases', [])
        phase = ClinicalTrialsParser.normalize_phase(phases)
        study_type = design_module.get('studyType', '')

        enrollment_info = design_module.get('enrollmentInfo', {})
        enrollment = enrollment_info.get('count')

        # Interventions
        interventions = []
        arms_module = protocol.get('armsInterventionsModule', {})
        for intervention in arms_module.get('interventions', []):
            interventions.append({
                'type': intervention.get('type', ''),
                'name': intervention.get('name', ''),
                'description': intervention.get('description', '')
            })

        # Outcomes
        outcomes_module = protocol.get('outcomesModule', {})
        primary_outcomes = [
            {
                'measure': o.get('measure', ''),
                'timeFrame': o.get('timeFrame', ''),
                'description': o.get('description', '')
            }
            for o in outcomes_module.get('primaryOutcomes', [])
        ]

        secondary_outcomes = [
            {
                'measure': o.get('measure', ''),
                'timeFrame': o.get('timeFrame', ''),
                'description': o.get('description', '')
            }
            for o in outcomes_module.get('secondaryOutcomes', [])
        ]

        # Conditions
        conditions_module = protocol.get('conditionsModule', {})
        conditions = conditions_module.get('conditions', [])

        # Description
        desc_module = protocol.get('descriptionModule', {})
        brief_summary = desc_module.get('briefSummary', '')
        detailed_description = desc_module.get('detailedDescription', '')

        # Eligibility
        eligibility_module = protocol.get('eligibilityModule', {})
        eligibility_criteria = eligibility_module.get('eligibilityCriteria', '')
        min_age = eligibility_module.get('minimumAge', '')
        max_age = eligibility_module.get('maximumAge', '')

        # Contacts
        contacts_module = protocol.get('contactsLocationsModule', {})
        locations = []
        for loc in contacts_module.get('locations', []):
            locations.append({
                'facility': loc.get('facility', ''),
                'city': loc.get('city', ''),
                'state': loc.get('state', ''),
                'country': loc.get('country', '')
            })

        return {
            'nct_id': nct_id,
            'title': title,
            'official_title': official_title,
            'status': status,
            'phase': phase,
            'study_type': study_type,
            'enrollment': enrollment,
            'start_date': start_date,
            'completion_date': completion_date,
            'lead_sponsor': lead_sponsor,
            'collaborators': collaborators,
            'interventions': interventions,
            'primary_outcomes': primary_outcomes,
            'secondary_outcomes': secondary_outcomes,
            'conditions': conditions,
            'brief_summary': brief_summary,
            'detailed_description': detailed_description,
            'eligibility_criteria': eligibility_criteria,
            'min_age': min_age,
            'max_age': max_age,
            'locations': locations
        }

    @staticmethod
    def normalize_phase(phases: List[str]) -> Optional[str]:
        """
        Normalize phase list to readable string.

        Args:
            phases: List of phase identifiers

        Returns:
            Normalized phase string
        """
        if not phases:
            return None

        normalized = []
        for phase in phases:
            normalized.append(
                ClinicalTrialsParser.PHASE_MAP.get(phase, phase)
            )

        return '/'.join(normalized)

    @staticmethod
    def _extract_date(date_struct: Dict) -> Optional[str]:
        """Extract date from date struct."""
        if not date_struct:
            return None
        return date_struct.get('date')
```

### Phase 3: ClinicalTrials.gov Scraper (TDD)

#### Test Cases

```python
# tests/test_clinicaltrials_scraper.py

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

    trials = await scraper.search_by_institution(
        "Boston Children's Hospital",
        max_results=100
    )

    assert len(trials) == 2
    assert trials[0]['nct_id'] == 'NCT123'

@pytest.mark.asyncio
async def test_search_with_status_filter(scraper, mocker):
    """Should filter by trial status."""
    mock_client = mocker.patch.object(scraper, 'http_client')
    mock_client.get = AsyncMock(return_value={"studies": [], "nextPageToken": None})

    await scraper.search_by_institution(
        "Test Hospital",
        status=["RECRUITING", "ACTIVE_NOT_RECRUITING"]
    )

    # Verify filter was applied
    call_args = mock_client.get.call_args
    assert "RECRUITING" in str(call_args)

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

    trials = await scraper.search_by_institution(
        "Test Hospital",
        max_results=1000
    )

    # Should make 2 requests
    assert mock_client.get.call_count == 2
    assert len(trials) == 2

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

    trials = await scraper.search_by_institution(
        "Test Hospital",
        max_results=150
    )

    # Should stop after getting 150 results
    assert len(trials) <= 150

@pytest.mark.asyncio
async def test_search_by_sponsor_with_collaborator(scraper, mocker):
    """Should search by sponsor or collaborator."""
    mock_client = mocker.patch.object(scraper, 'http_client')
    mock_client.get = AsyncMock(return_value={"studies": [], "nextPageToken": None})

    await scraper.search_by_sponsor(
        "Boston Children's Hospital",
        include_collaborator=True
    )

    call_args = mock_client.get.call_args
    # Should search both lead sponsor and collaborator
    assert call_args is not None

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

    study = await scraper.get_study_details("NCT12345678")

    assert study['nct_id'] == 'NCT12345678'
    assert study['title'] == 'Test Study'

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
```

#### Implementation

```python
# src/scrapers/clinicaltrials_scraper.py

import json
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger

from .http_client import HTTPClient
from .clinicaltrials_parser import ClinicalTrialsParser
from .exceptions import APIError

class ClinicalTrialsScraper:
    """
    Scrape ClinicalTrials.gov by institution/sponsor.

    Features:
    - Search by lead sponsor or collaborator
    - Filter by status, phase, study type
    - Handle pagination automatically
    - Extract comprehensive trial metadata

    Args:
        page_size: Results per page (max 1000)
        timeout: Request timeout in seconds

    Example:
        >>> scraper = ClinicalTrialsScraper()
        >>> trials = await scraper.search_by_institution(
        ...     "Boston Children's Hospital",
        ...     status=["RECRUITING"],
        ...     max_results=500
        ... )
    """

    API_BASE_URL = "https://clinicaltrials.gov/api/v2"

    def __init__(
        self,
        page_size: int = 100,
        timeout: float = 30.0
    ):
        if page_size > 1000:
            logger.warning("Page size > 1000, capping at 1000")
            page_size = 1000

        self.page_size = page_size
        self.http_client = HTTPClient(
            timeout=timeout,
            base_url=self.API_BASE_URL
        )
        self.parser = ClinicalTrialsParser()

    async def search_by_institution(
        self,
        institution: str,
        status: Optional[List[str]] = None,
        phase: Optional[List[str]] = None,
        max_results: int = 1000
    ) -> List[Dict]:
        """
        Search trials by lead sponsor institution.

        Args:
            institution: Institution name
                Example: "Boston Children's Hospital"
            status: Filter by status
                Options: ["RECRUITING", "COMPLETED", "ACTIVE_NOT_RECRUITING", etc.]
            phase: Filter by phase
                Options: ["PHASE1", "PHASE2", "PHASE3", "PHASE4"]
            max_results: Maximum trials to return

        Returns:
            List of trial dictionaries
        """
        logger.info(f"Searching ClinicalTrials.gov: {institution}")

        query_params = {
            "query.lead": institution,
            "pageSize": min(self.page_size, max_results)
        }

        # Add filters
        if status:
            query_params["filter.overallStatus"] = ",".join(status)

        if phase:
            query_params["filter.phase"] = ",".join(phase)

        trials = []
        next_token = None

        while len(trials) < max_results:
            # Add pagination token
            if next_token:
                query_params["pageToken"] = next_token

            try:
                response = await self.http_client.get(
                    "/studies",
                    params=query_params
                )

                # Parse studies
                for study in response.get("studies", []):
                    trial = self.parser.parse_study(study)
                    trials.append(trial)

                    if len(trials) >= max_results:
                        break

                # Check for more pages
                next_token = response.get("nextPageToken")
                if not next_token:
                    break

                logger.debug(f"Fetched {len(trials)} trials, continuing...")

            except APIError as e:
                logger.error(f"API error during search: {e}")
                break

        logger.info(f"Found {len(trials)} trials for {institution}")
        return trials[:max_results]

    async def search_by_sponsor(
        self,
        sponsor: str,
        include_collaborator: bool = False,
        status: Optional[List[str]] = None,
        max_results: int = 1000
    ) -> List[Dict]:
        """
        Search trials by sponsor (lead or collaborator).

        Args:
            sponsor: Sponsor organization name
            include_collaborator: Also search collaborator field
            status: Filter by status
            max_results: Maximum trials to return

        Returns:
            List of trial dictionaries
        """
        if include_collaborator:
            # Search both lead sponsor and collaborators
            query = f'"{sponsor}"'
            query_params = {
                "query.cond": query,
                "pageSize": min(self.page_size, max_results)
            }
        else:
            # Search only lead sponsor
            return await self.search_by_institution(
                sponsor, status=status, max_results=max_results
            )

        if status:
            query_params["filter.overallStatus"] = ",".join(status)

        trials = []
        next_token = None

        while len(trials) < max_results:
            if next_token:
                query_params["pageToken"] = next_token

            try:
                response = await self.http_client.get(
                    "/studies",
                    params=query_params
                )

                for study in response.get("studies", []):
                    # Filter to include only if sponsor matches
                    protocol = study.get("protocolSection", {})
                    sponsor_module = protocol.get("sponsorCollaboratorsModule", {})

                    lead = sponsor_module.get("leadSponsor", {}).get("name", "")
                    collaborators = [
                        c.get("name", "")
                        for c in sponsor_module.get("collaborators", [])
                    ]

                    if sponsor in lead or sponsor in " ".join(collaborators):
                        trial = self.parser.parse_study(study)
                        trials.append(trial)

                    if len(trials) >= max_results:
                        break

                next_token = response.get("nextPageToken")
                if not next_token:
                    break

            except APIError as e:
                logger.error(f"API error: {e}")
                break

        logger.info(f"Found {len(trials)} trials for sponsor {sponsor}")
        return trials[:max_results]

    async def get_study_details(self, nct_id: str) -> Dict:
        """
        Fetch full details for specific trial by NCT ID.

        Args:
            nct_id: NCT identifier (e.g., "NCT00000102")

        Returns:
            Trial dictionary

        Raises:
            APIError: If study not found
        """
        logger.info(f"Fetching study: {nct_id}")

        try:
            response = await self.http_client.get(f"/studies/{nct_id}")

            # API returns single study (not array)
            study = response

            trial = self.parser.parse_study(study)
            return trial

        except APIError as e:
            logger.error(f"Failed to fetch {nct_id}: {e}")
            raise

    async def scrape_and_save(
        self,
        institution: str,
        output_file: str,
        status: Optional[List[str]] = None,
        phase: Optional[List[str]] = None,
        max_results: int = 1000,
        include_collaborator: bool = False
    ) -> None:
        """
        Complete workflow: search, save to JSON.

        Args:
            institution: Institution/sponsor name
            output_file: Path to save JSON results
            status: Filter by status
            phase: Filter by phase
            max_results: Maximum trials to retrieve
            include_collaborator: Search collaborator field too

        Example:
            >>> await scraper.scrape_and_save(
            ...     institution="Boston Children's Hospital",
            ...     output_file="data/bch_trials.json",
            ...     status=["RECRUITING", "ACTIVE_NOT_RECRUITING"],
            ...     max_results=500
            ... )
        """
        logger.info(f"Starting scrape: {institution}")

        # Search
        if include_collaborator:
            trials = await self.search_by_sponsor(
                institution,
                include_collaborator=True,
                status=status,
                max_results=max_results
            )
        else:
            trials = await self.search_by_institution(
                institution,
                status=status,
                phase=phase,
                max_results=max_results
            )

        if not trials:
            logger.warning("No trials found")
            return

        # Save
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(trials, f, indent=2)

        logger.info(f"✓ Saved {len(trials)} trials to {output_file}")

    async def close(self):
        """Close HTTP client."""
        await self.http_client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
```

### Phase 4: Custom Exceptions

```python
# src/scrapers/exceptions.py (extended)

class APIError(Exception):
    """Base exception for API errors."""
    pass

class NetworkError(APIError):
    """Raised when network request fails."""
    pass

class ParseError(APIError):
    """Raised when response parsing fails."""
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

# Run specific tests
pytest tests/test_clinicaltrials_scraper.py -v

# Run async tests
pytest tests/ -v -m asyncio
```

## Integration Testing

```python
# tests/integration/test_real_clinicaltrials.py

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
```

## Usage Examples

### Basic Usage

```python
import asyncio
from src.scrapers.clinicaltrials_scraper import ClinicalTrialsScraper

async def main():
    async with ClinicalTrialsScraper() as scraper:
        # Search by institution
        trials = await scraper.search_by_institution(
            "Boston Children's Hospital",
            status=["RECRUITING", "ACTIVE_NOT_RECRUITING"],
            max_results=100
        )

        print(f"Found {len(trials)} trials")

        # Print first trial
        if trials:
            trial = trials[0]
            print(f"\nNCT ID: {trial['nct_id']}")
            print(f"Title: {trial['title']}")
            print(f"Phase: {trial['phase']}")
            print(f"Status: {trial['status']}")
            print(f"Enrollment: {trial['enrollment']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Search with Filters

```python
# Search for recruiting Phase 2/3 trials
trials = await scraper.search_by_institution(
    "Mayo Clinic",
    status=["RECRUITING"],
    phase=["PHASE2", "PHASE3"],
    max_results=500
)
```

### Search Including Collaborators

```python
# Find trials where institution is lead sponsor OR collaborator
trials = await scraper.search_by_sponsor(
    "Harvard Medical School",
    include_collaborator=True,
    max_results=1000
)
```

### Save to File

```python
await scraper.scrape_and_save(
    institution="Stanford University",
    output_file="data/raw/stanford_trials.json",
    status=["RECRUITING"],
    max_results=1000
)
```

## Success Criteria

✅ **Must Have:**
1. All unit tests passing (>90% coverage)
2. Can search by lead sponsor/institution
3. Handles pagination correctly
4. Extracts all key fields: NCT ID, title, phase, status, interventions, outcomes
5. Saves results to valid JSON
6. Robust error handling

✅ **Should Have:**
7. Integration tests with real API
8. Search by collaborator
9. Filter by status and phase
10. Retry logic for transient failures

✅ **Nice to Have:**
11. Rate limiting (if needed)
12. Caching mechanism
13. Export to CSV/Excel

## Validation Checklist

- [ ] All unit tests pass
- [ ] Integration test with real API works
- [ ] Can scrape 100 trials successfully
- [ ] Handles pagination for large result sets
- [ ] JSON output is valid and complete
- [ ] Error handling tested
- [ ] Code follows PEP 8
- [ ] Docstrings complete

## Deliverables

1. **Source code**:
   - `src/scrapers/http_client.py`
   - `src/scrapers/clinicaltrials_scraper.py`
   - `src/scrapers/clinicaltrials_parser.py`

2. **Tests**:
   - `tests/test_http_client.py`
   - `tests/test_clinicaltrials_scraper.py`
   - `tests/test_clinicaltrials_parser.py`
   - `tests/integration/test_real_clinicaltrials.py`

3. **Documentation**:
   - README with usage examples
   - Sample output JSON

4. **Validation**:
   - Test coverage report (>90%)
   - Sample dataset (100-1000 trials)

## Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install httpx aiohttp pydantic loguru pytest pytest-asyncio pytest-mock respx tenacity

# Run tests
pytest tests/ -v --cov=src/scrapers
```

## API Reference

### Key Endpoints

**Search studies:**
```
GET /api/v2/studies?query.lead={institution}&pageSize=100
```

**Get specific study:**
```
GET /api/v2/studies/{nct_id}
```

**Common query parameters:**
- `query.lead`: Lead sponsor name
- `query.cond`: Condition/disease
- `filter.overallStatus`: Trial status
- `filter.phase`: Trial phase
- `pageSize`: Results per page (max 1000)
- `pageToken`: Pagination token

## Troubleshooting

**No results returned:**
- Check institution name spelling
- Try partial name match
- Use `include_collaborator=True`

**Pagination issues:**
- Verify `nextPageToken` handling
- Check `max_results` limit

**Parsing errors:**
- Log raw JSON for debugging
- Check for missing required fields
- Handle optional fields gracefully

## Additional Resources

- **API Documentation**: https://clinicaltrials.gov/data-api/api
- **API Explorer**: https://clinicaltrials.gov/data-api/query-builder
- **Study Structure Guide**: https://clinicaltrials.gov/data-api/about-api/study-data-structure

---

**Task completion**: When all tests pass, integration test works, and you can successfully scrape 100+ trials with comprehensive metadata extraction.
