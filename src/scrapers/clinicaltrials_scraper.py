"""Scrape ClinicalTrials.gov by institution/sponsor."""

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
