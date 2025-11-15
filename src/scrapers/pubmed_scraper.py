"""
PubMed scraper for institutional affiliation searches.

Scrape PubMed articles by institutional affiliation using NCBI E-utilities API.
Features rate limiting, batch processing, and robust error handling.
"""

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
