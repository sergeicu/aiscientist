"""HTTP client with retry logic and error handling."""

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
