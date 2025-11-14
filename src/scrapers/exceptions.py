"""
Custom exceptions for PubMed scraper.
"""


class PubMedAPIError(Exception):
    """Base exception for PubMed API errors."""
    pass


class RateLimitError(PubMedAPIError):
    """Raised when rate limit is exceeded."""
    pass


class ParseError(PubMedAPIError):
    """Raised when XML parsing fails."""
    pass
