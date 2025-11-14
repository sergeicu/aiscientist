"""Custom exceptions for scrapers module."""


class APIError(Exception):
    """Base exception for API errors."""
    pass


class NetworkError(APIError):
    """Raised when network request fails."""
    pass


class ParseError(APIError):
    """Raised when response parsing fails."""
    pass
