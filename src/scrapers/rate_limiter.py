"""
Token bucket rate limiter for API calls.

Implements a token bucket algorithm to enforce rate limits while
allowing burst traffic. Thread-safe for async operations.
"""

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
                self.last_update = time.time()  # Update after wait
            else:
                self.tokens -= 1.0

    def reset(self) -> None:
        """Reset to full burst capacity."""
        self.tokens = float(self.burst_size)
        self.last_update = time.time()
