"""
Tests for RateLimiter class.

Tests token bucket rate limiting for PubMed API calls.
"""

import pytest
import asyncio
import time
from src.scrapers.rate_limiter import RateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_allows_burst():
    """Should allow burst_size requests immediately."""
    limiter = RateLimiter(requests_per_second=3.0, burst_size=10)

    start = time.time()
    for _ in range(10):
        await limiter.acquire()
    elapsed = time.time() - start

    # Should complete almost instantly (< 0.1s)
    assert elapsed < 0.1


@pytest.mark.asyncio
async def test_rate_limiter_enforces_rate():
    """Should enforce rate limit after burst exhausted."""
    limiter = RateLimiter(requests_per_second=3.0, burst_size=3)

    # Exhaust burst
    for _ in range(3):
        await limiter.acquire()

    # Next 3 requests should take ~1 second
    start = time.time()
    for _ in range(3):
        await limiter.acquire()
    elapsed = time.time() - start

    # Should take approximately 1 second (3 req / 3 req/s)
    assert 0.9 < elapsed < 1.2


@pytest.mark.asyncio
async def test_rate_limiter_refills_tokens():
    """Tokens should refill over time."""
    limiter = RateLimiter(requests_per_second=10.0, burst_size=10)

    # Exhaust tokens
    for _ in range(10):
        await limiter.acquire()

    # Wait for refill
    await asyncio.sleep(0.5)  # Should refill 5 tokens

    # Should be able to make 5 more requests quickly
    start = time.time()
    for _ in range(5):
        await limiter.acquire()
    elapsed = time.time() - start

    assert elapsed < 0.1


@pytest.mark.asyncio
async def test_rate_limiter_concurrent_safety():
    """Should be thread-safe for concurrent requests."""
    limiter = RateLimiter(requests_per_second=10.0, burst_size=20)

    async def make_request():
        await limiter.acquire()
        return True

    # 50 concurrent requests
    tasks = [make_request() for _ in range(50)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 50
    assert all(results)


def test_rate_limiter_invalid_params():
    """Should raise ValueError for invalid parameters."""
    with pytest.raises(ValueError, match="requests_per_second must be positive"):
        RateLimiter(requests_per_second=0)

    with pytest.raises(ValueError, match="burst_size must be positive"):
        RateLimiter(requests_per_second=3.0, burst_size=0)


def test_rate_limiter_reset():
    """Should reset to full burst capacity."""
    limiter = RateLimiter(requests_per_second=3.0, burst_size=10)

    # Manually reduce tokens
    limiter.tokens = 0

    # Reset
    limiter.reset()

    assert limiter.tokens == 10.0
