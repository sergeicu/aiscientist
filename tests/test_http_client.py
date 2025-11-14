"""Tests for HTTP client with retry logic."""

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
    await client.close()


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
    await client.close()


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
    await client.close()


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
    await client.close()


@pytest.mark.asyncio
async def test_timeout_handling(respx_mock):
    """Should handle request timeouts."""
    respx_mock.get("https://api.example.com/timeout").mock(
        side_effect=httpx.TimeoutException("Timeout")
    )

    client = HTTPClient(timeout=5.0, max_retries=1)

    with pytest.raises(NetworkError):
        await client.get("https://api.example.com/timeout")

    await client.close()


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
    await client.close()
