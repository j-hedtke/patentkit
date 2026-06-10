"""Shared HTTP plumbing for all patentkit connectors.

Centralizes the things every connector needs:

- :func:`request_json` / :func:`request_text` / :func:`download` built on
  ``httpx`` with a tenacity retry policy (3 attempts, exponential backoff)
  that retries transport errors, timeouts, HTTP 429 and 5xx responses,
- :class:`RateLimiter`, a simple thread-safe minimum-interval sleeper so
  scraping and free-tier APIs stay polite,
- a default ``User-Agent`` of ``patentkit/0.1``.

Connectors pass their own auth headers; no keys live in this module.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "patentkit/0.1"
DEFAULT_TIMEOUT = 30.0
DOWNLOAD_TIMEOUT = 300.0


class RetryableHTTPStatusError(httpx.HTTPStatusError):
    """An HTTP status (429 or 5xx) worth retrying."""


class RateLimiter:
    """Thread-safe minimum-interval sleeper.

    ``RateLimiter(0.5).wait()`` guarantees at least 0.5s between successive
    returns from :meth:`wait`, sleeping as needed. A ``min_interval_s`` of 0
    disables limiting entirely.
    """

    def __init__(self, min_interval_s: float = 0.0):
        self.min_interval_s = min_interval_s
        self._last: float = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.min_interval_s <= 0:
            return
        with self._lock:
            now = time.monotonic()
            remaining = self._last + self.min_interval_s - now
            if remaining > 0:
                time.sleep(remaining)
                now = time.monotonic()
            self._last = now


def _merge_headers(headers: Optional[Mapping[str, str]]) -> dict[str, str]:
    merged = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        merged.update(headers)
    return merged


_retry_policy = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(
        (httpx.TransportError, httpx.TimeoutException, RetryableHTTPStatusError)
    ),
    reraise=True,
)


@_retry_policy
def _request(
    method: str,
    url: str,
    *,
    headers: Optional[Mapping[str, str]] = None,
    params: Optional[Mapping[str, Any]] = None,
    json: Any = None,
    data: Any = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> httpx.Response:
    response = httpx.request(
        method,
        url,
        headers=_merge_headers(headers),
        params=params,
        json=json,
        data=data,
        timeout=timeout,
        follow_redirects=True,
    )
    if response.status_code == 429 or response.status_code >= 500:
        logger.warning("Retryable HTTP %s from %s", response.status_code, url)
        raise RetryableHTTPStatusError(
            f"HTTP {response.status_code} from {url}",
            request=response.request,
            response=response,
        )
    response.raise_for_status()
    return response


def request_json(
    method: str,
    url: str,
    *,
    headers: Optional[Mapping[str, str]] = None,
    params: Optional[Mapping[str, Any]] = None,
    json: Any = None,
    data: Any = None,
    timeout: float = DEFAULT_TIMEOUT,
    rate_limiter: Optional[RateLimiter] = None,
) -> Any:
    """Make an HTTP request and return the decoded JSON body."""
    if rate_limiter:
        rate_limiter.wait()
    return _request(
        method, url, headers=headers, params=params, json=json, data=data, timeout=timeout
    ).json()


def request_text(
    url: str,
    *,
    headers: Optional[Mapping[str, str]] = None,
    params: Optional[Mapping[str, Any]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    rate_limiter: Optional[RateLimiter] = None,
) -> str:
    """GET a URL and return the response body as text."""
    if rate_limiter:
        rate_limiter.wait()
    return _request("GET", url, headers=headers, params=params, timeout=timeout).text


def download(
    url: str,
    dest: Optional[Union[str, Path]] = None,
    *,
    headers: Optional[Mapping[str, str]] = None,
    params: Optional[Mapping[str, Any]] = None,
    timeout: float = DOWNLOAD_TIMEOUT,
    rate_limiter: Optional[RateLimiter] = None,
) -> Union[bytes, Path]:
    """GET a URL's raw bytes; write them to ``dest`` if given.

    Returns the bytes when ``dest`` is None, otherwise the destination Path.
    """
    if rate_limiter:
        rate_limiter.wait()
    response = _request("GET", url, headers=headers, params=params, timeout=timeout)
    if dest is None:
        return response.content
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(response.content)
    return dest_path
