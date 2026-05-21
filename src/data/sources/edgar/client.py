"""Async HTTP client for SEC EDGAR with a token-bucket rate limiter.

SEC's documented limit is "no more than 10 requests per second" with a
User-Agent identifying the requester (Sample User-Agent string from
https://www.sec.gov/os/accessing-edgar-data). We aim for 9 req/s sustained
to stay comfortably under the cap while still saturating throughput. A single
process-wide bucket coordinates all callers so that parallel modalities
(financials + form4 + filings text) share the budget rather than racing.

The bucket is `await`-driven: a coroutine asking for a token blocks just long
enough for one to be available, then proceeds. No busy-waiting, no global lock
contention. The bucket is created lazily by `get_client()` so importing this
module does not start an aiohttp session.
"""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import aiohttp
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

# SEC fair-use policy requires a contact in the User-Agent. Override via env
# if you need to attribute requests differently.
DEFAULT_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "norn-bwm research rohit@dragonscale.ai"
)
SEC_RPS = 9.0  # ceiling 10/s per SEC docs; leave ~1/s of headroom


class TokenBucket:
    """Process-wide async token bucket with steady refill at `rate` tokens/s.

    Tokens accumulate up to `capacity`, so short bursts above `rate` are
    permitted until the bucket drains. This matches SEC's observed tolerance
    for bursting within a one-second window.
    """

    def __init__(self, rate: float, capacity: Optional[float] = None) -> None:
        self.rate = rate
        self.capacity = capacity if capacity is not None else rate
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, n: float = 1.0) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._last) * self.rate
                )
                self._last = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
                deficit = n - self._tokens
                wait_s = deficit / self.rate
            # Sleep outside the lock so other coroutines can refill-check.
            await asyncio.sleep(wait_s)


_bucket: Optional[TokenBucket] = None


def _get_bucket() -> TokenBucket:
    global _bucket
    if _bucket is None:
        _bucket = TokenBucket(rate=SEC_RPS)
    return _bucket


class EdgarClient:
    """Thin async client over aiohttp. One instance per coroutine tree.

    Usage:
        async with EdgarClient() as client:
            body = await client.get_bytes("https://data.sec.gov/submissions/CIK0000320193.json")
    """

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_s: float = 60.0,
        max_concurrency: int = 16,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._sem = asyncio.Semaphore(max_concurrency)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "EdgarClient":
        self._session = aiohttp.ClientSession(
            timeout=self.timeout,
            headers={
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _request(self, method: str, url: str) -> aiohttp.ClientResponse:
        assert self._session is not None, "use 'async with EdgarClient() as c:'"
        await _get_bucket().acquire()
        async with self._sem:
            resp = await self._session.request(method, url)
            # Retry-After is sometimes returned on 429 / 503; respect it
            if resp.status in (429, 503):
                retry_after = float(resp.headers.get("Retry-After", "2"))
                await asyncio.sleep(min(retry_after, 30.0))
                resp.release()
                resp = await self._session.request(method, url)
            resp.raise_for_status()
            return resp

    async def get_bytes(self, url: str) -> bytes:
        """GET with retry on transient errors, return raw body bytes."""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(4),
            wait=wait_exponential_jitter(initial=1.0, max=20.0),
            retry=retry_if_exception_type(
                (aiohttp.ClientError, asyncio.TimeoutError)
            ),
            reraise=True,
        ):
            with attempt:
                resp = await self._request("GET", url)
                try:
                    return await resp.read()
                finally:
                    resp.release()
        raise RuntimeError("unreachable")  # pragma: no cover


@asynccontextmanager
async def edgar_client(**kwargs) -> AsyncIterator[EdgarClient]:
    """Convenience: `async with edgar_client() as c: ...`"""
    async with EdgarClient(**kwargs) as c:
        yield c
