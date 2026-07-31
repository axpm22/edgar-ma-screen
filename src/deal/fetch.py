"""Rate-limited HTTP with a disk cache.

Every download goes through here. Two reasons: the SEC bans IPs that exceed
10 requests/second, and a parser bug should never cost a re-download of
several GB.
"""
import hashlib
import threading
import time
from pathlib import Path
from typing import Callable

import httpx

from . import config

CACHE_ROOT = Path("data/raw")


class RateLimiter:
    """Spaces calls to at most per_second. Thread-safe."""

    def __init__(self, per_second: float):
        self.per_second = per_second
        self._interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
            self._next = max(now, self._next) + self._interval


# SEC's published ceiling is 10/s. Sitting at 8 leaves headroom.
SEC_LIMITER = RateLimiter(per_second=8.0)


def cache_path(source: str, key: str) -> Path:
    # Keys are URLs; hash them so slashes cannot escape into the directory tree.
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    return CACHE_ROOT / source / digest


def cached(source: str, key: str, fetcher: Callable[[], bytes]) -> bytes:
    path = cache_path(source, key)
    if path.exists():
        return path.read_bytes()
    data = fetcher()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def sec_get(url: str) -> bytes:
    def _go() -> bytes:
        SEC_LIMITER.wait()
        r = httpx.get(
            url,
            headers={"User-Agent": config.EDGAR_UA},
            timeout=300,
            follow_redirects=True,
        )
        r.raise_for_status()
        return r.content

    return cached("sec", url, _go)
