"""A local System One server, for the decision-model tests and the documentation's example.

A Kev-enabled `llama-server` answers System One requests at `SYSTEM_ONE_URL`. Tests that need it
skip when nothing listens there (`requires_system_one`, root `conftest.py`).
"""

from __future__ import annotations

import functools
import socket
from urllib.parse import urlsplit

SYSTEM_ONE_URL = "http://127.0.0.1:8080/v1/systemone"


@functools.cache
def system_one_reachable() -> bool:
    """Whether a server accepts connections at `SYSTEM_ONE_URL`'s host and port."""
    parts = urlsplit(SYSTEM_ONE_URL)
    try:
        socket.create_connection((parts.hostname, parts.port), timeout=1).close()
    except OSError:
        return False
    return True
