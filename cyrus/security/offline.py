"""Process-local network denial used by Cyrus runtime commands and tests."""

from __future__ import annotations

import contextlib
import socket
from collections.abc import Iterator
from typing import Any


class OfflineViolation(PermissionError):
    """Raised before Cyrus runtime code can initiate an outbound connection."""


def _blocked(*_args: Any, **_kwargs: Any) -> None:
    raise OfflineViolation("outbound networking is disabled by Cyrus offline policy")


@contextlib.contextmanager
def offline_guard(enabled: bool = True) -> Iterator[None]:
    """Temporarily deny socket connection APIs in the current Python process.

    This is a defense-in-depth runtime guard, not an operating-system sandbox. For
    high-assurance air-gapped runs, combine it with container/network namespace or
    firewall isolation as documented in docs/SECURITY.md.
    """

    if not enabled:
        yield
        return

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection
    socket.socket.connect = _blocked  # type: ignore[method-assign]
    socket.socket.connect_ex = _blocked  # type: ignore[method-assign]
    socket.create_connection = _blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]
        socket.create_connection = original_create_connection

