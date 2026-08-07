"""Logging helpers for the datactrl package."""

from __future__ import annotations

import logging
import sys
from typing import TextIO

LOGGER_NAME = "datactrl"

_NOISY_LOGGERS = ("imgtools", "PIL", "urllib3", "httpx", "httpcore")


def configure(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
    fmt: str = "%(levelname)s %(name)s: %(message)s",
    quiet_third_party: bool = True,
) -> None:
    """Configure a stream handler for ``datactrl`` loggers.

    Safe to call from scripts or tests; library code does not invoke this
    automatically so importing ``datactrl`` stays quiet by default.
    """
    resolved = getattr(logging, str(level).upper()) if isinstance(level, str) else level

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger(LOGGER_NAME)
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved)
    root.propagate = False

    if quiet_third_party:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
