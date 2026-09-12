"""Minimal, consistent logging setup used across the project."""
from __future__ import annotations

import logging

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str | int = "INFO") -> None:
    """Configure the root logger once, idempotently."""
    global _CONFIGURED
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if not _CONFIGURED:
        logging.basicConfig(level=level, format=_FORMAT)
        _CONFIGURED = True
    else:
        logging.getLogger().setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
