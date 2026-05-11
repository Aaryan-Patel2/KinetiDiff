"""Package-level logging helpers."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a logger scoped to the kinetidiff namespace."""
    return logging.getLogger(name)
