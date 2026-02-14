"""General logger for the wonkyconn package."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def _setup_logger(log_level: str = "INFO") -> logging.Logger:
    """Create and configure the package-wide logger with rich output."""
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )

    return logging.getLogger("wonkyconn")


logger = _setup_logger()


def set_verbosity(verbosity: int | list[int]) -> None:
    """Set the logger verbosity level (0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG)."""
    if isinstance(verbosity, list):
        verbosity = verbosity[0]
    if verbosity == 0:
        logger.setLevel("ERROR")
    elif verbosity == 1:
        logger.setLevel("WARNING")
    elif verbosity == 2:
        logger.setLevel("INFO")
    elif verbosity == 3:
        logger.setLevel("DEBUG")
