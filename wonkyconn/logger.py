"""General logger for the cohort_creator package."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def gc_logger(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )

    return logging.getLogger("giga_connectome")


gc_log = gc_logger()


def set_verbosity(verbosity: int | list[int]) -> None:
    if isinstance(verbosity, list):
        verbosity = verbosity[0]
    if verbosity == 0:
        gc_log.setLevel("ERROR")
    elif verbosity == 1:
        gc_log.setLevel("WARNING")
    elif verbosity == 2:
        gc_log.setLevel("INFO")
    elif verbosity == 3:
        gc_log.setLevel("DEBUG")
