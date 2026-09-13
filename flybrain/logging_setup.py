"""One place for the command-line scripts to configure loguru."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

CONSOLE_FORMAT = "<dim>{time:HH:mm:ss}</dim> <level>{level:<7}</level> {message}"


def configure(verbose: bool = False, log_file: str | Path | None = None) -> None:
    """
    Configure loguru for a command-line run.

    :param verbose: Console at DEBUG instead of INFO.
    :param log_file: Optional file that receives everything at DEBUG.
    """
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO", format=CONSOLE_FORMAT, colorize=True)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(str(log_file), level="DEBUG", rotation="10 MB", retention=5,
                   format="{time:YYYY-MM-DD HH:mm:ss} {level:<7} {name}:{function} {message}")
