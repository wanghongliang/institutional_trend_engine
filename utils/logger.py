"""
Logging utilities for the engine.

Provides a configured logger that writes to both console and file.
"""

import logging
import os
import sys
from pathlib import Path


def setup_logger(
    name: str = "engine",
    level: str = "INFO",
    log_file: str = "logs/engine.log",
) -> logging.Logger:
    """
    Create or retrieve a configured logger.

    Parameters
    ----------
    name : str
        Logger name (hierarchical, dot-separated).
    level : str
        Logging level: DEBUG / INFO / WARNING / ERROR.
    log_file : str
        Path to the log file. Parent dirs are created automatically.

    Returns
    -------
    logging.Logger
        A configured logger with console + file handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
