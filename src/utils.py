"""
Utility Functions for AI-Powered Cybersecurity Threat Detection System.

Provides logging setup, path resolution, and helper routines across modules.
"""

import logging
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """
    Return the root directory of the project.

    Returns
    -------
    Path
        Absolute path to the repository root.
    """
    return Path(__file__).resolve().parent.parent


def setup_logger(
    name: str = "threat_detection",
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configure and return a standardized logger instance.

    Parameters
    ----------
    name : str, default "threat_detection"
        Name of the logger.
    log_file : Optional[Path], default None
        Optional path to write log output.
    level : int, default logging.INFO
        Logging verbosity level.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_file))
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
