"""Structured logging setup for the ML monitoring stack."""

import logging
import sys


def setup_logging(
    level: int = logging.INFO,
    log_format: str | None = None,
) -> logging.Logger:
    """Configure structured logging for the application.

    Args:
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        log_format: Custom log format string. Uses a sensible default
            if not provided.

    Returns:
        The root logger instance configured for the application.
    """
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"

    root_logger = logging.getLogger("src")
    if root_logger.handlers:
        return root_logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger under the application namespace.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A logger instance.
    """
    return logging.getLogger(name)
