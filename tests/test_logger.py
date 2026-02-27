"""Tests for structured logging setup."""

import logging

from src.utils.logger import get_logger, setup_logging


class TestSetupLogging:
    """Tests for logging configuration."""

    def test_setup_returns_logger(self):
        """setup_logging should return a logger instance."""
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "src"

    def test_setup_sets_level(self):
        """Logger level should match the requested level."""
        # Remove existing handlers first
        root = logging.getLogger("src")
        root.handlers.clear()

        logger = setup_logging(level=logging.DEBUG)
        assert logger.level == logging.DEBUG

        # Cleanup
        root.handlers.clear()

    def test_custom_format(self):
        """Custom log format should be applied."""
        root = logging.getLogger("src")
        root.handlers.clear()

        custom_fmt = "%(levelname)s - %(message)s"
        logger = setup_logging(log_format=custom_fmt)
        assert logger.handlers[0].formatter._fmt == custom_fmt

        root.handlers.clear()

    def test_idempotent_setup(self):
        """Calling setup_logging twice should not duplicate handlers."""
        root = logging.getLogger("src")
        root.handlers.clear()

        setup_logging()
        handler_count = len(root.handlers)
        setup_logging()
        assert len(root.handlers) == handler_count

        root.handlers.clear()


class TestGetLogger:
    """Tests for named logger retrieval."""

    def test_returns_named_logger(self):
        """get_logger should return a logger with the given name."""
        logger = get_logger("test.module")
        assert logger.name == "test.module"
