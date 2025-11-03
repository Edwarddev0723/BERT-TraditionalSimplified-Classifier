from __future__ import annotations

import logging

from rich.logging import RichHandler


def setup_logger(name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Create a rich logger.

    Args:
        name: Logger name. If None, root logger is configured.
        level: Logging level.
    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="%H:%M:%S",
            handlers=[RichHandler(rich_tracebacks=True, show_level=True, show_time=True)],
        )
    logger.setLevel(level)
    return logger
