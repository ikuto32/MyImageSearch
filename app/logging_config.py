import logging
from typing import Optional

from rich.logging import RichHandler


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure application-wide logging with a Rich handler.

    The configuration is idempotent: if a ``RichHandler`` is already
    attached to the root logger, the function simply updates the log level
    and returns without adding duplicate handlers.
    """

    root_logger = logging.getLogger()
    existing_rich_handler: Optional[logging.Handler] = next(
        (handler for handler in root_logger.handlers if isinstance(handler, RichHandler)),
        None,
    )

    if existing_rich_handler:
        root_logger.setLevel(level)
        return root_logger

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )

    return logging.getLogger()
