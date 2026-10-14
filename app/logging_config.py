import logging

from rich.logging import RichHandler

# 時刻とレベルはRichHandlerが列として表示する。
LOG_FORMAT = "%(name)s: %(message)s"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure application-wide logging with a Rich handler.

    The configuration is idempotent: if a ``RichHandler`` is already
    attached to the root logger, the function simply updates the log level
    and returns without adding duplicate handlers.
    """

    root_logger = logging.getLogger()
    if not any(isinstance(handler, RichHandler) for handler in root_logger.handlers):
        logging.basicConfig(
            format=LOG_FORMAT,
            handlers=[RichHandler(rich_tracebacks=True, markup=False)],
            force=True,
        )
    root_logger.setLevel(level)
    return root_logger
