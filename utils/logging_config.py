"""
utils/logging_config.py
-----------------------
Centralised logging setup for IntelliMoE.

Call ``setup_logging()`` once at application startup (in main.py or
ui/app.py) before any other module is imported. All subsequent calls
are no-ops thanks to the idempotency guard.

Design:
  - Single StreamHandler → stdout (compatible with Streamlit and Docker).
  - Format and level pulled from config.settings → one place to change.
  - Idempotent: safe to call multiple times across modules.
"""

import logging
import sys

from config.settings import LOG_DATE_FMT, LOG_FORMAT, LOG_LEVEL


def setup_logging(level: int = LOG_LEVEL) -> None:
    """
    Configure the root logger for the IntelliMoE application.

    Parameters
    ----------
    level : int
        Logging level (e.g. ``logging.INFO``, ``logging.DEBUG``).
        Defaults to the value in ``config.settings.LOG_LEVEL``.
    """
    root_logger = logging.getLogger()

    # Idempotency guard — do nothing if handlers are already attached.
    if root_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FMT)
    )

    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    logging.getLogger(__name__).debug("Logging initialised at level %s.", level)
