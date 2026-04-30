from __future__ import annotations

from loguru import logger
import os


def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logger.remove()
    logger.add(os.path.join(log_dir, "app.log"), rotation="10 MB", retention=10, level="INFO")


__all__ = ["logger", "setup_logging"]


