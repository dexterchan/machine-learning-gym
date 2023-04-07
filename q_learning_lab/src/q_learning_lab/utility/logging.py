import os
import os
from typing import Any, Union
from pythonjsonlogger import jsonlogger
from cachetools import cached, LRUCache
import logging

LOG_FILE: str = os.environ.get("LOG_FILE", "info.log")

format_str: str = "%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)"
formatter = logging.Formatter(format_str)

logging.basicConfig(format=format_str, level=logging.getLevelName("INFO"))


@cached(cache=LRUCache(maxsize=1024 * 1024))
def get_logger(name: str, level: Union[str, int] = logging.INFO) -> logging.Logger:
    """Produces a new logger with the application logging parameter preset.

    Args:
        name (str): [description]
        level (Union[str, int], optional): [description]. Defaults to LOG_LEVEL.

    Returns:
        logging.Logger: [description]
    """
    logger = logging.getLogger(name)
    _level = logging.getLevelName(level)
    logger.setLevel(_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(_level)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(_level)
    logger.addHandler(file_handler)
    return logger
