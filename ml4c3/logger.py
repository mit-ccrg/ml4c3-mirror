"""Provides config settings for the logger and a way to load them"""

# Imports: standard library
import os
import sys
import errno
import logging.config


def load_config(log_level, log_dir, log_file_basename):
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{log_file_basename}.log"
    logger = logging.getLogger(__name__)
    try:
        logger_config = _create_logger_config(log_level=log_level, log_file=log_file)
        logging.config.dictConfig(logger_config)
        success_msg = (
            "Logging configuration was loaded. "
            f"Log messages can be found at {log_file}."
        )
        logger.info(success_msg)
    except Exception as error:
        logger.error("Failed to load logging config!")
        raise error


def _create_logger_config(log_level: str, log_file: str):
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "format": (
                    "%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
                ),
            },
        },
        "handlers": {
            "console": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": sys.stdout,
            },
            "file": {
                "level": log_level,
                "class": "logging.FileHandler",
                "formatter": "simple",
                "filename": log_file,
                "mode": "w",
            },
        },
        "loggers": {"": {"handlers": ["console", "file"], "level": log_level}},
    }
