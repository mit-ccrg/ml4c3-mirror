"""Provides config settings for the logger and a way to load them"""

# Imports: standard library
import os
import sys
import errno
import logging.config


def load_config(log_level, log_dir, log_file_basename):
    try:
        os.makedirs(log_dir)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise error

    logger = logging.getLogger(__name__)

    log_file = f"{log_dir}/{log_file_basename}.log"

    try:
        logging.config.dictConfig(_create_config(log_level, log_file))
        success_msg = (
            "Logging configuration was loaded. "
            f"Log messages can be found at {log_file}."
        )
        logger.info(success_msg)
    except Exception as error:
        logger.error("Failed to load logging config!")
        raise error


def _create_config(log_level, log_file):
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": (
                    "%(asctime)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s"
                ),
            },
            "detailed": {
                "format": "%(name)s:%(levelname)s %(module)s:%(lineno)d:  %(message)s",
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
