import logging
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / f'log_{datetime.now().strftime("%Y-%m-%d")}.log'

logging.basicConfig(
    filename=LOG_FILE,
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def get_logger(name):
    """
    Returns a logger instance with the specified name.

    Args:
        name (str): The name of the logger, typically __name__ of the module.

    Returns:
        logging.Logger: A logger object that can be used to log messages.

    Example:
        from logger import get_logger

        logger = get_logger(__name__)
        logger.info('Starting the application')
        logger.error('This is an error')
    """
    return logging.getLogger(name)
