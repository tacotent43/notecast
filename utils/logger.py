import logging
import sys

def setup_logger(
    name: str = __name__,
    level: int = logging.INFO,
    log_to_file: bool = False,
    filename: str = "app.log"
) -> logging.Logger:
    """
    Logger configuration with output to command line and (optional) to file

    :param name: logger name
    :param level: logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param log_to_file: logging to filename
    :param filename: filename for logs
    :return: logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        file_handler = logging.FileHandler(filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger