import logging
import os


def setup_logger(log_dir, logger_name="train_log", log_level=logging.INFO):
    """
    Initialize and return a logger and its console handler.

    :param log_dir: directory where the log file is saved
    :param logger_name: name of the logger
    :param log_level: logging level
    :return: the logger object and the console handler object
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # File handler (writes logs to a file).
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{logger_name}.log"))
    file_handler.setLevel(log_level)

    # Log format.
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Attach the file handler to the logger.
    logger.addHandler(file_handler)

    # Create the console handler (not attached by default).
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    return logger, console_handler


def enable_console_logging(logger, console_handler, enable=True):
    """
    Enable or disable console log output at runtime.

    :param logger: the logger to control
    :param console_handler: the console handler
    :param enable: True to enable console output, False to disable it
    """
    if enable:
        if console_handler not in logger.handlers:
            logger.addHandler(console_handler)  # attach the console handler
    else:
        if console_handler in logger.handlers:
            logger.removeHandler(console_handler)  # detach the console handler
