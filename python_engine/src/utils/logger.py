import logging
from rich.logging import RichHandler
from src.utils.paths import LOG_PATH

def setup_logging(name: str, log_file: str = LOG_PATH) -> logging.Logger:
    """Create and configure a named logger.

    Args:
        name: Logger name (usually module or component name).
        log_file: Path-like or string pointing to a log file.

    Returns:
        logging.Logger: Configured logger instance with both console and file handlers.
    """
    logger = logging.getLogger(name)  # named logger for this module/component
    logger.setLevel(logging.INFO)

    # Handlers
    shell_handler = RichHandler(rich_tracebacks=True)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_formatter = logging.Formatter(f'[PYTH-{name}]  %(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    # Add them
    logger.addHandler(shell_handler)
    logger.addHandler(file_handler)
    return logger