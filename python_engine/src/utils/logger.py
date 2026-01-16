import logging
from rich.logging import RichHandler
from src.utils.paths import LOG_PATH

def setup_logging(name: str, log_file: str = LOG_PATH) -> logging.Logger:
    logger = logging.getLogger(name) # Give it a specific name
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