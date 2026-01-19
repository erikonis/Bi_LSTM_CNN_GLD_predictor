from pathlib import Path
import os

"""Path helpers used to locate project, application, and data directories.

Functions return Path objects and ensure required folders exist.
"""


def get_project_root() -> Path:
    """Find the project root by locating `pyproject.toml` in parent folders."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback to current directory if not found
    return Path.cwd()


def get_app_root() -> Path:
    """Find the application root by locating a `java_brain` folder in parents."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "java_brain").exists():
            return parent
    # Fallback to current directory if not found
    return Path.cwd()


# Global folder constants
root_dir = get_app_root()
project_root = get_project_root()

# -- ROOT
data_dir = root_dir / "data"
config_dir = root_dir / "config"
logs_dir = root_dir / "logs"
processed_dir = data_dir / "processed"
raw_dir = data_dir / "raw"

config_dir.mkdir(exist_ok=True)
logs_dir.mkdir(exist_ok=True)
processed_dir.mkdir(exist_ok=True)
raw_dir.mkdir(exist_ok=True)

# -- NESTED
results_dir_name = "results"
MARKET_DATA_DIR = raw_dir / "market"
NEWS_DATA_DIR = raw_dir / "news"
DATASETS_DIR = processed_dir / "datasets"
MODELS_DIR = data_dir / "models"
quarantine_name = "quarantine"
QUARANTINE_DIR = MODELS_DIR / "quarantine"

CONFIG_TRAIN_PATH = config_dir / "train_config.yaml"
CONFIG_TICKERS_PATH = config_dir / "tickers.json"
LOG_PATH = logs_dir / "python_engine.log"
LOG_COMMS_PATH = logs_dir / "comms.log"
LOG_BRAIN_PATH = logs_dir / "python_engine.log"


MARKET_DATA_DIR.mkdir(exist_ok=True)
NEWS_DATA_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
QUARANTINE_DIR.mkdir(exist_ok=True)


def get_model_dir(ticker: str, model_name: str):
    """Return (model_dir, results_dir), creating them if necessary.

    Args:
        ticker: Stock ticker symbol.
        model_name: Model identifier.

    Returns:
        Tuple[Path, Path]: model directory and results directory paths.
    """
    model_folder = MODELS_DIR / ticker / model_name
    os.makedirs(model_folder, exist_ok=True)
    result_folder = model_folder / results_dir_name
    os.makedirs(result_folder, exist_ok=True)
    return model_folder, result_folder


def get_dataset_dir(ticker: str):
    """Return dataset folder for `ticker`, creating it if missing.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Path: dataset path for the ticker.
    """
    ticker = ticker.upper()
    data_folder = DATASETS_DIR / ticker
    os.makedirs(data_folder, exist_ok=True)
    return data_folder
