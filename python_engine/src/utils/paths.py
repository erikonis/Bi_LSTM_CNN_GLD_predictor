from pathlib import Path
import os


# 1. Start at THIS file's location
# 2. Go up until you find the folder containing 'pyproject.toml'
def get_project_root() -> Path:
    """Finds the root of the project by looking for pyproject.toml."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback to current directory if not found
    return Path.cwd()


def get_app_root() -> Path:
    """Finds the root of the application by looking for 'java_brain' folder."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "java_brain").exists():
            return parent
    # Fallback to current directory if not found
    return Path.cwd()


# Define global constants for your folders
root_dir = get_app_root()
project_root = get_project_root()

# -- ROOT
data_dir = root_dir / "data"
config_dir = root_dir / "config"
logs_dir = data_dir / "logs"
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
QUARANTINE_DIR = MODELS_DIR / "quarantine"

CONFIG_TRAIN_PATH = config_dir / "train_config.yaml"
CONFIG_TICKERS_PATH = config_dir / "tickers.json"


MARKET_DATA_DIR.mkdir(exist_ok=True)
NEWS_DATA_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
QUARANTINE_DIR.mkdir(exist_ok=True)


def get_model_dir(ticker: str, model_name: str):
    """
    Get or create the model and results directories for a specific ticker and model name.
    Returns both the model directory and the results directory paths.
        Args:
            ticker (str): The stock ticker symbol.
            model_name (str): The name of the model.

        Returns:
            Tuple[Path, Path]: The paths to the model directory and results directory for the given ticker and model name.

    """
    model_folder = MODELS_DIR / ticker / model_name
    os.makedirs(model_folder, exist_ok=True)
    result_folder = model_folder / results_dir_name
    os.makedirs(result_folder, exist_ok=True)
    return model_folder, result_folder


def get_dataset_dir(ticker: str):
    """
    Get or create the dataset directory for a specific ticker.
     Args:
        ticker (str): The stock ticker symbol.

     Returns:
        Path: The path to the dataset directory for the given ticker.
    """
    ticker = ticker.upper()
    data_folder = DATASETS_DIR / ticker
    os.makedirs(data_folder, exist_ok=True)
    return data_folder
