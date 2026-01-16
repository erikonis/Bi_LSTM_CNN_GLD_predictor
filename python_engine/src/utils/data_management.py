import json
import os
import shutil

from src.utils.paths import CONFIG_TICKERS_PATH, get_model_dir, QUARANTINE_DIR, quarantine_name
from src.utils.MetaConstants import MetaKeys, ConfigKeys, UNKNOWN

def update_available_models(model_name, ticker, timestamp, date_range):
    """Updates the config/tickers.json file with the latest trained model info."""
    if os.path.exists(CONFIG_TICKERS_PATH):
        with open(CONFIG_TICKERS_PATH, "r") as f:
            try:
                tickers_config = json.load(f)
            except json.JSONDecodeError:
                tickers_config = {ConfigKeys.AVAILABLE_TICKERS: [], ConfigKeys.MODELS: {}}
    else:
        tickers_config = {ConfigKeys.AVAILABLE_TICKERS: [], ConfigKeys.MODELS: {}}

    tickers_config[ConfigKeys.MODELS][model_name] = {
        ConfigKeys.TICKER: ticker,
        ConfigKeys.TRAINED_AT: timestamp,
        ConfigKeys.DATE_RANGE : date_range
    }

    if ticker not in tickers_config[ConfigKeys.AVAILABLE_TICKERS]:
        tickers_config[ConfigKeys.AVAILABLE_TICKERS].append(ticker)
    with open(CONFIG_TICKERS_PATH, "w") as f:
        json.dump(tickers_config, f, indent=4)

    print(f"Updated {CONFIG_TICKERS_PATH} with model {model_name} for ticker {ticker}")

def sync_available_models():
    """Reads the models from the models directory and syncs with config/tickers.json."""
    models_root = get_model_dir("", "")[0]  # Get the base models directory



    available_models = {}
    for ticker in os.listdir(models_root):
        if ticker == quarantine_name:
            continue
        ticker_dir = models_root / ticker
        if ticker_dir.is_dir():
            for model_name in os.listdir(ticker_dir):
                model_dir = ticker_dir / model_name
                if model_dir.is_dir():
                    metadata_path = model_dir / MetaKeys.METADATA_FILE
                    if metadata_path.exists():
                        with open(metadata_path, "r", encoding="utf-8") as f:
                            try:
                                metadata = json.load(f)
                                details = metadata.get(MetaKeys.MODEL_DETAILS, {})
                                
                                if not details:
                                    print(f"Warning: {metadata_path} has no '{MetaKeys.MODEL_DETAILS}'. Skipping.")
                                    print(f"Moving broken model {model_name} to quarantine...")
                                    shutil.move(str(model_dir), str(QUARANTINE_DIR / f"{ticker}_{model_name}"))
                                    continue

                                trained = details.get(MetaKeys.CREATED_AT, UNKNOWN)
                                date_range = details.get(MetaKeys.DATE_RANGE, UNKNOWN)
                                
                                available_models[model_name] = {
                                    ConfigKeys.TICKER: ticker,
                                    ConfigKeys.TRAINED_AT: trained,
                                    ConfigKeys.DATE_RANGE: date_range
                                }
                            except json.JSONDecodeError:
                                print(f"Error: {metadata_path} is empty or corrupted. Skipping.")
                                print(f"Moving broken model {model_name} to quarantine...")
                                shutil.move(str(model_dir), str(QUARANTINE_DIR / f"{ticker}_{model_name}"))
                                continue
                    else:
                        print(f"Moving broken model {model_name} to quarantine...")
                        shutil.move(str(model_dir), str(QUARANTINE_DIR / f"{ticker}_{model_name}"))

    # Now update the config file
    if os.path.exists(CONFIG_TICKERS_PATH):
        with open(CONFIG_TICKERS_PATH, "r") as f:
            try:
                tickers_config = json.load(f)
            except json.JSONDecodeError:
                tickers_config = {ConfigKeys.AVAILABLE_TICKERS: [], ConfigKeys.MODELS: {}}
    else:
        tickers_config = {ConfigKeys.AVAILABLE_TICKERS: [], ConfigKeys.MODELS: {}}

    tickers_config[ConfigKeys.MODELS] = available_models
    tickers_config[ConfigKeys.AVAILABLE_TICKERS] = list(
        set(tickers_config.get(ConfigKeys.AVAILABLE_TICKERS, [])).union(
            {info[ConfigKeys.TICKER] for info in available_models.values()}
        )
    )

    with open(CONFIG_TICKERS_PATH, "w") as f:
        json.dump(tickers_config, f, indent=4)

    print(f"Synchronized available models to {CONFIG_TICKERS_PATH}")

def delete_model(model_name):
    """Deletes a model directory and updates the config file."""
    models_root = get_model_dir("", "")[0]  # Get the base models directory

    model_found = False
    for ticker in os.listdir(models_root):
        ticker_dir = models_root / ticker
        if ticker_dir.is_dir():
            model_dir = ticker_dir / model_name
            if model_dir.is_dir():
                # Delete the model directory
                shutil.rmtree(model_dir)
                model_found = True
                print(f"Deleted model directory: {model_dir}")

    if not model_found:
        raise ValueError(f"Model {model_name} not found in any ticker directory.")

    # Now update the config file
    if os.path.exists(CONFIG_TICKERS_PATH):
        with open(CONFIG_TICKERS_PATH, "r") as f:
            try:
                tickers_config = json.load(f)
            except json.JSONDecodeError:
                tickers_config = {ConfigKeys.AVAILABLE_TICKERS: [], ConfigKeys.MODELS: {}}
    else:
        tickers_config = {ConfigKeys.AVAILABLE_TICKERS: [], ConfigKeys.MODELS: {}}

    if model_name in tickers_config[ConfigKeys.MODELS]:
        del tickers_config[ConfigKeys.MODELS][model_name]

    with open(CONFIG_TICKERS_PATH, "w") as f:
        json.dump(tickers_config, f, indent=4)

    print(f"Updated {CONFIG_TICKERS_PATH} after deleting model {model_name}")

def generate_metadata(
    model, optimizer, model_name, information, performance, hyperparams, dataset_details, path, timestamp
):
    """
    Creates a JSON file with model architecture, training params, and performance.
    """
    metadata = {
        MetaKeys.MODEL_DETAILS: {
            MetaKeys.ARCH: model.__class__.__name__,
            MetaKeys.NAME: model_name,
            MetaKeys.INFORMATION: information,
            MetaKeys.MKT_DIM: getattr(model, "mkt_feat_dim", UNKNOWN),
            MetaKeys.SENT_DIM: getattr(model, "sent_feat_dim", UNKNOWN),
            MetaKeys.HIDDEN_DIM: getattr(model, "hidden_dim", UNKNOWN),
            MetaKeys.TARGET_DIM: getattr(model, "target_dim", UNKNOWN),
            MetaKeys.CREATED_AT: timestamp,
        },
        MetaKeys.HYPERPARAMS: hyperparams,
        MetaKeys.PERFORMANCE: {
            # Convert numpy floats to standard python floats so JSON can save them
            MetaKeys.RMSE: float(performance.get(MetaKeys.RMSE, 0)),
            MetaKeys.ACCURACY: float(performance.get(MetaKeys.ACCURACY, 0)),
            MetaKeys.EPSILON_ACCURACY: float(performance.get(MetaKeys.EPSILON_ACCURACY, 0)),
            MetaKeys.MAE: float(performance.get(MetaKeys.MAE, 0)),
            MetaKeys.MAPE: float(performance.get(MetaKeys.MAPE, 0)),
        },
        MetaKeys.DATASET: dataset_details,
        MetaKeys.OPTIMIZER: {
            MetaKeys.TYPE: optimizer.__class__.__name__,
            MetaKeys.LEARNING_RATE: optimizer.param_groups[0]["lr"],
            MetaKeys.WEIGHT_DECAY: optimizer.param_groups[0].get("weight_decay", 0),
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {path}")

def get_available_tickers():
    """Returns a list of available tickers from the config file."""
    if os.path.exists(CONFIG_TICKERS_PATH):
        with open(CONFIG_TICKERS_PATH, "r") as f:
            try:
                tickers_config = json.load(f)
                return tickers_config.get(ConfigKeys.AVAILABLE_TICKERS, [])
            except json.JSONDecodeError:
                return []
    return []

if __name__ == "__main__":
    print(get_available_tickers())