import json
import os
import shutil

from src.utils.paths import CONFIG_TICKERS_PATH, get_model_dir, QUARANTINE_DIR

def update_available_models(model_name, ticker, timestamp, date_range):
    """Updates the config/tickers.json file with the latest trained model info."""
    if os.path.exists(CONFIG_TICKERS_PATH):
        with open(CONFIG_TICKERS_PATH, "r") as f:
            try:
                tickers_config = json.load(f)
            except json.JSONDecodeError:
                tickers_config = {"available_tickers": [], "models": {}}
    else:
        tickers_config = {"available_tickers": [], "models": {}}

    tickers_config["models"][model_name] = {
        "ticker": ticker,
        "trained": timestamp,
        "date_range" : date_range
    }

    if ticker not in tickers_config["available_tickers"]:
        tickers_config["available_tickers"].append(ticker)

    with open(CONFIG_TICKERS_PATH, "w") as f:
        json.dump(tickers_config, f, indent=4)

    print(f"Updated {CONFIG_TICKERS_PATH} with model {model_name} for ticker {ticker}")

def sync_available_models():
    """Reads the models from the models directory and syncs with config/tickers.json."""
    models_root = get_model_dir("", "")[0]  # Get the base models directory



    available_models = {}
    for ticker in os.listdir(models_root):
        if ticker == "quarantine":
            continue
        ticker_dir = models_root / ticker
        if ticker_dir.is_dir():
            for model_name in os.listdir(ticker_dir):
                model_dir = ticker_dir / model_name
                if model_dir.is_dir():
                    metadata_path = model_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, "r", encoding="utf-8") as f:
                            try:
                                metadata = json.load(f)
                                details = metadata.get("model_details", {})
                                
                                if not details:
                                    print(f"Warning: {metadata_path} has no 'model_details'. Skipping.")
                                    print(f"Moving broken model {model_name} to quarantine...")
                                    shutil.move(str(model_dir), str(QUARANTINE_DIR / f"{ticker}_{model_name}"))
                                    continue

                                trained = details.get("created_at", "UNKNOWN")
                                date_range = details.get("date_range", "UNKNOWN")
                                
                                available_models[model_name] = {
                                    "ticker": ticker,
                                    "trained": trained,
                                    "date_range": date_range
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
                tickers_config = {"available_tickers": [], "models": {}}
    else:
        tickers_config = {"available_tickers": [], "models": {}}

    tickers_config["models"] = available_models
    tickers_config["available_tickers"] = list(
        set(tickers_config.get("available_tickers", [])).union(
            {info["ticker"] for info in available_models.values()}
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
                tickers_config = {"available_tickers": [], "models": {}}
    else:
        tickers_config = {"available_tickers": [], "models": {}}

    if model_name in tickers_config["models"]:
        del tickers_config["models"][model_name]

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
        "model_details": {
            "architecture": model.__class__.__name__,
            "model_name": model_name,
            "information": information,
            "mkt_feat_dim": getattr(model, "mkt_feat_dim", "unknown"),
            "sent_feat_dim": getattr(model, "sent_feat_dim", "unknown"),
            "hidden_dim": getattr(model, "hidden_dim", "unknown"),
            "target_dim": getattr(model, "target_dim", "unknown"),
            "created_at": timestamp,
        },
        "hyperparameters": hyperparams,
        "performance_metrics": {
            # Convert numpy floats to standard python floats so JSON can save them
            "rmse": float(performance.get("rmse", 0)),
            "directional_accuracy": float(performance.get("directional_accuracy", 0)),
            "epsilon_accuracy": float(performance.get("epsilon_accuracy", 0)),
            "mae": float(performance.get("mae", 0)),
            "mape": float(performance.get("mape", 0)),
        },
        "dataset_details": dataset_details,
        "optimizer": {
            "type": optimizer.__class__.__name__,
            "lr": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
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
                return tickers_config.get("available_tickers", [])
            except json.JSONDecodeError:
                return []
    return []

if __name__ == "__main__":
    print(get_available_tickers())