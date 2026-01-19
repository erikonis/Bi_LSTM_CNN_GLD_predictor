from datetime import date, datetime, timedelta
import json
import logging
import os
import shutil
import pandas as pd
import numpy as np
import dtale

from src.python_engine.training.Constants import ColNames
from src.python_engine.fetch_data.News.yahoo_fetch import news_inference_pipeline
from src.python_engine.fetch_data.stock.fetch_data import fetchStockOHLCVdataframe
from src.utils.paths import CONFIG_TICKERS_PATH, get_model_dir, QUARANTINE_DIR, quarantine_name
from src.utils.MetaConstants import MetaKeys, ConfigKeys, UNKNOWN
from src.python_engine.training.dataset_former import MarketDataset

logger = logging.getLogger("BRAIN")

def update_available_models(model_name, ticker, timestamp, date_range):
    """Add or update a trained model entry in the tickers config file.

    Args:
        model_name: Identifier for the trained model.
        ticker: Stock ticker symbol the model was trained for.
        timestamp: Training timestamp (string or ISO format).
        date_range: Date range used for training (string or tuple).

    Returns:
        None
    """
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
    """Scan the models directory and rebuild the tickers config metadata.

    Reads per-model metadata files and moves broken entries to quarantine.

    Returns:
        None
    """
    models_root = get_model_dir("", "")[0]



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
                                    print(f"Warning: {metadata_path} missing '{MetaKeys.MODEL_DETAILS}'. Moving to quarantine.")
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
                                print(f"Error: {metadata_path} is corrupted. Moving to quarantine.")
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
    """Delete a model directory and remove its entry from the tickers config.

    Args:
        model_name: Name of the model to remove.

    Raises:
        ValueError: If the model is not found in any ticker folder.

    Returns:
        None
    """
    models_root = get_model_dir("", "")[0]

    model_found = False
    for ticker in os.listdir(models_root):
        ticker_dir = models_root / ticker
        if ticker_dir.is_dir():
            model_dir = ticker_dir / model_name
            if model_dir.is_dir():
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
    """Serialize model metadata to JSON for recording and reproducibility.

    Args:
        model: Trained model object (used to infer architecture and dims).
        optimizer: Optimizer instance (used to record learning rate, etc.).
        model_name: Human-readable model identifier.
        information: Short free-text description of the model/run.
        performance: Dict of numeric performance metrics (RMSE, MAE, etc.).
        hyperparams: Dict of hyperparameters used for training.
        dataset_details: Dict describing dataset (dates, sizes, etc.).
        path: Filesystem path where metadata JSON will be written.
        timestamp: Creation timestamp string.

    Returns:
        None
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
            # Ensure types are JSON serializable (convert numpy types to native floats)
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
    """Read available tickers from the config file.

    Returns:
        list: List of ticker strings; empty list if the config file is missing or invalid.
    """
    if os.path.exists(CONFIG_TICKERS_PATH):
        with open(CONFIG_TICKERS_PATH, "r") as f:
            try:
                tickers_config = json.load(f)
                return tickers_config.get(ConfigKeys.AVAILABLE_TICKERS, [])
            except json.JSONDecodeError:
                return []
    return []

def update_data(ticker):
    """Fetch new market OHLCV and per-day sentiment, append and save the dataset.

    Args:
        ticker: Stock ticker symbol to update (string).

    Returns:
        None
    """
    prefix = "update_data:"
    logger.info(f"{prefix} Updating data for ticker: {ticker}")
    
    dataset = MarketDataset.load(ticker)
    last_date = dataset.end_date
    start_date = last_date + timedelta(days=1)
    end_date = date.today()

    if start_date > end_date:
        logger.info(f"{prefix} Data for {ticker} is already up to date.")
        return

    # Fetch market price data
    logger.info(f"{prefix} Fetching market data for {ticker}")
    new_data = fetchStockOHLCVdataframe(ticker, start_date, end_date)
    
    if new_data.empty:
        logger.info(f"{prefix} No new market data found.")
        return

    # Fetch and map sentiment data (create Sentiment and News Volume columns)
    sentiment_scores = []
    news_volumes = []

    logger.info(f"{prefix} Processing sentiment for {len(new_data)} days...")
    
    if not isinstance(new_data.index, pd.DatetimeIndex):
        new_data.index = pd.to_datetime(new_data.index)

    for row_date in new_data.index:
        # Days difference from today; used to determine news fetch window
        delta_days = (datetime.now().date() - row_date.date()).days
        
        # Fetch news for that day; `news_inference_pipeline` returns (score, count)
        score, count = news_inference_pipeline(ticker, days=delta_days + 1)
        logger.info(f"{prefix} Date: {row_date.date()} | Sentiment: {score:.4f} | Articles: {count}")
        sentiment_scores.append(score)
        news_volumes.append(count)


    # Attach new columns and persist dataset
    new_data.index.name = "Date"
    new_data[ColNames.SENTIMENT] = sentiment_scores
    new_data[ColNames.SENTIMENT_VOL] = news_volumes
    
    new_data[ColNames.YEAR] = new_data.index.year.astype(np.int32)

    new_data = new_data.reset_index()
    # 4. Save to dataset
    dataset.add_data(new_data)
    logger.info(f"{prefix} Successfully updated {ticker} with market and sentiment data.")
    dataset.save()

if __name__ == "__main__":
    update_data("GLD")
    #print(get_available_tickers())