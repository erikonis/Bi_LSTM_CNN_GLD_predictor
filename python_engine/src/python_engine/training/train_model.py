import os
from typing import List, Any, Type
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
import logging
import torch.nn as nn
import torch

from src.python_engine.training.models import *
from src.python_engine.training.Constants import ColNames, Training
import src.python_engine.training.dataset_former as dataset_former
from src.python_engine.training.analysis import *
from src.utils.paths import get_dataset_dir, get_model_dir

# @dataclass
# class TrainingConfig:
#     # Required arguments (no defaults)
#     predictor_class: Type[Any]
#     targets: List[str]
#     not_considered_feat: List[str]
#     ticker: str
#     model_name: str
    
#     # Optional arguments (with defaults)
#     auto_feat_engineering: bool = False
#     early_stop: bool = False
#     hidden_dim: int = 64
#     batch_size: int = 64
#     epochs: int = 100
#     learning_rate: float = 0.001
#     dropout_rate: float = 0.3
#     feature_threshold: float = 0.0
#     data_split: str = Training.DATASPLIT_EXPAND
#     information: str = "None"

#     def to_dict(self):
#         """Useful for saving to metadata.json later"""
#         return asdict(self)

def feature_engineering(
    predictor_class,
    train_loader,
    val_loader,
    mkt_cols,
    sent_cols,
    device,
    logger,
    dropout_rate=0.3,
    hidden_dim=64,
    target_dim=4,
    threshold=0,
):
    # --- PHASE 1: Signal Discovery (Quick Run) ---
    logger.info("PHASE 1: Identifying feature signals...")

    # Initialize a temporary model to test importance
    mkt_dim = len(mkt_cols)  # Initial count
    sent_dim = len(sent_cols)
    temp_model = predictor_class(
        mkt_dim,
        sent_dim,
        hidden_dim=hidden_dim,
        target_dim=target_dim,
        dropout_rate=dropout_rate,
    ).to(device)

    train_model(
        temp_model,
        device,
        train_loader,
        val_loader,
        mkt_cols,
        sent_cols,
        epochs=50,
        lr=0.001,
        logger=logger,
        early_stop=False,
    )

    # Calculate initial importance using your existing function
    importances = calculate_permutation_importance(
        temp_model, val_loader, mkt_cols, sent_cols, fold="Discovery"
    )

    # Determine which features to keep/drop
    to_keep, to_drop = identify_low_importance_features(
        importances, threshold=threshold
    )
    logger.info(f"Dropping noise features: {to_drop}")
    logger.info(f"Training with signal features: {to_keep}")

    # Update global feature list for the final model
    filtered_mkt = [f for f in to_keep if f in mkt_cols]
    filtered_sent = [f for f in to_keep if f in sent_cols]

    print(":::::::::::::::::::: Filtered features ::::::::::::::::::::")
    print(f"To keep features: {to_keep}")
    print(f"Dropping noise features: {to_drop}")
    print(f"Total market filtered features: {filtered_mkt}")
    print(f"Total sentiment filtered features: {filtered_sent}")

    # --- PHASE 2: Optimized Training ---
    logger.info("PHASE 2: Starting optimized training with Early Stopping...")
    final_model = predictor_class(
        len(filtered_mkt),
        len(filtered_sent),
        hidden_dim=hidden_dim,
        target_dim=target_dim,
        dropout_rate=dropout_rate,
    ).to(device)

    return final_model, filtered_mkt, filtered_sent


def setup_logger(filename: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(filename),
            logging.StreamHandler(),  # Still prints to console
        ],
    )
    return logging.getLogger()


def save_predictions_csv(preds, targets, filename: str):
    df = pd.DataFrame(
        {
            "Predicted_Return": preds.flatten() / 100,
            "Actual_Return": targets.flatten() / 100,
        }
    )
    # Add a column to see the error magnitude
    df["Error"] = df["Actual_Return"] - df["Predicted_Return"]

    df.to_csv(filename, index=False)
    print(f"saved {len(df)} predictions to {filename}")


def train_model(
    model,
    device,
    train_loader,
    val_loader,
    market_cols,
    sent_cols,
    epochs=50,
    lr=0.0005,
    logger=None,
    early_stop=False,
    dataset_denorm_fn=None,
):
    """
    Trains the model with the given data loaders.
    :param model: nn.Module - the model to train
    :param device: torch.device - device to run the training on
    :param train_loader: DataLoader - training data loader
    :param val_loader: DataLoader - validation data loader
    :param market_cols: list - list of market feature names
    :param sent_cols: list - list of sentiment feature names
    :param epochs: int - number of training epochs
    :param lr: float - learning rate
    :param logger: logging.Logger - logger for logging info
    :param early_stop: bool - whether to use early stopping
    :param dataset_denorm_fn: function - function to denormalize dataset values
    :return: history dict, optimizer, weight history dict
    """
    model.to(device)

    feature_names = market_cols + sent_cols

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Allow for adaptive learning rate:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    if early_stop:
        stopper = EarlyStopping(patience=10)
    criterion = LogOHLCLoss()  # Robust to financial outliers

    history = {"train_loss": [], "val_loss": []}
    weight_history = {name: [] for name in feature_names}

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for mkt_data, sent_data, targets, all_real_prices in train_loader:
            mkt_data, sent_data, targets = (
                mkt_data.to(device),
                sent_data.to(device),
                targets.to(device),
            )

            optimizer.zero_grad()
            outputs = model(mkt_data, sent_data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []

        all_val_preds = []
        all_val_targets = []
        all_real_prices = []

        with torch.no_grad():
            # Extract conv input-channel importance if available
            conv_weights = _extract_conv_feature_weights(model)
            if conv_weights.size:
                all_current_weights = conv_weights
            else:
                # fallback: try to probe common attribute names
                try:
                    all_current_weights = (
                        model.cnn5.weight.abs().mean(dim=(0, 2)).cpu().numpy()
                    )
                except Exception:
                    all_current_weights = np.zeros(len(feature_names))

            for i, name in enumerate(feature_names):
                if i < len(all_current_weights):
                    weight_history[name].append(float(all_current_weights[i]))
                else:
                    weight_history[name].append(0.0)

            # Validation loop
            for mkt_data, sent_data, targets, real_price in val_loader:
                mkt_data, sent_data, targets = (
                    mkt_data.to(device),
                    sent_data.to(device),
                    targets.to(device),
                )
                outputs = model(mkt_data, sent_data)
                v_loss = criterion(outputs, targets)
                val_losses.append(v_loss.item())

                all_val_preds.append(outputs.cpu())
                all_val_targets.append(targets.cpu())
                all_real_prices.append(real_price)

        # Concatenate all batches
        all_val_preds = torch.cat(all_val_preds, dim=0).numpy()
        all_val_targets = torch.cat(all_val_targets, dim=0).numpy()
        all_real_prices = torch.cat(all_real_prices, dim=0).numpy()

        # Sanity check"
        logger.info("\nSanity CHECK:")
        logger.info(
            f"DBG shapes: {all_val_preds.shape}, {all_val_targets.shape}, {all_real_prices.shape}"
        )
        logger.info(
            f"DBG preds stats: min={np.min(all_val_preds)}, max={np.max(all_val_preds)}, median={np.median(all_val_preds)}, std={np.std(all_val_preds)}"
        )
        logger.info(
            f"DBG actuals stats: min={np.min(all_val_targets)}, max={np.max(all_val_targets)}, median={np.median(all_val_targets)}, std={np.std(all_val_targets)}"
        )

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        scheduler.step(avg_val_loss)

        current_lr = scheduler.get_last_lr()[0]

        # Verbose logging every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
            )

            if logger:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
                )
                logger.info(f"Current LR: {current_lr:.6f}")

        # Calculate trading metrics on validation set using normalized arrays
        epsilon_pct = 0.002
        val_metrics = calculate_trading_metrics(
            all_val_preds,
            all_val_targets,
            epsilon_pct=epsilon_pct,
            last_close_prices=all_real_prices,
            dataset_denorm_func=dataset_denorm_fn,
        )

        # Adding val metrics to the history for performance plotting
        history.setdefault("directional_accuracy", []).append(
            val_metrics.get("directional_accuracy", np.nan)
        )
        history.setdefault("epsilon_accuracy", []).append(
            val_metrics.get("epsilon_accuracy", np.nan)
        )
        history.setdefault("mape", []).append(val_metrics.get("mape", np.nan))

        if logger:
            logger.info(f"Epoch {epoch}:")
            logger.info(f"  - Range Coverage: {val_metrics['range_coverage']:.2%}")
            logger.info(
                f"  - Epsilon Hit ({epsilon_pct * 100}%): {val_metrics['epsilon_accuracy']:.2%}"
            )
            logger.info(
                f"  - Directional Acc: {val_metrics['directional_accuracy']:.2%}"
            )
        else:
            print(f"Epoch {epoch}:")
            print(f"  - Range Coverage: {val_metrics['range_coverage']:.2%}")
            print(
                f"  - Epsilon Hit ({epsilon_pct * 100}%): {val_metrics['epsilon_accuracy']:.2%}"
            )
            print(f"  - Directional Acc: {val_metrics['directional_accuracy']:.2%}")

        # Early stop logic
        if early_stop:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                logger.info(f"Best model updated at Epoch {epoch}")

            stopper(avg_val_loss)
            if stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    if early_stop:
        model.load_state_dict(best_model_state)

    return history, optimizer, weight_history


def _extract_conv_feature_weights(model: nn.Module) -> np.ndarray:
    """Extract mean absolute weights per input channel from the first Conv1d layer.

    Returns an array of length equal to input channels. If no Conv1d found,
    returns an empty numpy array.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            # weight shape: (out_channels, in_channels, kernel_size)
            w = m.weight.data.abs().mean(dim=(0, 2)).cpu().numpy()
            return w
    return np.array([])


def evaluate_test_set(
    model,
    test_loader,
    filename: str,
    logger=None,
    scaler_target=None,
    debug=False,
    dataset_denorm_fn=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_preds = []
    all_actuals = []
    all_real_prices = []

    with torch.no_grad():
        for mkt_data, sent_data, targets, real_price in test_loader:
            mkt_data, sent_data, targets = (
                mkt_data.to(device),
                sent_data.to(device),
                targets.to(device),
            )

            outputs = model(mkt_data, sent_data)

            # Inverse transform to get back to real dollar prices if you scaled them
            if scaler_target is not None:
                actuals = scaler_target.inverse_transform(targets.cpu().numpy())
                preds = scaler_target.inverse_transform(outputs.cpu().numpy())
            else:
                actuals = targets.cpu().numpy()
                preds = outputs.cpu().numpy()

            all_preds.append(torch.from_numpy(preds))
            all_actuals.append(torch.from_numpy(actuals))
            all_real_prices.append(real_price)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_actuals = torch.cat(all_actuals, dim=0).numpy()
    all_real_prices = torch.cat(all_real_prices, dim=0).numpy()

    # --- Metric 1: RMSE (Regression Error) ---
    mse = np.mean((all_preds - all_actuals) ** 2)
    rmse = np.sqrt(mse)

    # --- Metric 2: Trading Metrics ---
    epsilon_pct = 0.002
    val_metrics = calculate_trading_metrics(
        all_preds,
        all_actuals,
        epsilon_pct=epsilon_pct,
        last_close_prices=all_real_prices,
        dataset_denorm_func=dataset_denorm_fn,
    )

    # --- MEtric 3: Backtest Strategy ---

    print(
        "preds range:",
        all_preds.min(),
        all_preds.max(),
        "median:",
        np.median(np.abs(all_preds)),
    )
    print(
        "targets range:",
        all_actuals.min(),
        all_actuals.max(),
        "median:",
        np.median(np.abs(all_actuals)),
    )

    results = backtest_with_costs(all_preds, all_actuals)

    # --- Metric 4: Threshold Sensitivity Analysis ---
    find_best_threshold(all_preds, all_actuals, logger)

    if not logger:
        print("--- Final Test Results ---")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"  - Range Coverage: {val_metrics['range_coverage']:.2%}")
        print(
            f"  - Epsilon Hit ({epsilon_pct * 100}%): {val_metrics['epsilon_accuracy']:.2%}"
        )
        print(f"  - Directional Acc: {val_metrics['directional_accuracy']:.2%}")
        print(f"  - MAE: {val_metrics['mae']:.4f}")
        print(f"  - MAPE: {val_metrics['mape']:.2f}%")
        print(
            f"  - Max Predicted Move (log-return): {val_metrics['max_pred_move']:.4f}"
        )
        print(
            f"  - Avg Predicted Move (log-return): {val_metrics['avg_pred_move']:.4f}"
        )

        print("\n--- Trading Performance (with 0.03% fee and $1000 capital) ---")
        print(f"Total Trades: {results['num_trades']}")
        print(f"Strategy Final Value: ${results['equity_curve'][-1]:.2f}")
        print(f"Buy & Hold Final Value: ${results['buy_and_hold'][-1]:.2f}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")

        if results["final_value"] > (results["buy_and_hold"][-1]):
            print("STRATEGY OUTPERFORMED MARKET")
        else:
            print("MARKET OUTPERFORMED STRATEGY")
    else:
        logger.info("--- Final Test Results ---")
        logger.info(f"Test RMSE: {rmse:.4f}")
        logger.info(f"  - Range Coverage: {val_metrics['range_coverage']:.2%}")
        logger.info(
            f"  - Epsilon Hit ({epsilon_pct * 100}%): {val_metrics['epsilon_accuracy']:.2%}"
        )
        logger.info(f"  - Directional Acc: {val_metrics['directional_accuracy']:.2%}")
        logger.info(f"  - MAE: {val_metrics['mae']:.4f}")
        logger.info(f"  - MAPE: {val_metrics['mape']:.2f}%")
        logger.info(
            f"  - Max Predicted Move (log-return): {val_metrics['max_pred_move']:.4f}"
        )
        logger.info(
            f"  - Avg Predicted Move (log-return): {val_metrics['avg_pred_move']:.4f}"
        )

        logger.info("\n--- Trading Performance (with 0.03% fee and $1000 capital) ---")
        logger.info(f"Total Trades: {results['num_trades']}")
        logger.info(f"Strategy Final Value: ${results['final_value']:.2f}")
        logger.info(f"Buy & Hold Final Value: ${results['buy_and_hold'][-1]:.2f}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")

        if results["final_value"] > results["buy_and_hold"][-1]:
            logger.info("STRATEGY OUTPERFORMED MARKET")
        else:
            logger.info("MARKET OUTPERFORMED STRATEGY")

    # plotting
    plot_model_results(all_preds, all_actuals, results, filename=filename)

    return all_preds, all_actuals, val_metrics, all_real_prices


def main(
    predictor_class,
    targets,
    not_considered_feat,
    ticker: str,
    model_name: str,
    auto_feat_engineering: bool = False,
    early_stop: bool = False,
    hidden_dim: int = 64,
    batch_size: int = 64,
    epochs: int = 100,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.3,
    feature_threshold: float = 0.0,
    data_split: str = Training.DATASPLIT_EXPAND,
    information: str = "None",
):
    # ---------  SETUP   ---------
    model_dir, result_dir = get_model_dir(ticker, model_name)
    dataset_dir = get_dataset_dir(ticker)
    dataset = dataset_former.MarketDataset.load(dataset_dir)

    mkt_cols = [f for f in dataset._market_cols if f not in not_considered_feat]
    sent_cols = [f for f in dataset._sent_cols if f not in not_considered_feat]

    dataset.set_active_cols(
        new_market_cols=mkt_cols, new_sent_cols=sent_cols, new_target_cols=targets
    )

    mkt_feat_dim = len(mkt_cols)
    sent_feat_dim = len(sent_cols)
    target_dim = len(targets)
    feature_names = mkt_cols + sent_cols
    logger = setup_logger(result_dir / "train.log")
    # -----------------------------------------
    # ---- Log report ----
    logger.info("Hyperparameters:")
    logger.info(f"  - Automatic Feature Engineering: {auto_feat_engineering}")
    logger.info(f"  - Early Stopping: {early_stop}")
    logger.info("----------  FEATURES  --------")
    logger.info(f"  - Targets: {targets}")
    logger.info(f"  - Market Features: {mkt_cols}")
    logger.info(f"  - Sentiment Features: {sent_cols}")
    logger.info(f"  - Not considered Features: {not_considered_feat}")
    logger.info("-------------------------------")
    logger.info("Model Configuration:")
    logger.info(f"  - Hidden Dim: {hidden_dim}")
    logger.info(f"  - Batch Size: {batch_size}")
    logger.info(f"  - Epochs: {epochs}")
    logger.info(f"  - Learning Rate: {learning_rate}")
    logger.info(f"  - Feature Threshold: {feature_threshold}")
    logger.info(f"  - Output Folder: {model_dir}")
    logger.info(f"  - Data Split: {data_split}")
    logger.info("  - Architecture 2: CNN + LSTM + Attention")
    logger.info("-" * 50)
    logger.info("Additional information:")
    logger.info("  - cnn5 only")
    logger.info(f"  - Predictor Class: {predictor_class.__name__}")
    # Initialize Device and Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if auto_feat_engineering:
        for train, val, test, test_year in dataset.get_loaders(
            training_setting=data_split, batch_size=batch_size
        ):
            last_train = train
            last_val = val
        model, mkt_cols, sent_cols = feature_engineering(
            predictor_class,
            last_train,
            last_val,
            mkt_cols,
            sent_cols,
            device,
            logger,
            dropout_rate=dropout_rate,
            hidden_dim=hidden_dim,
            target_dim=target_dim,
            threshold=feature_threshold,
        )
        dataset.set_active_cols(mkt_cols, sent_cols)
        feature_names = mkt_cols + sent_cols
    else:
        model = predictor_class(
            mkt_feat_dim,
            sent_feat_dim,
            hidden_dim,
            target_dim,
            dropout_rate=dropout_rate,
        )

    # Weights collection
    weights = {name: [] for name in feature_names}

    # Cross-Validation Folds
    fold_num = 0
    for train_loader, val_loader, test_loader, test_year in dataset.get_loaders(
        training_setting=data_split, batch_size=batch_size
    ):
        fold_num += 1

        fold_dir = result_dir / f"fold_{fold_num}"
        os.makedirs(fold_dir, exist_ok=True)

        logger.info(
            f"\n\n ==============> Starting Fold {fold_num} | Test Year: {test_year}"
        )

        # Train Model
        history, optimizer, fold_weights = train_model(
            model,
            device,
            train_loader,
            val_loader,
            market_cols=mkt_cols,
            sent_cols=sent_cols,
            epochs=epochs,
            lr=learning_rate,
            logger=logger,
            early_stop=early_stop,
            dataset_denorm_fn=dataset.unnormalize_price,
        )

        # metrics plotting:
        # metrics_to_plot = {k: history[k] for k in ('directional_accuracy','epsilon_accuracy','mape') if k in history}
        plot_training_history(
            history, filename=f"{fold_dir}/training_history{fold_num}.png"
        )  # metrics=metrics_to_plot,

        for name in feature_names:
            weights[name].extend(fold_weights[name])

        # Evaluate on Test Set
        all_preds, all_actuals, performance, all_real_prices = evaluate_test_set(
            model,
            test_loader,
            logger=logger,
            dataset_denorm_fn=dataset.unnormalize_price,
            filename=f"{fold_dir}/performance_summary{fold_num}.png",
        )

        # --- Metric: Plotting pred vs. actual close price
        last_close = all_real_prices[:, 3]
        plot_close_pred_vs_actual(
            all_preds,
            all_actuals,
            last_close=last_close,
            denorm_fn=dataset.unnormalize_price,
            c_id=-1,
            filename=f"{fold_dir}/pred_vs_actual_close{fold_num}.png",
        )

        logger.info(f"Completed Fold {fold_num}\n")

        save_predictions_csv(
            all_preds,
            all_actuals,
            filename=f"{fold_dir}/gold_predictions_fold_{fold_num}.csv",
        )
        # === FEATURE ANALYSIS ===

        # 3. Run Permutation Test (How much MAE drops when a feature is "broken")
        p_importance = calculate_permutation_importance(
            model,
            val_loader,
            mkt_cols,
            sent_cols,
            fold_num,
            filename=f"{fold_dir}/fold_{fold_num}_permutation_importance.png",
        )

        # Sort and log results
        sorted_imp = sorted(p_importance.items(), key=lambda x: x[1], reverse=True)

        logger.info(f"\n--- Permutation Importance Fold {fold_num} ---")
        for feat, imp in sorted_imp:
            # We multiply by 1000 to make the small log-return errors easier to read
            logger.info(f"{feat:<15}: {imp * 1000:.6f} (scaled x1000)")
        plot_feature_time_heatmap(
            model,
            val_loader,
            feature_names=feature_names,
            filename=f"{fold_dir}/fold_{fold_num}_saliency_heatmap.png",
        )
        plot_feature_weights(
            model,
            feature_names,
            filename=f"{fold_dir}/fold_{fold_num}_feature_weights.png",
        )
    # After folding
    plot_maw_progression(
        weights, fold_num="1-5", filename=f"{result_dir}/feature_weights_over_folds.png"
    )

    current_hyperparams = {
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "early_stop": early_stop,
        "auto_feature_engineering": auto_feat_engineering,
        "feature_threshold": feature_threshold,
        "dropout_rate": dropout_rate,
        "early_stop_patience": 10 if early_stop else None,
        "data_split": data_split,
        "targets": targets,
        "mkt_cols": mkt_cols,
        "sent_cols": sent_cols,
    }

    dataset_details = {
        "dataset_architecture": dataset.__class__.__name__,
        "ticker": ticker,
        "num_samples": len(dataset),
        "date_range": (str(dataset.start_date), str(dataset.end_date)),
    }

    model.save(
        optimizer,
        model_name=model_name,
        information=information,
        performance=performance,
        hyperparams=current_hyperparams,
        dataset_details=dataset_details,
        path=f"{model_dir}/{model_name}.pth",
    )


if __name__ == "__main__":
    model_name = "bilstmcnn"
    information = "Bi-LSTM-CNN Gold Predictor"
    predictor_class = PredictorBiLSTMcnn
    ticker = "GLD"
    automatic_feature_engineering = False
    early_stop = False

    targets = [ColNames.TARGET_C_NORM]
    not_considered_feat = [
        ColNames.MACD_SIG_NORM,
        ColNames.RSI_NORM,
        ColNames.SMA_20_NORM,
        ColNames.SMA_50_NORM,
    ]

    target_dim = len(targets)
    hidden_dim = 64
    batch_size = 64
    epochs = 100
    learning_rate = 0.001
    feature_threshold = 0
    dropout_rate = 0.3
    data_split = Training.DATASPLIT_EXPAND

    main(
        predictor_class,
        targets,
        not_considered_feat,
        ticker,
        model_name,
        automatic_feature_engineering,
        early_stop,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        feature_threshold=feature_threshold,
        data_split=data_split,
        information=information,
    )
