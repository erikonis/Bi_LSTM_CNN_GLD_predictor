import sys
from src.python_engine.training.train_model import main
from src.python_engine.training.Constants import ColNames, Training
from src.utils.data_management import get_available_tickers
from src.python_engine.training.models import name_to_class, ModelNames

import argparse

"""E.g. Usage:
uv run -m src.gui_backend.train_interface --model predictor_bilstmCNN --name predictor_bilstmCNN_GLD --ticker GLD --epochs 50 --lr 0.001 --hidden 64 --auto_feat --early_stop --batch 64 --dropout 0.3 --feat_threshold 0.01 --data_split expand --targets CLOSE_NORM --not_considered_feat SMA_20_NORM ATR_NORM MACD_SIG_NORM --info "BilstmCNN model for GLD  with auto feature selection"
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gold Predictor Models")

    # Define CLI arguments (invoked by Java or CLI)
    parser.add_argument("--model", type=str, required=True, choices=ModelNames.MODELS)
    parser.add_argument("--name", type=str, required=True, help="Model name to save as")
    parser.add_argument(
        "--auto_feat",
        action="store_true",
        default=False,
        help="Whether to enable automatic feature selection",
    )
    parser.add_argument(
        "--early_stop",
        action="store_true",
        default=False,
        help="Whether to enable early stopping",
    )
    parser.add_argument("--batch", type=int, default=64, help="Batch size for training")
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout rate for the model"
    )
    parser.add_argument(
        "--feat_threshold", type=float, default=0, help="Feature selection threshold"
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default=Training.DATASPLIT_EXPAND,
        choices=[Training.DATASPLIT_EXPAND, Training.DATASPLIT_SLIDE],
        help=f"Datasplit option. Choices: {Training.DATASPLIT_EXPAND}, {Training.DATASPLIT_SLIDE}",
    )
    parser.add_argument(
        "--ticker", type=str, choices=get_available_tickers(), required=True
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument(
        "--targets",
        type=list,
        nargs="+",
        default=[ColNames.CLOSE_NORM],
        help=f"Either of the following: {ColNames.TARGET_COLS_NORM}, {ColNames.TARGET_COLS}",
    )
    parser.add_argument(
        "--not_considered_feat",
        type=list,
        nargs="+",
        default=[
            ColNames.SMA_20_NORM,
            ColNames.SMA_50_NORM,
            ColNames.ATR_NORM,
            ColNames.MACD_SIG_NORM,
        ],
        help=f"Either of the following: {ColNames.TECHNICAL_COLS}, {ColNames.SENTIMENT_COLS}, or {ColNames.MARKET_COLS}",
    )
    parser.add_argument("--info", type=str, default="None")

    args = parser.parse_args()

    try:
        model = name_to_class(args.model)

        main(
            predictor_class=model,
            model_name=args.name,
            auto_feat_engineering=args.auto_feat,
            early_stop=args.early_stop,
            batch_size=args.batch,
            dropout_rate=args.dropout,
            feature_threshold=args.feat_threshold,
            data_split=args.data_split,
            targets=args.targets,
            not_considered_feat=args.not_considered_feat,
            ticker=args.ticker,
            epochs=args.epochs,
            learning_rate=args.lr,
            hidden_dim=args.hidden,
            information=args.info,
        )
    except Exception as e:
        print(f"PYTHON_EXECUTION_ERROR: {e}")
        sys.exit(1)
