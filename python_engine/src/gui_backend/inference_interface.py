import sys
import argparse
import json

from src.python_engine.inference.inference import ModelInference
from src.utils.data_management import get_available_tickers



"""
E.g. Usage:
uv run -m src.gui_backend.inference_interface --action get_inputs --ticker GLD --model predictor_bilstmCNN_GLD --json
uv run -m src.gui_backend.inference_interface --action infer --ticker GLD --model predictor_bilstmCNN_GLD --open 150 --high 155 --low 149 --close 154 --volume 1000000 --json
"""

def main():
    parser = argparse.ArgumentParser(description="Run model inference or query required inputs")

    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=["get_inputs", "infer"],
        help="Action to perform: 'get_inputs' returns required cols; 'infer' runs prediction",
    )

    parser.add_argument("--ticker", type=str, required=True, choices=get_available_tickers())
    parser.add_argument("--model", type=str, required=True, help="Model folder/name to load (e.g. predictor_bilstmCNN_GLD)")

    # OHLCV for inference
    parser.add_argument("--open", type=float, help="Open price (required for infer)")
    parser.add_argument("--high", type=float, help="High price (required for infer)")
    parser.add_argument("--low", type=float, help="Low price (required for infer)")
    parser.add_argument("--close", type=float, help="Close price (required for infer)")
    parser.add_argument("--volume", type=float, help="Volume (required for infer)")

    parser.add_argument("--json", action="store_true", help="Print output as JSON")

    args = parser.parse_args()

    try:
        if args.action == "get_inputs":
            engine = ModelInference(args.ticker, args.model)
            mkt_cols, sent_cols = engine.get_required_inputs()
            out = {"market_cols": mkt_cols, "sentiment_cols": sent_cols}
            if args.json:
                print(json.dumps(out))
            else:
                print("Market Columns:", mkt_cols)
                print("Sentiment Columns:", sent_cols)

        elif args.action == "infer":
            required = [args.open, args.high, args.low, args.close, args.volume]
            if any(v is None for v in required):
                raise ValueError("For 'infer' action you must provide --open --high --low --close --volume")

            engine = ModelInference(args.ticker, args.model)
            preds = engine.infer(args.open, args.high, args.low, args.close, args.volume)

            # preds may be a numpy array or list
            try:
                serializable = [float(x) for x in preds]
            except Exception:
                serializable = preds

            if args.json:
                print(json.dumps({"predictions": serializable}))
            else:
                print("Predictions:", serializable)

    except Exception as e:
        print(f"PYTHON_EXECUTION_ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
