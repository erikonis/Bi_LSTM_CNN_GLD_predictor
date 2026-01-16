from src.python_engine.training.Constants import ModelConst, ColNames
import pandas as pd
from datetime import datetime
import torch
import json
from pathlib import Path
import numpy as np
from src.utils.paths import get_model_dir
from src.python_engine.training.models import name_to_class
from src.utils.MetaConstants import MetaKeys
from src.python_engine.training.dataset_former import MarketDataset, unnormalize_price
from src.python_engine.fetch_data.News.near_live.yahoo_fetch import news_inference_pipeline


class ModelInference:
    def __init__(self, ticker: str, model_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.ticker = ticker
        self.mkt_cols = []
        self.sent_cols = []

        # 1. Locate files
        model_folder, _ = get_model_dir(ticker, model_name)
        weights_path = model_folder / f"{model_name}.pth"
        meta_path = model_folder / MetaKeys.METADATA_FILE
        # 2. Load Metadata
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
        
        # 3. Initialize the correct Class dynamically
        # (Assuming your classes are imported or in a registry)
        arch_name = self.meta[MetaKeys.MODEL_DETAILS][MetaKeys.ARCH]
        self.model = self._instantiate_model(arch_name)
        
        # 4. Load Weights
        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
        checkpoint = torch.load(weights_path, map_location=self.device)
        # Handle if you saved the whole state dict or just the model
        if ModelConst.MODEL_STATE_DICT in checkpoint:
            self.model.load_state_dict(checkpoint[ModelConst.MODEL_STATE_DICT])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        print(f"Model {model_name} loaded successfully on {self.device}")

    def _instantiate_model(self, name):
        # Using the logic we discussed earlier to pick the right class
        
        details = self.meta[MetaKeys.MODEL_DETAILS]
        hyperparams = self.meta[MetaKeys.HYPERPARAMS]
        self.mkt_cols = hyperparams[MetaKeys.MKT_COLS]
        self.sent_cols = hyperparams[MetaKeys.SENT_COLS]
        return name_to_class(name)(
            mkt_feat_dim=details[MetaKeys.MKT_DIM],
            sent_feat_dim=details[MetaKeys.SENT_DIM],
            hidden_dim=details[MetaKeys.HIDDEN_DIM],
            target_dim=details[MetaKeys.TARGET_DIM],
            dropout_rate=hyperparams[MetaKeys.DROPOUT]
        )
    
    def get_required_inputs(self):
        """Returns the required market and sentiment feature columns."""
        return self.mkt_cols, self.sent_cols

    def form_input(self, Open, High, Low, Close, Volume):
        """Forms the input numpy arrays for market and sentiment data.

        Loads the saved `MarketDataset`, appends the incoming OHLCV row,
        recomputes technicals and normalizations on a recent window,
        activates the market/sentiment columns required by the model and
        returns two numpy arrays `(mkt_data, sent_data)` shaped
        `(seq_len, n_features)` ready for `predict()`.
        """
        # Load saved dataset and determine seq length

        sentiment, count = news_inference_pipeline(self.ticker, 1)
        
        dataset = MarketDataset.load(ticker=self.ticker)
        mkt, sent = dataset.form_infere_sample(Open, High, Low, Close, Volume, mkt_cols=self.mkt_cols, sent_cols=self.sent_cols,
                                   sentiment=sentiment, sentiment_vol=count)
        return mkt, sent

    @torch.no_grad()
    def predict(self, mkt_data: np.ndarray, sent_data: np.ndarray):
        """
        Expects numpy arrays of shape (seq_len, features)
        """
        # Convert to Tensors and add Batch dimension (1, seq_len, features)
        mkt_tensor = torch.FloatTensor(mkt_data).unsqueeze(0).to(self.device)
        sent_tensor = torch.FloatTensor(sent_data).unsqueeze(0).to(self.device)
        #print(mkt_data[:2])
        #print(sent_data[:2])
        #print(mkt_tensor.shape, sent_tensor.shape)

        output = self.model(mkt_tensor, sent_tensor)
        #print(output)

        # Return as a simple list for Java to read easily
        return output.cpu().numpy().flatten().tolist()
    
    def infer(self, Open, High, Low, Close, Volume):
        """Convenience method to form input and predict in one call."""
        mkt_data, sent_data = self.form_input(Open, High, Low, Close, Volume)

        preds = np.array(self.predict(mkt_data, sent_data))
        return unnormalize_price(preds, Close)
    
if __name__ == "__main__":
    # Example usage
    ticker = "GLD"
    model_name = "predictor_bilstmCNN_GLD"
    
    inference_engine = ModelInference(ticker, model_name)
    
    mkt_cols, sent_cols = inference_engine.get_required_inputs()
    print(f"Market Columns: {mkt_cols}")
    print(f"Sentiment Columns: {sent_cols}")
    
    # Example OHLCV data (these would come from your data source)
    Open = 150.0
    High = 155.0
    Low = 149.0
    Close = 154.0
    Volume = 1000000
    
    predictions = inference_engine.infer(Open, High, Low, Close, Volume)
    print(f"Predictions: {predictions}")