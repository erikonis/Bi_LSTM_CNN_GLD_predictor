from src.python_engine.training.Constants import ModelConst
from src.utils.MetaConstants import UNKNOWN, MetaKeys
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import os
from datetime import datetime
from src.utils.data_management import (
    generate_metadata,
    update_available_models,
)

class LogOHLCLoss(nn.Module):
    def __init__(self, penalty_weight=0.5):
        super(LogOHLCLoss, self).__init__()
        self.mse = nn.HuberLoss()
        self.penalty_weight = penalty_weight

    def forward(self, pred, target):
        # 1. Base Regression Loss (Standard Huber)
        base_loss = self.mse(pred, target)

        # If model predicts O,H,L,C per sample (shape: [batch, 4]), apply structural penalties.
        # Index mapping: 0=Open, 1=High, 2=Low, 3=Close
        total_penalty = 0.0

        if pred.ndim == 2 and pred.size(1) >= 4:
            # High should be >= Open, Low, Close -> penalize violations
            h_o_penalty = torch.mean(F.relu(pred[:, 0] - pred[:, 1]))
            h_l_penalty = torch.mean(F.relu(pred[:, 2] - pred[:, 1]))
            h_c_penalty = torch.mean(F.relu(pred[:, 3] - pred[:, 1]))

            # Low should be <= Open, High, Close -> penalize violations
            l_o_penalty = torch.mean(F.relu(pred[:, 2] - pred[:, 0]))
            l_c_penalty = torch.mean(F.relu(pred[:, 2] - pred[:, 3]))

            total_penalty = (
                h_o_penalty + h_l_penalty + h_c_penalty + l_o_penalty + l_c_penalty
            )

        return base_loss + (self.penalty_weight * total_penalty)


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_dim)
        attn_weights = torch.tanh(self.attn(lstm_output))  # (batch, seq_len, 1)
        soft_attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_output * soft_attn_weights, dim=1)
        return context, soft_attn_weights


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class PredictorSkeleton(ABC, nn.Module):
    def __init__(self):
        super(PredictorSkeleton, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def save(
        self,
        optimizer,
        model_name,
        information,
        performance,
        hyperparams,
        dataset_details,
        path,
    ):
        """Saves the model state and training context + JSON metadata."""
        timestamp = datetime.now().isoformat()
        state = {
            ModelConst.INFORMATION: information,
            ModelConst.MODEL_STATE_DICT: self.state_dict(),
            ModelConst.OPTIMIZER_STATE_DICT: optimizer.state_dict(),
            ModelConst.PERFORMANCE: sanitize_dict(performance),
            ModelConst.HYPERPARAMS: sanitize_dict(hyperparams),
            ModelConst.DATASET_DETAILS: sanitize_dict(dataset_details),
            ModelConst.TIMESTAMP: timestamp,
        }
        torch.save(state, path)

        # Define the json path (e.g., model.pth -> model_metadata.json)
        json_path = os.path.dirname(path) + f"/{MetaKeys.METADATA_FILE}"

        # Call our new metadata function
        generate_metadata(
            self,
            optimizer,
            model_name,
            information,
            performance,
            hyperparams,
            dataset_details,
            json_path,
            timestamp,
        )

        update_available_models(
            model_name,
            dataset_details.get(MetaKeys.TICKER, UNKNOWN),
            timestamp,
            dataset_details.get(MetaKeys.DATE_RANGE, UNKNOWN),
        )

        print(f"Model and Metadata saved to {os.path.dirname(path)}")

    def load_checkpoint(self, path: str, optimizer=None):
        if not os.path.exists(path):
            print(f"No saved model found at {path}")
            return None

        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint[ModelConst.MODEL_STATE_DICT])

        if optimizer:
            optimizer.load_state_dict(checkpoint[ModelConst.OPTIMIZER_STATE_DICT])

        print(f"Model loaded from Fold {checkpoint.get('fold', UNKNOWN)}")
        return checkpoint.get("fold", 0)

def sanitize_dict(d):
    """Recursively convert numpy types to native Python types."""
    import numpy as np
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            new_dict[k] = sanitize_dict(v)
        elif isinstance(v, (np.float32, np.float64, np.float16)):
            new_dict[k] = float(v)
        elif isinstance(v, (np.int32, np.int64, np.int16)):
            new_dict[k] = int(v)
        elif isinstance(v, np.ndarray):
            new_dict[k] = v.tolist()
        else:
            new_dict[k] = v
    return new_dict

class PredictorBiLSTMcnnA(PredictorSkeleton):
    def __init__(
        self, mkt_feat_dim, sent_feat_dim, hidden_dim=64, target_dim=4, dropout_rate=0.3
    ):
        super(PredictorBiLSTMcnnA, self).__init__()

        self.total_feat_dim = mkt_feat_dim + sent_feat_dim
        self.mkt_feat_dim = mkt_feat_dim
        self.sent_feat_dim = sent_feat_dim
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        self.dropout_rate = dropout_rate

        # 1. CNN Branch: Feature Extraction
        self.cnn5 = nn.Conv1d(
            self.total_feat_dim, self.hidden_dim, kernel_size=5, padding=2
        )
        self.relu_cnn = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn = nn.Dropout(self.dropout_rate)

        # 2. LSTM Branch: Temporal Dependencies
        # Bidirectional LSTM doubles the hidden dimension for the output
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, batch_first=True, bidirectional=True
        )

        # 3. Attention Layer
        # It takes the Bi-LSTM output (hidden_dim * 2) and calculates weights
        self.attention = Attention(self.hidden_dim * 2)

        # 4. Dense Head: Final Prediction
        self.fc_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.target_dim),
        )

    def forward(self, mkt_seq, sent_vec):
        # Early Fusion of Market + Sentiment
        combined = torch.cat((mkt_seq, sent_vec), dim=2)  # (batch, seq_len, total_dim)

        # CNN Processing
        x = combined.transpose(1, 2)  # (batch, total_dim, seq_len)
        x = self.relu_cnn(self.cnn5(x))
        x = self.pool(x)
        x = self.dropout_cnn(x)

        # LSTM Processing
        x = x.transpose(1, 2)  # (batch, reduced_seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)  # (batch, reduced_seq_len, hidden_dim * 2)

        # --- ATTENTION STEP ---
        # Instead of taking just the last hidden state, we weigh the whole sequence
        # context shape: (batch, hidden_dim * 2)
        context, attn_weights = self.attention(lstm_out)

        # Dense Head
        output = self.fc_fusion(context)

        return output


class PredictorBiLSTMcnn(PredictorSkeleton):
    def __init__(
        self, mkt_feat_dim, sent_feat_dim, hidden_dim=64, target_dim=4, dropout_rate=0.3
    ):
        super(PredictorBiLSTMcnn, self).__init__()

        self.total_feat_dim = mkt_feat_dim + sent_feat_dim
        self.mkt_feat_dim = mkt_feat_dim
        self.sent_feat_dim = sent_feat_dim
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        self.dropout_rate = dropout_rate

        # 1. CNN Branch
        # Input -> CNN + ReLU
        self.cnn5 = nn.Conv1d(
            self.total_feat_dim, self.hidden_dim, kernel_size=5, padding=2
        )
        self.relu_cnn = nn.ReLU()

        # 2. Pooling + Dropout
        # Using MaxPool1d to reduce temporal dimensionality/noise
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn = nn.Dropout(self.dropout_rate)

        # 3. LSTM Branch
        # Adjusting input_size because Pooling reduced the sequence length,
        # but hidden_dim (channels) remains the same.
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, batch_first=True, bidirectional=True
        )
        self.relu_lstm = nn.ReLU()

        # 4. Dense Head
        # hidden_dim * 2 because LSTM is bidirectional
        self.fc_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.target_dim),  # No Softmax for regression
        )

    def forward(self, mkt_seq, sent_vec):
        # Merge Market and Sentiment (Early Fusion)
        # mkt_seq: (batch, seq_len, mkt_dim)
        # sent_vec: (batch, seq_len, sent_dim)
        combined = torch.cat((mkt_seq, sent_vec), dim=2)

        # CNN expects (batch, channels, seq_len)
        x = combined.transpose(1, 2)

        # Input -> CNN + ReLU
        x = self.relu_cnn(self.cnn5(x))

        # -> Pooling -> Dropout
        x = self.pool(x)
        x = self.dropout_cnn(x)

        # -> LSTM + ReLU
        # Back to (batch, seq_len, hidden_dim) for LSTM
        x = x.transpose(1, 2)
        lstm_out, (hn, cn) = self.lstm(x)

        # We use the final hidden state for the Dense layer
        # (Concatenate forward and backward last hidden states)
        # Shape of hn: (num_layers * num_directions, batch, hidden_dim)
        last_hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        x = self.relu_lstm(last_hidden)

        # -> Dense + ReLU -> Dropout -> Output
        output = self.fc_fusion(x)

        return output

class ModelNames:
    BILSTM_CNN = PredictorBiLSTMcnn.__name__.lower()
    BILSTM_CNN_A = PredictorBiLSTMcnnA.__name__.lower()
    MODELS = [BILSTM_CNN, BILSTM_CNN_A]

def name_to_class(name: str):
    match name.lower():
        case ModelNames.BILSTM_CNN:
            return PredictorBiLSTMcnn
        case ModelNames.BILSTM_CNN_A:
            return PredictorBiLSTMcnnA
        case _:
            raise ValueError(f"Unknown model name: {name}")
