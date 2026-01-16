UNKNOWN = "unknown"


class MetaKeys:
    METADATA_FILE = "metadata.json"
    
    # Top-level Sections
    MODEL_DETAILS = "model_details"
    HYPERPARAMS = "hyperparameters"
    PERFORMANCE = "performance_metrics"
    DATASET = "dataset_details"
    OPTIMIZER = "optimizer"

    # Model Details Sub-keys
    ARCH = "architecture"
    NAME = "model_name"
    MKT_DIM = "mkt_feat_dim"
    SENT_DIM = "sent_feat_dim"
    HIDDEN_DIM = "hidden_dim"
    TARGET_DIM = "target_dim"
    CREATED_AT = "created_at"
    INFORMATION = "information"

    # Hyperparameters Sub-keys (Architecture Critical)
    MKT_COLS = "mkt_cols"
    SENT_COLS = "sent_cols"
    TARGETS = "targets"
    DROPOUT = "dropout_rate"
    BATCH_SIZE = "batch_size"
    EPOCHS = "epochs"
    EARLY_STOP = "early_stop"
    EARLY_STOP_PATIENCE = "early_stop_patience"
    FEATURE_THRESHOLD = "feature_threshold"
    DATA_SPLIT = "data_split"
    AUTO_FEATURE_ENGINEERING = "auto_feature_engineering"
    
    # Dataset Details
    TICKER = "ticker"
    DATE_RANGE = "date_range"
    NUM_SAMPLES = "num_samples"

    # Performance
    ACCURACY = "directional_accuracy"
    RMSE = "rmse"
    MAE = "mae"
    EPSILON_ACCURACY = "epsilon_accuracy"
    MAPE = "mape"

    # Optimizer
    TYPE = "type"
    LEARNING_RATE = "lr"
    WEIGHT_DECAY = "weight_decay"

class ConfigKeys:
    # Top-level Sections
    MODELS = "models"
    AVAILABLE_TICKERS = "available_tickers"

    # Model Instance Sub-keys (Inside each model name)
    TICKER = "ticker"
    TRAINED_AT = "trained"
    DATE_RANGE = "date_range"
    