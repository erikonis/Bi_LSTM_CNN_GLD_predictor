class ColNames:
    DATE = "Date"
    YEAR = "Year"

    OPEN = "Open"
    HIGH = "High"
    LOW = "Low"
    CLOSE = "Close"
    VOLUME = "Volume"
    RSI = "RSI"
    SMA_20 = 'SMA_20_Ratio'
    SMA_50 = 'SMA_50_Ratio'
    MACD = 'MACD'
    MACD_SIG = 'MACD_Sig'
    ATR = 'ATR'
    BB_LOWER = 'BB_Lower_Ratio'
    BB_UPPER = 'BB_Upper_Ratio'
    
    OPEN_NORM = 'Log_Ret_Open'
    HIGH_NORM = "Log_Ret_High"
    LOW_NORM = "Log_Ret_Low"
    CLOSE_NORM = "Log_Ret_Close"
    VOLUME_NORM = "Volume_Z"
    RSI_NORM = "RSI_Z"
    SMA_20_NORM = 'SMA_20_Rel'
    SMA_50_NORM = 'SMA_50_Rel'
    MACD_NORM = 'MACD_Z'
    MACD_SIG_NORM = 'MACD_Sig_Z'
    ATR_NORM = 'ATR_Pct'
    BB_LOWER_NORM = 'BB_Low_Rel'
    BB_UPPER_NORM = 'BB_High_Rel'

    SENTIMENT = "Sentiment"
    SENTIMENT_NORM = "Sentiment"
    SENTIMENT_VOL = "Count"
    SENTIMENT_VOL_NORM = "News_Vol_Z"

    TARGET_O = "Target_Open"
    TARGET_H = "Target_High"
    TARGET_L = "Target_Low"
    TARGET_C = "Target_Close"
    TARGET_O_NORM = "Target_Open_Norm"
    TARGET_H_NORM = "Target_High_Norm"
    TARGET_L_NORM = "Target_Low_Norm"
    TARGET_C_NORM = "Target_Close_Norm"

    TECHNICAL_COLS = [RSI, MACD, MACD_SIG, ATR, SMA_20, SMA_50, BB_LOWER, BB_UPPER]
    TECHNICAL_COLS_NORM = [RSI_NORM, MACD_NORM, MACD_SIG_NORM, ATR_NORM, SMA_20_NORM, SMA_50_NORM, BB_LOWER_NORM, BB_UPPER_NORM]
    
    MARKET_COLS = [OPEN, HIGH, LOW, CLOSE, VOLUME]
    MARKET_COLS_NORM = [OPEN_NORM, HIGH_NORM, LOW_NORM, CLOSE_NORM, VOLUME_NORM]

    SENTIMENT_COLS = [SENTIMENT, SENTIMENT_VOL]
    SENTIMENT_COLS_NORM =[SENTIMENT_NORM, SENTIMENT_VOL_NORM]

    TARGET_COLS = [TARGET_O, TARGET_H, TARGET_L, TARGET_C]
    TARGET_COLS_NORM = [TARGET_O_NORM, TARGET_H_NORM, TARGET_L_NORM, TARGET_C_NORM]

    PRICES_COLS = [OPEN, HIGH, LOW, CLOSE, VOLUME]

    NORM_TO_NOT_MAP = {
        OPEN_NORM: OPEN,
        HIGH_NORM: HIGH,
        LOW_NORM: LOW,
        CLOSE_NORM: CLOSE,
        VOLUME_NORM: VOLUME,
        RSI_NORM: RSI,
        SMA_20_NORM: SMA_20,
        SMA_50_NORM: SMA_50,
        MACD_NORM: MACD,
        MACD_SIG_NORM: MACD_SIG,
        ATR_NORM: ATR,
        BB_LOWER_NORM: BB_LOWER,
        BB_UPPER_NORM: BB_UPPER,
        SENTIMENT_NORM: SENTIMENT,
        SENTIMENT_VOL_NORM: SENTIMENT_VOL
        }
    
    NOT_TO_NORM_MAP = {v: k for k, v in NORM_TO_NOT_MAP.items()}

class Training:
    DATASPLIT_EXPAND = "expanding_window"
    DATASPLIT_SLIDE = "sliding_window"
    BILSTM_CNN = "predictorbilstmcnn"
    BILSTM_CNN_A = "predictorbilstmcnna"
    MODELS = [BILSTM_CNN, BILSTM_CNN_A]