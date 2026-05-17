import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-itransformer-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

# Macro columns (for conditioning)
MACRO_COLUMNS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# Rolling windows (days)
WINDOWS = [63, 252, 504, 1008, 2016]

# iTransformer hyperparameters
D_MODEL = 64          # embedding dimension
N_HEADS = 8
N_LAYERS = 3
DROPOUT = 0.1
LR = 1e-3
EPOCHS = 30
BATCH_SIZE = 32
SEQ_LEN = 20          # sequence length for time steps

# Conformal prediction: use 0.9 confidence level (90% prediction interval)
CONFIDENCE = 0.9

TOP_N = 3
