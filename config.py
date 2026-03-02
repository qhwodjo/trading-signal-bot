"""
config.py - Central configuration for the Trading Signal Bot
"""

# ─── ACCOUNT SETTINGS ────────────────────────────────────────────────────────
TELEGRAM_TOKEN = "YOUR_TELEGRAM_TOKEN_HERE"
CHAT_ID        = "YOUR_CHAT_ID_HERE"
TWELVEDATA_KEY = "YOUR_TWELVEDATA_KEY_HERE"
NEWS_API_KEY   = "YOUR_NEWS_API_KEY_HERE"
COINMARKETCAP_KEY = "YOUR_COINMARKETCAP_KEY_HERE"
MT5_LOGIN      = 0
MT5_PASSWORD   = ""
MT5_SERVER     = ""

# Set your real account balance here — used to calculate suggested lot sizes
ACCOUNT_BALANCE    = 935.0         # Your current MT5 account balance in USD
RISK_PERCENT       = 1.0           # Default risk per trade (% of balance)
MAX_OPEN_SIGNALS   = 5             # Max simultaneous open trades

# Lot size suggestions sent with every signal (based on ACCOUNT_BALANCE above)
# The bot will suggest 3 options: conservative, recommended, aggressive
LOT_RISK_CONSERVATIVE  = 0.5      # 0.5% of balance
LOT_RISK_RECOMMENDED   = 1.0      # 1.0% of balance  ← default suggestion
LOT_RISK_AGGRESSIVE    = 2.0      # 2.0% of balance

# ─── SCANNING SETTINGS ────────────────────────────────────────────────────────
SCAN_INTERVAL_SECONDS  = 1800
NEWS_REFRESH_SECONDS   = 900
SIGNAL_COOLDOWN_HOURS  = 4

# Per-asset-class confidence thresholds
MIN_CONFIDENCE         = 6.0
MIN_CONFIDENCE_FOREX   = 8.0
MIN_CONFIDENCE_CRYPTO  = 9.0
MIN_CONFIDENCE_STOCKS  = 9.0
MIN_CONFIDENCE_COMMOD  = 7.0

# ─── ASSETS ──────────────────────────────────────────────────────────────────
FOREX_PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
    "USDCAD=X", "USDCHF=X", "NZDUSD=X",
    "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X",
    "CADJPY=X", "CHFJPY=X", "NZDJPY=X",
    "EURCHF=X", "EURAUD=X", "EURCAD=X",
]

CRYPTO_PAIRS = [
    "BTC-USD", "ETH-USD", "XRP-USD",
]

STOCKS = []   # not traded on prop broker

COMMODITIES = [
    "GC=F",   # Gold — best performer
]

ALL_ASSETS = FOREX_PAIRS + CRYPTO_PAIRS + STOCKS + COMMODITIES

# Asset class lookup
ASSET_CLASS = {}
for s in FOREX_PAIRS:  ASSET_CLASS[s] = "forex"
for s in CRYPTO_PAIRS: ASSET_CLASS[s] = "crypto"
for s in STOCKS:       ASSET_CLASS[s] = "stocks"
for s in COMMODITIES:  ASSET_CLASS[s] = "commodities"

# ─── TIMEFRAMES ───────────────────────────────────────────────────────────────
TIMEFRAMES = {
    "1m":  {"interval": "1m",  "period": "1d",   "weight": 0.5},
    "5m":  {"interval": "5m",  "period": "5d",   "weight": 0.7},
    "15m": {"interval": "15m", "period": "5d",   "weight": 0.8},
    "1h":  {"interval": "1h",  "period": "60d",  "weight": 1.0},
    "4h":  {"interval": "60m", "period": "60d",  "weight": 1.3},
    "1d":  {"interval": "1d",  "period": "365d", "weight": 1.6},
}

PRIMARY_TIMEFRAMES = ["15m", "1h", "4h"]

# ─── INDICATOR WEIGHTS (default) ─────────────────────────────────────────────
INDICATOR_WEIGHTS = {
    "ema_cross": 1.4, "supertrend": 1.2, "ichimoku": 1.5,
    "adx": 1.0, "parabolic_sar": 1.0, "aroon": 0.9,
    "rsi": 1.2, "macd": 1.3, "stoch_rsi": 1.1,
    "cci": 0.9, "williams_r": 0.9, "awesome_osc": 0.8, "tsi": 0.8,
    "bb_squeeze": 1.0, "keltner": 0.9, "atr_breakout": 1.0, "donchian": 0.9,
    "obv_divergence": 1.0, "mfi": 0.9, "cmf": 0.8, "force_index": 0.6,
    "chart_pattern": 1.5, "candlestick": 1.1,
    "support_resistance": 1.4, "fvg": 1.2, "order_block": 1.5,
    "bos": 1.3, "choch": 1.4, "liquidity_sweep": 1.2, "fibonacci": 1.1,
    "news_confirm": 0.5, "news_conflict": -2.0,
}

# ─── FOREX-SPECIFIC INDICATOR WEIGHTS ────────────────────────────────────────
FOREX_INDICATOR_WEIGHTS = {
    "ema_cross": 1.5, "supertrend": 1.3, "ichimoku": 1.6,
    "adx": 1.2, "parabolic_sar": 1.1, "aroon": 1.0,
    "rsi": 1.3, "macd": 1.4, "stoch_rsi": 1.2,
    "cci": 1.0, "williams_r": 1.0, "awesome_osc": 0.9, "tsi": 0.9,
    "bb_squeeze": 1.1, "keltner": 1.0, "atr_breakout": 1.1, "donchian": 1.0,
    "obv_divergence": 0.2, "mfi": 0.2, "cmf": 0.2, "force_index": 0.1,
    "chart_pattern": 1.6, "candlestick": 1.2,
    "support_resistance": 1.6, "fvg": 1.4, "order_block": 1.7,
    "bos": 1.5, "choch": 1.6, "liquidity_sweep": 1.4, "fibonacci": 1.3,
    "news_confirm": 0.5, "news_conflict": -2.0,
}

# ─── BACKTESTING ──────────────────────────────────────────────────────────────
BACKTEST_PERIOD_MONTHS = 6
BACKTEST_ASSETS = ["SPY", "BTC-USD", "EURUSD=X", "GC=F", "AAPL"]

# ─── LOGGING ─────────────────────────────────────────────────────────────────
SIGNALS_LOG_FILE     = "signals_log.csv"
PERFORMANCE_LOG_FILE = "performance_log.csv"
APP_LOG_FILE         = "bot.log"

# ─── WEBHOOK ──────────────────────────────────────────────────────────────────
WEBHOOK_ENABLED = False
WEBHOOK_URL     = ""

# ─── ECONOMIC CALENDAR ────────────────────────────────────────────────────────
PAUSE_BEFORE_NEWS_MINUTES = 30
PAUSE_AFTER_NEWS_MINUTES  = 30
HIGH_IMPACT_EVENTS = ["NFP", "CPI", "FOMC", "GDP", "PPI", "Unemployment"]

# Approval timeout — if you don't respond to a signal within this many seconds,
# it is automatically DECLINED (not traded). Set to 0 to disable timeout.
APPROVAL_TIMEOUT_SECONDS = 300     # 5 minutes to approve/decline