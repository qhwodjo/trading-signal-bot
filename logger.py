"""
logger.py - Centralized logging for the Trading Signal Bot
"""

import logging
import csv
import os
from datetime import datetime
from config import SIGNALS_LOG_FILE, PERFORMANCE_LOG_FILE, APP_LOG_FILE

# ─── APP LOGGER ──────────────────────────────────────────────────────────────
def setup_logger(name: str = "trading_bot") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(APP_LOG_FILE, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

logger = setup_logger()

# ─── SIGNAL CSV LOGGER ───────────────────────────────────────────────────────
SIGNAL_FIELDS = [
    "timestamp", "asset", "direction", "timeframe", "signal_type",
    "entry", "tp1", "tp2", "tp3", "stop_loss", "risk_reward",
    "confidence", "confluences", "patterns", "news_sentiment"
]

def log_signal(signal: dict):
    """Append a signal dict to the signals CSV log."""
    file_exists = os.path.isfile(SIGNALS_LOG_FILE)
    try:
        with open(SIGNALS_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SIGNAL_FIELDS)
            if not file_exists:
                writer.writeheader()
            row = {k: signal.get(k, "") for k in SIGNAL_FIELDS}
            writer.writerow(row)
    except Exception as e:
        logger.error(f"Failed to log signal: {e}")

# ─── PERFORMANCE CSV LOGGER ──────────────────────────────────────────────────
PERF_FIELDS = [
    "signal_id", "asset", "direction", "entry", "tp1", "tp2", "tp3",
    "stop_loss", "confidence", "timestamp_sent",
    "result",           # TP1_HIT / TP2_HIT / TP3_HIT / SL_HIT / OPEN / EXPIRED
    "exit_price", "exit_time", "pnl_pct"
]

def log_performance(record: dict):
    """Append a performance record to the performance CSV."""
    file_exists = os.path.isfile(PERFORMANCE_LOG_FILE)
    try:
        with open(PERFORMANCE_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PERF_FIELDS)
            if not file_exists:
                writer.writeheader()
            row = {k: record.get(k, "") for k in PERF_FIELDS}
            writer.writerow(row)
    except Exception as e:
        logger.error(f"Failed to log performance: {e}")

def load_signals_log() -> list:
    """Load all logged signals as a list of dicts."""
    if not os.path.isfile(SIGNALS_LOG_FILE):
        return []
    with open(SIGNALS_LOG_FILE, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_performance_log() -> list:
    """Load all performance records."""
    if not os.path.isfile(PERFORMANCE_LOG_FILE):
        return []
    with open(PERFORMANCE_LOG_FILE, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))
