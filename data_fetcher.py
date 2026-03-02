"""
data_fetcher.py - Fetch OHLCV data from yfinance, ccxt, twelvedata, alpha_vantage.
Includes proper 4h resampling since yfinance has no native 4h interval.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Optional, Dict
import pandas as pd
import yfinance as yf

from config import (
    ALPHA_VANTAGE_KEY, TWELVEDATA_KEY, TIMEFRAMES, CRYPTO_PAIRS
)
from logger import logger


# ─── 4H RESAMPLER ─────────────────────────────────────────────────────────────

def resample_to_4h(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Resample a 1h OHLCV DataFrame into 4h candles.
    yfinance has no native 4h interval — this fixes the 4h weight not working.
    """
    try:
        df_4h = df.resample("4h").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna()
        return df_4h if len(df_4h) >= 20 else None
    except Exception as e:
        logger.debug(f"4h resample error: {e}")
        return None


# ─── YFINANCE (primary source) ────────────────────────────────────────────────

def fetch_yfinance(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from yfinance. Returns DataFrame or None."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(interval=interval, period=period, auto_adjust=True)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index, utc=True)
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        if len(df) < 30:
            return None
        return df
    except Exception as e:
        logger.warning(f"yfinance fetch failed for {symbol}/{interval}: {e}")
        return None


# ─── CCXT (crypto fallback) ────────────────────────────────────────────────────

def fetch_ccxt(symbol: str, timeframe: str = "1h", limit: int = 300) -> Optional[pd.DataFrame]:
    """Fetch crypto OHLCV via ccxt (Binance free, no key needed)."""
    try:
        import ccxt
        ccxt_symbol = symbol.replace("-USD", "/USDT")
        exchange = ccxt.binance({"enableRateLimit": True})
        ohlcv = exchange.fetch_ohlcv(ccxt_symbol, timeframe=timeframe, limit=limit)
        if not ohlcv:
            return None
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df.dropna()
    except Exception as e:
        logger.warning(f"ccxt fetch failed for {symbol}: {e}")
        return None


# ─── TWELVEDATA (fallback for forex/stocks) ────────────────────────────────────

def fetch_twelvedata(symbol: str, interval: str = "1h", outputsize: int = 300) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from Twelvedata free API."""
    if not TWELVEDATA_KEY:
        return None
    try:
        import requests
        # Convert yfinance symbol format to Twelvedata format
        td_symbol = (symbol
                     .replace("=X", "")      # EURUSD=X → EURUSD
                     .replace("-USD", "/USD") # BTC-USD  → BTC/USD
                     .replace("=F", ""))      # GC=F     → GC (futures)
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol":     td_symbol,
            "interval":   interval,
            "outputsize": outputsize,
            "apikey":     TWELVEDATA_KEY,
            "format":     "JSON"
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if "values" not in data:
            return None
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.set_index("datetime", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df.sort_index().dropna()
    except Exception as e:
        logger.warning(f"Twelvedata fetch failed for {symbol}: {e}")
        return None


# ─── ALPHA VANTAGE (fallback for stocks) ──────────────────────────────────────

def fetch_alpha_vantage(symbol: str, interval: str = "60min") -> Optional[pd.DataFrame]:
    if not ALPHA_VANTAGE_KEY or ALPHA_VANTAGE_KEY == "demo":
        return None
    try:
        import requests
        url = "https://www.alphavantage.co/query"
        params = {
            "function":   "TIME_SERIES_INTRADAY",
            "symbol":     symbol,
            "interval":   interval,
            "outputsize": "full",
            "apikey":     ALPHA_VANTAGE_KEY
        }
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        key = f"Time Series ({interval})"
        if key not in data:
            return None
        df = pd.DataFrame(data[key]).T
        df.index = pd.to_datetime(df.index, utc=True)
        df.columns = ["open", "high", "low", "close", "volume"]
        return df.astype(float).sort_index().dropna()
    except Exception as e:
        logger.warning(f"Alpha Vantage fetch failed for {symbol}: {e}")
        return None


# ─── MAIN MULTI-SOURCE FETCHER ────────────────────────────────────────────────

CCXT_TF_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "1h": "1h", "4h": "4h", "1d": "1d"
}

def get_ohlcv(symbol: str, tf_key: str) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV for a symbol and timeframe key.
    For the '4h' key: fetches 60m data from yfinance and resamples to true 4h candles.
    Falls back to ccxt (crypto), then twelvedata.
    """
    tf       = TIMEFRAMES[tf_key]
    interval = tf["interval"]
    period   = tf["period"]

    is_crypto = "-USD" in symbol or "-USDT" in symbol

    # Special handling for 4h: fetch 1h then resample
    if tf_key == "4h":
        df_1h = fetch_yfinance(symbol, "1h", "60d")
        if df_1h is not None and len(df_1h) >= 40:
            df_4h = resample_to_4h(df_1h)
            if df_4h is not None:
                return df_4h
        # Crypto fallback: fetch native 4h from Binance via ccxt
        if is_crypto:
            df = fetch_ccxt(symbol, timeframe="4h", limit=200)
            if df is not None:
                return df
        # Twelvedata supports 4h natively
        df = fetch_twelvedata(symbol, interval="4h", outputsize=200)
        if df is not None:
            return df
        return None

    # All other timeframes: standard fetch chain
    df = fetch_yfinance(symbol, interval, period)
    if df is not None and len(df) >= 50:
        return df

    if is_crypto:
        ccxt_tf = CCXT_TF_MAP.get(tf_key, "1h")
        df = fetch_ccxt(symbol, timeframe=ccxt_tf, limit=500)
        if df is not None and len(df) >= 50:
            return df

    df = fetch_twelvedata(symbol, interval=interval, outputsize=300)
    if df is not None and len(df) >= 50:
        return df

    av_map = {"1h": "60min", "1d": "daily"}
    av_tf = av_map.get(tf_key)
    if av_tf:
        df = fetch_alpha_vantage(symbol, interval=av_tf)
        if df is not None and len(df) >= 50:
            return df

    logger.warning(f"All data sources failed for {symbol} / {tf_key}")
    return None


def get_all_timeframes(symbol: str) -> Dict[str, Optional[pd.DataFrame]]:
    """Fetch all configured timeframes for a symbol."""
    result = {}
    for tf_key in TIMEFRAMES:
        result[tf_key] = get_ohlcv(symbol, tf_key)
        time.sleep(0.1)
    return result


# ─── ASYNC WRAPPER ────────────────────────────────────────────────────────────

async def async_get_ohlcv(symbol: str, tf_key: str) -> tuple:
    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(None, get_ohlcv, symbol, tf_key)
    return (symbol, tf_key, df)


async def fetch_batch(symbols: list, tf_key: str) -> Dict[str, Optional[pd.DataFrame]]:
    tasks = [async_get_ohlcv(sym, tf_key) for sym in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    output = {}
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"Batch fetch error: {r}")
            continue
        symbol, _, df = r
        output[symbol] = df
    return output