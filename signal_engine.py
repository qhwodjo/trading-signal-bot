"""
signal_engine.py - Multi-timeframe weighted signal scoring engine.
Aggregates indicator and pattern signals into a Confidence score and trading signal.
Uses per-asset-class confidence thresholds and indicator weights.
"""

import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

from config import (
    TIMEFRAMES, INDICATOR_WEIGHTS, FOREX_INDICATOR_WEIGHTS,
    MIN_CONFIDENCE, MIN_CONFIDENCE_FOREX, MIN_CONFIDENCE_CRYPTO,
    MIN_CONFIDENCE_STOCKS, MIN_CONFIDENCE_COMMOD,
    PRIMARY_TIMEFRAMES, ASSET_CLASS
)
from indicators import compute_all_indicators, atr_value
from patterns import compute_all_patterns
from logger import logger


# ─── ASSET CLASS HELPERS ──────────────────────────────────────────────────────

def get_asset_class(symbol: str) -> str:
    return ASSET_CLASS.get(symbol, "stocks")

def get_min_confidence(symbol: str) -> float:
    cls = get_asset_class(symbol)
    return {
        "forex":       MIN_CONFIDENCE_FOREX,
        "crypto":      MIN_CONFIDENCE_CRYPTO,
        "stocks":      MIN_CONFIDENCE_STOCKS,
        "commodities": MIN_CONFIDENCE_COMMOD,
    }.get(cls, MIN_CONFIDENCE)

def get_weights(symbol: str) -> dict:
    """Return the appropriate indicator weight dict for this symbol."""
    if get_asset_class(symbol) == "forex":
        return FOREX_INDICATOR_WEIGHTS
    return INDICATOR_WEIGHTS


# ─── SIGNAL TYPES ─────────────────────────────────────────────────────────────

SIGNAL_TYPES = {
    "reversal":     ["Head & Shoulders", "Inverse H&S", "Double Top", "Double Bottom",
                     "Morning Star", "Evening Star", "Doji", "Hammer", "Shooting Star",
                     "Bullish Engulfing", "Bearish Engulfing",
                     "CHoCH Bullish", "CHoCH Bearish",
                     "Liquidity Sweep High", "Liquidity Sweep Low"],
    "continuation": ["Three White Soldiers", "Three Black Crows", "Marubozu",
                     "Ascending Triangle", "Descending Triangle", "Flag", "Pennant",
                     "Cup & Handle", "BOS Bullish", "BOS Bearish"],
    "breakout":     ["Symmetrical Triangle", "Gap Up", "Gap Down",
                     "Donchian Breakout UP", "Donchian Breakout DOWN",
                     "BB Oversold", "BB Overbought"],
}


def classify_signal_type(triggered_labels: List[str]) -> str:
    counts = {"Reversal": 0, "Continuation": 0, "Breakout": 0}
    for lbl in triggered_labels:
        for stype, keywords in SIGNAL_TYPES.items():
            if any(kw.lower() in lbl.lower() for kw in keywords):
                counts[stype.capitalize()] += 1
    if not any(counts.values()):
        return "Continuation"
    return max(counts, key=counts.get)


# ─── POSITION SIZING ──────────────────────────────────────────────────────────

def calculate_position_size(entry: float, stop_loss: float,
                             account_balance: float, risk_pct: float) -> float:
    risk_amount = account_balance * (risk_pct / 100)
    risk_per_unit = abs(entry - stop_loss)
    if risk_per_unit <= 0:
        return 0.0
    return risk_amount / risk_per_unit


# ─── TP / SL CALCULATION ──────────────────────────────────────────────────────

def calculate_levels(direction: int, entry: float, atr: float) -> Tuple[float, float, float, float, float]:
    if direction == 1:
        tp1 = entry + 2 * atr
        tp2 = entry + 3.5 * atr
        tp3 = entry + 5.0 * atr
        sl  = entry - 1.5 * atr
    else:
        tp1 = entry - 2 * atr
        tp2 = entry - 3.5 * atr
        tp3 = entry - 5.0 * atr
        sl  = entry + 1.5 * atr
    rr = round(abs(tp1 - entry) / abs(entry - sl), 2) if sl != entry else 1.0
    return tp1, tp2, tp3, sl, rr

        # After calculate_levels():
    if rr < 1.5:
        return None   # skip signals with poor R:R


# ─── SINGLE-TIMEFRAME SCORING ─────────────────────────────────────────────────

def score_timeframe(df, tf_key: str, weights: dict) -> Dict:
    """Score a single timeframe using the given weight dict."""
    tf_weight = TIMEFRAMES[tf_key]["weight"]
    ind_results = compute_all_indicators(df)
    pat_results = compute_all_patterns(df)

    bull_score = 0.0
    bear_score = 0.0
    triggered  = []

    indicator_map = {
        "ema_cross":     "ema_cross",
        "supertrend":    "supertrend",
        "ichimoku":      "ichimoku",
        "adx":           "adx",
        "parabolic_sar": "parabolic_sar",
        "aroon":         "aroon",
        "rsi":           "rsi",
        "macd":          "macd",
        "stoch_rsi":     "stoch_rsi",
        "cci":           "cci",
        "williams_r":    "williams_r",
        "awesome_osc":   "awesome_osc",
        "tsi":           "tsi",
        "bollinger":     "bb_squeeze",
        "keltner":       "keltner",
        "donchian":      "donchian",
        "obv":           "obv_divergence",
        "mfi":           "mfi",
        "cmf":           "cmf",
        "force_index":   "force_index",
        "stochastic":    "adx",
        "demarker":      "cci",
        "ppo":           "macd",
        "stc":           "rsi",
        "roc":           "rsi",
        "ultimate_osc":  "rsi",
        "momentum":      "rsi",
        "eom":           "force_index",
    }

    for ind_key, weight_key in indicator_map.items():
        sig = ind_results.get(ind_key, {})
        d = sig.get("direction", 0)
        s = sig.get("strength", 0.5)
        w = weights.get(weight_key, 0.7) * tf_weight
        if d == 1:
            bull_score += w * s
            triggered.append(sig.get("label", ind_key))
        elif d == -1:
            bear_score += w * s
            triggered.append(sig.get("label", ind_key))

    # Candlestick patterns
    for cs in pat_results.get("candlestick", []):
        d = cs.get("direction", 0)
        s = cs.get("strength", 0.7)
        w = weights.get("candlestick", 0.9) * tf_weight
        if d == 1:
            bull_score += w * s; triggered.append(cs.get("label", "Candlestick"))
        elif d == -1:
            bear_score += w * s; triggered.append(cs.get("label", "Candlestick"))

    # Chart patterns
    for cp in pat_results.get("chart_patterns", []):
        d = cp.get("direction", 0)
        s = cp.get("strength", 0.9)
        w = weights.get("chart_pattern", 1.5) * tf_weight
        if d == 1:
            bull_score += w * s; triggered.append(cp.get("label", "Chart Pattern"))
        elif d == -1:
            bear_score += w * s; triggered.append(cp.get("label", "Chart Pattern"))

    # ICT / Price Action
    ict_keys = {
        "support_resistance": "support_resistance",
        "bos_choch":          "bos",
        "liquidity_sweep":    "liquidity_sweep",
        "fibonacci":          "fibonacci",
        "premium_discount":   "support_resistance",
    }
    for pat_key, wt_key in ict_keys.items():
        sig = pat_results.get(pat_key, {})
        d = sig.get("direction", 0)
        s = sig.get("strength", 0.8)
        w = weights.get(wt_key, 0.9) * tf_weight
        if d == 1:
            bull_score += w * s; triggered.append(sig.get("label", pat_key))
        elif d == -1:
            bear_score += w * s; triggered.append(sig.get("label", pat_key))

    for fvg in pat_results.get("fvg", []):
        d = fvg.get("direction", 0)
        s = fvg.get("strength", 0.9)
        w = weights.get("fvg", 1.0) * tf_weight
        if d == 1: bull_score += w * s; triggered.append(fvg.get("type", "FVG"))
        elif d == -1: bear_score += w * s; triggered.append(fvg.get("type", "FVG"))

    for ob in pat_results.get("order_blocks", []):
        d = ob.get("direction", 0)
        s = ob.get("strength", 1.0)
        w = weights.get("order_block", 1.3) * tf_weight
        if d == 1: bull_score += w * s; triggered.append(ob.get("type", "Order Block"))
        elif d == -1: bear_score += w * s; triggered.append(ob.get("type", "Order Block"))

    return {
        "bull_score": bull_score,
        "bear_score": bear_score,
        "triggered":  triggered,
        "tf_weight":  tf_weight,
    }


# ─── MULTI-TIMEFRAME SIGNAL ENGINE ────────────────────────────────────────────

def generate_signal(symbol: str, tf_data: Dict[str, Any],
                    news_sentiment: str = "Neutral",
                    news_headline: str = "",
                    account_balance: float = 1000.0,
                    risk_pct: float = 1.0) -> Optional[Dict]:
    """
    Generate a full trading signal for a symbol given multi-TF data.
    Uses per-asset-class confidence thresholds and indicator weights.
    Returns signal dict or None if confidence < threshold.
    """
    weights      = get_weights(symbol)
    min_conf     = get_min_confidence(symbol)
    asset_class  = get_asset_class(symbol)

    total_bull = 0.0
    total_bear = 0.0
    all_triggered = []
    dominant_tf   = "1h"
    best_tf_score = 0.0

    for tf_key, df in tf_data.items():
        if df is None or len(df) < 30:
            continue
        try:
            tf_result = score_timeframe(df, tf_key, weights)
        except Exception as e:
            logger.debug(f"Scoring error {symbol}/{tf_key}: {e}")
            continue

        total_bull += tf_result["bull_score"]
        total_bear += tf_result["bear_score"]
        all_triggered.extend(tf_result["triggered"])

        tf_net = abs(tf_result["bull_score"] - tf_result["bear_score"])
        if tf_net > best_tf_score:
            best_tf_score = tf_net
            dominant_tf   = tf_key

    if total_bull == 0 and total_bear == 0:
        return None

    direction  = 1 if total_bull > total_bear else -1
    raw_score  = max(total_bull, total_bear)
    confidence = min(raw_score / 10, 10.0)
    confluence = len([t for t in all_triggered if t])

    # News adjustment
    news_adjustment = 0.0
    news_conf_key = "news_confirm" if (
        (news_sentiment == "Bullish" and direction == 1) or
        (news_sentiment == "Bearish" and direction == -1)
    ) else ("news_conflict" if news_sentiment != "Neutral" else "")

    if news_conf_key:
        news_adjustment = weights.get(news_conf_key, 0)

    confidence = max(0.0, min(10.0, confidence + news_adjustment))

    # Apply per-asset-class threshold
    if confidence < min_conf:
        return None

    # Entry from most recent available timeframe
    df_primary = None
    for tf_key in ["1h", "4h", "1d", "15m", "5m", "1m"]:
        if tf_data.get(tf_key) is not None:
            df_primary = tf_data[tf_key]
            break

    if df_primary is None:
        return None

    entry = float(df_primary["close"].iloc[-1])
    atr   = atr_value(df_primary) or entry * 0.01

    tp1, tp2, tp3, sl, rr = calculate_levels(direction, entry, atr)
    pos_size = calculate_position_size(entry, sl, account_balance, risk_pct)

    # Top 3 unique triggered labels
    seen = set()
    top_patterns = []
    for lbl in all_triggered:
        if lbl and lbl not in seen:
            seen.add(lbl)
            top_patterns.append(lbl)
        if len(top_patterns) == 3:
            break

    signal_type = classify_signal_type(all_triggered)

    warnings = []
    if news_adjustment < 0:
        warnings.append(f"News conflict ({news_sentiment})")
    if atr / entry > 0.03:
        warnings.append("High volatility — wider stops")
    if asset_class == "forex":
        warnings.append("Forex: volume indicators excluded from scoring")
    vol_series = df_primary.get("volume") if hasattr(df_primary, "get") else df_primary["volume"]
    if vol_series is not None and vol_series.iloc[-5:].mean() < vol_series.mean() * 0.5:
        warnings.append("Low volume — reduced conviction")

    return {
        "timestamp":      datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "asset":          symbol,
        "direction":      "LONG" if direction == 1 else "SHORT",
        "timeframe":      dominant_tf,
        "signal_type":    signal_type,
        "entry":          round(entry, 6),
        "tp1":            round(tp1, 6),
        "tp2":            round(tp2, 6),
        "tp3":            round(tp3, 6),
        "stop_loss":      round(sl, 6),
        "risk_reward":    rr,
        "confidence":     round(confidence, 1),
        "confluences":    confluence,
        "patterns":       ", ".join(top_patterns) if top_patterns else "Multiple indicators",
        "news_sentiment": news_sentiment,
        "news_headline":  news_headline,
        "warnings":       " | ".join(warnings) if warnings else "None",
        "position_size":  round(pos_size, 4),
        "asset_class":    asset_class,
        # Internal
        "_direction_int": direction,
        "_atr":           atr,
    }