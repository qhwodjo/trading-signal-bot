"""
patterns.py - Chart patterns, candlestick patterns, price action & ICT concepts.
Each function returns: {"direction": 1/-1/0, "strength": float, "label": str, "found": bool}
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from logger import logger


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _body(df: pd.DataFrame, i: int) -> float:
    return abs(df["close"].iloc[i] - df["open"].iloc[i])

def _range(df: pd.DataFrame, i: int) -> float:
    return df["high"].iloc[i] - df["low"].iloc[i]

def _is_bullish(df: pd.DataFrame, i: int) -> bool:
    return df["close"].iloc[i] > df["open"].iloc[i]

def find_swing_highs(df: pd.DataFrame, window: int = 5) -> List[int]:
    """Find local swing high indices."""
    highs = []
    h = df["high"].values
    for i in range(window, len(h) - window):
        if h[i] == max(h[i - window:i + window + 1]):
            highs.append(i)
    return highs

def find_swing_lows(df: pd.DataFrame, window: int = 5) -> List[int]:
    """Find local swing low indices."""
    lows = []
    l = df["low"].values
    for i in range(window, len(l) - window):
        if l[i] == min(l[i - window:i + window + 1]):
            lows.append(i)
    return lows


# ─── CANDLESTICK PATTERNS ─────────────────────────────────────────────────────

def candlestick_patterns(df: pd.DataFrame) -> List[Dict]:
    """Detect candlestick patterns on the last N candles."""
    patterns = []
    if len(df) < 4:
        return patterns

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    i = -1  # last candle

    body   = abs(c[i] - o[i])
    rng    = h[i] - l[i]
    upper_wick = h[i] - max(o[i], c[i])
    lower_wick = min(o[i], c[i]) - l[i]

    if rng == 0:
        return patterns

    # Doji
    if body / rng < 0.1:
        patterns.append({"label": "Doji", "direction": 0, "strength": 0.5})

    # Hammer (bullish)
    if lower_wick > 2 * body and upper_wick < body * 0.5 and not _is_bullish(df, i):
        patterns.append({"label": "Hammer", "direction": 1, "strength": 0.8})

    # Shooting Star (bearish)
    if upper_wick > 2 * body and lower_wick < body * 0.5 and _is_bullish(df, i):
        patterns.append({"label": "Shooting Star", "direction": -1, "strength": 0.8})

    # Marubozu
    if body / rng > 0.95:
        d = 1 if _is_bullish(df, i) else -1
        patterns.append({"label": "Marubozu", "direction": d, "strength": 0.9})

    # Spinning Top
    if 0.1 < body / rng < 0.4 and upper_wick > body * 0.5 and lower_wick > body * 0.5:
        patterns.append({"label": "Spinning Top", "direction": 0, "strength": 0.3})

    if len(df) < 2:
        return patterns

    # 2-candle patterns
    prev_body = abs(c[-2] - o[-2])
    prev_bull = _is_bullish(df, -2)

    # Bullish Engulfing
    if not prev_bull and _is_bullish(df, i) and c[i] > o[-2] and o[i] < c[-2]:
        patterns.append({"label": "Bullish Engulfing", "direction": 1, "strength": 1.0})

    # Bearish Engulfing
    if prev_bull and not _is_bullish(df, i) and c[i] < o[-2] and o[i] > c[-2]:
        patterns.append({"label": "Bearish Engulfing", "direction": -1, "strength": 1.0})

    # Bullish Harami
    if not prev_bull and _is_bullish(df, i) and c[i] < o[-2] and o[i] > c[-2]:
        patterns.append({"label": "Bullish Harami", "direction": 1, "strength": 0.6})

    # Bearish Harami
    if prev_bull and not _is_bullish(df, i) and c[i] > o[-2] and o[i] < c[-2]:
        patterns.append({"label": "Bearish Harami", "direction": -1, "strength": 0.6})

    # Piercing Line (bullish)
    if not prev_bull and _is_bullish(df, i):
        mid_prev = (o[-2] + c[-2]) / 2
        if o[i] < c[-2] and c[i] > mid_prev:
            patterns.append({"label": "Piercing Line", "direction": 1, "strength": 0.7})

    # Dark Cloud Cover (bearish)
    if prev_bull and not _is_bullish(df, i):
        mid_prev = (o[-2] + c[-2]) / 2
        if o[i] > c[-2] and c[i] < mid_prev:
            patterns.append({"label": "Dark Cloud Cover", "direction": -1, "strength": 0.7})

    # Tweezer Top (bearish)
    if prev_bull and not _is_bullish(df, i) and abs(h[-1] - h[-2]) / rng < 0.05:
        patterns.append({"label": "Tweezer Top", "direction": -1, "strength": 0.7})

    # Tweezer Bottom (bullish)
    if not prev_bull and _is_bullish(df, i) and abs(l[-1] - l[-2]) / rng < 0.05:
        patterns.append({"label": "Tweezer Bottom", "direction": 1, "strength": 0.7})

    if len(df) < 3:
        return patterns

    # 3-candle patterns
    # Morning Star (bullish)
    if (not _is_bullish(df, -3) and
        abs(c[-2] - o[-2]) / (h[-2] - l[-2]) < 0.3 and
        _is_bullish(df, i) and
        c[i] > (o[-3] + c[-3]) / 2):
        patterns.append({"label": "Morning Star", "direction": 1, "strength": 1.0})

    # Evening Star (bearish)
    if (_is_bullish(df, -3) and
        abs(c[-2] - o[-2]) / (h[-2] - l[-2]) < 0.3 and
        not _is_bullish(df, i) and
        c[i] < (o[-3] + c[-3]) / 2):
        patterns.append({"label": "Evening Star", "direction": -1, "strength": 1.0})

    # Three White Soldiers
    if all(_is_bullish(df, j) for j in [-3, -2, -1]):
        if c[-1] > c[-2] > c[-3]:
            patterns.append({"label": "Three White Soldiers", "direction": 1, "strength": 1.0})

    # Three Black Crows
    if all(not _is_bullish(df, j) for j in [-3, -2, -1]):
        if c[-1] < c[-2] < c[-3]:
            patterns.append({"label": "Three Black Crows", "direction": -1, "strength": 1.0})

    # Abandoned Baby (bullish)
    if (not _is_bullish(df, -3) and
        abs(c[-2] - o[-2]) / max(h[-2] - l[-2], 1e-9) < 0.1 and
        l[-2] > h[-3] and _is_bullish(df, i) and l[i] > h[-2]):
        patterns.append({"label": "Abandoned Baby Bull", "direction": 1, "strength": 1.0})

    return patterns


# ─── CHART PATTERNS ───────────────────────────────────────────────────────────

def detect_double_top_bottom(df: pd.DataFrame) -> Optional[Dict]:
    """Detect Double Top or Double Bottom."""
    try:
        sh = find_swing_highs(df, window=5)
        sl = find_swing_lows(df, window=5)
        h_vals = df["high"].values
        l_vals = df["low"].values
        tol    = 0.005  # 0.5% tolerance

        # Double Top
        if len(sh) >= 2:
            t1, t2 = sh[-2], sh[-1]
            if t2 > t1 and abs(h_vals[t1] - h_vals[t2]) / h_vals[t1] < tol:
                return {"label": "Double Top", "direction": -1, "strength": 1.0, "found": True}

        # Double Bottom
        if len(sl) >= 2:
            b1, b2 = sl[-2], sl[-1]
            if b2 > b1 and abs(l_vals[b1] - l_vals[b2]) / l_vals[b1] < tol:
                return {"label": "Double Bottom", "direction": 1, "strength": 1.0, "found": True}
    except Exception as e:
        logger.debug(f"Double top/bottom error: {e}")
    return None


def detect_head_shoulders(df: pd.DataFrame) -> Optional[Dict]:
    """Detect Head & Shoulders or Inverse H&S."""
    try:
        sh = find_swing_highs(df, window=5)
        sl = find_swing_lows(df, window=5)
        h_vals = df["high"].values
        l_vals = df["low"].values
        tol    = 0.02

        # H&S (bearish)
        if len(sh) >= 3:
            l_sh, m_sh, r_sh = sh[-3], sh[-2], sh[-1]
            lh, mh, rh = h_vals[l_sh], h_vals[m_sh], h_vals[r_sh]
            if mh > lh and mh > rh and abs(lh - rh) / lh < tol:
                return {"label": "Head & Shoulders", "direction": -1, "strength": 1.2, "found": True}

        # Inverse H&S (bullish)
        if len(sl) >= 3:
            l_sl, m_sl, r_sl = sl[-3], sl[-2], sl[-1]
            ll, ml, rl = l_vals[l_sl], l_vals[m_sl], l_vals[r_sl]
            if ml < ll and ml < rl and abs(ll - rl) / ll < tol:
                return {"label": "Inverse H&S", "direction": 1, "strength": 1.2, "found": True}
    except Exception as e:
        logger.debug(f"H&S error: {e}")
    return None


def detect_triangle(df: pd.DataFrame) -> Optional[Dict]:
    """Detect ascending, descending, or symmetrical triangle."""
    try:
        n = min(50, len(df))
        sub = df.iloc[-n:]
        highs = sub["high"].values
        lows  = sub["low"].values
        x     = np.arange(n)

        # Linear regression on highs and lows
        slope_h, _ = np.polyfit(x, highs, 1)
        slope_l, _ = np.polyfit(x, lows,  1)

        tol = 0.0002

        if abs(slope_h) < tol and slope_l > tol:
            return {"label": "Ascending Triangle", "direction": 1, "strength": 0.9, "found": True}
        if abs(slope_l) < tol and slope_h < -tol:
            return {"label": "Descending Triangle", "direction": -1, "strength": 0.9, "found": True}
        if slope_h < -tol and slope_l > tol:
            return {"label": "Symmetrical Triangle", "direction": 0, "strength": 0.6, "found": True}
    except Exception as e:
        logger.debug(f"Triangle error: {e}")
    return None


def detect_wedge(df: pd.DataFrame) -> Optional[Dict]:
    """Rising wedge (bearish) / Falling wedge (bullish)."""
    try:
        n = min(40, len(df))
        sub = df.iloc[-n:]
        x   = np.arange(n)
        slope_h, _ = np.polyfit(x, sub["high"].values, 1)
        slope_l, _ = np.polyfit(x, sub["low"].values,  1)

        if slope_h > 0 and slope_l > 0 and slope_l > slope_h:
            return {"label": "Rising Wedge", "direction": -1, "strength": 0.8, "found": True}
        if slope_h < 0 and slope_l < 0 and slope_h < slope_l:
            return {"label": "Falling Wedge", "direction": 1, "strength": 0.8, "found": True}
    except Exception as e:
        logger.debug(f"Wedge error: {e}")
    return None


def detect_flag_pennant(df: pd.DataFrame) -> Optional[Dict]:
    """Flag or pennant after sharp move."""
    try:
        if len(df) < 30:
            return None
        # Look for prior strong trend (last 10 candles before the last 10)
        prior  = df.iloc[-20:-10]
        recent = df.iloc[-10:]
        trend_move = (prior["close"].iloc[-1] - prior["close"].iloc[0]) / prior["close"].iloc[0]
        consolidation_range = (recent["high"].max() - recent["low"].min()) / prior["close"].iloc[-1]
        if abs(trend_move) > 0.03 and consolidation_range < 0.015:
            direction = 1 if trend_move > 0 else -1
            return {"label": "Flag" if consolidation_range < 0.01 else "Pennant",
                    "direction": direction, "strength": 0.9, "found": True}
    except Exception as e:
        logger.debug(f"Flag/Pennant error: {e}")
    return None


def detect_cup_handle(df: pd.DataFrame) -> Optional[Dict]:
    """Cup & Handle (bullish continuation)."""
    try:
        n = min(100, len(df))
        sub = df.iloc[-n:]
        close = sub["close"].values
        mid = len(close) // 2
        left_high = max(close[:mid // 2])
        cup_low   = min(close[mid // 4: 3 * mid // 4])
        right_high = max(close[3 * mid // 4:])
        handle_recent = max(close[-5:])

        if (right_high >= left_high * 0.97 and
                cup_low < left_high * 0.85 and
                handle_recent < right_high and
                handle_recent > cup_low):
            return {"label": "Cup & Handle", "direction": 1, "strength": 1.0, "found": True}
    except Exception as e:
        logger.debug(f"Cup & Handle error: {e}")
    return None


def detect_gaps(df: pd.DataFrame) -> Optional[Dict]:
    """Detect price gaps."""
    try:
        if len(df) < 3:
            return None
        prev_close = df["close"].iloc[-2]
        curr_open  = df["open"].iloc[-1]
        gap_pct    = (curr_open - prev_close) / prev_close * 100

        if abs(gap_pct) > 0.3:
            direction = 1 if gap_pct > 0 else -1
            return {"label": f"Gap {'Up' if gap_pct > 0 else 'Down'} ({abs(gap_pct):.2f}%)",
                    "direction": direction, "strength": min(abs(gap_pct) / 2, 1.0), "found": True}
    except Exception as e:
        logger.debug(f"Gap error: {e}")
    return None


# ─── PRICE ACTION / ICT CONCEPTS ──────────────────────────────────────────────

def detect_support_resistance(df: pd.DataFrame, window: int = 5) -> Dict:
    """Detect key S/R levels and whether price is near them."""
    result = {"levels": [], "direction": 0, "strength": 0, "label": "S/R"}
    try:
        sh = find_swing_highs(df, window)
        sl = find_swing_lows(df, window)
        h_vals = df["high"].values
        l_vals = df["low"].values
        price  = df["close"].iloc[-1]

        resistance = [h_vals[i] for i in sh[-5:]]
        support    = [l_vals[i] for i in sl[-5:]]
        result["levels"] = {"support": support, "resistance": resistance}

        # Check proximity (within 0.5% of price)
        near_resistance = any(abs(r - price) / price < 0.005 for r in resistance)
        near_support    = any(abs(s - price) / price < 0.005 for s in support)

        if near_support:
            result.update({"direction": 1, "strength": 0.9, "label": "Near Support"})
        elif near_resistance:
            result.update({"direction": -1, "strength": 0.9, "label": "Near Resistance"})
    except Exception as e:
        logger.debug(f"S/R error: {e}")
    return result


def detect_fair_value_gaps(df: pd.DataFrame) -> List[Dict]:
    """ICT Fair Value Gaps (FVG) — 3-candle imbalance."""
    fvgs = []
    try:
        for i in range(2, len(df)):
            prev2_high = df["high"].iloc[i - 2]
            prev2_low  = df["low"].iloc[i - 2]
            curr_high  = df["high"].iloc[i]
            curr_low   = df["low"].iloc[i]

            # Bullish FVG: curr_low > prev2_high
            if curr_low > prev2_high:
                fvgs.append({"type": "Bullish FVG", "low": prev2_high, "high": curr_low,
                              "direction": 1, "strength": 0.9})
            # Bearish FVG: curr_high < prev2_low
            elif curr_high < prev2_low:
                fvgs.append({"type": "Bearish FVG", "low": curr_high, "high": prev2_low,
                              "direction": -1, "strength": 0.9})
    except Exception as e:
        logger.debug(f"FVG error: {e}")
    return fvgs[-3:] if fvgs else []  # return last 3


def detect_order_blocks(df: pd.DataFrame) -> List[Dict]:
    """ICT Order Blocks — last bullish/bearish candle before impulsive move."""
    blocks = []
    try:
        for i in range(3, len(df) - 1):
            # Bullish Order Block: bearish candle followed by strong bullish move
            if not _is_bullish(df, i - 1) and _is_bullish(df, i):
                move = (df["close"].iloc[i] - df["open"].iloc[i]) / df["close"].iloc[i - 1]
                if move > 0.003:
                    blocks.append({
                        "type": "Bullish OB",
                        "top": df["open"].iloc[i - 1],
                        "bottom": df["close"].iloc[i - 1],
                        "direction": 1, "strength": 1.0
                    })
            # Bearish Order Block: bullish candle followed by strong bearish move
            elif _is_bullish(df, i - 1) and not _is_bullish(df, i):
                move = (df["open"].iloc[i] - df["close"].iloc[i]) / df["open"].iloc[i - 1]
                if move > 0.003:
                    blocks.append({
                        "type": "Bearish OB",
                        "top": df["close"].iloc[i - 1],
                        "bottom": df["open"].iloc[i - 1],
                        "direction": -1, "strength": 1.0
                    })
    except Exception as e:
        logger.debug(f"Order block error: {e}")
    return blocks[-3:] if blocks else []


def detect_bos_choch(df: pd.DataFrame) -> Dict:
    """Break of Structure (BOS) and Change of Character (CHoCH)."""
    result = {"direction": 0, "strength": 0, "label": "BOS/CHoCH"}
    try:
        sh = find_swing_highs(df, 5)
        sl = find_swing_lows(df, 5)
        price = df["close"].iloc[-1]
        h_vals = df["high"].values
        l_vals = df["low"].values

        if len(sh) >= 2 and len(sl) >= 2:
            last_high = h_vals[sh[-1]]
            prev_high = h_vals[sh[-2]]
            last_low  = l_vals[sl[-1]]
            prev_low  = l_vals[sl[-2]]

            # Bullish BOS: price breaks above last swing high
            if price > last_high and last_high > prev_high:
                result = {"direction": 1, "strength": 1.0, "label": "BOS Bullish"}
            # Bearish BOS: price breaks below last swing low
            elif price < last_low and last_low < prev_low:
                result = {"direction": -1, "strength": 1.0, "label": "BOS Bearish"}
            # CHoCH: break of prior structure in opposite direction
            elif price > last_high and last_high < prev_high:
                result = {"direction": 1, "strength": 1.1, "label": "CHoCH Bullish"}
            elif price < last_low and last_low > prev_low:
                result = {"direction": -1, "strength": 1.1, "label": "CHoCH Bearish"}
    except Exception as e:
        logger.debug(f"BOS/CHoCH error: {e}")
    return result


def detect_liquidity_sweep(df: pd.DataFrame) -> Dict:
    """Liquidity Sweep — price briefly breaks a key level then reverses."""
    result = {"direction": 0, "strength": 0, "label": "Liq Sweep"}
    try:
        sh = find_swing_highs(df, 5)
        sl = find_swing_lows(df, 5)
        if not sh or not sl:
            return result

        last_swing_high = df["high"].values[sh[-1]]
        last_swing_low  = df["low"].values[sl[-1]]
        recent = df.iloc[-3:]

        # Sweep high and reject
        if recent["high"].max() > last_swing_high and recent["close"].iloc[-1] < last_swing_high:
            result = {"direction": -1, "strength": 1.0, "label": "Liquidity Sweep High"}
        # Sweep low and reject
        elif recent["low"].min() < last_swing_low and recent["close"].iloc[-1] > last_swing_low:
            result = {"direction": 1, "strength": 1.0, "label": "Liquidity Sweep Low"}
    except Exception as e:
        logger.debug(f"Liquidity sweep error: {e}")
    return result


def detect_fibonacci(df: pd.DataFrame) -> Dict:
    """Auto Fibonacci retracement from last major swing."""
    result = {"direction": 0, "strength": 0, "label": "Fibonacci", "levels": {}}
    try:
        sh = find_swing_highs(df, 5)
        sl = find_swing_lows(df, 5)
        if not sh or not sl:
            return result

        swing_high = df["high"].values[sh[-1]]
        swing_low  = df["low"].values[sl[-1]]
        price      = df["close"].iloc[-1]
        rng        = swing_high - swing_low

        if rng <= 0:
            return result

        # Determine trend direction
        trend = 1 if sh[-1] < sl[-1] else -1  # simplistic

        levels = {}
        for ratio in [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]:
            if trend == 1:
                levels[ratio] = swing_low + ratio * rng
            else:
                levels[ratio] = swing_high - ratio * rng

        # Check if price is near a key fib level
        for ratio, lvl in levels.items():
            if abs(price - lvl) / price < 0.003:
                fib_direction = -1 if ratio in [0.618, 0.786] else 1
                result = {
                    "direction": fib_direction,
                    "strength": 0.8,
                    "label": f"Fib {ratio:.3f} ({lvl:.4f})",
                    "levels": levels
                }
                break
        else:
            result["levels"] = levels

    except Exception as e:
        logger.debug(f"Fibonacci error: {e}")
    return result


def detect_premium_discount(df: pd.DataFrame) -> Dict:
    """Premium & Discount zones based on Fibonacci 50%."""
    result = {"direction": 0, "strength": 0, "label": "Premium/Discount"}
    try:
        sh = find_swing_highs(df, 5)
        sl = find_swing_lows(df, 5)
        if not sh or not sl:
            return result
        swing_high = df["high"].values[sh[-1]]
        swing_low  = df["low"].values[sl[-1]]
        midpoint   = (swing_high + swing_low) / 2
        price      = df["close"].iloc[-1]

        if price < midpoint:
            result = {"direction": 1, "strength": 0.7, "label": "Discount Zone (Buy)"}
        else:
            result = {"direction": -1, "strength": 0.7, "label": "Premium Zone (Sell)"}
    except Exception as e:
        logger.debug(f"Premium/Discount error: {e}")
    return result


def detect_supply_demand_zones(df: pd.DataFrame) -> List[Dict]:
    """Supply and demand zones from strong impulse candles."""
    zones = []
    try:
        closes = df["close"].values
        opens  = df["open"].values
        for i in range(10, len(df) - 5):
            move = (closes[i] - opens[i]) / opens[i]
            if move > 0.01:  # 1% bullish candle → demand zone below
                zones.append({
                    "type": "Demand Zone",
                    "top": opens[i],
                    "bottom": df["low"].iloc[i],
                    "direction": 1, "strength": 0.9
                })
            elif move < -0.01:  # 1% bearish candle → supply zone above
                zones.append({
                    "type": "Supply Zone",
                    "top": df["high"].iloc[i],
                    "bottom": opens[i],
                    "direction": -1, "strength": 0.9
                })
    except Exception as e:
        logger.debug(f"S/D zones error: {e}")
    return zones[-5:]


# ─── MASTER PATTERN RUNNER ────────────────────────────────────────────────────

def compute_all_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Run all pattern detectors. Returns structured results."""
    if df is None or len(df) < 20:
        return {}

    results = {}

    # Candlestick patterns
    cs = candlestick_patterns(df)
    results["candlestick"] = cs  # list of dicts

    # Chart patterns
    chart_patterns = []
    for fn in [detect_double_top_bottom, detect_head_shoulders, detect_triangle,
                detect_wedge, detect_flag_pennant, detect_cup_handle, detect_gaps]:
        r = fn(df)
        if r:
            chart_patterns.append(r)
    results["chart_patterns"] = chart_patterns

    # Price action / ICT
    results["support_resistance"] = detect_support_resistance(df)
    results["fvg"]                = detect_fair_value_gaps(df)
    results["order_blocks"]       = detect_order_blocks(df)
    results["bos_choch"]          = detect_bos_choch(df)
    results["liquidity_sweep"]    = detect_liquidity_sweep(df)
    results["fibonacci"]          = detect_fibonacci(df)
    results["premium_discount"]   = detect_premium_discount(df)
    results["supply_demand"]      = detect_supply_demand_zones(df)

    return results
