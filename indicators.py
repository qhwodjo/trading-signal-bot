"""
indicators.py - Comprehensive technical analysis engine.
Computes all trend, momentum, volatility, volume, and oscillator indicators.
Returns structured signal data with direction and strength.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings("ignore")

try:
    import pandas_ta as ta
except ImportError:
    ta = None

from logger import logger


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _safe(val):
    """Return float or None safely."""
    try:
        v = float(val)
        return v if not np.isnan(v) else None
    except Exception:
        return None

def _last(series: pd.Series):
    return _safe(series.iloc[-1]) if len(series) else None

def _prev(series: pd.Series, n: int = 1):
    return _safe(series.iloc[-(n+1)]) if len(series) > n else None


# ─── TREND INDICATORS ─────────────────────────────────────────────────────────

def ema_signals(df: pd.DataFrame) -> dict:
    """EMA 9/21/50/200 cross signals."""
    signals = {}
    c = df["close"]
    for p in [9, 21, 50, 200]:
        df[f"ema{p}"] = c.ewm(span=p, adjust=False).mean()

    ema9   = _last(df["ema9"])
    ema21  = _last(df["ema21"])
    ema50  = _last(df["ema50"])
    ema200 = _last(df["ema200"])
    price  = _last(c)

    if None in [ema9, ema21, ema50, ema200, price]:
        return {"direction": 0, "strength": 0, "label": "EMA"}

    bull_count = sum([
        price > ema9, price > ema21, price > ema50, price > ema200,
        ema9 > ema21, ema21 > ema50, ema50 > ema200
    ])
    # Symmetric: >=5 → LONG, <=2 → SHORT, 3-4 = no signal (weak/mixed)
    if bull_count >= 5:
        direction = 1
    elif bull_count <= 2:
        direction = -1
    else:
        direction = 0
    strength = abs(bull_count - 3.5) / 3.5
    return {"direction": direction, "strength": strength, "label": "EMA Cross"}


def sma_signals(df: pd.DataFrame) -> dict:
    c = df["close"]
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    price  = _last(c)
    s50    = _last(sma50)
    s200   = _last(sma200)
    if None in [price, s50, s200]:
        return {"direction": 0, "strength": 0.5, "label": "SMA"}
    direction = 1 if (price > s50 > s200) else (-1 if price < s50 < s200 else 0)
    return {"direction": direction, "strength": 0.6, "label": "SMA"}


def vwap_signal(df: pd.DataFrame) -> dict:
    """VWAP signal — price above/below VWAP."""
    if "volume" not in df.columns or df["volume"].sum() == 0:
        return {"direction": 0, "strength": 0, "label": "VWAP"}
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical * df["volume"]).cumsum() / df["volume"].cumsum()
    price = _last(df["close"])
    v     = _last(vwap)
    if None in [price, v]:
        return {"direction": 0, "strength": 0, "label": "VWAP"}
    direction = 1 if price > v else -1
    strength  = min(abs(price - v) / v * 10, 1.0) if v else 0
    return {"direction": direction, "strength": strength, "label": "VWAP"}


def supertrend_signal(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> dict:
    """Supertrend indicator."""
    try:
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        hl2 = (high + low) / 2
        upper_band = hl2 + mult * atr
        lower_band = hl2 - mult * atr

        supertrend = pd.Series(index=df.index, dtype=float)
        direction  = pd.Series(index=df.index, dtype=int)
        for i in range(period, len(df)):
            prev_st = supertrend.iloc[i-1] if i > period else upper_band.iloc[i]
            if close.iloc[i] > prev_st:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1

        d = _last(direction)
        return {"direction": int(d) if d else 0, "strength": 0.8, "label": "Supertrend"}
    except Exception as e:
        logger.debug(f"Supertrend error: {e}")
        return {"direction": 0, "strength": 0, "label": "Supertrend"}


def ichimoku_signal(df: pd.DataFrame) -> dict:
    """Ichimoku Cloud — Tenkan, Kijun, Senkou A/B, Chikou."""
    try:
        h, l, c = df["high"], df["low"], df["close"]
        tenkan  = (h.rolling(9).max()  + l.rolling(9).min())  / 2
        kijun   = (h.rolling(26).max() + l.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)

        p  = _last(c)
        t  = _last(tenkan)
        k  = _last(kijun)
        sa = _last(senkou_a)
        sb = _last(senkou_b)
        if None in [p, t, k, sa, sb]:
            return {"direction": 0, "strength": 0, "label": "Ichimoku"}

        bull = sum([p > max(sa, sb), t > k, p > t, p > k])
        bear = sum([p < min(sa, sb), t < k, p < t, p < k])
        direction = 1 if bull >= 3 else (-1 if bear >= 3 else 0)
        strength  = max(bull, bear) / 4
        return {"direction": direction, "strength": strength, "label": "Ichimoku"}
    except Exception as e:
        logger.debug(f"Ichimoku error: {e}")
        return {"direction": 0, "strength": 0, "label": "Ichimoku"}


def adx_signal(df: pd.DataFrame, period: int = 14) -> dict:
    """ADX + DI+/DI-."""
    try:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs()
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(period).mean()

        dmp = (h - h.shift()).clip(lower=0)
        dmm = (l.shift() - l).clip(lower=0)
        dmp[dmp < dmm] = 0
        dmm[dmm < dmp] = 0

        di_plus  = 100 * dmp.rolling(period).mean() / atr14
        di_minus = 100 * dmm.rolling(period).mean() / atr14
        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
        adx = dx.rolling(period).mean()

        adx_val  = _last(adx)
        dip      = _last(di_plus)
        dim      = _last(di_minus)
        if None in [adx_val, dip, dim]:
            return {"direction": 0, "strength": 0, "label": "ADX"}

        if adx_val < 20:
            return {"direction": 0, "strength": 0.2, "label": "ADX (weak trend)"}
        direction = 1 if dip > dim else -1
        strength  = min(adx_val / 50, 1.0)
        return {"direction": direction, "strength": strength, "label": f"ADX({adx_val:.1f})"}
    except Exception as e:
        logger.debug(f"ADX error: {e}")
        return {"direction": 0, "strength": 0, "label": "ADX"}


def parabolic_sar_signal(df: pd.DataFrame) -> dict:
    """Parabolic SAR — price above SAR = bull."""
    try:
        if ta:
            sar = ta.psar(df["high"], df["low"], df["close"])
            if sar is not None and not sar.empty:
                col = [c for c in sar.columns if "PSARl" in c or "PSARs" in c]
                if col:
                    price = _last(df["close"])
                    sar_val = _last(sar[col[0]])
                    if price and sar_val:
                        direction = 1 if price > sar_val else -1
                        return {"direction": direction, "strength": 0.7, "label": "Parabolic SAR"}

        # Manual fallback
        af = 0.02; max_af = 0.2
        bull = df["close"].iloc[0] > df["open"].iloc[0]
        ep  = df["high"].iloc[0] if bull else df["low"].iloc[0]
        sar_val = df["low"].iloc[0] if bull else df["high"].iloc[0]
        for i in range(1, len(df)):
            sar_val = sar_val + af * (ep - sar_val)
            if bull:
                if df["low"].iloc[i] < sar_val:
                    bull = False; sar_val = ep; ep = df["low"].iloc[i]; af = 0.02
                else:
                    if df["high"].iloc[i] > ep:
                        ep = df["high"].iloc[i]; af = min(af + 0.02, max_af)
            else:
                if df["high"].iloc[i] > sar_val:
                    bull = True; sar_val = ep; ep = df["high"].iloc[i]; af = 0.02
                else:
                    if df["low"].iloc[i] < ep:
                        ep = df["low"].iloc[i]; af = min(af + 0.02, max_af)

        direction = 1 if bull else -1
        return {"direction": direction, "strength": 0.7, "label": "Parabolic SAR"}
    except Exception as e:
        logger.debug(f"PSAR error: {e}")
        return {"direction": 0, "strength": 0, "label": "Parabolic SAR"}


def aroon_signal(df: pd.DataFrame, period: int = 25) -> dict:
    """Aroon Up/Down."""
    try:
        h, l = df["high"], df["low"]
        aroon_up   = 100 * h.rolling(period + 1).apply(lambda x: (period - x[::-1].argmax()) / period)
        aroon_down = 100 * l.rolling(period + 1).apply(lambda x: (period - x[::-1].argmin()) / period)
        up   = _last(aroon_up)
        down = _last(aroon_down)
        if None in [up, down]:
            return {"direction": 0, "strength": 0, "label": "Aroon"}
        direction = 1 if up > 70 and down < 30 else (-1 if down > 70 and up < 30 else 0)
        strength  = abs(up - down) / 100
        return {"direction": direction, "strength": strength, "label": f"Aroon({up:.0f}/{down:.0f})"}
    except Exception as e:
        logger.debug(f"Aroon error: {e}")
        return {"direction": 0, "strength": 0, "label": "Aroon"}


# ─── MOMENTUM INDICATORS ──────────────────────────────────────────────────────

def rsi_signal(df: pd.DataFrame, period: int = 14) -> dict:
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    rsi   = 100 - (100 / (1 + rs))
    val   = _last(rsi)
    if val is None:
        return {"direction": 0, "strength": 0, "label": "RSI"}
    if val < 30:
        return {"direction": 1,  "strength": (30 - val) / 30, "label": f"RSI({val:.1f}) Oversold"}
    if val > 70:
        return {"direction": -1, "strength": (val - 70) / 30, "label": f"RSI({val:.1f}) Overbought"}
    return {"direction": 0, "strength": 0.3, "label": f"RSI({val:.1f}) Neutral"}


def stoch_rsi_signal(df: pd.DataFrame) -> dict:
    try:
        if ta:
            s = ta.stochrsi(df["close"])
            if s is not None and not s.empty:
                k_col = [c for c in s.columns if "STOCHRSIk" in c]
                d_col = [c for c in s.columns if "STOCHRSId" in c]
                if k_col and d_col:
                    k = _last(s[k_col[0]])
                    d = _last(s[d_col[0]])
                    if None not in [k, d]:
                        bull  = k < 20 and d < 20 and k > d
                        bear  = k > 80 and d > 80 and k < d
                        direction = 1 if bull else (-1 if bear else 0)
                        strength  = 0.7 if bull or bear else 0.3
                        return {"direction": direction, "strength": strength, "label": f"StochRSI({k:.1f})"}
    except Exception:
        pass
    return {"direction": 0, "strength": 0, "label": "StochRSI"}


def macd_signal(df: pd.DataFrame) -> dict:
    c = df["close"]
    exp1 = c.ewm(span=12, adjust=False).mean()
    exp2 = c.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal

    m  = _last(macd)
    s  = _last(signal)
    h  = _last(hist)
    ph = _prev(hist)
    if None in [m, s, h, ph]:
        return {"direction": 0, "strength": 0, "label": "MACD"}

    # Crossover
    if h > 0 and ph <= 0:
        return {"direction": 1,  "strength": 0.9, "label": "MACD Bullish Cross"}
    if h < 0 and ph >= 0:
        return {"direction": -1, "strength": 0.9, "label": "MACD Bearish Cross"}
    direction = 1 if m > s else -1
    strength  = min(abs(h) / abs(m) if m else 0.5, 1.0)
    return {"direction": direction, "strength": strength, "label": "MACD"}


def cci_signal(df: pd.DataFrame, period: int = 20) -> dict:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    mean    = typical.rolling(period).mean()
    mad     = typical.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    cci     = (typical - mean) / (0.015 * mad)
    val     = _last(cci)
    if val is None:
        return {"direction": 0, "strength": 0, "label": "CCI"}
    direction = 1 if val < -100 else (-1 if val > 100 else 0)
    strength  = min(abs(val) / 200, 1.0)
    return {"direction": direction, "strength": strength, "label": f"CCI({val:.0f})"}


def williams_r_signal(df: pd.DataFrame, period: int = 14) -> dict:
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    wr = -100 * (hh - df["close"]) / (hh - ll)
    val = _last(wr)
    if val is None:
        return {"direction": 0, "strength": 0, "label": "Williams%R"}
    direction = 1 if val < -80 else (-1 if val > -20 else 0)
    return {"direction": direction, "strength": 0.6, "label": f"Williams%R({val:.1f})"}


def awesome_oscillator_signal(df: pd.DataFrame) -> dict:
    median = (df["high"] + df["low"]) / 2
    ao   = median.rolling(5).mean() - median.rolling(34).mean()
    val  = _last(ao)
    prev = _prev(ao)
    if None in [val, prev]:
        return {"direction": 0, "strength": 0, "label": "AO"}
    # Zero-line cross (strong signal)
    if val > 0 and prev <= 0:
        return {"direction": 1,  "strength": 0.8, "label": "AO Zero Cross Up"}
    if val < 0 and prev >= 0:
        return {"direction": -1, "strength": 0.8, "label": "AO Zero Cross Down"}
    # Trending
    if val > 0 and val > prev:
        return {"direction": 1,  "strength": 0.5, "label": "AO Bullish"}
    if val < 0 and val < prev:
        return {"direction": -1, "strength": 0.5, "label": "AO Bearish"}
    return {"direction": 0, "strength": 0, "label": "AO"}


def tsi_signal(df: pd.DataFrame) -> dict:
    try:
        if ta:
            tsi = ta.tsi(df["close"])
            if tsi is not None:
                col = tsi.columns[0] if hasattr(tsi, "columns") else None
                val = _last(tsi[col]) if col else _last(tsi)
                if val is not None:
                    direction = 1 if val > 0 else -1
                    return {"direction": direction, "strength": min(abs(val)/25, 1.0), "label": f"TSI({val:.1f})"}
    except Exception:
        pass
    return {"direction": 0, "strength": 0, "label": "TSI"}


def roc_signal(df: pd.DataFrame, period: int = 12) -> dict:
    roc = df["close"].pct_change(period) * 100
    val = _last(roc)
    if val is None:
        return {"direction": 0, "strength": 0, "label": "ROC"}
    # Only fire when meaningfully positive or negative (> 0.3%)
    if abs(val) < 0.3:
        return {"direction": 0, "strength": 0, "label": f"ROC({val:.2f}%) Flat"}
    direction = 1 if val > 0 else -1
    # Scale: 0.3%→0.4 strength, 2.5%→1.0 strength
    strength  = min(0.4 + abs(val) / 4, 1.0)
    return {"direction": direction, "strength": strength, "label": f"ROC({val:.2f}%)"}


def ultimate_oscillator_signal(df: pd.DataFrame) -> dict:
    try:
        if ta:
            uo = ta.uo(df["high"], df["low"], df["close"])
            val = _last(uo)
            if val:
                direction = 1 if val < 30 else (-1 if val > 70 else 0)
                return {"direction": direction, "strength": 0.6, "label": f"UO({val:.1f})"}
    except Exception:
        pass
    return {"direction": 0, "strength": 0, "label": "UO"}


def momentum_signal(df: pd.DataFrame, period: int = 10) -> dict:
    mom   = df["close"] - df["close"].shift(period)
    val   = _last(mom)
    price = _last(df["close"])
    if val is None or price is None or price == 0:
        return {"direction": 0, "strength": 0, "label": "Momentum"}
    pct_move = abs(val) / price * 100
    # Only fire when momentum is meaningful (> 0.2% move)
    if pct_move < 0.2:
        return {"direction": 0, "strength": 0, "label": "Momentum Flat"}
    direction = 1 if val > 0 else -1
    strength  = min(0.4 + pct_move / 5, 1.0)
    return {"direction": direction, "strength": strength, "label": f"Momentum({'↑' if direction==1 else '↓'})"}


# ─── VOLATILITY INDICATORS ────────────────────────────────────────────────────

def bollinger_signal(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> dict:
    c    = df["close"]
    mid  = c.rolling(period).mean()
    band = c.rolling(period).std()
    upper = mid + std * band
    lower = mid - std * band
    price = _last(c)
    u     = _last(upper)
    l     = _last(lower)
    m     = _last(mid)
    if None in [price, u, l, m]:
        return {"direction": 0, "strength": 0, "label": "BB"}
    if price < l:
        return {"direction": 1,  "strength": 0.8, "label": "BB Oversold"}
    if price > u:
        return {"direction": -1, "strength": 0.8, "label": "BB Overbought"}
    # Squeeze detection
    bandwidth = (u - l) / m
    if bandwidth < 0.02:
        return {"direction": 0, "strength": 0.5, "label": "BB Squeeze"}
    return {"direction": 0, "strength": 0.2, "label": "BB Neutral"}


def atr_value(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return _last(tr.rolling(period).mean())


def keltner_signal(df: pd.DataFrame) -> dict:
    try:
        if ta:
            kc = ta.kc(df["high"], df["low"], df["close"])
            if kc is not None and not kc.empty:
                ku_col = [c for c in kc.columns if "KCUe" in c]
                kl_col = [c for c in kc.columns if "KCLe" in c]
                if ku_col and kl_col:
                    price = _last(df["close"])
                    ku = _last(kc[ku_col[0]])
                    kl = _last(kc[kl_col[0]])
                    if None not in [price, ku, kl]:
                        direction = 1 if price < kl else (-1 if price > ku else 0)
                        return {"direction": direction, "strength": 0.7, "label": "Keltner Channel"}
    except Exception:
        pass
    return {"direction": 0, "strength": 0, "label": "Keltner Channel"}


def donchian_signal(df: pd.DataFrame, period: int = 20) -> dict:
    h_max = df["high"].rolling(period).max()
    l_min = df["low"].rolling(period).min()
    price = _last(df["close"])
    h     = _last(h_max)
    l     = _last(l_min)
    if None in [price, h, l]:
        return {"direction": 0, "strength": 0, "label": "Donchian"}
    if price >= h:
        return {"direction": 1,  "strength": 0.8, "label": "Donchian Breakout UP"}
    if price <= l:
        return {"direction": -1, "strength": 0.8, "label": "Donchian Breakout DOWN"}
    return {"direction": 0, "strength": 0.3, "label": "Donchian Mid"}


# ─── VOLUME INDICATORS ────────────────────────────────────────────────────────

def obv_signal(df: pd.DataFrame) -> dict:
    try:
        direction_arr = np.sign(df["close"].diff().fillna(0))
        obv    = (direction_arr * df["volume"]).cumsum()
        obv_ma = obv.rolling(20).mean()
        o  = _last(obv)
        om = _last(obv_ma)
        if None in [o, om]:
            return {"direction": 0, "strength": 0, "label": "OBV"}
        # Divergence: OBV vs price trend
        price_up = df["close"].iloc[-1] > df["close"].iloc[-10]
        obv_up   = o > obv_ma.iloc[-10] if len(obv_ma) > 10 else (o > om)
        if price_up and obv_up:
            return {"direction": 1,  "strength": 0.7, "label": "OBV Bullish"}
        if not price_up and not obv_up:
            return {"direction": -1, "strength": 0.7, "label": "OBV Bearish"}
        # Divergence (opposing direction = early reversal signal)
        if price_up and not obv_up:
            return {"direction": -1, "strength": 0.6, "label": "OBV Bearish Divergence"}
        if not price_up and obv_up:
            return {"direction": 1,  "strength": 0.6, "label": "OBV Bullish Divergence"}
        return {"direction": 0, "strength": 0, "label": "OBV"}
    except Exception:
        return {"direction": 0, "strength": 0, "label": "OBV"}


def mfi_signal(df: pd.DataFrame, period: int = 14) -> dict:
    try:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        raw_mf  = typical * df["volume"]
        positive = raw_mf.where(typical > typical.shift(), 0).rolling(period).sum()
        negative = raw_mf.where(typical < typical.shift(), 0).rolling(period).sum()
        mfr = positive / negative
        mfi = 100 - 100 / (1 + mfr)
        val = _last(mfi)
        if val is None:
            return {"direction": 0, "strength": 0, "label": "MFI"}
        direction = 1 if val < 20 else (-1 if val > 80 else 0)
        return {"direction": direction, "strength": abs(val - 50) / 50, "label": f"MFI({val:.1f})"}
    except Exception:
        return {"direction": 0, "strength": 0, "label": "MFI"}


def cmf_signal(df: pd.DataFrame, period: int = 20) -> dict:
    try:
        mf_mult = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
        mf_vol  = mf_mult * df["volume"]
        cmf     = mf_vol.rolling(period).sum() / df["volume"].rolling(period).sum()
        val     = _last(cmf)
        if val is None:
            return {"direction": 0, "strength": 0, "label": "CMF"}
        direction = 1 if val > 0 else -1
        return {"direction": direction, "strength": min(abs(val) * 2, 1.0), "label": f"CMF({val:.2f})"}
    except Exception:
        return {"direction": 0, "strength": 0, "label": "CMF"}


def force_index_signal(df: pd.DataFrame, period: int = 13) -> dict:
    try:
        fi = df["close"].diff() * df["volume"]
        fi_ema = fi.ewm(span=period, adjust=False).mean()
        val = _last(fi_ema)
        if val is None:
            return {"direction": 0, "strength": 0, "label": "Force Index"}
        direction = 1 if val > 0 else -1
        return {"direction": direction, "strength": 0.5, "label": "Force Index"}
    except Exception:
        return {"direction": 0, "strength": 0, "label": "Force Index"}


def ease_of_movement_signal(df: pd.DataFrame) -> dict:
    try:
        box_ratio = df["volume"] / (df["high"] - df["low"])
        mid_move  = df["high"].add(df["low"]).div(2).diff()
        eom       = mid_move / box_ratio
        eom_ma    = eom.rolling(14).mean()
        val       = _last(eom_ma)
        if val is None:
            return {"direction": 0, "strength": 0, "label": "EOM"}
        direction = 1 if val > 0 else -1
        return {"direction": direction, "strength": 0.5, "label": "EOM"}
    except Exception:
        return {"direction": 0, "strength": 0, "label": "EOM"}


# ─── OSCILLATORS ──────────────────────────────────────────────────────────────

def stochastic_signal(df: pd.DataFrame, k: int = 14, d: int = 3) -> dict:
    hh = df["high"].rolling(k).max()
    ll = df["low"].rolling(k).min()
    K  = 100 * (df["close"] - ll) / (hh - ll)
    D  = K.rolling(d).mean()
    kv  = _last(K)
    dv  = _last(D)
    pkv = _prev(K)
    if None in [kv, dv]:
        return {"direction": 0, "strength": 0, "label": "Stochastic"}
    # Oversold cross up
    if kv < 20 and kv > dv:
        return {"direction": 1,  "strength": 0.8, "label": f"Stoch Oversold ({kv:.0f})"}
    # Overbought cross down
    if kv > 80 and kv < dv:
        return {"direction": -1, "strength": 0.8, "label": f"Stoch Overbought ({kv:.0f})"}
    # Mid-range momentum
    if 20 <= kv <= 50 and kv > dv:
        return {"direction": 1,  "strength": 0.5, "label": f"Stoch Bull ({kv:.0f})"}
    if 50 <= kv <= 80 and kv < dv:
        return {"direction": -1, "strength": 0.5, "label": f"Stoch Bear ({kv:.0f})"}
    return {"direction": 0, "strength": 0, "label": f"Stoch Neutral ({kv:.0f})"}


def demarker_signal(df: pd.DataFrame, period: int = 14) -> dict:
    try:
        demax = (df["high"] - df["high"].shift()).clip(lower=0).rolling(period).mean()
        demin = (df["low"].shift() - df["low"]).clip(lower=0).rolling(period).mean()
        dem   = demax / (demax + demin)
        val   = _last(dem)
        if val is None:
            return {"direction": 0, "strength": 0, "label": "DeMarker"}
        direction = 1 if val < 0.3 else (-1 if val > 0.7 else 0)
        return {"direction": direction, "strength": abs(val - 0.5) * 2, "label": f"DeMarker({val:.2f})"}
    except Exception:
        return {"direction": 0, "strength": 0, "label": "DeMarker"}


def ppo_signal(df: pd.DataFrame) -> dict:
    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    ppo  = (exp1 - exp2) / exp2 * 100
    signal = ppo.ewm(span=9, adjust=False).mean()
    val    = _last(ppo)
    sig    = _last(signal)
    if None in [val, sig]:
        return {"direction": 0, "strength": 0, "label": "PPO"}
    direction = 1 if val > sig else -1
    return {"direction": direction, "strength": 0.6, "label": f"PPO({val:.2f})"}


def schaff_trend_cycle(df: pd.DataFrame) -> dict:
    try:
        if ta:
            stc = ta.stc(df["close"])
            if stc is not None:
                col = stc.columns[0] if hasattr(stc, "columns") else None
                val = _last(stc[col]) if col else None
                if val is not None:
                    direction = 1 if val < 25 else (-1 if val > 75 else 0)
                    return {"direction": direction, "strength": abs(val - 50) / 50, "label": f"STC({val:.1f})"}
    except Exception:
        pass
    return {"direction": 0, "strength": 0, "label": "STC"}


# ─── MASTER INDICATOR RUNNER ──────────────────────────────────────────────────

def compute_all_indicators(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Run all indicators on a DataFrame. Returns dict keyed by indicator name."""
    if df is None or len(df) < 50:
        return {}

    results = {}

    # Trend
    results["ema_cross"]    = ema_signals(df)
    results["sma"]          = sma_signals(df)
    results["vwap"]         = vwap_signal(df)
    results["supertrend"]   = supertrend_signal(df)
    results["ichimoku"]     = ichimoku_signal(df)
    results["adx"]          = adx_signal(df)
    results["parabolic_sar"]= parabolic_sar_signal(df)
    results["aroon"]        = aroon_signal(df)

    # Momentum
    results["rsi"]          = rsi_signal(df)
    results["stoch_rsi"]    = stoch_rsi_signal(df)
    results["macd"]         = macd_signal(df)
    results["cci"]          = cci_signal(df)
    results["williams_r"]   = williams_r_signal(df)
    results["awesome_osc"]  = awesome_oscillator_signal(df)
    results["tsi"]          = tsi_signal(df)
    results["roc"]          = roc_signal(df)
    results["ultimate_osc"] = ultimate_oscillator_signal(df)
    results["momentum"]     = momentum_signal(df)

    # Volatility
    results["bollinger"]    = bollinger_signal(df)
    results["keltner"]      = keltner_signal(df)
    results["donchian"]     = donchian_signal(df)
    results["atr_val"]      = {"direction": 0, "strength": 0, "label": f"ATR({atr_value(df):.4f})" if atr_value(df) else "ATR"}

    # Volume
    results["obv"]          = obv_signal(df)
    results["mfi"]          = mfi_signal(df)
    results["cmf"]          = cmf_signal(df)
    results["force_index"]  = force_index_signal(df)
    results["eom"]          = ease_of_movement_signal(df)

    # Oscillators
    results["stochastic"]   = stochastic_signal(df)
    results["demarker"]     = demarker_signal(df)
    results["ppo"]          = ppo_signal(df)
    results["stc"]          = schaff_trend_cycle(df)

    return results