"""
backtester.py - Backtest top strategies on 6 months of historical data.
Uses vectorbt (or manual simulation as fallback).
Reports win rate, max drawdown, and Sharpe ratio for top 5 strategies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import yfinance as yf

from config import BACKTEST_ASSETS, BACKTEST_PERIOD_MONTHS
from indicators import compute_all_indicators, atr_value, rsi_signal, macd_signal, ema_signals
from logger import logger


# ─── STRATEGY DEFINITIONS ─────────────────────────────────────────────────────

def strategy_ema_cross(df: pd.DataFrame) -> pd.Series:
    """EMA 9/21 cross strategy. Returns +1/0/-1 signal series."""
    ema9  = df["close"].ewm(span=9,  adjust=False).mean()
    ema21 = df["close"].ewm(span=21, adjust=False).mean()
    signal = pd.Series(0, index=df.index)
    signal[ema9 > ema21]  =  1
    signal[ema9 <= ema21] = -1
    # Only trade on crossovers
    cross = signal.diff()
    entries = pd.Series(0, index=df.index)
    entries[cross == 2]  =  1   # crossed up
    entries[cross == -2] = -1   # crossed down
    return entries


def strategy_rsi_reversion(df: pd.DataFrame) -> pd.Series:
    """RSI oversold/overbought reversion strategy."""
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - 100 / (1 + gain / loss)
    signal = pd.Series(0, index=df.index)
    signal[rsi < 30] =  1   # buy oversold
    signal[rsi > 70] = -1   # sell overbought
    return signal


def strategy_macd_cross(df: pd.DataFrame) -> pd.Series:
    """MACD histogram cross strategy."""
    exp1   = df["close"].ewm(span=12, adjust=False).mean()
    exp2   = df["close"].ewm(span=26, adjust=False).mean()
    macd   = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    entries = pd.Series(0, index=df.index)
    entries[(hist > 0) & (hist.shift() <= 0)] =  1
    entries[(hist < 0) & (hist.shift() >= 0)] = -1
    return entries


def strategy_bb_reversion(df: pd.DataFrame) -> pd.Series:
    """Bollinger Band reversion at extremes."""
    mid   = df["close"].rolling(20).mean()
    std   = df["close"].rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    entries = pd.Series(0, index=df.index)
    entries[df["close"] < lower] =  1
    entries[df["close"] > upper] = -1
    return entries


def strategy_supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.Series:
    """Supertrend-based entries."""
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    hl2 = (h + l) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    direction = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if c.iloc[i] > upper.iloc[i-1]:
            direction.iloc[i] = 1
        elif c.iloc[i] < lower.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
    # Entries on direction change
    entries = pd.Series(0, index=df.index)
    entries[direction.diff() == 2]  =  1
    entries[direction.diff() == -2] = -1
    return entries


# ─── BACKTEST SIMULATOR ───────────────────────────────────────────────────────

def simulate_strategy(df: pd.DataFrame, signals: pd.Series,
                       atr_mult_tp: float = 2.0,
                       atr_mult_sl: float = 1.5) -> Dict:
    """
    Simulate a signal series on OHLCV data.
    Returns: win_rate, total_trades, max_drawdown, sharpe_ratio, returns list.
    """
    if df is None or len(df) < 50:
        return {"win_rate": 0, "total_trades": 0, "max_drawdown": 0, "sharpe_ratio": 0}

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().fillna(method="bfill")

    pnl_list  = []
    equity    = [1.0]
    in_trade  = False
    direction = 0
    entry_price = 0.0
    tp = 0.0
    sl = 0.0

    for i in range(20, len(df)):
        price = df["close"].iloc[i]
        atr_v = atr.iloc[i]
        sig   = signals.iloc[i]

        if in_trade:
            high = df["high"].iloc[i]
            low  = df["low"].iloc[i]
            if direction == 1:
                if high >= tp:
                    pnl = (tp - entry_price) / entry_price
                    pnl_list.append(pnl)
                    equity.append(equity[-1] * (1 + pnl))
                    in_trade = False
                elif low <= sl:
                    pnl = (sl - entry_price) / entry_price
                    pnl_list.append(pnl)
                    equity.append(equity[-1] * (1 + pnl))
                    in_trade = False
            else:
                if low <= tp:
                    pnl = (entry_price - tp) / entry_price
                    pnl_list.append(pnl)
                    equity.append(equity[-1] * (1 + pnl))
                    in_trade = False
                elif high >= sl:
                    pnl = (entry_price - sl) / entry_price
                    pnl_list.append(pnl)
                    equity.append(equity[-1] * (1 + pnl))
                    in_trade = False
        else:
            if sig == 1 and atr_v > 0:
                in_trade = True; direction = 1; entry_price = price
                tp = price + atr_mult_tp * atr_v
                sl = price - atr_mult_sl * atr_v
            elif sig == -1 and atr_v > 0:
                in_trade = True; direction = -1; entry_price = price
                tp = price - atr_mult_tp * atr_v
                sl = price + atr_mult_sl * atr_v

    if not pnl_list:
        return {"win_rate": 0, "total_trades": 0, "max_drawdown": 0, "sharpe_ratio": 0, "equity": [1.0]}

    wins         = sum(1 for p in pnl_list if p > 0)
    total_trades = len(pnl_list)
    win_rate     = wins / total_trades * 100

    eq = pd.Series(equity)
    peak = eq.cummax()
    drawdown = (eq - peak) / peak
    max_dd   = float(drawdown.min()) * 100

    returns = pd.Series(pnl_list)
    sharpe  = float((returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0)

    return {
        "win_rate":     round(win_rate, 1),
        "total_trades": total_trades,
        "max_drawdown": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "equity":       equity,
        "avg_pnl":      round(float(returns.mean()) * 100, 3),
    }


# ─── VECTORBT (optional, better perf) ────────────────────────────────────────

def run_vectorbt(df: pd.DataFrame, entries_long: pd.Series,
                  entries_short: pd.Series) -> Optional[Dict]:
    """Use vectorbt if available for faster simulation."""
    try:
        import vectorbt as vbt
        price = df["close"]
        pf = vbt.Portfolio.from_signals(
            price,
            entries=entries_long.astype(bool),
            exits=entries_short.astype(bool),
            short_entries=entries_short.astype(bool),
            short_exits=entries_long.astype(bool),
            freq="1D"
        )
        return {
            "win_rate":     round(pf.trades.win_rate * 100, 1),
            "total_trades": pf.trades.count(),
            "max_drawdown": round(pf.max_drawdown() * 100, 2),
            "sharpe_ratio": round(pf.sharpe_ratio(), 2),
            "total_return": round(pf.total_return() * 100, 2),
        }
    except Exception as e:
        logger.debug(f"vectorbt not available or error: {e}")
        return None


# ─── MAIN BACKTEST RUNNER ─────────────────────────────────────────────────────

STRATEGIES = {
    "EMA Cross":       strategy_ema_cross,
    "RSI Reversion":   strategy_rsi_reversion,
    "MACD Cross":      strategy_macd_cross,
    "BB Reversion":    strategy_bb_reversion,
    "Supertrend":      strategy_supertrend,
}


def run_all_backtests() -> Dict[str, Dict]:
    """Run all 5 strategies on BACKTEST_ASSETS and print results."""
    logger.info("=" * 60)
    logger.info("🔬 Running backtests on 6 months of data...")
    logger.info("=" * 60)

    end   = datetime.now()
    start = end - timedelta(days=BACKTEST_PERIOD_MONTHS * 30)

    all_results = {}

    for asset in BACKTEST_ASSETS:
        logger.info(f"\n  📊 Asset: {asset}")
        try:
            ticker = yf.Ticker(asset)
            df = ticker.history(start=start.strftime("%Y-%m-%d"),
                                end=end.strftime("%Y-%m-%d"),
                                interval="1d", auto_adjust=True)
            if df.empty or len(df) < 60:
                logger.warning(f"  ⚠️ Not enough data for {asset}")
                continue
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna()

            asset_results = {}
            for strat_name, strat_fn in STRATEGIES.items():
                try:
                    signals = strat_fn(df)
                    result  = simulate_strategy(df, signals)
                    asset_results[strat_name] = result

                    logger.info(
                        f"    [{strat_name:<20}] "
                        f"Win: {result['win_rate']:5.1f}% | "
                        f"Trades: {result['total_trades']:3d} | "
                        f"MaxDD: {result['max_drawdown']:6.2f}% | "
                        f"Sharpe: {result['sharpe_ratio']:5.2f}"
                    )
                except Exception as e:
                    logger.warning(f"    ⚠️ {strat_name} failed: {e}")

            all_results[asset] = asset_results

        except Exception as e:
            logger.error(f"  ❌ Backtest failed for {asset}: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Backtests complete.")
    logger.info("=" * 60)

    return all_results
