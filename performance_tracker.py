"""
performance_tracker.py - Track whether signals hit TP1/TP2/TP3 or SL within 24 hours.
Runs asynchronously after each signal is issued.
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import yfinance as yf
import pandas as pd

from logger import logger, log_performance, load_performance_log


# ─── SINGLE SIGNAL TRACKER ────────────────────────────────────────────────────

async def track_signal_outcome(signal: dict, check_interval: int = 900) -> dict:
    """
    Poll price every check_interval seconds for up to 24h.
    Returns performance record with result (TP1_HIT, TP2_HIT, TP3_HIT, SL_HIT, EXPIRED).
    """
    asset     = signal["asset"]
    direction = signal["_direction_int"]  # 1 = long, -1 = short
    entry     = float(signal["entry"])
    tp1       = float(signal["tp1"])
    tp2       = float(signal["tp2"])
    tp3       = float(signal["tp3"])
    sl        = float(signal["stop_loss"])
    sent_time = datetime.now(timezone.utc)
    expiry    = sent_time + timedelta(hours=24)

    logger.info(f"📡 Tracking signal: {asset} {signal['direction']}")

    result     = "OPEN"
    exit_price = entry
    exit_time  = None

    while datetime.now(timezone.utc) < expiry:
        await asyncio.sleep(check_interval)
        try:
            ticker = yf.Ticker(asset)
            hist   = ticker.history(period="1d", interval="5m", auto_adjust=True)
            if hist.empty:
                continue

            current_high  = float(hist["High"].max())
            current_low   = float(hist["Low"].min())
            current_price = float(hist["Close"].iloc[-1])

            if direction == 1:  # LONG
                if current_low <= sl:
                    result = "SL_HIT"; exit_price = sl; break
                elif current_high >= tp3:
                    result = "TP3_HIT"; exit_price = tp3; break
                elif current_high >= tp2:
                    result = "TP2_HIT"; exit_price = tp2
                    # Don't break — keep tracking for TP3 possibility
                elif current_high >= tp1:
                    result = "TP1_HIT"; exit_price = tp1
            else:  # SHORT
                if current_high >= sl:
                    result = "SL_HIT"; exit_price = sl; break
                elif current_low <= tp3:
                    result = "TP3_HIT"; exit_price = tp3; break
                elif current_low <= tp2:
                    result = "TP2_HIT"; exit_price = tp2
                elif current_low <= tp1:
                    result = "TP1_HIT"; exit_price = tp1

        except Exception as e:
            logger.debug(f"Performance tracking error for {asset}: {e}")

    if result == "OPEN":
        result = "EXPIRED"

    exit_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pnl_pct   = ((exit_price - entry) / entry * direction * 100) if entry else 0.0

    record = {
        "signal_id":      signal.get("timestamp", ""),
        "asset":          asset,
        "direction":      signal["direction"],
        "entry":          entry,
        "tp1":            tp1,
        "tp2":            tp2,
        "tp3":            tp3,
        "stop_loss":      sl,
        "confidence":     signal.get("confidence", 0),
        "timestamp_sent": signal.get("timestamp", ""),
        "result":         result,
        "exit_price":     round(exit_price, 6),
        "exit_time":      exit_time,
        "pnl_pct":        round(pnl_pct, 3),
    }

    log_performance(record)
    logger.info(
        f"📊 Outcome — {asset} {signal['direction']}: "
        f"{result} | PnL: {pnl_pct:+.2f}% | Exit: {exit_price:.6f}"
    )
    return record


# ─── ASYNC SCHEDULER ──────────────────────────────────────────────────────────

async def schedule_tracking(signal: dict):
    """Fire-and-forget task to track signal outcome."""
    asyncio.create_task(track_signal_outcome(signal))


# ─── DAILY STATS ─────────────────────────────────────────────────────────────

def compute_daily_stats() -> dict:
    """Compute win/loss stats from performance log."""
    records = load_performance_log()
    if not records:
        return {"total": 0, "tp_hits": 0, "sl_hits": 0, "open": 0,
                "win_rate": 0.0, "best_asset": "N/A"}

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_records = [
        r for r in records
        if r.get("timestamp_sent", "").startswith(today_str)
    ]

    total  = len(today_records)
    tp_hits = sum(1 for r in today_records if "TP" in r.get("result", ""))
    sl_hits = sum(1 for r in today_records if "SL" in r.get("result", ""))
    open_c  = sum(1 for r in today_records if r.get("result", "") in ["OPEN", "EXPIRED"])

    win_rate = (tp_hits / total * 100) if total else 0.0

    # Best performing asset today
    asset_pnl: Dict[str, List[float]] = {}
    for r in today_records:
        a = r.get("asset", "")
        p = float(r.get("pnl_pct", 0))
        if a not in asset_pnl:
            asset_pnl[a] = []
        asset_pnl[a].append(p)

    best_asset = "N/A"
    if asset_pnl:
        best_asset = max(asset_pnl, key=lambda a: sum(asset_pnl[a]))

    return {
        "total":      total,
        "tp_hits":    tp_hits,
        "sl_hits":    sl_hits,
        "open":       open_c,
        "win_rate":   round(win_rate, 1),
        "best_asset": best_asset,
    }


def overall_stats() -> dict:
    """Compute all-time stats."""
    records = load_performance_log()
    if not records:
        return {}
    closed = [r for r in records if r.get("result") not in ["OPEN", ""]]
    total  = len(closed)
    wins   = sum(1 for r in closed if "TP" in r.get("result", ""))
    losses = sum(1 for r in closed if "SL" in r.get("result", ""))
    pnl    = [float(r.get("pnl_pct", 0)) for r in closed]
    return {
        "total_closed": total,
        "wins":         wins,
        "losses":       losses,
        "win_rate":     round(wins / total * 100, 1) if total else 0,
        "avg_pnl":      round(sum(pnl) / len(pnl), 3) if pnl else 0,
        "total_pnl":    round(sum(pnl), 2),
    }
