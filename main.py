"""
main.py - Trading Signal Bot with Telegram approval-before-execution flow.

Signal Flow:
    1. Bot scans assets every SCAN_INTERVAL_SECONDS
    2. For each qualifying signal, sends Telegram message with:
       ✅ Accept (recommended lot) | lot options | ✏️ Custom | ❌ Decline
    3. You tap a button (or type a custom lot size)
    4. Bot places the trade on MT5 automatically
    5. If no response within APPROVAL_TIMEOUT_SECONDS → signal auto-declines
"""

import asyncio
import sys
import traceback
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from config import (
    ALL_ASSETS, TIMEFRAMES, SCAN_INTERVAL_SECONDS,
    NEWS_REFRESH_SECONDS, SIGNAL_COOLDOWN_HOURS,
    ACCOUNT_BALANCE, RISK_PERCENT,
    PAUSE_AFTER_NEWS_MINUTES,
    MT5_AUTO_TRADE, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,
    MT5_MAX_DAILY_LOSS, MT5_RISK_PERCENT, MAX_OPEN_SIGNALS,
    LOT_RISK_CONSERVATIVE, LOT_RISK_RECOMMENDED, LOT_RISK_AGGRESSIVE
)
from data_fetcher import get_ohlcv
from signal_engine import generate_signal
from news_engine import (
    get_asset_news_sentiment, check_economic_calendar, async_get_all_news
)
from telegram_bot import (
    send_signal_for_approval, send_alert,
    send_startup_message, send_daily_summary,
    send_market_bias_alert, send_economic_event_warning,
    send_api_failure_alert, send_message
)
from approval_handler import (
    run_approval_listener, register_approve_callback,
    add_pending_signal, generate_signal_id
)
from mt5_trader import (
    connect_mt5, disconnect_mt5, execute_signal,
    suggest_lot_sizes, check_daily_loss_limit,
    get_open_positions, get_daily_pnl, get_account_balance,
    is_connected
)
from performance_tracker import schedule_tracking, compute_daily_stats
from backtester import run_all_backtests
from logger import logger, log_signal


# ─── STATE ───────────────────────────────────────────────────────────────────

signal_cooldown: Dict[str, datetime] = {}
last_news_time:  Optional[datetime]  = None
current_news:    List[Dict]          = []
last_daily_report_date: Optional[str] = None


# ─── APPROVAL CALLBACK ───────────────────────────────────────────────────────

async def on_trade_approved(signal: dict, lot: float):
    """
    Called by approval_handler when user taps Accept or submits custom lot.
    Places the trade on MT5 and sends confirmation to Telegram.
    """
    asset = signal["asset"]
    direc = signal["direction"]

    # Final safety checks
    if MT5_AUTO_TRADE:
        if not check_daily_loss_limit(MT5_MAX_DAILY_LOSS):
            await send_alert(
                f"🛑 Daily loss limit hit — cannot place {asset} {direc} trade.", "🔴"
            )
            return

        open_pos = get_open_positions()
        if len(open_pos) >= MAX_OPEN_SIGNALS:
            await send_alert(
                f"⚠️ Max open positions ({MAX_OPEN_SIGNALS}) reached. "
                f"Cannot place {asset} {direc}.", "⚠️"
            )
            return

        order = execute_signal(signal, lot=lot, risk_pct=MT5_RISK_PERCENT)
        if order:
            await send_message(
                f"✅ <b>Trade Placed on MT5!</b>\n\n"
                f"📌 <b>{asset}</b> {direc}\n"
                f"🎫 Ticket: <code>{order['ticket']}</code>\n"
                f"📦 Lot: <b>{order['lot']}</b>\n"
                f"✅ Entry: <b>{order['price']}</b>\n"
                f"🎯 TP: {order['tp']}\n"
                f"🛑 SL: {order['sl']}\n\n"
                f"💰 Balance: ${get_account_balance():.2f} | "
                f"Today P&L: ${get_daily_pnl():+.2f}"
            )
            logger.info(
                f"✅ MT5 order: {asset} {direc} | "
                f"Ticket:{order['ticket']} Lot:{order['lot']}"
            )
        else:
            await send_alert(
                f"❌ MT5 order FAILED for {asset} {direc}.\n"
                f"Check MT5 terminal — auto trading may be disabled.", "🔴"
            )
    else:
        # Signal-only mode — just confirm
        await send_message(
            f"📝 <b>Signal Noted</b> — {asset} {direc} @ {lot} lots\n"
            f"(MT5 auto-trade is disabled — enter manually)"
        )

    # Track outcome regardless
    await schedule_tracking(signal)


# ─── COOLDOWN ────────────────────────────────────────────────────────────────

def is_on_cooldown(asset: str, direction: str) -> bool:
    last = signal_cooldown.get(f"{asset}_{direction}")
    if not last:
        return False
    return (datetime.now(timezone.utc) - last).total_seconds() / 3600 < SIGNAL_COOLDOWN_HOURS

def mark_cooldown(asset: str, direction: str):
    signal_cooldown[f"{asset}_{direction}"] = datetime.now(timezone.utc)


# ─── NEWS ────────────────────────────────────────────────────────────────────

async def refresh_news():
    global current_news, last_news_time
    now = datetime.now(timezone.utc)
    if last_news_time is None or \
       (now - last_news_time).total_seconds() > NEWS_REFRESH_SECONDS:
        try:
            current_news   = await async_get_all_news()
            last_news_time = now
        except Exception as e:
            logger.warning(f"News refresh failed: {e}")


# ─── CALENDAR PAUSE ──────────────────────────────────────────────────────────

_signals_paused_until: Optional[datetime] = None

async def check_calendar():
    global _signals_paused_until
    try:
        events = check_economic_calendar()
        if events:
            await send_economic_event_warning(events)
            _signals_paused_until = (
                datetime.now(timezone.utc) +
                timedelta(minutes=PAUSE_AFTER_NEWS_MINUTES)
            )
    except Exception as e:
        logger.debug(f"Calendar error: {e}")

def signals_paused() -> bool:
    return (_signals_paused_until is not None and
            datetime.now(timezone.utc) < _signals_paused_until)


# ─── ASSET SCANNER ───────────────────────────────────────────────────────────

async def scan_asset(asset: str) -> Optional[dict]:
    tf_data = {}
    loop    = asyncio.get_event_loop()
    for tf_key in TIMEFRAMES:
        df = await loop.run_in_executor(None, get_ohlcv, asset, tf_key)
        tf_data[tf_key] = df
    news_sent, news_head = get_asset_news_sentiment(asset, current_news)
    signal = generate_signal(
        symbol=asset, tf_data=tf_data,
        news_sentiment=news_sent, news_headline=news_head,
        account_balance=ACCOUNT_BALANCE, risk_pct=RISK_PERCENT
    )
    if signal:
        signal["_account_balance"] = ACCOUNT_BALANCE
    return signal


# ─── MAIN SCAN LOOP ──────────────────────────────────────────────────────────

async def scan_all_assets():
    if signals_paused():
        logger.info("⏸️  Signals paused — news window active.")
        return

    logger.info(f"🔍 Scanning {len(ALL_ASSETS)} assets...")
    bull_assets, bear_assets = [], []
    sent_count = failed = 0

    for asset in ALL_ASSETS:
        try:
            signal = await scan_asset(asset)
            if not signal:
                continue

            direction  = signal["direction"]
            confidence = signal["confidence"]

            if is_on_cooldown(asset, direction) and confidence < 8.5:
                continue

            log_signal(signal)
            mark_cooldown(asset, direction)

            # Calculate lot size suggestions
            lots = suggest_lot_sizes(
                asset,
                signal["entry"],
                signal["stop_loss"],
                ACCOUNT_BALANCE
            )

            # Send to Telegram with approval buttons
            signal_id = generate_signal_id()
            msg_id    = await send_signal_for_approval(signal, signal_id, lots)

            if msg_id:
                add_pending_signal(signal_id, signal, lots, msg_id)
                sent_count += 1
                logger.info(
                    f"📤 Signal pending approval: {asset} {direction} | "
                    f"Conf: {confidence}/10 | "
                    f"Suggested lot: {lots['recommended']}"
                )

            if direction == "LONG":
                bull_assets.append(asset)
            else:
                bear_assets.append(asset)

        except Exception as e:
            failed += 1
            logger.debug(f"Scan error {asset}: {e}")

    if len(bull_assets) >= 5:
        await send_market_bias_alert("LONG", bull_assets)
    elif len(bear_assets) >= 5:
        await send_market_bias_alert("SHORT", bear_assets)

    if failed >= 3:
        await send_api_failure_alert([f"{failed} assets failed"])

    logger.info(f"🏁 Scan done. {sent_count} signal(s) sent for approval | {failed} failed")


# ─── DAILY SUMMARY ───────────────────────────────────────────────────────────

async def maybe_daily_summary():
    global last_daily_report_date
    now   = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    if now.hour >= 22 and last_daily_report_date != today:
        stats = compute_daily_stats()
        if is_connected():
            stats["mt5_balance"] = get_account_balance()
            stats["mt5_pnl"]     = get_daily_pnl()
        await send_daily_summary(stats)
        last_daily_report_date = today


# ─── MAIN ────────────────────────────────────────────────────────────────────

async def main():
    logger.info("=" * 60)
    logger.info("🤖 Trading Signal Bot Starting Up...")
    logger.info("=" * 60)

    # Register approval callback
    register_approve_callback(on_trade_approved)

    # Connect MT5
    if MT5_AUTO_TRADE:
        logger.info("🔌 Connecting to MetaTrader 5...")
        ok = connect_mt5(MT5_LOGIN or None, MT5_PASSWORD or None, MT5_SERVER or None)
        if ok:
            bal = get_account_balance()
            logger.info(f"✅ MT5 connected | Balance: ${bal:.2f}")
        else:
            logger.warning("⚠️ MT5 not connected — will retry on each trade approval.")

    # Backtests
    try:
        run_all_backtests()
    except Exception as e:
        logger.error(f"Backtester error: {e}")

    # Startup Telegram message
    try:
        await send_startup_message(len(ALL_ASSETS), len(TIMEFRAMES))
    except Exception as e:
        logger.error(f"Startup message failed: {e}")

    await refresh_news()

    logger.info(f"🚀 Bot live! Scanning every {SCAN_INTERVAL_SECONDS}s.")
    logger.info(f"👆 Waiting for your approvals in Telegram.")

    # Start approval listener as background task
    listener_task = asyncio.create_task(run_approval_listener())

    loop_count = 0
    try:
        while True:
            loop_count += 1
            logger.info(
                f"\n{'='*40}\n"
                f"🔄 Loop #{loop_count} — "
                f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )
            try:
                await refresh_news()
                if loop_count % 3 == 0:
                    await check_calendar()
                await scan_all_assets()
                await maybe_daily_summary()
            except Exception as e:
                logger.error(f"Loop error: {e}\n{traceback.format_exc()[:400]}")
                try:
                    await send_alert(f"Bot error: {str(e)[:200]}", "🔴")
                except Exception:
                    pass

            logger.info(f"⏰ Next scan in {SCAN_INTERVAL_SECONDS}s...")
            await asyncio.sleep(SCAN_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("🛑 Stopped by user.")
        await send_alert("🛑 Trading Bot stopped.")
    finally:
        listener_task.cancel()
        if MT5_AUTO_TRADE:
            disconnect_mt5()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped.")
        sys.exit(0)