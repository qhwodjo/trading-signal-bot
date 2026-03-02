"""
telegram_bot.py - Telegram messaging with inline keyboard approval buttons.
Handles signal delivery AND the accept/decline/custom-lot approval flow.
Uses ThreadedResolver to fix Windows DNS issues.
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timezone
from typing import Optional, Dict, List

from config import (
    TELEGRAM_TOKEN, CHAT_ID,
    WEBHOOK_ENABLED, WEBHOOK_URL
)
from logger import logger

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


# ─── WINDOWS-COMPATIBLE CONNECTOR ────────────────────────────────────────────

def _make_connector() -> aiohttp.TCPConnector:
    return aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())


# ─── CORE SENDERS ─────────────────────────────────────────────────────────────

async def send_message(text: str, chat_id: str = CHAT_ID,
                       parse_mode: str = "HTML",
                       reply_markup: dict = None) -> Optional[int]:
    """Send a message. Returns message_id on success, None on failure."""
    url     = f"{BASE_URL}/sendMessage"
    payload = {
        "chat_id":                  chat_id,
        "text":                     text,
        "parse_mode":               parse_mode,
        "disable_web_page_preview": True,
    }
    if reply_markup:
        payload["reply_markup"] = json.dumps(reply_markup)

    try:
        async with aiohttp.ClientSession(connector=_make_connector()) as session:
            async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                data = await resp.json()
                if not data.get("ok"):
                    logger.error(f"Telegram error: {data.get('description')}")
                    return None
                return data["result"]["message_id"]
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return None


async def edit_message(chat_id: str, message_id: int,
                       text: str, parse_mode: str = "HTML") -> bool:
    """Edit an existing message (used to update approval messages)."""
    url     = f"{BASE_URL}/editMessageText"
    payload = {
        "chat_id":                  chat_id,
        "message_id":               message_id,
        "text":                     text,
        "parse_mode":               parse_mode,
        "disable_web_page_preview": True,
    }
    try:
        async with aiohttp.ClientSession(connector=_make_connector()) as session:
            async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                data = await resp.json()
                return data.get("ok", False)
    except Exception as e:
        logger.error(f"Edit message failed: {e}")
        return False


async def answer_callback_query(callback_query_id: str,
                                text: str = "") -> bool:
    """Acknowledge a button press (removes the loading spinner)."""
    url = f"{BASE_URL}/answerCallbackQuery"
    try:
        async with aiohttp.ClientSession(connector=_make_connector()) as session:
            async with session.post(
                url,
                json={"callback_query_id": callback_query_id, "text": text},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                data = await resp.json()
                return data.get("ok", False)
    except Exception as e:
        logger.error(f"Answer callback failed: {e}")
        return False


async def get_updates(offset: int = 0, timeout: int = 30) -> list:
    """Long-poll for Telegram updates (button presses and messages)."""
    url = f"{BASE_URL}/getUpdates"
    try:
        async with aiohttp.ClientSession(connector=_make_connector()) as session:
            async with session.post(
                url,
                json={"offset": offset, "timeout": timeout, "allowed_updates": ["callback_query", "message"]},
                timeout=aiohttp.ClientTimeout(total=timeout + 5)
            ) as resp:
                data = await resp.json()
                return data.get("result", []) if data.get("ok") else []
    except Exception:
        return []


# ─── SIGNAL FORMATTER ─────────────────────────────────────────────────────────

def format_signal_message(signal: dict, lots: dict = None) -> str:
    """Format a trading signal with lot size suggestions."""
    direction = signal.get("direction", "LONG")
    icon      = "📈" if direction == "LONG" else "📉"
    news_sent = signal.get("news_sentiment", "Neutral")
    news_head = signal.get("news_headline", "")
    news_line = f"{news_sent} — {news_head[:80]}" if news_head else news_sent
    warnings  = signal.get("warnings", "None")

    msg = (
        f"🚨 <b>TRADING SIGNAL</b> 🚨\n\n"
        f"📌 <b>Asset:</b> {signal.get('asset', 'N/A')}\n"
        f"{icon} <b>Direction:</b> {direction}\n"
        f"⏱ <b>Timeframe:</b> {signal.get('timeframe', 'N/A')}\n"
        f"🔰 <b>Signal Type:</b> {signal.get('signal_type', 'N/A')}\n\n"
        f"✅ <b>Entry:</b> {signal.get('entry', 'N/A')}\n"
        f"🎯 <b>TP1:</b> {signal.get('tp1', 'N/A')} | "
        f"<b>TP2:</b> {signal.get('tp2', 'N/A')} | "
        f"<b>TP3:</b> {signal.get('tp3', 'N/A')}\n"
        f"🛑 <b>Stop Loss:</b> {signal.get('stop_loss', 'N/A')}\n"
        f"⚖️ <b>Risk/Reward:</b> 1:{signal.get('risk_reward', 'N/A')}\n\n"
        f"📊 <b>Confidence:</b> {signal.get('confidence', 0)}/10\n"
        f"🔢 <b>Confluences:</b> {signal.get('confluences', 0)} factors\n\n"
        f"📐 <b>Patterns:</b> {signal.get('patterns', 'N/A')}\n\n"
        f"📰 <b>News:</b> {news_line}\n\n"
        f"⚠️ <b>Notes:</b> {warnings}\n\n"
        f"🕐 <b>Time:</b> {signal.get('timestamp', 'N/A')}"
    )

    # Append lot size suggestions if provided
    if lots:
        msg += (
            f"\n\n{'─'*30}\n"
            f"💼 <b>Suggested Lot Sizes</b>\n"
            f"(Based on ${signal.get('_account_balance', 1000):.0f} balance)\n\n"
            f"🟢 Conservative (0.5% risk): <b>{lots['conservative']} lots</b>\n"
            f"🔵 Recommended  (1.0% risk): <b>{lots['recommended']} lots</b>\n"
            f"🔴 Aggressive   (2.0% risk): <b>{lots['aggressive']} lots</b>\n"
            f"\n👇 <b>Choose an action below:</b>"
        )

    return msg


def build_approval_keyboard(signal_id: str, lots: dict) -> dict:
    """Build the inline keyboard with Accept / Custom Lot / Decline buttons."""
    rec_lot = lots["recommended"]
    return {
        "inline_keyboard": [
            [
                {
                    "text": f"✅ Accept ({rec_lot} lots)",
                    "callback_data": f"accept:{signal_id}:{rec_lot}"
                },
            ],
            [
                {
                    "text": f"🟢 {lots['conservative']} lots (conservative)",
                    "callback_data": f"accept:{signal_id}:{lots['conservative']}"
                },
                {
                    "text": f"🔴 {lots['aggressive']} lots (aggressive)",
                    "callback_data": f"accept:{signal_id}:{lots['aggressive']}"
                },
            ],
            [
                {
                    "text": "✏️ Custom lot size",
                    "callback_data": f"custom:{signal_id}"
                },
                {
                    "text": "❌ Decline",
                    "callback_data": f"decline:{signal_id}"
                },
            ]
        ]
    }


# ─── SIGNAL SENDER (with approval buttons) ───────────────────────────────────

async def send_signal_for_approval(signal: dict, signal_id: str,
                                    lots: dict) -> Optional[int]:
    """
    Send a signal message with Accept/Decline/Custom lot buttons.
    Returns the message_id so we can edit it later.
    """
    text     = format_signal_message(signal, lots)
    keyboard = build_approval_keyboard(signal_id, lots)
    msg_id   = await send_message(text, reply_markup=keyboard)
    return msg_id


async def send_webhook(signal: dict) -> bool:
    if not WEBHOOK_ENABLED or not WEBHOOK_URL:
        return False
    try:
        async with aiohttp.ClientSession(connector=_make_connector()) as session:
            async with session.post(
                WEBHOOK_URL, json=signal,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                return resp.status == 200
    except Exception as e:
        logger.warning(f"Webhook failed: {e}")
        return False


# ─── GENERIC ALERT SENDERS ────────────────────────────────────────────────────

async def send_alert(message: str, emoji: str = "⚠️") -> None:
    await send_message(f"{emoji} {message}")


async def send_startup_message(asset_count: int, tf_count: int) -> None:
    from config import ACCOUNT_BALANCE, MT5_AUTO_TRADE, APPROVAL_TIMEOUT_SECONDS
    mode = "🤖 Auto-trade with approval" if MT5_AUTO_TRADE else "📡 Signal only"
    msg  = (
        f"✅ <b>Trading Bot is LIVE</b>\n\n"
        f"🔍 Scanning <b>{asset_count}</b> assets × <b>{tf_count}</b> timeframes\n"
        f"💰 Account balance: <b>${ACCOUNT_BALANCE:.2f}</b>\n"
        f"⚙️ Mode: {mode}\n"
        f"⏱ Approval timeout: {APPROVAL_TIMEOUT_SECONDS}s\n"
        f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    await send_message(msg)


async def send_daily_summary(stats: dict) -> None:
    mt5_line = ""
    if "mt5_balance" in stats:
        mt5_line = (
            f"\n💰 MT5 Balance: ${stats['mt5_balance']:.2f}"
            f" | Today P&L: ${stats.get('mt5_pnl', 0):+.2f}"
        )
    msg = (
        f"📋 <b>Daily Summary</b>\n\n"
        f"📊 Signals sent: {stats.get('total', 0)}\n"
        f"✅ TP hits: {stats.get('tp_hits', 0)}\n"
        f"🛑 SL hits: {stats.get('sl_hits', 0)}\n"
        f"⏳ Open: {stats.get('open', 0)}\n"
        f"📈 Win rate: {stats.get('win_rate', 0):.1f}%\n"
        f"🏆 Best asset: {stats.get('best_asset', 'N/A')}"
        f"{mt5_line}\n\n"
        f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    await send_message(msg)


async def send_market_bias_alert(direction: str, assets: List[str]) -> None:
    icon = "📈" if direction == "LONG" else "📉"
    await send_message(
        f"{icon} <b>Market Bias Alert!</b>\n\n"
        f"<b>{len(assets)} assets</b> aligned <b>{direction}</b>\n"
        f"Assets: {', '.join(assets[:10])}\n\n"
        f"⚠️ Consider reducing position sizes.\n"
        f"🕐 {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
    )


async def send_economic_event_warning(events: list) -> None:
    if not events:
        return
    lines = "\n".join(
        f"• {e['time']} — {e['country']} {e['title']} [{e['impact'].upper()}]"
        for e in events[:5]
    )
    await send_message(
        f"📅 <b>High-Impact Events Upcoming</b>\n\n"
        f"{lines}\n\n"
        f"⚠️ Signals paused ±30 min around events."
    )


async def send_api_failure_alert(failed_sources: List[str]) -> None:
    await send_message(
        f"🔴 <b>API Failures</b>\n"
        f"Failed: {', '.join(failed_sources)}\n"
        f"Bot continuing with available sources."
    )