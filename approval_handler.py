"""
approval_handler.py - Listens for Telegram button presses and text replies.

Flow:
    1. Bot sends signal with ✅ Accept / ✏️ Custom / ❌ Decline buttons
    2. This module polls for updates in the background
    3. When a button is pressed:
       - Accept → executes trade immediately with chosen lot
       - Custom → asks user to type a lot size, waits for their reply
       - Decline → cancels, updates message
    4. Pending signals expire after APPROVAL_TIMEOUT_SECONDS
"""

import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Callable

from config import APPROVAL_TIMEOUT_SECONDS, CHAT_ID, MT5_RISK_PERCENT
from telegram_bot import (
    get_updates, answer_callback_query,
    edit_message, send_message
)
from logger import logger


# ─── PENDING SIGNAL STORE ────────────────────────────────────────────────────
# signal_id → {signal, lots, message_id, created_at, future, awaiting_custom_lot}

_pending: Dict[str, dict] = {}

# When bot is waiting for a custom lot reply from user
_awaiting_custom: Optional[str] = None   # signal_id currently waiting for custom lot

# Callback registered by main.py — called when a signal is approved
# Signature: async def on_approve(signal: dict, lot: float) -> None
_on_approve_cb: Optional[Callable] = None

# Update offset for long-polling
_update_offset = 0


def register_approve_callback(cb: Callable):
    """Register the function to call when user approves a trade."""
    global _on_approve_cb
    _on_approve_cb = cb


def generate_signal_id() -> str:
    return str(uuid.uuid4())[:8]


def add_pending_signal(signal_id: str, signal: dict,
                       lots: dict, message_id: int):
    """Register a signal as pending approval."""
    _pending[signal_id] = {
        "signal":     signal,
        "lots":       lots,
        "message_id": message_id,
        "created_at": datetime.now(timezone.utc),
        "awaiting_custom": False,
    }
    logger.info(f"⏳ Signal {signal_id} ({signal['asset']} {signal['direction']}) pending approval")


def _remove_pending(signal_id: str):
    _pending.pop(signal_id, None)


# ─── UPDATE PROCESSOR ─────────────────────────────────────────────────────────

async def _handle_callback_query(update: dict):
    """Process a button press."""
    global _awaiting_custom

    cq         = update["callback_query"]
    cq_id      = cq["id"]
    data       = cq.get("data", "")
    chat_id    = str(cq["message"]["chat"]["id"])
    message_id = cq["message"]["message_id"]

    # Only handle callbacks from our chat
    if chat_id != str(CHAT_ID).lstrip("-") and chat_id != CHAT_ID:
        # Group chat IDs start with -, compare both ways
        if chat_id.lstrip("-") != str(CHAT_ID).lstrip("-"):
            return

    parts = data.split(":")
    if len(parts) < 2:
        return

    action    = parts[0]
    signal_id = parts[1]
    pending   = _pending.get(signal_id)

    if pending is None:
        await answer_callback_query(cq_id, "⚠️ Signal expired or already processed.")
        return

    signal = pending["signal"]
    asset  = signal["asset"]
    direc  = signal["direction"]

    # ── ACCEPT ────────────────────────────────────────────────────────────────
    if action == "accept":
        lot = float(parts[2]) if len(parts) > 2 else pending["lots"]["recommended"]
        await answer_callback_query(cq_id, f"✅ Placing {direc} {asset} @ {lot} lots...")
        await edit_message(
            chat_id, message_id,
            f"✅ <b>ACCEPTED</b> — {asset} {direc}\n"
            f"📦 Lot: <b>{lot}</b>\n"
            f"🔄 Placing order on MT5..."
        )
        _remove_pending(signal_id)
        _awaiting_custom = None

        if _on_approve_cb:
            try:
                await _on_approve_cb(signal, lot)
            except Exception as e:
                logger.error(f"Approve callback error: {e}")
                await send_message(f"❌ Trade execution error: {e}")

    # ── CUSTOM LOT ────────────────────────────────────────────────────────────
    elif action == "custom":
        _awaiting_custom = signal_id
        pending["awaiting_custom"] = True
        await answer_callback_query(cq_id, "✏️ Type your lot size...")
        await edit_message(
            chat_id, message_id,
            f"✏️ <b>Custom lot size</b> — {asset} {direc}\n\n"
            f"💬 <b>Reply with your desired lot size</b>\n"
            f"(e.g. <code>0.05</code> or <code>0.1</code>)\n\n"
            f"Suggested lots:\n"
            f"🟢 Conservative: {pending['lots']['conservative']}\n"
            f"🔵 Recommended:  {pending['lots']['recommended']}\n"
            f"🔴 Aggressive:   {pending['lots']['aggressive']}"
        )

    # ── DECLINE ───────────────────────────────────────────────────────────────
    elif action == "decline":
        await answer_callback_query(cq_id, "❌ Signal declined.")
        await edit_message(
            chat_id, message_id,
            f"❌ <b>DECLINED</b> — {asset} {direc}\n"
            f"Signal skipped."
        )
        _remove_pending(signal_id)
        _awaiting_custom = None
        logger.info(f"❌ Signal {signal_id} ({asset} {direc}) declined by user.")


async def _handle_message(update: dict):
    """Process a text message — used for custom lot size input."""
    global _awaiting_custom

    msg     = update.get("message", {})
    chat_id = str(msg.get("chat", {}).get("id", ""))
    text    = msg.get("text", "").strip()

    # Only handle messages from our chat
    if chat_id.lstrip("-") != str(CHAT_ID).lstrip("-"):
        return

    # Not waiting for custom lot
    if _awaiting_custom is None:
        return

    signal_id = _awaiting_custom
    pending   = _pending.get(signal_id)
    if pending is None or not pending.get("awaiting_custom"):
        _awaiting_custom = None
        return

    # Try to parse lot size
    try:
        lot = float(text.replace(",", "."))
        if lot <= 0:
            raise ValueError("Lot must be > 0")
    except ValueError:
        await send_message(
            f"⚠️ Invalid lot size: <code>{text}</code>\n"
            f"Please enter a number like <code>0.05</code>"
        )
        return

    signal = pending["signal"]
    asset  = signal["asset"]
    direc  = signal["direction"]

    # Confirm the trade
    await send_message(
        f"✅ <b>Custom lot accepted</b>\n"
        f"📌 {asset} {direc} @ <b>{lot} lots</b>\n"
        f"🔄 Placing order on MT5..."
    )

    _awaiting_custom = None
    _remove_pending(signal_id)

    if _on_approve_cb:
        try:
            await _on_approve_cb(signal, lot)
        except Exception as e:
            logger.error(f"Approve callback error: {e}")
            await send_message(f"❌ Trade execution error: {e}")


# ─── EXPIRY CHECKER ───────────────────────────────────────────────────────────

async def _expire_pending():
    """Mark signals as expired if timeout exceeded."""
    global _awaiting_custom
    if APPROVAL_TIMEOUT_SECONDS <= 0:
        return

    now     = datetime.now(timezone.utc)
    expired = []
    for sid, p in _pending.items():
        age = (now - p["created_at"]).total_seconds()
        if age > APPROVAL_TIMEOUT_SECONDS:
            expired.append(sid)

    for sid in expired:
        p     = _pending.pop(sid)
        sig   = p["signal"]
        try:
            await edit_message(
                CHAT_ID, p["message_id"],
                f"⏰ <b>EXPIRED</b> — {sig['asset']} {sig['direction']}\n"
                f"No response within {APPROVAL_TIMEOUT_SECONDS}s. Signal skipped."
            )
        except Exception:
            pass
        if _awaiting_custom == sid:
            _awaiting_custom = None
        logger.info(f"⏰ Signal {sid} ({sig['asset']}) expired.")


# ─── MAIN POLLING LOOP ────────────────────────────────────────────────────────

async def run_approval_listener():
    """
    Background task — long-polls Telegram for button presses and messages.
    Run this as an asyncio task alongside the main scanning loop.
    """
    global _update_offset
    logger.info("🎧 Approval listener started.")

    while True:
        try:
            updates = await get_updates(offset=_update_offset, timeout=10)
            for update in updates:
                _update_offset = update["update_id"] + 1
                if "callback_query" in update:
                    await _handle_callback_query(update)
                elif "message" in update:
                    await _handle_message(update)

            await _expire_pending()

        except asyncio.CancelledError:
            logger.info("Approval listener stopped.")
            break
        except Exception as e:
            logger.debug(f"Approval listener error: {e}")
            await asyncio.sleep(2)

        await asyncio.sleep(0.5)