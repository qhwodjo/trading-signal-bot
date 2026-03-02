"""
mt5_trader.py - MetaTrader 5 trade execution module.
Handles connection, lot size calculation, order placement and position management.

Requirements:
    pip install MetaTrader5

IMPORTANT:
    - MT5 terminal must be running on the SAME Windows machine as the bot
    - Enable: MT5 → Tools → Options → Expert Advisors → Allow automated trading
    - The green "Auto Trading" button in MT5 toolbar must be ON
"""

from datetime import datetime, timezone, date
from typing import Optional, Dict
from logger import logger

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 not installed. Run: pip install MetaTrader5")


# ─── SYMBOL MAP ───────────────────────────────────────────────────────────────
# Maps bot symbol names → your broker's exact MT5 symbol names.
# Check Market Watch in MT5 for exact names (right-click → Show All Symbols).

SYMBOL_MAP = {
    "BTC-USD":  "BTCUSD.e",
    "ETH-USD":  "ETHUSD.e",
    "XRP-USD":  "XRPUSD.e",
    "BNB-USD":  "BNBUSD.e",
    "SOL-USD":  "SOLUSD.e",
    "GC=F":     "XAUUSD.e",
    "SI=F":     "XAGUSD.e",
    "CL=F":     "XTIUSD.e",
    "BZ=F":     "XBRUSD.e",
    "EURUSD=X": "EURUSD.e",
    "GBPUSD=X": "GBPUSD.e",
    "USDJPY=X": "USDJPY.e",
    "AUDUSD=X": "AUDUSD.e",
    "USDCAD=X": "USDCAD.e",
    "USDCHF=X": "USDCHF.e",
    "NZDUSD=X": "NZDUSD.e",
    "EURGBP=X": "EURGBP.e",
    "EURJPY=X": "EURJPY.e",
    "GBPJPY=X": "GBPJPY.e",
    "AUDJPY=X": "AUDJPY.e",
    "EURAUD=X": "EURAUD.e",
    "EURCAD=X": "EURCAD.e",
    "EURCHF=X": "EURCHF.e",
    "CADJPY=X": "CADJPY.e",
    "CHFJPY=X": "CHFJPY.e",
    "NZDJPY=X": "NZDJPY.e",
}

# ─── CONNECTION ───────────────────────────────────────────────────────────────

_connected = False

def connect_mt5(login: int = None, password: str = None,
                server: str = None) -> bool:
    global _connected
    if not MT5_AVAILABLE:
        logger.error("MetaTrader5 package not installed.")
        return False
    if _connected:
        return True
    try:
        if login and password and server:
            ok = mt5.initialize(login=login, password=password, server=server)
        else:
            ok = mt5.initialize()

        if not ok:
            logger.error(f"MT5 init failed: {mt5.last_error()}")
            return False

        info = mt5.account_info()
        if info is None:
            logger.error("MT5 connected but no account found. Is MT5 logged in?")
            return False

        logger.info(
            f"✅ MT5 connected | #{info.login} | "
            f"${info.balance:.2f} {info.currency} | {info.server}"
        )
        _connected = True
        return True
    except Exception as e:
        logger.error(f"MT5 connection error: {e}")
        return False


def disconnect_mt5():
    global _connected
    if MT5_AVAILABLE and _connected:
        mt5.shutdown()
        _connected = False


def is_connected() -> bool:
    return _connected


def get_account_info() -> Optional[Dict]:
    if not _connected:
        return None
    info = mt5.account_info()
    if not info:
        return None
    return {
        "balance":  info.balance,
        "equity":   info.equity,
        "margin":   info.margin,
        "free_margin": info.margin_free,
        "currency": info.currency,
        "login":    info.login,
        "server":   info.server,
    }


def get_account_balance() -> float:
    info = get_account_info()
    return info["balance"] if info else 0.0


# ─── LOT SIZE CALCULATOR ─────────────────────────────────────────────────────

def calculate_lot_size(symbol: str, entry: float, stop_loss: float,
                       risk_pct: float, account_balance: float = None) -> float:
    """
    Calculate the correct lot size for a given risk percentage.

    Formula:
        risk_usd   = balance * (risk_pct / 100)
        lot        = risk_usd / (sl_distance_in_price * value_per_lot)

    Falls back to a safe minimum if MT5 data is unavailable.
    """
    try:
        if account_balance is None:
            account_balance = get_account_balance()
        if account_balance <= 0:
            account_balance = 1000.0   # safe fallback

        risk_usd    = account_balance * (risk_pct / 100)
        sl_distance = abs(entry - stop_loss)

        if sl_distance == 0:
            return 0.01

        if _connected:
            sym_info = mt5.symbol_info(symbol)
            if sym_info:
                tick_value = sym_info.trade_tick_value
                tick_size  = sym_info.trade_tick_size
                if tick_size > 0 and tick_value > 0:
                    value_per_lot = (tick_value / tick_size) * sl_distance
                    if value_per_lot > 0:
                        lot = risk_usd / value_per_lot
                        # Clamp to broker limits
                        lot = round(lot / sym_info.volume_step) * sym_info.volume_step
                        lot = max(sym_info.volume_min, min(sym_info.volume_max, lot))
                        return round(lot, 2)

        # Fallback calculation (no MT5 connection / symbol info)
        # Approximate: for most forex pairs 1 lot = $10/pip, SL in price units
        lot = risk_usd / (sl_distance * 100000)
        lot = max(0.01, round(lot, 2))
        return lot

    except Exception as e:
        logger.error(f"Lot size calculation error: {e}")
        return 0.01


def suggest_lot_sizes(symbol: str, entry: float, stop_loss: float,
                      account_balance: float) -> Dict[str, float]:
    """
    Return 3 lot size suggestions: conservative, recommended, aggressive.
    Based on 0.5%, 1.0%, 2.0% risk of account_balance.
    """
    from config import LOT_RISK_CONSERVATIVE, LOT_RISK_RECOMMENDED, LOT_RISK_AGGRESSIVE

    # Get MT5 symbol name
    mt5_symbol = SYMBOL_MAP.get(symbol, symbol)

    return {
        "conservative": calculate_lot_size(
            mt5_symbol, entry, stop_loss, LOT_RISK_CONSERVATIVE, account_balance
        ),
        "recommended": calculate_lot_size(
            mt5_symbol, entry, stop_loss, LOT_RISK_RECOMMENDED, account_balance
        ),
        "aggressive": calculate_lot_size(
            mt5_symbol, entry, stop_loss, LOT_RISK_AGGRESSIVE, account_balance
        ),
    }


# ─── ORDER EXECUTION ──────────────────────────────────────────────────────────

def execute_signal(signal: dict, lot: float = None,
                   risk_pct: float = 1.0) -> Optional[Dict]:
    """
    Place a market order on MT5.
    If lot is provided, uses that exact lot size.
    Otherwise calculates from risk_pct.
    Returns order result dict or None on failure.
    """
    if not MT5_AVAILABLE:
        logger.error("MT5 not available.")
        return None
    if not _connected:
        if not connect_mt5():
            return None

    bot_symbol = signal.get("asset", "")
    mt5_symbol = SYMBOL_MAP.get(bot_symbol)
    if not mt5_symbol:
        logger.warning(
            f"No MT5 symbol mapping for '{bot_symbol}'. "
            f"Add it to SYMBOL_MAP in mt5_trader.py"
        )
        return None

    # Ensure symbol visible
    if not mt5.symbol_select(mt5_symbol, True):
        logger.error(f"Cannot select {mt5_symbol} in MT5 Market Watch")
        return None

    direction  = signal.get("direction", "LONG")
    tp1        = float(signal.get("tp1", 0))
    sl         = float(signal.get("stop_loss", 0))
    order_type = mt5.ORDER_TYPE_BUY if direction == "LONG" else mt5.ORDER_TYPE_SELL

    # Live price
    tick = mt5.symbol_info_tick(mt5_symbol)
    if tick is None:
        logger.error(f"Cannot get tick for {mt5_symbol}")
        return None
    price = tick.ask if direction == "LONG" else tick.bid

    # Lot size
    if lot is None:
        lot = calculate_lot_size(
            mt5_symbol, price, sl,
            risk_pct, get_account_balance()
        )
    if lot <= 0:
        logger.error("Invalid lot size.")
        return None

    sym_info = mt5.symbol_info(mt5_symbol)
    digits   = sym_info.digits if sym_info else 5

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       mt5_symbol,
        "volume":       lot,
        "type":         order_type,
        "price":        price,
        "sl":           round(sl,  digits),
        "tp":           round(tp1, digits),
        "deviation":    20,
        "magic":        234000,
        "comment":      f"Bot {signal.get('confidence', 0)}/10",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    logger.info(
        f"📤 {direction} {mt5_symbol} | "
        f"Price:{price} TP:{tp1} SL:{sl} Lot:{lot}"
    )

    result = mt5.order_send(request)
    if result is None:
        logger.error(f"order_send returned None: {mt5.last_error()}")
        return None

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(
            f"❌ Order failed | {mt5_symbol} | "
            f"Code:{result.retcode} | {result.comment}"
        )
        return None

    logger.info(
        f"✅ Order placed | {mt5_symbol} {direction} | "
        f"Ticket:{result.order} Lot:{lot} Price:{result.price}"
    )
    return {
        "ticket":    result.order,
        "symbol":    mt5_symbol,
        "direction": direction,
        "lot":       lot,
        "price":     result.price,
        "tp":        tp1,
        "sl":        sl,
        "time":      datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


# ─── POSITION MANAGEMENT ─────────────────────────────────────────────────────

def get_open_positions(magic: int = 234000) -> list:
    if not _connected:
        return []
    pos = mt5.positions_get()
    return [p for p in pos if p.magic == magic] if pos else []


def close_position(ticket: int) -> bool:
    if not _connected:
        return False
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return False
    pos        = positions[0]
    close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
    tick       = mt5.symbol_info_tick(pos.symbol)
    price      = tick.bid if pos.type == 0 else tick.ask
    request    = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       pos.symbol,
        "volume":       pos.volume,
        "type":         close_type,
        "position":     ticket,
        "price":        price,
        "deviation":    20,
        "magic":        234000,
        "comment":      "Bot close",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE


# ─── DAILY P&L / CIRCUIT BREAKER ─────────────────────────────────────────────

def get_daily_pnl() -> float:
    if not _connected:
        return 0.0
    today = datetime.combine(date.today(), datetime.min.time())
    deals = mt5.history_deals_get(today, datetime.now())
    if not deals:
        return 0.0
    return sum(d.profit for d in deals if d.magic == 234000)


def check_daily_loss_limit(max_loss: float = 45.0) -> bool:
    """Returns True if safe to trade, False if limit breached."""
    pnl = get_daily_pnl()
    if pnl <= -max_loss:
        logger.warning(f"🛑 Daily loss limit: ${pnl:.2f} (limit -${max_loss})")
        return False
    logger.info(f"💰 Daily P&L: ${pnl:+.2f} | Buffer: ${max_loss + pnl:.2f}")
    return True