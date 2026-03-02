# 🤖 AlgoTrader — Multi-Asset Trading Signal Bot

A production-grade algorithmic trading system built in Python. Scans 40+ assets across 6 timeframes, generates high-confidence trading signals using 30+ technical indicators, NLP news sentiment, and ICT price action concepts — then delivers them to Telegram with MT5 auto-execution support.

> Built and maintained by a self-taught developer in Ghana 🇬🇭

---

## 🎯 Live Results

| Metric | Value |
|--------|-------|
| Assets scanned | 40 (Forex, Crypto, Gold, Commodities) |
| Timeframes | 6 (1m, 5m, 15m, 1h, 4h, 1d) |
| Profitable symbols | XAUUSD, BTCUSD, XRPUSD, ETHUSD |
| Best signal confidence | 8.7/10 (ETH-USD LONG) |
| Prop challenge | VPropTrader $1,000 — in progress |

---

## 🏗️ Architecture

```
main.py                  — Async event loop, scan scheduler
├── data_fetcher.py      — Multi-source OHLCV (yfinance → ccxt → Twelvedata)
├── indicators.py        — 30+ technical indicators (trend, momentum, volatility, volume)
├── patterns.py          — Candlestick, chart patterns, ICT concepts (FVG, OB, BOS, CHoCH)
├── signal_engine.py     — Multi-TF weighted scoring, per-asset-class thresholds
├── news_engine.py       — RSS + NewsAPI aggregation, FinBERT/VADER sentiment
├── telegram_bot.py      — Signal delivery with inline keyboard approval buttons
├── approval_handler.py  — Long-poll listener for accept/decline/custom-lot responses
├── mt5_trader.py        — MetaTrader 5 auto-execution with lot size calculation
├── backtester.py        — 5 strategies × 5 assets × 6 months on startup
├── performance_tracker.py — 24h outcome polling (TP1/TP2/TP3/SL hit tracking)
└── logger.py            — Structured logging + signals_log.csv + performance_log.csv
```

---

## ⚙️ Technical Stack

- **Language:** Python 3.12 async
- **Data:** yfinance, ccxt (Binance), Twelvedata API, AlphaVantage
- **NLP/ML:** HuggingFace Transformers — ProsusAI/FinBERT for financial sentiment
- **Technical Analysis:** pandas-ta, custom indicator implementations
- **Execution:** MetaTrader5 Python API
- **Messaging:** Telegram Bot API with inline keyboards
- **Concurrency:** asyncio, ThreadPoolExecutor

---

## 📊 Signal Engine

Every signal goes through a 4-stage pipeline:

### 1. Multi-Timeframe Data Fetch
Fetches OHLCV for all assets across all timeframes concurrently with automatic fallback chain.
4h candles are resampled from 1h data (yfinance has no native 4h interval).

### 2. Indicator Scoring
30+ indicators computed per timeframe, each returning `{direction: 1/-1/0, strength: 0-1}`.
Scores are weighted by:
- Indicator weight (e.g. Ichimoku 1.6×, MACD 1.4×)
- Timeframe weight (1d=1.6×, 4h=1.3×, 1h=1.0×)
- Asset class (volume indicators near-zero for forex — no centralised exchange)

### 3. Normalized Confidence Score
```
confidence = (dominant_score - opposing_score) / total_score × 10
```
This formula is symmetric — LONG and SHORT compete on equal footing.
Prevents the long bias common in most indicator-based systems.

### 4. News Sentiment Adjustment
FinBERT classifies headlines as Bullish/Bearish/Neutral per asset.
- Confirms signal direction: +0.5 confidence
- Conflicts with signal: -2.0 confidence (strong override)

---

## 📱 Telegram Approval Flow

```
Signal fires
    ↓
Telegram message with full signal details + lot size suggestions
    ↓
    ┌─────────────────────────────────────┐
    │  ✅ Accept (0.05 lots)              │
    │  🟢 0.03 lots   🔴 0.10 lots       │
    │  ✏️ Custom lot   ❌ Decline         │
    └─────────────────────────────────────┘
    ↓
User taps button (or types custom lot)
    ↓
MT5 order placed with exact entry, TP1, SL
    ↓
Confirmation sent to Telegram
```

Signals auto-expire after configurable timeout (default 5 min).

---

## 💡 Key Features

- **Per-asset-class confidence thresholds** — Forex 7.0, Crypto 8.0, Gold 8.0
- **Symmetric long/short scoring** — fixes the long bias in standard implementations
- **Real 4h candles** via resampling — not just 1h data mislabelled as 4h
- **ICT concepts** — Fair Value Gaps, Order Blocks, BOS, CHoCH, Liquidity Sweeps
- **Daily loss circuit breaker** — stops trading at configurable daily loss limit
- **Economic calendar** — pauses signals ±30min around NFP/CPI/FOMC/GDP
- **Performance tracking** — polls price for 24h per signal, logs TP/SL outcomes
- **Backtesting** — runs on startup, 5 strategies × 5 assets × 6 months

---

## 🚀 Setup

```bash
git clone https://github.com/qhwodjo/trading-signal-bot
cd trading-signal-bot
pip install -r requirements.txt
```

Edit `config.py`:
```python
TELEGRAM_TOKEN  = "your_token"
CHAT_ID         = "your_chat_id"
ACCOUNT_BALANCE = 1000.0
MT5_AUTO_TRADE  = True   # requires MetaTrader5 desktop on Windows
```

Run:
```bash
python main.py
```

---

## 📁 Output Files

| File | Contents |
|------|----------|
| `signals_log.csv` | Every signal: asset, direction, entry, TP/SL, confidence, patterns, sentiment |
| `performance_log.csv` | Outcomes: TP1/TP2/TP3/SL hit, exit price, P&L % |
| `bot.log` | Full application log |

---

## ⚠️ Disclaimer

This project is for educational purposes. Trading involves significant risk. Past signal performance does not guarantee future results. Always use proper risk management.

---

## 👤 Author

**Sodji John** — IT Management student, Ghana 🇬🇭
- GitHub: [@qhwodjo](https://github.com/qhwodjo)
- Email: sodjijohn1@gmail.com
- Built entirely on a broken-screen Lenovo Flex 5

*Open to remote Python/fintech developer roles and collaboration. Feel free to reach out.*