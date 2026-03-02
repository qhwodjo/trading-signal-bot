"""
news_engine.py - Live financial news fetching + NLP sentiment analysis.
Uses RSS feeds and NewsAPI. Sentiment via FinBERT (transformers) or VADER fallback.
"""

import re
import time
import asyncio
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import feedparser

from config import NEWS_API_KEY
from logger import logger

# ─── RSS FEEDS ────────────────────────────────────────────────────────────────

RSS_FEEDS = {
    "reuters":      "https://feeds.reuters.com/reuters/businessNews",
    "marketwatch":  "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "cointelegraph": "https://cointelegraph.com/rss",
    "investing":    "https://www.investing.com/rss/news.rss",
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
}

# ─── SENTIMENT MODEL ──────────────────────────────────────────────────────────

_finbert_pipe = None
_vader        = None
_model_loaded = False

def _load_sentiment_model():
    """Load FinBERT or fall back to VADER."""
    global _finbert_pipe, _vader, _model_loaded
    if _model_loaded:
        return
    try:
        from transformers import pipeline
        logger.info("Loading FinBERT sentiment model (first run may take a moment)...")
        _finbert_pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,          # CPU
            max_length=512,
            truncation=True
        )
        logger.info("FinBERT loaded successfully.")
    except Exception as e:
        logger.warning(f"FinBERT unavailable ({e}), falling back to VADER.")
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            import nltk
            nltk.download("vader_lexicon", quiet=True)
            _vader = SentimentIntensityAnalyzer()
        except Exception as e2:
            logger.warning(f"VADER also unavailable: {e2}")
    _model_loaded = True


def analyze_sentiment(text: str) -> Tuple[str, float]:
    """
    Return (sentiment_label, score).
    sentiment_label: 'Bullish', 'Bearish', or 'Neutral'
    score: -1.0 to 1.0
    """
    _load_sentiment_model()
    if not text:
        return "Neutral", 0.0

    text = text[:500]

    # FinBERT
    if _finbert_pipe:
        try:
            result = _finbert_pipe(text)[0]
            label = result["label"].lower()
            score = result["score"]
            if label == "positive":
                return "Bullish", score
            elif label == "negative":
                return "Bearish", -score
            return "Neutral", 0.0
        except Exception as e:
            logger.debug(f"FinBERT inference error: {e}")

    # VADER fallback
    if _vader:
        try:
            scores = _vader.polarity_scores(text)
            compound = scores["compound"]
            if compound >= 0.05:
                return "Bullish", compound
            elif compound <= -0.05:
                return "Bearish", compound
            return "Neutral", compound
        except Exception:
            pass

    # Simple keyword fallback
    text_lower = text.lower()
    bull_words = ["rally", "surge", "gain", "rise", "bullish", "breakout", "recovery", "growth", "up"]
    bear_words = ["fall", "drop", "crash", "decline", "bearish", "sell-off", "loss", "down", "recession"]
    b = sum(1 for w in bull_words if w in text_lower)
    s = sum(1 for w in bear_words if w in text_lower)
    if b > s:
        return "Bullish", 0.3
    elif s > b:
        return "Bearish", -0.3
    return "Neutral", 0.0


# ─── RSS FETCHER ──────────────────────────────────────────────────────────────

def fetch_rss_news(feed_url: str, max_items: int = 10) -> List[Dict]:
    """Fetch and parse RSS feed."""
    try:
        feed = feedparser.parse(feed_url)
        items = []
        for entry in feed.entries[:max_items]:
            title   = entry.get("title", "")
            summary = entry.get("summary", entry.get("description", ""))
            link    = entry.get("link", "")
            pub     = entry.get("published", str(datetime.now(timezone.utc)))
            items.append({
                "title":   title,
                "summary": summary,
                "link":    link,
                "source":  feed_url,
                "published": pub
            })
        return items
    except Exception as e:
        logger.warning(f"RSS fetch failed for {feed_url}: {e}")
        return []


def fetch_newsapi(query: str = "stock market finance") -> List[Dict]:
    """Fetch from NewsAPI free tier."""
    if not NEWS_API_KEY:
        return []
    try:
        import requests
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 20
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("status") != "ok":
            return []
        items = []
        for art in data.get("articles", []):
            items.append({
                "title":   art.get("title", ""),
                "summary": art.get("description", ""),
                "link":    art.get("url", ""),
                "source":  art.get("source", {}).get("name", "newsapi"),
                "published": art.get("publishedAt", "")
            })
        return items
    except Exception as e:
        logger.warning(f"NewsAPI fetch failed: {e}")
        return []


# ─── NEWS CACHE ───────────────────────────────────────────────────────────────

_news_cache: Dict[str, List[Dict]] = {}
_last_fetch: Optional[datetime] = None
_CACHE_TTL_SECONDS = 900  # 15 min


def get_all_news(force: bool = False) -> List[Dict]:
    """Fetch all news sources and return combined, deduped list."""
    global _news_cache, _last_fetch
    now = datetime.now(timezone.utc)

    if not force and _last_fetch and (now - _last_fetch).total_seconds() < _CACHE_TTL_SECONDS:
        return [item for items in _news_cache.values() for item in items]

    all_articles = []
    failed_sources = 0

    for name, url in RSS_FEEDS.items():
        items = fetch_rss_news(url)
        if items:
            _news_cache[name] = items
            all_articles.extend(items)
        else:
            failed_sources += 1

    # NewsAPI
    na_items = fetch_newsapi("stock market crypto forex gold oil")
    if na_items:
        _news_cache["newsapi"] = na_items
        all_articles.extend(na_items)
    else:
        failed_sources += 1

    _last_fetch = now

    # Deduplicate by title hash
    seen = set()
    deduped = []
    for art in all_articles:
        key = hashlib.md5(art["title"].encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            deduped.append(art)

    if failed_sources >= 3:
        logger.warning(f"⚠️ {failed_sources}+ news sources failed simultaneously.")

    logger.info(f"Fetched {len(deduped)} unique news articles from {len(RSS_FEEDS)+1} sources.")
    return deduped


# ─── ASSET-SPECIFIC NEWS LOOKUP ───────────────────────────────────────────────

ASSET_KEYWORDS = {
    "BTC-USD":  ["bitcoin", "btc", "crypto"],
    "ETH-USD":  ["ethereum", "eth", "ether"],
    "GC=F":     ["gold", "xau", "precious metals"],
    "CL=F":     ["oil", "crude", "wti", "petroleum"],
    "EURUSD=X": ["euro", "eur", "ecb"],
    "GBPUSD=X": ["pound", "gbp", "boe", "bank of england"],
    "USDJPY=X": ["yen", "jpy", "bank of japan", "boj"],
    "SPY":      ["s&p 500", "sp500", "spy", "stock market"],
    "QQQ":      ["nasdaq", "tech stocks", "qqq"],
}

def get_asset_news_sentiment(symbol: str, all_news: List[Dict]) -> Tuple[str, str]:
    """
    Find relevant news for an asset and return (sentiment, headline).
    """
    # Build keyword list
    keywords = ASSET_KEYWORDS.get(symbol, [symbol.replace("=X", "").replace("-USD", "").lower()])

    relevant = []
    for art in all_news:
        text = f"{art['title']} {art['summary']}".lower()
        if any(kw in text for kw in keywords):
            relevant.append(art)

    if not relevant:
        return "Neutral", ""

    # Analyze sentiment of top 3 relevant articles
    sentiments = []
    for art in relevant[:3]:
        text = f"{art['title']}. {art['summary']}"
        sentiment, score = analyze_sentiment(text)
        sentiments.append((sentiment, score, art["title"]))

    if not sentiments:
        return "Neutral", ""

    # Average score
    avg_score = sum(s for _, s, _ in sentiments) / len(sentiments)
    top_headline = sentiments[0][2]

    if avg_score >= 0.05:
        return "Bullish", top_headline
    elif avg_score <= -0.05:
        return "Bearish", top_headline
    return "Neutral", top_headline


# ─── ECONOMIC CALENDAR ────────────────────────────────────────────────────────

def check_economic_calendar() -> List[Dict]:
    """
    Scrape ForexFactory (or Investing.com) for upcoming high-impact events.
    Returns list of events in the next 2 hours.
    """
    events = []
    try:
        import requests
        from bs4 import BeautifulSoup
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        r = requests.get(url, timeout=10)
        data = r.json()
        now = datetime.now(timezone.utc)
        window = now + timedelta(hours=2)
        for item in data:
            if item.get("impact", "").lower() in ["high", "medium"]:
                try:
                    t = datetime.fromisoformat(item.get("date", ""))
                    if now <= t <= window:
                        events.append({
                            "title": item.get("title", ""),
                            "country": item.get("country", ""),
                            "impact": item.get("impact", ""),
                            "time": str(t)
                        })
                except Exception:
                    pass
    except Exception as e:
        logger.debug(f"Economic calendar fetch error: {e}")
    return events


async def async_get_all_news(force: bool = False) -> List[Dict]:
    """Async wrapper for get_all_news."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_all_news, force)
