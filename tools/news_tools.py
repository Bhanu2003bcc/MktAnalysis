"""
tools/news_tools.py — News scraping and NLP sentiment analysis
"""

import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Any
from loguru import logger
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()

# ── RSS feed templates ────────────────────────────────────────────────────────
RSS_FEEDS = {
    "yahoo_ticker": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    "yahoo_business": "https://finance.yahoo.com/rss/topfinstories",
    "reuters_biz": "https://feeds.reuters.com/reuters/businessNews",
    "seeking_alpha": "https://seekingalpha.com/api/sa/combined/{ticker}.xml",
    "investing_com": "https://www.investing.com/rss/news.rss",
}


def fetch_news(ticker: str, company_name: str, max_articles: int = 20) -> List[Dict]:
    """Fetch news from multiple RSS sources and scrape summaries."""
    articles = []
    seen_titles = set()

    feeds_to_try = [
        RSS_FEEDS["yahoo_ticker"].format(ticker=ticker),
        RSS_FEEDS["yahoo_business"],
        RSS_FEEDS["reuters_biz"],
        RSS_FEEDS["seeking_alpha"].format(ticker=ticker),
    ]

    for feed_url in feeds_to_try:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                title = entry.get("title", "").strip()
                if not title or title in seen_titles:
                    continue

                # Filter: keep if relevant to ticker or company
                title_lower = title.lower()
                company_lower = company_name.lower().split()[0]
                if not any(kw in title_lower for kw in [
                    ticker.lower(), company_lower, "market", "stock", "earnings",
                    "revenue", "profit", "rate", "fed", "economy"
                ]):
                    continue

                seen_titles.add(title)
                published = _parse_date(entry)
                summary = entry.get("summary", "") or entry.get("description", "")
                summary = _clean_html(summary)[:500]

                articles.append({
                    "title": title,
                    "url": entry.get("link", ""),
                    "published": published,
                    "source": feed.feed.get("title", "Unknown"),
                    "summary": summary,
                })

                if len(articles) >= max_articles:
                    break
        except Exception as e:
            logger.warning(f"Feed {feed_url} failed: {e}")
            continue

        if len(articles) >= max_articles:
            break

    # Fallback: Yahoo Finance scraping
    if len(articles) < 5:
        articles.extend(_scrape_yahoo_news(ticker, company_name))

    return articles[:max_articles]


def analyse_sentiment(articles: List[Dict]) -> Dict:
    """Run VADER sentiment on all article titles + summaries."""
    if not articles:
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "Neutral",
            "sentiment_breakdown": {"positive": 0, "negative": 0, "neutral": 0},
            "key_themes": [],
        }

    scores = []
    breakdown = {"positive": 0, "negative": 0, "neutral": 0}

    for article in articles:
        text = article["title"] + " " + article.get("summary", "")
        vs = _vader.polarity_scores(text)
        compound = vs["compound"]
        scores.append(compound)

        article["sentiment_score"] = round(compound, 3)
        article["sentiment_label"] = (
            "Positive" if compound >= 0.05 else
            "Negative" if compound <= -0.05 else "Neutral"
        )

        if compound >= 0.05:
            breakdown["positive"] += 1
        elif compound <= -0.05:
            breakdown["negative"] += 1
        else:
            breakdown["neutral"] += 1

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "sentiment_score": round(avg_score, 3),
        "sentiment_label": (
            "Bullish 🟢" if avg_score >= 0.05 else
            "Bearish 🔴" if avg_score <= -0.05 else "Neutral ⚪"
        ),
        "sentiment_breakdown": breakdown,
        "key_themes": _extract_themes(articles),
    }


def _extract_themes(articles: List[Dict]) -> List[str]:
    """Extract recurring themes/topics from headlines."""
    theme_keywords = {
        "Earnings & Revenue": ["earnings", "revenue", "profit", "eps", "quarterly", "guidance"],
        "Leadership & Strategy": ["ceo", "executive", "strategy", "plan", "acquisition", "merger"],
        "Market Performance": ["rally", "surge", "drop", "decline", "gain", "loss", "52-week"],
        "Macro / Economy": ["fed", "interest rate", "inflation", "gdp", "recession", "economic"],
        "Product / Innovation": ["launch", "product", "ai", "technology", "patent", "innovation"],
        "Analyst Ratings": ["upgrade", "downgrade", "target", "buy", "sell", "hold", "analyst"],
        "ESG / Regulatory": ["regulation", "sec", "lawsuit", "esg", "sustainability", "fine"],
        "Dividends & Buybacks": ["dividend", "buyback", "repurchase", "yield", "payout"],
    }

    all_text = " ".join(
        (a["title"] + " " + a.get("summary", "")).lower()
        for a in articles
    )

    themes = []
    for theme, keywords in theme_keywords.items():
        if any(kw in all_text for kw in keywords):
            themes.append(theme)

    return themes[:6]


def _parse_date(entry) -> str:
    for attr in ["published", "updated", "created"]:
        val = getattr(entry, attr, None)
        if val:
            return str(val)[:20]
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _clean_html(text: str) -> str:
    if not text:
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _scrape_yahoo_news(ticker: str, company_name: str) -> List[Dict]:
    """Lightweight fallback scraper for Yahoo Finance news."""
    articles = []
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/news/"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        for item in soup.select("h3.Mb\\(5px\\)")[:10]:
            title = item.get_text(strip=True)
            if title:
                articles.append({
                    "title": title,
                    "url": url,
                    "published": datetime.now().strftime("%Y-%m-%d"),
                    "source": "Yahoo Finance",
                    "summary": "",
                })
    except Exception as e:
        logger.warning(f"Yahoo news scrape failed for {ticker}: {e}")
    return articles
