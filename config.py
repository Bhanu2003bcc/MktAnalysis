"""
config.py — Central configuration for Financial Analysis Agent Crew
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
REPORT_DIR = Path(os.getenv("REPORT_OUTPUT_DIR", BASE_DIR / "reports"))
CHART_DIR = Path(os.getenv("CHART_OUTPUT_DIR", REPORT_DIR / "charts"))

REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

# ─── Gemini ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))

# ─── Agent behaviour ──────────────────────────────────────────────────────────
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT_SECONDS", "120"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ─── News RSS feeds (free, no API key needed) ─────────────────────────────────
NEWS_RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://rss.cnn.com/rss/money_latest.rss",
]

# ─── KPI definitions ──────────────────────────────────────────────────────────
KPIS = [
    "trailingPE", "forwardPE", "priceToBook", "returnOnEquity",
    "returnOnAssets", "debtToEquity", "currentRatio", "quickRatio",
    "revenueGrowth", "earningsGrowth", "profitMargins", "grossMargins",
    "operatingMargins", "dividendYield", "beta", "marketCap",
    "enterpriseValue", "trailingEps", "forwardEps",
]

KPI_LABELS = {
    "trailingPE": "P/E Ratio (Trailing)",
    "forwardPE": "P/E Ratio (Forward)",
    "priceToBook": "Price / Book",
    "returnOnEquity": "Return on Equity",
    "returnOnAssets": "Return on Assets",
    "debtToEquity": "Debt / Equity",
    "currentRatio": "Current Ratio",
    "quickRatio": "Quick Ratio",
    "revenueGrowth": "Revenue Growth (YoY)",
    "earningsGrowth": "Earnings Growth (YoY)",
    "profitMargins": "Net Profit Margin",
    "grossMargins": "Gross Margin",
    "operatingMargins": "Operating Margin",
    "dividendYield": "Dividend Yield",
    "beta": "Beta (Volatility)",
    "marketCap": "Market Cap",
    "enterpriseValue": "Enterprise Value",
    "trailingEps": "EPS (Trailing)",
    "forwardEps": "EPS (Forward)",
}

BRAND_COLORS = {
    "primary": "#6C3EF4",
    "secondary": "#00C896",
    "accent": "#FF6B6B",
    "dark": "#1A1A2E",
    "mid": "#16213E",
    "light": "#E8F4FD",
    "text": "#2D3748",
    "muted": "#718096",
}
