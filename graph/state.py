"""
graph/state.py — Shared LangGraph state definition for all agents
"""

from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime


class AgentMessage(TypedDict):
    agent: str
    content: str
    timestamp: str
    status: str   # "running" | "done" | "error"


class StockData(TypedDict, total=False):
    ticker: str
    company_name: str
    sector: str
    industry: str
    current_price: float
    price_change: float
    price_change_pct: float
    market_cap: float
    volume: int
    avg_volume: int
    fifty_two_week_high: float
    fifty_two_week_low: float
    hist_prices: List[Dict]      # [{date, Open, High, Low, Close, Volume}]
    kpis: Dict[str, Any]
    financials: Dict[str, Any]   # income / balance / cashflow summary


class NewsData(TypedDict, total=False):
    articles: List[Dict]         # [{title, url, published, source, summary}]
    sentiment_score: float       # -1 to +1
    sentiment_label: str         # Bearish / Neutral / Bullish
    sentiment_breakdown: Dict    # {positive, negative, neutral counts}
    key_themes: List[str]


class AnalysisData(TypedDict, total=False):
    kpi_table: List[Dict]        # [{metric, value, benchmark, signal}]
    chart_paths: List[str]       # local file paths
    price_trend: str             # Uptrend / Downtrend / Sideways
    volatility_level: str
    technical_signals: Dict
    fundamental_score: float     # 0-100
    analyst_summary: str


class ReportData(TypedDict, total=False):
    executive_summary: str
    market_research_section: str
    data_analysis_section: str
    recommendation: str          # Buy / Hold / Sell
    confidence_score: float      # 0-100
    risk_factors: List[str]
    catalysts: List[str]
    target_price: Optional[float]
    pdf_path: str


class FinancialAgentState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────
    ticker: str
    company_name: str
    start_date: str
    end_date: str

    # ── Agent outputs ─────────────────────────────────────────
    stock_data: Optional[StockData]
    news_data: Optional[NewsData]
    analysis_data: Optional[AnalysisData]
    report_data: Optional[ReportData]

    # ── Collaboration log ─────────────────────────────────────
    messages: List[AgentMessage]

    # ── Control flow ──────────────────────────────────────────
    current_agent: str
    retry_count: int
    errors: List[str]
    completed: bool
