"""
agents/data_analyst.py — Data Analyst Agent
Computes KPIs, generates all charts, and produces numerical insights.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any
from loguru import logger

from graph.state import FinancialAgentState, AgentMessage
from tools.financial_tools import compute_kpis
from tools.chart_tools import (
    chart_price_history,
    chart_kpi_bars,
    chart_returns_distribution,
    chart_sentiment_gauge,
    chart_financials,
)

AGENT_NAME = "Data Analyst"


def _log(state: FinancialAgentState, content: str, status: str = "running") -> list:
    msgs = list(state.get("messages", []))
    msgs.append(AgentMessage(
        agent=AGENT_NAME,
        content=content,
        timestamp=datetime.now().strftime("%H:%M:%S"),
        status=status,
    ))
    return msgs


def data_analyst_node(state: FinancialAgentState) -> FinancialAgentState:
    """
    LangGraph node: Data Analyst Agent.
    Populates state.analysis_data with KPIs, charts, and insights.
    """
    ticker = state["ticker"]
    company_name = state.get("company_name", ticker)
    stock_data = state.get("stock_data") or {}
    news_data = state.get("news_data") or {}
    errors = list(state.get("errors", []))

    logger.info(f"[{AGENT_NAME}] Starting analysis for {ticker}")
    msgs = _log(state, f"🧮 Starting quantitative analysis for **{ticker}**…")

    # ── Reconstruct DataFrame ──────────────────────────────────
    hist_records = stock_data.get("hist_prices", [])
    if hist_records:
        df = pd.DataFrame(hist_records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        # Normalize all column names to Title Case so that
        # both old (lowercase) and new (Title Case) records work
        df.columns = [col.capitalize() if col.islower() else col for col in df.columns]
        # Handle 'volume' -> 'Volume' specifically (capitalize is enough)
        rename_map = {col: col.title() for col in df.columns
                      if col.lower() in ("open", "high", "low", "close", "volume")}
        df = df.rename(columns=rename_map)
    else:
        df = pd.DataFrame()

    # ── Step 1: KPI computation ────────────────────────────────
    msgs = _log({**state, "messages": msgs}, "📐 Computing KPIs, ratios, and technical indicators…")
    kpi_table = compute_kpis(df, stock_data)
    msgs = _log({**state, "messages": msgs}, f"✅ {len(kpi_table)} KPI signals computed.")

    # ── Step 2: Charts ─────────────────────────────────────────
    chart_paths = []

    msgs = _log({**state, "messages": msgs}, "📊 Generating price history chart with MA overlays…")
    currency = stock_data.get("currency", "USD")
    p = chart_price_history(df, ticker, company_name, currency=currency)
    if p:
        chart_paths.append(p)
        msgs = _log({**state, "messages": msgs}, f"✅ Price chart saved.")

    msgs = _log({**state, "messages": msgs}, "📊 Generating KPI fundamentals chart…")
    p = chart_kpi_bars(kpi_table, ticker)
    if p:
        chart_paths.append(p)
        msgs = _log({**state, "messages": msgs}, "✅ KPI chart saved.")

    msgs = _log({**state, "messages": msgs}, "📊 Generating returns distribution histogram…")
    p = chart_returns_distribution(df, ticker)
    if p:
        chart_paths.append(p)
        msgs = _log({**state, "messages": msgs}, "✅ Returns distribution saved.")

    msgs = _log({**state, "messages": msgs}, "📊 Generating sentiment gauge…")
    p = chart_sentiment_gauge(
        news_data.get("sentiment_score", 0),
        news_data.get("sentiment_label", "Neutral"),
        ticker,
        news_data.get("sentiment_breakdown", {}),
    )
    if p:
        chart_paths.append(p)
        msgs = _log({**state, "messages": msgs}, "✅ Sentiment gauge saved.")

    msgs = _log({**state, "messages": msgs}, "📊 Generating financials summary chart…")
    p = chart_financials(stock_data.get("financials", {}), ticker, currency=currency)
    if p:
        chart_paths.append(p)
        msgs = _log({**state, "messages": msgs}, "✅ Financials chart saved.")

    msgs = _log({**state, "messages": msgs}, f"✅ {len(chart_paths)}/5 charts generated.")

    # ── Step 3: Trend & volatility analysis ───────────────────
    price_trend = "Insufficient Data"
    volatility_level = "Unknown"
    fundamental_score = 0.0
    technical_signals = {}

    if not df.empty and len(df) >= 20 and "Close" in df.columns:
        close = df["Close"]
        ret = close.pct_change().dropna()

        # Trend
        # ma20 = close.rolling(20).mean().iloc[-1]
        # ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else ma20
        # price_trend = (
        #     "Uptrend 📈" if close.iloc[-1] > ma20 > ma50 else
        #     "Downtrend 📉" if close.iloc[-1] < ma20 < ma50 else
        #     "Sideways ↔"
        # )
    
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else ma20
        price_trend = (
            "Uptrend [+]" if close.iloc[-1] > ma20 > ma50 else
            "Downtrend [-]" if close.iloc[-1] < ma20 < ma50 else
            "Sideways [~]"
        )

        # Volatility
        vol = ret.std() * np.sqrt(252) * 100
        volatility_level = (
            "Low (< 20%)" if vol < 20 else
            "High (> 40%)" if vol > 40 else
            "Moderate (20-40%)"
        )

        technical_signals = {
            "price_vs_20ma": "Above" if close.iloc[-1] > ma20 else "Below",
            "price_vs_50ma": "Above" if close.iloc[-1] > ma50 else "Below",
            "annualised_volatility_pct": round(vol, 2),
            "period_return_pct": round(((close.iloc[-1] / close.iloc[0]) - 1) * 100, 2),
            "52w_high": stock_data.get("fifty_two_week_high"),
            "52w_low": stock_data.get("fifty_two_week_low"),
        }

    # ── Step 4: Fundamental score (0-100) ─────────────────────
    green = sum(1 for k in kpi_table if "🟢" in k.get("signal", ""))
    red   = sum(1 for k in kpi_table if "🔴" in k.get("signal", ""))
    total_signals = len(kpi_table) or 1
    fundamental_score = round((green / total_signals) * 100, 1)

    msgs = _log({**state, "messages": msgs},
                f"📈 Trend: {price_trend} | Volatility: {volatility_level} | "
                f"Fundamental Score: {fundamental_score}/100 "
                f"({green} bullish, {red} bearish signals)")

    # ── Build analyst summary text ─────────────────────────────
    analyst_summary = _build_analyst_summary(
        ticker, company_name, stock_data, kpi_table,
        price_trend, volatility_level, fundamental_score, technical_signals
    )

    msgs = _log({**state, "messages": msgs},
                "🏁 Analysis complete. Handing off to **Report Writer**.", "done")

    analysis_data = {
        "kpi_table": kpi_table,
        "chart_paths": chart_paths,
        "price_trend": price_trend,
        "volatility_level": volatility_level,
        "technical_signals": technical_signals,
        "fundamental_score": fundamental_score,
        "analyst_summary": analyst_summary,
    }

    return {
        **state,
        "analysis_data": analysis_data,
        "messages": msgs,
        "errors": errors,
        "current_agent": "report_writer",
    }


def _build_analyst_summary(ticker, company_name, stock_data, kpi_table,
                            price_trend, volatility_level, score, signals):
    price = stock_data.get("current_price", "N/A")
    chg_pct = stock_data.get("price_change_pct", 0)
    sector = stock_data.get("sector", "N/A")

    green_kpis = [k["metric"] for k in kpi_table if "🟢" in k.get("signal", "")]
    red_kpis   = [k["metric"] for k in kpi_table if "🔴" in k.get("signal", "")]

    summary = (
        f"{company_name} ({ticker}) is currently trading at ${price} ({chg_pct:+.2f}% today) "
        f"in the {sector} sector. "
        f"The stock exhibits a {price_trend} with {volatility_level} volatility. "
    )
    if green_kpis:
        summary += f"Positive indicators include: {', '.join(green_kpis[:4])}. "
    if red_kpis:
        summary += f"Areas of concern: {', '.join(red_kpis[:4])}. "
    summary += f"Overall fundamental health score: {score}/100."
    return summary
