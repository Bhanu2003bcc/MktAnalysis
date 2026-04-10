"""
agents/market_researcher.py — Market Researcher Agent
Gathers financial data, news, and sentiment for a given stock.
"""

from datetime import datetime
from typing import Any
from loguru import logger

from graph.state import FinancialAgentState, AgentMessage
from tools.financial_tools import fetch_stock_info, fetch_price_history, fetch_financials_summary
from tools.news_tools import fetch_news, analyse_sentiment


AGENT_NAME = "Market Researcher"


def _log(state: FinancialAgentState, content: str, status: str = "running") -> list:
    msgs = list(state.get("messages", []))
    msgs.append(AgentMessage(
        agent=AGENT_NAME,
        content=content,
        timestamp=datetime.now().strftime("%H:%M:%S"),
        status=status,
    ))
    return msgs


def market_researcher_node(state: FinancialAgentState) -> FinancialAgentState:
    """
    LangGraph node: Market Researcher Agent.
    Populates state.stock_data and state.news_data.
    """
    ticker = state["ticker"].upper().strip()
    company_name = state.get("company_name", ticker)
    start_date = state["start_date"]
    end_date = state["end_date"]
    errors = list(state.get("errors", []))

    logger.info(f"[{AGENT_NAME}] Starting research for {ticker}")
    msgs = _log(state, f"🔍 Starting market research for **{ticker}** ({company_name})…")

    # ── Step 1: Fetch stock info ───────────────────────────────
    msgs = _log({**state, "messages": msgs}, f"📊 Fetching stock fundamentals from Yahoo Finance…")
    stock_info = fetch_stock_info(ticker)

    if not stock_info.get("success"):
        err = f"Failed to fetch stock info for {ticker}: {stock_info.get('error', 'unknown')}"
        errors.append(err)
        logger.error(f"[{AGENT_NAME}] {err}")
        msgs = _log({**state, "messages": msgs}, f"⚠️ {err}", "error")
        return {**state, "messages": msgs, "errors": errors, "current_agent": "data_analyst"}

    company_name = stock_info.get("company_name", company_name)
    msgs = _log({**state, "messages": msgs},
                f"✅ Stock info retrieved: {company_name} | Sector: {stock_info.get('sector','N/A')} | "
                f"Price: ${stock_info.get('current_price','N/A')} "
                f"({stock_info.get('price_change_pct',0):+.2f}%)")

    # ── Step 2: Fetch price history ────────────────────────────
    msgs = _log({**state, "messages": msgs}, f"📈 Fetching price history ({start_date} → {end_date})…")
    hist_records, hist_df = fetch_price_history(ticker, start_date, end_date)

    if hist_records:
        msgs = _log({**state, "messages": msgs},
                    f"✅ {len(hist_records)} trading days of price data retrieved.")
    else:
        errors.append("Price history unavailable — charts may be limited.")
        msgs = _log({**state, "messages": msgs}, "⚠️ Price history unavailable.", "error")

    # ── Step 3: Fetch financials ───────────────────────────────
    msgs = _log({**state, "messages": msgs}, "💰 Fetching income statement & balance sheet…")
    financials = fetch_financials_summary(ticker)
    msgs = _log({**state, "messages": msgs},
                f"✅ Financials retrieved: Revenue ${financials.get('revenue', 0)/1e9:.2f}B | "
                f"Net Income ${financials.get('net_income', 0)/1e9:.2f}B")

    # ── Step 4: Fetch news & sentiment ────────────────────────
    msgs = _log({**state, "messages": msgs},
                f"📰 Scanning news sources for '{ticker}' mentions…")
    articles = fetch_news(ticker, company_name, max_articles=20)
    sentiment_result = analyse_sentiment(articles)

    msgs = _log({**state, "messages": msgs},
                f"✅ Found {len(articles)} relevant news articles. "
                f"Sentiment: **{sentiment_result['sentiment_label']}** "
                f"(score: {sentiment_result['sentiment_score']:+.3f}) | "
                f"Themes: {', '.join(sentiment_result.get('key_themes', [])[:3])}")

    # ── Build state ────────────────────────────────────────────
    stock_data = {
        **stock_info,
        "hist_prices": hist_records,
        "financials": financials,
        "_hist_df": hist_df,   # passed in-memory, not serialised to JSON
    }

    news_data = {
        "articles": articles,
        **sentiment_result,
    }

    msgs = _log({**state, "messages": msgs},
                f"🏁 Research complete. Handing off to **Data Analyst**.", "done")

    return {
        **state,
        "ticker": ticker,
        "company_name": company_name,
        "stock_data": stock_data,
        "news_data": news_data,
        "messages": msgs,
        "errors": errors,
        "current_agent": "data_analyst",
    }
