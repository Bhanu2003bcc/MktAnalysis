"""
agents/report_writer.py — Report Writer Agent
Uses Gemini to synthesize research + analysis into a professional report.
"""

import os
import json
from datetime import datetime
from typing import Any
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

import google.generativeai as genai_old  # Keep for potential backwards compat if needed elsewhere
from google import genai
from google.genai import types

from graph.state import FinancialAgentState, AgentMessage
AGENT_NAME = "Report Writer"

# ─── Gemini Client Helper ─────────────────────────────────────────────────────
def _get_client():
    from config import GEMINI_API_KEY
    key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if not key:
        logger.error("Gemini API Key is missing! Ensure it is set in the sidebar or .env file.")
        raise ValueError("Gemini API Key is missing.")
    return genai.Client(api_key=key)


def _log(state: FinancialAgentState, content: str, status: str = "running") -> list:
    msgs = list(state.get("messages", []))
    msgs.append(AgentMessage(
        agent=AGENT_NAME,
        content=content,
        timestamp=datetime.now().strftime("%H:%M:%S"),
        status=status,
    ))
    return msgs


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_gemini(prompt: str, schema: Any = None) -> str:
    from config import GEMINI_MODEL, GEMINI_TEMPERATURE
    try:
        client = _get_client()
        
        config_args = {
            "temperature": GEMINI_TEMPERATURE,
            "max_output_tokens": 2048,
        }
        if schema:
            config_args["response_mime_type"] = "application/json"
            config_args["response_schema"] = schema

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(**config_args)
            )
        except Exception as e:
            err_str = str(e).upper()
            # Fallback for 404 (Not Found) or 429 (Resource Exhausted / Quota)
            if ("NOT_FOUND" in err_str or "RESOURCE_EXHAUSTED" in err_str) and GEMINI_MODEL != "gemini-1.5-flash":
                logger.warning(f"Model {GEMINI_MODEL} failed (Quota/Missing). Falling back to gemini-1.5-flash.")
                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_args)
                )
            else:
                raise e

        if not response or not response.text:
            return ""
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error in _call_gemini: {e}")
        raise e


def report_writer_node(state: FinancialAgentState) -> FinancialAgentState:
    """
    LangGraph node: Report Writer Agent.
    Synthesises all data using Gemini and produces report sections.
    """
    ticker = state["ticker"]
    company_name = state.get("company_name", ticker)
    stock_data = state.get("stock_data") or {}
    news_data = state.get("news_data") or {}
    analysis_data = state.get("analysis_data") or {}
    errors = list(state.get("errors", []))

    logger.info(f"[{AGENT_NAME}] Starting report writing for {ticker}")
    msgs = _log(state, f"✍️ Starting report synthesis for **{ticker}** using Gemini…")

    # Build context for Gemini
    context = _build_context(ticker, company_name, stock_data, news_data, analysis_data)

    report_data = {}

    # ── Section 1: Executive Summary ──────────────────────────
    msgs = _log({**state, "messages": msgs}, "🤖 Gemini writing executive summary…")
    try:
        exec_prompt = f"""
You are a senior Wall Street analyst. Write a concise, professional executive summary (150-200 words) for a stock analysis report.

Context:
{context}

Requirements:
- Lead with the company's current position and recent performance
- Highlight 2-3 key strengths and risks
- Include the sector and macro context
- Professional, data-driven tone
- Do NOT include any recommendation yet

Executive Summary:
"""
        report_data["executive_summary"] = _call_gemini(exec_prompt)
        msgs = _log({**state, "messages": msgs}, "✅ Executive summary drafted.")
    except Exception as e:
        report_data["executive_summary"] = analysis_data.get("analyst_summary", "Summary unavailable.")
        errors.append(f"Gemini exec summary error: {e}")
        msgs = _log({**state, "messages": msgs}, f"⚠️ Gemini fallback used for executive summary.", "error")

    # ── Section 2: Market Research section ────────────────────
    msgs = _log({**state, "messages": msgs}, "🤖 Gemini writing market research section…")
    try:
        articles_text = "\n".join(
            f"- [{a.get('sentiment_label','?')}] {a['title']} ({a.get('published','')[:10]})"
            for a in news_data.get("articles", [])[:10]
        )
        research_prompt = f"""
You are a market researcher. Write a professional market research section (200-250 words) for a stock report.

Company: {company_name} ({ticker})
Sector: {stock_data.get('sector','N/A')}
News Sentiment: {news_data.get('sentiment_label','N/A')} (score: {news_data.get('sentiment_score', 0):+.3f})
Key Themes: {', '.join(news_data.get('key_themes', []))}
Sentiment Breakdown: {json.dumps(news_data.get('sentiment_breakdown', {}))}

Recent Headlines:
{articles_text}

Requirements:
- Summarise the news landscape and dominant narratives
- Comment on market sentiment with evidence from headlines
- Identify key themes driving investor perception
- Professional, balanced tone

Market Research Section:
"""
        report_data["market_research_section"] = _call_gemini(research_prompt)
        msgs = _log({**state, "messages": msgs}, "✅ Market research section drafted.")
    except Exception as e:
        report_data["market_research_section"] = f"Sentiment: {news_data.get('sentiment_label','N/A')}. Themes: {', '.join(news_data.get('key_themes',[]))}"
        errors.append(f"Gemini research section error: {e}")

    # ── Section 3: Data Analysis section ──────────────────────
    msgs = _log({**state, "messages": msgs}, "🤖 Gemini writing data analysis section…")
    try:
        kpi_text = "\n".join(
            f"- {k['metric']}: {k['value']} → {k['signal']}"
            for k in analysis_data.get("kpi_table", [])[:15]
        )
        # Financials (latest year)
        currency = stock_data.get("currency", "USD")
        financials = stock_data.get("financials", {})
        analysis_prompt = f"""
You are a quantitative financial analyst. Write a professional data analysis section (200-250 words) for {company_name} ({ticker}).
The currency for all values is {currency}.

Price Trend: {analysis_data.get('price_trend','N/A')}
Volatility: {analysis_data.get('volatility_level','N/A')}
Fundamental Score: {analysis_data.get('fundamental_score', 0)}/100

KPI Signals:
{kpi_text}

Financials (latest year in {currency}):
- Revenue: {financials.get('revenue',0)/1e9:.2f}B
- Net Income: {financials.get('net_income',0)/1e9:.2f}B
- Free Cash Flow: {financials.get('free_cf',0)/1e9:.2f}B

Requirements:
- Interpret the KPIs with professional context
- Comment on valuation, growth, and financial health in the context of {currency}
- Reference technical trend and volatility
- Use financial terminology appropriately

Data Analysis Section:
"""
        report_data["data_analysis_section"] = _call_gemini(analysis_prompt)
        msgs = _log({**state, "messages": msgs}, "✅ Data analysis section drafted.")
    except Exception as e:
        report_data["data_analysis_section"] = analysis_data.get("analyst_summary", "Analysis unavailable.")
        errors.append(f"Gemini analysis section error: {e}")

    # ── Section 4: Recommendation ──────────────────────────────
    msgs = _log({**state, "messages": msgs}, "🤖 Gemini generating investment recommendation…")
    try:
        # Define JSON schema for the recommendation
        rec_schema = {
            "type": "OBJECT",
            "properties": {
                "recommendation": {"type": "STRING", "enum": ["BUY", "HOLD", "SELL"]},
                "confidence_score": {"type": "NUMBER"},
                "target_price": {"type": "NUMBER", "nullable": True},
                "rationale": {"type": "STRING"},
                "risk_factors": {"type": "ARRAY", "items": {"type": "STRING"}},
                "catalysts": {"type": "ARRAY", "items": {"type": "STRING"}}
            },
            "required": ["recommendation", "confidence_score", "rationale", "risk_factors", "catalysts"]
        }

        rec_prompt = f"""
You are a senior investment analyst. Provide a clear, justified investment recommendation.
Base your recommendation on the fundamental score, price trend, and news sentiment provided.
All financial values are in {stock_data.get('currency', 'USD')}.

Company: {company_name} ({ticker})
Current Price: {stock_data.get('current_price','N/A')}
52-Week Range: {stock_data.get('fifty_two_week_low','N/A')} - {stock_data.get('fifty_two_week_high','N/A')}
Fundamental Score: {analysis_data.get('fundamental_score',0)}/100
Price Trend: {analysis_data.get('price_trend','N/A')}
Sentiment: {news_data.get('sentiment_label','N/A')}
Executive Summary: {report_data.get('executive_summary','')[:500]}
"""
        rec_text = _call_gemini(rec_prompt, schema=rec_schema)
        rec_json = json.loads(rec_text)

        report_data["recommendation"]   = rec_json.get("recommendation", "HOLD")
        report_data["confidence_score"] = float(rec_json.get("confidence_score", 50))
        report_data["target_price"]     = rec_json.get("target_price")
        report_data["risk_factors"]     = rec_json.get("risk_factors", [])
        report_data["catalysts"]        = rec_json.get("catalysts", [])
        report_data["recommendation_rationale"] = rec_json.get("rationale", "")

        msgs = _log({**state, "messages": msgs},
                    f"✅ Recommendation: **{report_data['recommendation']}** "
                    f"(Confidence: {report_data['confidence_score']:.0f}%) | "
                    f"Target: ${report_data.get('target_price','N/A')}")
    except Exception as e:
        logger.error(f"Gemini recommendation error: {e}")
        report_data["recommendation"] = "HOLD"
        report_data["confidence_score"] = 50.0
        report_data["risk_factors"] = ["Data limitations", "Market uncertainty"]
        report_data["catalysts"] = ["Earnings beat", "Sector tailwinds"]
        errors.append(f"Gemini recommendation error: {e}")
        msgs = _log({**state, "messages": msgs}, f"⚠️ Default recommendation used (Gemini error).", "error")

    msgs = _log({**state, "messages": msgs},
                "🏁 Report writing complete. Generating PDF…", "done")

    return {
        **state,
        "report_data": report_data,
        "messages": msgs,
        "errors": errors,
        "completed": True,
    }


def _build_context(ticker, company_name, stock_data, news_data, analysis_data) -> str:
    currency = stock_data.get('currency', 'USD')
    return f"""
Company: {company_name} ({ticker})
Sector: {stock_data.get('sector', 'N/A')} | Industry: {stock_data.get('industry','N/A')}
Current Price: {stock_data.get('current_price', 'N/A')} ({stock_data.get('price_change_pct', 0):+.2f}% today)
Market Cap: {currency} {(stock_data.get('market_cap') or 0)/1e9:.2f}B (Values in {currency})
52-Week Range: {stock_data.get('fifty_two_week_low','N/A')} - {stock_data.get('fifty_two_week_high','N/A')}

Price Trend: {analysis_data.get('price_trend', 'N/A')}
Volatility: {analysis_data.get('volatility_level', 'N/A')}
Fundamental Score: {analysis_data.get('fundamental_score', 0)}/100

News Sentiment: {news_data.get('sentiment_label', 'N/A')} (score: {news_data.get('sentiment_score', 0):+.3f})
Key Themes: {', '.join(news_data.get('key_themes', []))}
""".strip()
