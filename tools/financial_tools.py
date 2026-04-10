"""
tools/financial_tools.py — Yahoo Finance data fetching tools
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

from config import KPIS, KPI_LABELS


def fetch_stock_info(ticker: str) -> Dict[str, Any]:
    """Fetch comprehensive stock information from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        company_name = info.get("longName") or info.get("shortName") or ticker

        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose") or current_price
        price_change = current_price - prev_close
        price_change_pct = (price_change / prev_close * 100) if prev_close else 0

        kpis = {}
        for kpi in KPIS:
            val = info.get(kpi)
            if val is not None:
                kpis[kpi] = val

        return {
            "ticker": ticker.upper(),
            "company_name": company_name,
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "current_price": round(current_price, 2),
            "price_change": round(price_change, 2),
            "price_change_pct": round(price_change_pct, 2),
            "market_cap": info.get("marketCap"),
            "volume": info.get("volume"),
            "avg_volume": info.get("averageVolume"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "description": info.get("longBusinessSummary", ""),
            "website": info.get("website", ""),
            "employees": info.get("fullTimeEmployees"),
            "country": info.get("country", ""),
            "exchange": info.get("exchange", ""),
            "currency": info.get("currency", "USD"),
            "kpis": kpis,
            "success": True,
        }
    except Exception as e:
        logger.error(f"Error fetching stock info for {ticker}: {e}")
        return {"ticker": ticker, "success": False, "error": str(e)}


def fetch_price_history(ticker: str, start_date: str, end_date: str) -> Tuple[List[Dict], pd.DataFrame]:
    """Fetch OHLCV price history."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, auto_adjust=True)

        if df.empty:
            logger.warning(f"No price data found for {ticker}")
            return [], pd.DataFrame()

        df.index = pd.to_datetime(df.index)
        df = df.round(4)

        records = []
        for date, row in df.iterrows():
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "Open": row["Open"],
                "High": row["High"],
                "Low": row["Low"],
                "Close": row["Close"],
                "Volume": int(row["Volume"]),
            })

        return records, df
    except Exception as e:
        logger.error(f"Error fetching price history for {ticker}: {e}")
        return [], pd.DataFrame()


def compute_kpis(df: pd.DataFrame, info: Dict) -> List[Dict]:
    """Compute KPI table with benchmark signals."""
    kpi_table = []

    # Defensive: normalize column names to Title Case regardless of source
    if not df.empty:
        rename_map = {col: col.title() for col in df.columns
                      if col.lower() in ("open", "high", "low", "close", "volume")}
        if rename_map:
            df = df.rename(columns=rename_map)

    # ── Price-based metrics ───────────────────────────────────
    if not df.empty and len(df) >= 2 and "Close" in df.columns:
        close = df["Close"]
        returns = close.pct_change().dropna()

        # Moving averages
        ma_20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else None
        ma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
        ma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

        # Volatility (annualised)
        volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 1 else None

        # RSI
        rsi = _compute_rsi(close)

        # Total return
        total_return = ((close.iloc[-1] / close.iloc[0]) - 1) * 100

        if ma_20:
            kpi_table.append({
                "category": "Technical",
                "metric": "20-Day MA",
                "value": f"${ma_20:.2f}",
                "benchmark": "Price vs MA",
                "signal": _price_vs_ma_signal(close.iloc[-1], ma_20),
            })
        if ma_50:
            kpi_table.append({
                "category": "Technical",
                "metric": "50-Day MA",
                "value": f"${ma_50:.2f}",
                "benchmark": "Price vs MA",
                "signal": _price_vs_ma_signal(close.iloc[-1], ma_50),
            })
        if volatility:
            kpi_table.append({
                "category": "Technical",
                "metric": "Annualised Volatility",
                "value": f"{volatility:.1f}%",
                "benchmark": "< 20% Low, 20-40% Med, > 40% High",
                "signal": "Low" if volatility < 20 else "Medium" if volatility < 40 else "High",
            })
        if rsi:
            kpi_table.append({
                "category": "Technical",
                "metric": "RSI (14-day)",
                "value": f"{rsi:.1f}",
                "benchmark": "< 30 Oversold | > 70 Overbought",
                "signal": "Oversold 🟢" if rsi < 30 else "Overbought 🔴" if rsi > 70 else "Neutral ⚪",
            })
        kpi_table.append({
            "category": "Technical",
            "metric": "Period Return",
            "value": f"{total_return:+.1f}%",
            "benchmark": "vs S&P 500",
            "signal": "Positive 🟢" if total_return > 0 else "Negative 🔴",
        })

    # ── Fundamental metrics ───────────────────────────────────
    kpis = info.get("kpis", {})
    fundamental_defs = [
        ("trailingPE",     "P/E Ratio", lambda v: f"{v:.1f}x",     lambda v: "Fair" if 10 < v < 25 else "Overvalued 🔴" if v > 25 else "Cheap 🟢"),
        ("priceToBook",    "P/B Ratio", lambda v: f"{v:.2f}x",     lambda v: "Fair" if 1 < v < 3 else "Overvalued 🔴" if v > 3 else "Cheap 🟢"),
        ("returnOnEquity", "ROE",       lambda v: f"{v*100:.1f}%", lambda v: "Strong 🟢" if v > 0.15 else "Weak 🔴" if v < 0.08 else "Average ⚪"),
        ("debtToEquity",   "D/E Ratio", lambda v: f"{v:.2f}",      lambda v: "Safe 🟢" if v < 1 else "Risky 🔴" if v > 2 else "Moderate ⚪"),
        ("profitMargins",  "Net Margin",lambda v: f"{v*100:.1f}%", lambda v: "Strong 🟢" if v > 0.15 else "Thin 🔴" if v < 0.05 else "Average ⚪"),
        ("currentRatio",   "Current Ratio",lambda v: f"{v:.2f}",   lambda v: "Healthy 🟢" if v > 2 else "Risky 🔴" if v < 1 else "Adequate ⚪"),
        ("revenueGrowth",  "Rev. Growth",lambda v: f"{v*100:.1f}%",lambda v: "Strong 🟢" if v > 0.15 else "Declining 🔴" if v < 0 else "Moderate ⚪"),
        ("dividendYield",  "Div. Yield", lambda v: f"{v*100:.2f}%",lambda v: "Good 🟢" if v > 0.03 else "None/Low ⚪"),
        ("beta",           "Beta",       lambda v: f"{v:.2f}",      lambda v: "Low Vol 🟢" if v < 0.8 else "High Vol 🔴" if v > 1.5 else "Market 🟡"),
    ]

    for key, label, fmt, sig_fn in fundamental_defs:
        val = kpis.get(key)
        if val is not None:
            try:
                kpi_table.append({
                    "category": "Fundamental",
                    "metric": label,
                    "value": fmt(val),
                    "benchmark": "—",
                    "signal": sig_fn(val),
                    "raw_value": val,
                })
            except Exception:
                pass

    return kpi_table


def _compute_rsi(prices: pd.Series, period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def _price_vs_ma_signal(price: float, ma: float) -> str:
    if price > ma * 1.05:
        return "Above MA 🟢"
    elif price < ma * 0.95:
        return "Below MA 🔴"
    return "Near MA ⚪"


def fetch_financials_summary(ticker: str) -> Dict[str, Any]:
    """Fetch income/balance sheet/cashflow summaries."""
    try:
        stock = yf.Ticker(ticker)
        summary = {}

        try:
            income = stock.financials
            if not income.empty:
                latest = income.iloc[:, 0]
                summary["revenue"] = float(latest.get("Total Revenue", 0) or 0)
                summary["gross_profit"] = float(latest.get("Gross Profit", 0) or 0)
                summary["net_income"] = float(latest.get("Net Income", 0) or 0)
                summary["ebitda"] = float(latest.get("EBITDA", 0) or 0)
        except Exception:
            pass

        try:
            balance = stock.balance_sheet
            if not balance.empty:
                latest = balance.iloc[:, 0]
                summary["total_assets"] = float(latest.get("Total Assets", 0) or 0)
                summary["total_liabilities"] = float(latest.get("Total Liabilities Net Minority Interest", 0) or 0)
                summary["total_equity"] = float(latest.get("Stockholders Equity", 0) or 0)
                summary["cash"] = float(latest.get("Cash And Cash Equivalents", 0) or 0)
        except Exception:
            pass

        try:
            cf = stock.cashflow
            if not cf.empty:
                latest = cf.iloc[:, 0]
                summary["operating_cf"] = float(latest.get("Operating Cash Flow", 0) or 0)
                summary["free_cf"] = float(latest.get("Free Cash Flow", 0) or 0)
                summary["capex"] = float(latest.get("Capital Expenditure", 0) or 0)
        except Exception:
            pass

        return summary
    except Exception as e:
        logger.error(f"Error fetching financials for {ticker}: {e}")
        return {}
