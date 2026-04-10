"""
run.py — CLI runner for Financial Analysis Agent Crew
Usage: python run.py --ticker AAPL --start 2024-01-01 --end 2025-01-01
"""

import argparse
import sys
import os
from datetime import date, timedelta
from loguru import logger

from config import GEMINI_API_KEY


def main():
    parser = argparse.ArgumentParser(
        description="Financial Analysis Agent Crew — CLI Runner"
    )
    parser.add_argument("--ticker",  required=True,  help="Stock ticker (e.g. AAPL)")
    parser.add_argument("--name",    default="",     help="Company name (optional, auto-detected)")
    parser.add_argument("--start",   default=str(date.today() - timedelta(days=365)),
                        help="Start date YYYY-MM-DD (default: 1 year ago)")
    parser.add_argument("--end",     default=str(date.today()),
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--api-key", default=GEMINI_API_KEY,
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    args = parser.parse_args()

    if not args.api_key:
        logger.error("Gemini API key required. Set GEMINI_API_KEY or use --api-key")
        sys.exit(1)

    os.environ["GEMINI_API_KEY"] = args.api_key

    logger.info(f"Starting analysis: {args.ticker} | {args.start} → {args.end}")

    from graph.workflow import run_analysis
    state = run_analysis(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        company_name=args.name,
    )

    print("\n" + "="*60)
    print(f"  ANALYSIS COMPLETE: {state.get('company_name')} ({args.ticker})")
    print("="*60)

    report = state.get("report_data") or {}
    analysis = state.get("analysis_data") or {}
    print(f"  Recommendation : {report.get('recommendation','N/A')}")
    print(f"  Confidence     : {report.get('confidence_score', 0):.0f}%")
    print(f"  Target Price   : ${report.get('target_price','N/A')}")
    print(f"  Fund. Score    : {analysis.get('fundamental_score', 0):.0f}/100")
    print(f"  Price Trend    : {analysis.get('price_trend','N/A')}")
    print(f"  Sentiment      : {state.get('news_data',{}).get('sentiment_label','N/A')}")
    print(f"  PDF Report     : {report.get('pdf_path','Not generated')}")

    if state.get("errors"):
        print("\n  Warnings:")
        for e in state["errors"]:
            print(f"    ⚠ {e}")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
