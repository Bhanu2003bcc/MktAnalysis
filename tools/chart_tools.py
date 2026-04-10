"""
tools/chart_tools.py — Generate all charts as PNG files for PDF and Streamlit
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from datetime import datetime

from config import CHART_DIR, BRAND_COLORS


def _style_ax(ax, bg="#0D1117", grid_color="#21262D"):
    ax.set_facecolor(bg)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(grid_color)
    ax.grid(True, color=grid_color, linewidth=0.5, alpha=0.7)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


def _fig_bg(fig, color="#0D1117"):
    fig.patch.set_facecolor(color)


def _save(fig, name: str) -> str:
    path = CHART_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Chart saved: {path}")
    return str(path)


# ── 1. Price chart with volume & moving averages ──────────────────────────────
def chart_price_history(df: pd.DataFrame, ticker: str, company_name: str, currency: str = "USD") -> Optional[str]:
    if df.empty:
        return None
    try:
        # Normalize column names defensively
        if not df.empty:
            rename_map = {col: col.title() for col in df.columns
                          if col.lower() in ("open", "high", "low", "close", "volume")}
            if rename_map:
                df = df.rename(columns=rename_map)
        if "Close" not in df.columns:
            logger.warning("chart_price_history: 'Close' column missing, skipping chart.")
            return None

        fig = plt.figure(figsize=(14, 7), tight_layout=True)
        _fig_bg(fig)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        _style_ax(ax1); _style_ax(ax2)

        close = df["Close"]
        dates = df.index

        # Price line + fill
        ax1.plot(dates, close, color=BRAND_COLORS["primary"], linewidth=1.8, label="Close Price", zorder=3)
        ax1.fill_between(dates, close, close.min() * 0.98,
                         color=BRAND_COLORS["primary"], alpha=0.12)

        # Moving averages
        if len(close) >= 20:
            ma20 = close.rolling(20).mean()
            ax1.plot(dates, ma20, color=BRAND_COLORS["secondary"], linewidth=1.2,
                     linestyle="--", label="20-MA", alpha=0.9)
        if len(close) >= 50:
            ma50 = close.rolling(50).mean()
            ax1.plot(dates, ma50, color=BRAND_COLORS["accent"], linewidth=1.2,
                     linestyle="--", label="50-MA", alpha=0.9)

        ax1.set_title(f"{company_name} ({ticker}) — Price History", fontsize=13, pad=10, color="white", fontweight="bold")
        ax1.set_ylabel(f"Price ({currency})", color="white")
        ax1.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=8)
        plt.setp(ax1.get_xticklabels(), visible=False)

        # Volume bars
        colors = [BRAND_COLORS["secondary"] if c >= o else BRAND_COLORS["accent"]
                  for c, o in zip(df["Close"], df["Open"])]
        ax2.bar(dates, df["Volume"] / 1e6, color=colors, alpha=0.8, width=1)
        ax2.set_ylabel("Vol (M)", color="white", fontsize=8)

        fig.autofmt_xdate()
        return _save(fig, f"{ticker}_price")
    except Exception as e:
        logger.error(f"Price chart failed: {e}")
        return None


# ── 2. KPI radar / bar chart ──────────────────────────────────────────────────
def chart_kpi_bars(kpi_table: List[Dict], ticker: str) -> Optional[str]:
    fundamental = [k for k in kpi_table if k.get("category") == "Fundamental"]
    if not fundamental:
        return None
    try:
        labels = [k["metric"] for k in fundamental]
        values_str = [k["value"] for k in fundamental]
        signals = [k.get("signal", "—") for k in fundamental]
        colors = [
            BRAND_COLORS["secondary"] if "🟢" in s else
            BRAND_COLORS["accent"]    if "🔴" in s else
            "#A0AEC0"
            for s in signals
        ]

        fig, ax = plt.subplots(figsize=(12, 5))
        _fig_bg(fig); _style_ax(ax)

        x = np.arange(len(labels))
        bars = ax.bar(x, range(len(labels), 0, -1), color=colors, alpha=0.85, width=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9, color="white")
        ax.set_yticks([])
        ax.set_title(f"{ticker} — Fundamental KPI Signals", fontsize=13, color="white", fontweight="bold")

        for bar, val, sig in zip(bars, values_str, signals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val}\n{sig}", ha="center", va="bottom", color="white", fontsize=8)

        return _save(fig, f"{ticker}_kpis")
    except Exception as e:
        logger.error(f"KPI chart failed: {e}")
        return None


# ── 3. Returns distribution histogram ────────────────────────────────────────
def chart_returns_distribution(df: pd.DataFrame, ticker: str) -> Optional[str]:
    if df.empty or len(df) < 30:
        return None
    try:
        if "Close" not in df.columns:
            logger.warning("chart_returns_distribution: 'Close' column missing, skipping.")
            return None
        returns = df["Close"].pct_change().dropna() * 100
        fig, ax = plt.subplots(figsize=(10, 5))
        _fig_bg(fig); _style_ax(ax)

        n, bins, patches = ax.hist(returns, bins=50, edgecolor="none", alpha=0.85)
        for patch, left in zip(patches, bins):
            patch.set_facecolor(BRAND_COLORS["secondary"] if left >= 0 else BRAND_COLORS["accent"])

        mean_r = returns.mean()
        ax.axvline(mean_r, color="yellow", linewidth=1.5, linestyle="--", label=f"Mean: {mean_r:.2f}%")
        ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{ticker} — Daily Returns Distribution", fontsize=13, color="white", fontweight="bold")
        ax.set_xlabel("Daily Return (%)", color="white")
        ax.set_ylabel("Frequency", color="white")
        ax.legend(facecolor="#1A1A2E", labelcolor="white")
        return _save(fig, f"{ticker}_returns")
    except Exception as e:
        logger.error(f"Returns chart failed: {e}")
        return None


# ── 4. Sentiment gauge ────────────────────────────────────────────────────────
def chart_sentiment_gauge(score: float, label: str, ticker: str, breakdown: Dict) -> Optional[str]:
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        _fig_bg(fig); _style_ax(ax1); _style_ax(ax2)

        # Gauge
        theta = np.linspace(np.pi, 0, 200)
        ax1.set_xlim(-1.3, 1.3); ax1.set_ylim(-0.2, 1.3)
        ax1.set_aspect("equal"); ax1.axis("off")
        ax1.set_facecolor("#0D1117")

        for t1, t2, color in [
            (np.pi, np.pi * 2/3,     "#EF4444"),
            (np.pi * 2/3, np.pi/3,   "#F59E0B"),
            (np.pi / 3, 0,           "#22C55E"),
        ]:
            t = np.linspace(t1, t2, 50)
            ax1.fill_between(np.cos(t) * 1.1, np.sin(t) * 1.1,
                             np.cos(t) * 0.7,  np.sin(t) * 0.7, color=color, alpha=0.9)

        needle_angle = np.pi * (1 - (score + 1) / 2)
        ax1.annotate("", xy=(np.cos(needle_angle) * 0.9, np.sin(needle_angle) * 0.9),
                     xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="white", lw=2))
        ax1.text(0, -0.15, f"Score: {score:+.2f}", ha="center", color="white", fontsize=12, fontweight="bold")
        ax1.text(0, 0.35, label, ha="center", color="white", fontsize=11)
        ax1.set_title("Sentiment Gauge", color="white", fontsize=12)

        # Breakdown donut
        pos = breakdown.get("positive", 0)
        neu = breakdown.get("neutral", 0)
        neg = breakdown.get("negative", 0)
        total = pos + neu + neg or 1
        wedge_colors = [BRAND_COLORS["secondary"], "#A0AEC0", BRAND_COLORS["accent"]]
        wedges, texts, autotexts = ax2.pie(
            [pos, neu, neg], labels=["Positive", "Neutral", "Negative"],
            colors=wedge_colors, autopct="%1.0f%%", startangle=90,
            wedgeprops=dict(width=0.55),
            textprops={"color": "white", "fontsize": 9},
        )
        for at in autotexts:
            at.set_color("white")
        ax2.set_title(f"News Sentiment Breakdown\n({total} articles)", color="white", fontsize=12)
        ax2.set_facecolor("#0D1117")

        return _save(fig, f"{ticker}_sentiment")
    except Exception as e:
        logger.error(f"Sentiment chart failed: {e}")
        return None


# ── 5. Financials bar chart ───────────────────────────────────────────────────
def chart_financials(financials: Dict, ticker: str, currency: str = "USD") -> Optional[str]:
    keys = ["revenue", "gross_profit", "net_income", "ebitda", "operating_cf", "free_cf"]
    labels_map = {
        "revenue": "Revenue", "gross_profit": "Gross Profit",
        "net_income": "Net Income", "ebitda": "EBITDA",
        "operating_cf": "Operating CF", "free_cf": "Free CF",
    }
    vals = [(labels_map[k], financials.get(k, 0) / 1e9) for k in keys if financials.get(k)]
    if not vals:
        return None
    try:
        labels, amounts = zip(*vals)
        colors = [BRAND_COLORS["secondary"] if v >= 0 else BRAND_COLORS["accent"] for v in amounts]
        fig, ax = plt.subplots(figsize=(10, 5))
        _fig_bg(fig); _style_ax(ax)
        bars = ax.bar(labels, amounts, color=colors, alpha=0.85, width=0.6)
        for bar, val in zip(bars, amounts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.05 if val >= 0 else -0.15),
                    f"${val:.2f}B", ha="center", va="bottom", color="white", fontsize=9)
        ax.set_title(f"{ticker} — Financial Summary (Latest Year)", fontsize=13, color="white", fontweight="bold")
        ax.set_ylabel(f"{currency} Billions", color="white")
        ax.axhline(0, color="white", linewidth=0.5)
        return _save(fig, f"{ticker}_financials")
    except Exception as e:
        logger.error(f"Financials chart failed: {e}")
        return None
