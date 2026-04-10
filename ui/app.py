"""
ui/app.py — Streamlit Dashboard for Financial Analysis Agent Crew
Run: streamlit run ui/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
from pathlib import Path
import time
import json

from config import GEMINI_API_KEY, BRAND_COLORS

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Analysis Agent Crew",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --primary: #6C3EF4;
    --secondary: #00C896;
    --accent: #FF6B6B;
}
.main { background-color: #0F0F1A; }
.stApp { background-color: #0F0F1A; }

.metric-card {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
    border: 1px solid #2D2D4E;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 6px 0;
}
.metric-label { font-size: 11px; color: #8888aa; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-size: 22px; font-weight: 700; color: #ffffff; }
.metric-sub   { font-size: 12px; color: #aaaacc; }

.agent-msg {
    background: #1A1A2E;
    border-left: 3px solid var(--primary);
    border-radius: 0 8px 8px 0;
    padding: 8px 14px;
    margin: 4px 0;
    font-size: 13px;
}
.agent-researcher { border-color: #6C3EF4; }
.agent-analyst    { border-color: #00C896; }
.agent-writer     { border-color: #FF9500; }
.agent-pdf        { border-color: #FF6B6B; }
.agent-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }

.rec-buy  { background: linear-gradient(90deg,#166534,#14532d); color: #86efac; border-radius: 8px; padding: 12px 20px; }
.rec-hold { background: linear-gradient(90deg,#713f12,#451a03); color: #fde68a; border-radius: 8px; padding: 12px 20px; }
.rec-sell { background: linear-gradient(90deg,#7f1d1d,#450a0a); color: #fca5a5; border-radius: 8px; padding: 12px 20px; }

.section-header {
    background: linear-gradient(90deg, #6C3EF4, #8B5CF6);
    color: white;
    padding: 8px 16px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 14px;
    margin: 16px 0 8px 0;
}
.news-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 10px; border-radius: 6px; margin: 3px 0;
    background: #1A1A2E;
}
.tag-pos { background:#166534; color:#86efac; padding:2px 8px; border-radius:12px; font-size:11px; }
.tag-neg { background:#7f1d1d; color:#fca5a5; padding:2px 8px; border-radius:12px; font-size:11px; }
.tag-neu { background:#374151; color:#d1d5db; padding:2px 8px; border-radius:12px; font-size:11px; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _fmt_large(val):
    if val is None:
        return "N/A"
    try:
        v = float(val)
        if v >= 1e12: return f"${v/1e12:.2f}T"
        if v >= 1e9:  return f"${v/1e9:.2f}B"
        if v >= 1e6:  return f"${v/1e6:.2f}M"
        return f"${v:,.0f}"
    except Exception:
        return str(val)


def _rec_class(rec):
    return {"BUY": "rec-buy", "HOLD": "rec-hold", "SELL": "rec-sell"}.get(rec.upper(), "rec-hold")


def _get_symbol(currency_code):
    symbols = {
        "USD": "$", "INR": "₹", "EUR": "€", "GBP": "£",
        "JPY": "¥", "CAD": "C$", "AUD": "A$", "CNY": "¥"
    }
    return symbols.get(currency_code.upper(), currency_code)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Financial Agent Crew")
    st.markdown("*Powered by Gemini + LangGraph*")
    st.markdown("---")

    if not GEMINI_API_KEY:
        st.error("🔑 **Gemini API Key missing**")
        st.info("Please set `GEMINI_API_KEY` in your `.env` file or environment to enable analysis.")
    
    st.markdown("### 🔎 Analysis Parameters")
    ticker = st.text_input("Stock Ticker", value="AAPL", placeholder="e.g. TSLA, MSFT, NVDA").upper().strip()
    company_name = st.text_input("Company Name (optional)", placeholder="Auto-detected")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=date.today() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", value=date.today())

    st.markdown("---")
    # Disable button if API key is missing
    run_btn = st.button(
        "🚀 Run Analysis", 
        type="primary", 
        use_container_width=True,
        disabled=not GEMINI_API_KEY
    )

    st.markdown("---")
    st.markdown("""
**Agents:**
- 🔍 Market Researcher
- 🧮 Data Analyst
- ✍️ Report Writer (Gemini)
- 📄 PDF Generator

**Outputs:**
- Live Dashboard
- 📑 PDF Report
""")


# ─── Main content ──────────────────────────────────────────────────────────────
st.title("📈 Financial Analysis Agent Crew")
st.markdown("*A multi-agent AI system that researches, analyses, and writes professional stock reports*")

if not run_btn:
    # Landing screen
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
<div class='metric-card'>
<div class='metric-label'>Step 1</div>
<div class='metric-value'>🔍 Research</div>
<div class='metric-sub'>Market Researcher gathers price data, financials & news</div>
</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
<div class='metric-card'>
<div class='metric-label'>Step 2</div>
<div class='metric-value'>🧮 Analyse</div>
<div class='metric-sub'>Data Analyst computes KPIs, generates charts & insights</div>
</div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
<div class='metric-card'>
<div class='metric-label'>Step 3</div>
<div class='metric-value'>✍️ Report</div>
<div class='metric-sub'>Gemini writes the full report and investment recommendation</div>
</div>""", unsafe_allow_html=True)

    st.info("👈 Enter a stock ticker in the sidebar and click **Run Analysis** to get started.")
    st.stop()

# ─── Run Analysis ─────────────────────────────────────────────────────────────
if run_btn:
    from graph.workflow import run_analysis

    # Live agent log container
    st.markdown("## 🤖 Agent Collaboration Live Feed")
    log_container = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_log(messages, progress: int, status: str):
        progress_bar.progress(progress)
        status_text.markdown(f"**Status:** {status}")
        log_html = ""
        agent_class = {
            "Market Researcher": "agent-researcher",
            "Data Analyst": "agent-analyst",
            "Report Writer": "agent-writer",
            "PDF Generator": "agent-pdf",
        }
        agent_color = {
            "Market Researcher": "#6C3EF4",
            "Data Analyst":      "#00C896",
            "Report Writer":     "#FF9500",
            "PDF Generator":     "#FF6B6B",
        }
        for msg in messages[-20:]:
            agent = msg.get("agent", "System")
            content = msg.get("content", "")
            ts = msg.get("timestamp", "")
            css = agent_class.get(agent, "agent-researcher")
            color = agent_color.get(agent, "#aaaaff")
            log_html += f"""
<div class='agent-msg {css}'>
<span class='agent-label' style='color:{color}'>[{ts}] {agent}</span><br>
{content}
</div>"""
        log_container.markdown(log_html, unsafe_allow_html=True)

    with st.spinner("Running multi-agent analysis pipeline…"):
        status_text.markdown("**Status:** Initialising agents…")
        progress_bar.progress(5)

        try:
            final_state = run_analysis(
                ticker=ticker,
                start_date=str(start_date),
                end_date=str(end_date),
                company_name=company_name,
            )
            update_log(final_state.get("messages", []), 100, "✅ Analysis complete!")

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)
            st.stop()

    # ── Store results in session ───────────────────────────────
    st.session_state["final_state"] = final_state

# ─── Results ──────────────────────────────────────────────────────────────────
if "final_state" not in st.session_state:
    st.stop()

state = st.session_state["final_state"]
stock_data    = state.get("stock_data") or {}
news_data     = state.get("news_data") or {}
analysis_data = state.get("analysis_data") or {}
report_data   = state.get("report_data") or {}
errors        = state.get("errors", [])

if errors:
    with st.expander("⚠️ Warnings / Errors"):
        for e in errors:
            st.warning(e)

st.markdown("---")

# ── Headline metrics ───────────────────────────────────────────────────────────
st.markdown("## Dashboard")
c1, c2, c3, c4, c5, c6 = st.columns(6)
price = stock_data.get("current_price", 0)
chg   = stock_data.get("price_change_pct", 0)
curr  = stock_data.get("currency", "USD")
sym   = _get_symbol(curr)

c1.metric(f"Price ({curr})", f"{sym}{price}",         f"{chg:+.2f}%")
c2.metric("Market Cap",     f"{sym}{_fmt_large(stock_data.get('market_cap')).replace('$','')}")
c3.metric("Sector",         stock_data.get("sector","N/A"))
c4.metric("Sentiment",      news_data.get("sentiment_label","N/A"))
c5.metric("Fund. Score",    f"{analysis_data.get('fundamental_score',0):.0f}/100")
c6.metric("Recommendation", report_data.get("recommendation","—"))

# ── Recommendation banner ─────────────────────────────────────────────────────
rec  = report_data.get("recommendation", "HOLD")
conf = report_data.get("confidence_score", 50)
tgt  = report_data.get("target_price")
rationale = report_data.get("recommendation_rationale","")
rec_cls = _rec_class(rec)

upside_str = ""
if tgt and price:
    upside = ((float(tgt) - float(price)) / float(price)) * 100
    upside_str = f"&nbsp;&nbsp;|&nbsp;&nbsp; 12M Target: **{sym}{tgt}** ({upside:+.1f}%)"

st.markdown(f"""
<div class='{rec_cls}' style='margin:16px 0'>
<span style='font-size:24px;font-weight:800'>{rec}</span>
&nbsp;&nbsp;|&nbsp;&nbsp; Confidence: <b>{conf:.0f}%</b>{upside_str}
<br><small>{rationale}</small>
</div>""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Price & Charts", "🧮 KPI Analysis",
    "📰 News & Sentiment", "📝 Full Report", "🤖 Agent Log"
])

# ── TAB 1: Price & Charts ─────────────────────────────────────────────────────
with tab1:
    hist = stock_data.get("hist_prices", [])
    if hist:
        df = pd.DataFrame(hist)
        df["date"] = pd.to_datetime(df["date"])

        # Plotly price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["Close"],
            name="Close", line=dict(color="#6C3EF4", width=2),
            fill="tonexty", fillcolor="rgba(108,62,244,0.08)",
        ))
        if len(df) >= 20:
            df["ma20"] = df["Close"].rolling(20).mean()
            fig.add_trace(go.Scatter(x=df["date"], y=df["ma20"],
                name="20-MA", line=dict(color="#00C896", width=1.5, dash="dot")))
        if len(df) >= 50:
            df["ma50"] = df["Close"].rolling(50).mean()
            fig.add_trace(go.Scatter(x=df["date"], y=df["ma50"],
                name="50-MA", line=dict(color="#FF6B6B", width=1.5, dash="dash")))

        fig.update_layout(
            title=f"{stock_data.get('company_name',ticker)} Price History",
            paper_bgcolor="#0F0F1A", plot_bgcolor="#0D1117",
            font=dict(color="white"),
            legend=dict(bgcolor="#1A1A2E"),
            xaxis=dict(gridcolor="#21262D"),
            yaxis=dict(gridcolor="#21262D"),
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volume
        fig2 = go.Figure(go.Bar(
            x=df["date"], y=df["Volume"],
            marker_color=["#00C896" if c >= o else "#FF6B6B"
                          for c, o in zip(df["Close"], df["Open"])],
        ))
        fig2.update_layout(
            title="Volume", height=180,
            paper_bgcolor="#0F0F1A", plot_bgcolor="#0D1117",
            font=dict(color="white"),
            xaxis=dict(gridcolor="#21262D"),
            yaxis=dict(gridcolor="#21262D"),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Returns distribution
        returns = (df["Close"].pct_change().dropna() * 100).round(3)
        fig3 = px.histogram(
            returns, nbins=60,
            title="Daily Returns Distribution",
            color_discrete_sequence=["#6C3EF4"],
        )
        fig3.update_layout(
            paper_bgcolor="#0F0F1A", plot_bgcolor="#0D1117",
            font=dict(color="white"), height=300,
            xaxis=dict(title="Return %", gridcolor="#21262D"),
            yaxis=dict(gridcolor="#21262D"),
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("No price history available.")

# ── TAB 2: KPI Analysis ───────────────────────────────────────────────────────
with tab2:
    kpis = analysis_data.get("kpi_table", [])
    if kpis:
        tech = [k for k in kpis if k.get("category") == "Technical"]
        fund = [k for k in kpis if k.get("category") == "Fundamental"]

        col_t, col_f = st.columns(2)

        def render_kpi_table(items, title):
            rows = []
            for k in items:
                sig = k.get("signal","")
                emoji = "🟢" if ("🟢" in sig or "Strong" in sig or "Positive" in sig) else \
                        "🔴" if ("🔴" in sig or "Risk" in sig or "Negative" in sig) else "⚪"
                rows.append({"Metric": k["metric"], "Value": k["value"], "Signal": sig, "": emoji})
            df_kpi = pd.DataFrame(rows)
            st.markdown(f"**{title}**")
            st.dataframe(df_kpi, use_container_width=True, hide_index=True)

        with col_t:
            render_kpi_table(tech, "📐 Technical Indicators")
        with col_f:
            render_kpi_table(fund, "💰 Fundamental KPIs")

        # Bull/Bear gauge
        green = sum(1 for k in kpis if "🟢" in k.get("signal",""))
        red   = sum(1 for k in kpis if "🔴" in k.get("signal",""))
        neu   = len(kpis) - green - red

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=analysis_data.get("fundamental_score",0),
            title={"text": "Fundamental Health Score", "font": {"color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},
                "bar":  {"color": "#6C3EF4"},
                "steps": [
                    {"range": [0, 33],  "color": "#7f1d1d"},
                    {"range": [33, 66], "color": "#713f12"},
                    {"range": [66, 100],"color": "#166534"},
                ],
            },
            number={"font": {"color": "white"}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0F0F1A", font=dict(color="white"), height=300
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.info("KPI data not available.")

    # Financials
    fin = stock_data.get("financials", {})
    if fin:
        st.markdown("### 💵 Financial Summary")
        fin_data = {
            "Revenue":    fin.get("revenue", 0),
            "Gross Profit": fin.get("gross_profit", 0),
            "Net Income": fin.get("net_income", 0),
            "EBITDA":     fin.get("ebitda", 0),
            "Operating CF": fin.get("operating_cf", 0),
            "Free CF":    fin.get("free_cf", 0),
        }
        vals = {k: v/1e9 for k, v in fin_data.items() if v}
        if vals:
            fig_fin = go.Figure(go.Bar(
                x=list(vals.keys()), y=list(vals.values()),
                marker_color=["#00C896" if v >= 0 else "#FF6B6B" for v in vals.values()],
                text=[f"{sym}{v:.2f}B" for v in vals.values()],
                textposition="outside",
            ))
            fig_fin.update_layout(
                title=f"Financial Summary ({curr} Billions, Latest Year)",
                paper_bgcolor="#0F0F1A", plot_bgcolor="#0D1117",
                font=dict(color="white"), height=350,
                yaxis=dict(title=f"{curr} Billions", gridcolor="#21262D"),
                xaxis=dict(gridcolor="#21262D"),
            )
            st.plotly_chart(fig_fin, use_container_width=True)

# ── TAB 3: News & Sentiment ───────────────────────────────────────────────────
with tab3:
    sent_score = news_data.get("sentiment_score", 0)
    sent_label = news_data.get("sentiment_label", "N/A")
    breakdown  = news_data.get("sentiment_breakdown", {})
    themes     = news_data.get("key_themes", [])

    col_g, col_b = st.columns([1, 2])
    with col_g:
        fig_sent = go.Figure(go.Indicator(
            mode="gauge+number",
            value=(sent_score + 1) * 50,
            title={"text": f"Sentiment: {sent_label}", "font": {"color":"white"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor":"white",
                         "tickvals": [0,25,50,75,100],
                         "ticktext": ["Bearish","","Neutral","","Bullish"]},
                "bar":  {"color": "#6C3EF4"},
                "steps": [
                    {"range": [0, 33],  "color": "#7f1d1d"},
                    {"range": [33, 66], "color": "#374151"},
                    {"range": [66, 100],"color": "#166534"},
                ],
            },
            number={"suffix": "", "font": {"color":"white"},
                    "valueformat": ".0f"},
        ))
        fig_sent.update_layout(paper_bgcolor="#0F0F1A", font=dict(color="white"), height=280)
        st.plotly_chart(fig_sent, use_container_width=True)

    with col_b:
        if breakdown:
            fig_pie = go.Figure(go.Pie(
                labels=["Positive","Neutral","Negative"],
                values=[breakdown.get("positive",0),
                        breakdown.get("neutral",0),
                        breakdown.get("negative",0)],
                marker_colors=["#00C896","#6B7280","#FF6B6B"],
                hole=0.5,
            ))
            fig_pie.update_layout(
                title="News Breakdown",
                paper_bgcolor="#0F0F1A",
                font=dict(color="white"), height=280,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    if themes:
        st.markdown("**Key Market Themes:**")
        cols = st.columns(min(len(themes), 4))
        for i, theme in enumerate(themes[:4]):
            cols[i].markdown(f"<span style='background:#1A1A2E;padding:4px 12px;border-radius:20px;border:1px solid #6C3EF4;font-size:12px'>{theme}</span>", unsafe_allow_html=True)
        st.markdown("")

    articles = news_data.get("articles", [])
    if articles:
        st.markdown("### 📰 Recent Headlines")
        for art in articles[:15]:
            sent = art.get("sentiment_label","Neutral")
            tag = "tag-pos" if "Positive" in sent else "tag-neg" if "Negative" in sent else "tag-neu"
            title = art.get("title","")
            date_str = str(art.get("published",""))[:10]
            source = art.get("source","")
            url = art.get("url","#")
            st.markdown(f"""
<div class='news-row'>
<span style='font-size:13px'><a href='{url}' target='_blank' style='color:#c0c0ff;text-decoration:none'>{title}</a></span>
<span><span class='{tag}'>{sent}</span>&nbsp;<small style='color:#666'>{source} · {date_str}</small></span>
</div>""", unsafe_allow_html=True)

# ── TAB 4: Full Report ────────────────────────────────────────────────────────
with tab4:
    company_display = stock_data.get("company_name", ticker)
    st.markdown(f"# {company_display} ({ticker}) — Investment Report")
    st.markdown(f"*Generated {datetime.now().strftime('%B %d, %Y at %H:%M')}*")

    if report_data.get("executive_summary"):
        st.markdown("## Executive Summary")
        st.markdown(report_data["executive_summary"])

    if report_data.get("market_research_section"):
        st.markdown("## Market Research")
        st.markdown(report_data["market_research_section"])

    if report_data.get("data_analysis_section"):
        st.markdown("## Data Analysis")
        st.markdown(report_data["data_analysis_section"])

    st.markdown("## Investment Recommendation")
    rec_col, conf_col, tgt_col = st.columns(3)
    rec_col.metric("Recommendation", rec)
    conf_col.metric("Confidence", f"{conf:.0f}%")
    tgt_col.metric("12M Target Price", f"{sym}{tgt}" if tgt else "N/A")

    if report_data.get("risk_factors"):
        st.markdown("### ⚠️ Risk Factors")
        for r in report_data["risk_factors"]:
            st.markdown(f"- {r}")

    if report_data.get("catalysts"):
        st.markdown("### 🚀 Potential Catalysts")
        for c in report_data["catalysts"]:
            st.markdown(f"- {c}")

    # PDF download
    pdf_path = report_data.get("pdf_path")
    if pdf_path and Path(pdf_path).exists():
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download PDF Report",
                data=f.read(),
                file_name=Path(pdf_path).name,
                mime="application/pdf",
                type="primary",
            )
    else:
        st.info("PDF report not available.")

# ── TAB 5: Agent Log ──────────────────────────────────────────────────────────
with tab5:
    st.markdown("### 🤖 Inter-Agent Conversation Log")
    st.caption("Watch how the AI agents collaborated to produce this report.")
    messages = state.get("messages", [])
    agent_colors = {
        "Market Researcher": "#6C3EF4",
        "Data Analyst":      "#00C896",
        "Report Writer":     "#FF9500",
        "PDF Generator":     "#FF6B6B",
    }
    for msg in messages:
        agent  = msg.get("agent","System")
        content= msg.get("content","")
        ts     = msg.get("timestamp","")
        status = msg.get("status","running")
        color  = agent_colors.get(agent,"#aaaaff")
        icon   = "✅" if status == "done" else "⚠️" if status == "error" else "🔄"
        st.markdown(f"""
<div class='agent-msg' style='border-left-color:{color}'>
<span style='color:{color};font-size:10px;font-weight:700'>[{ts}] {agent}</span>
<span style='float:right;font-size:11px'>{icon}</span><br>
<span style='font-size:13px'>{content}</span>
</div>""", unsafe_allow_html=True)

    if errors:
        st.markdown("### Errors")
        for e in errors:
            st.error(e)
