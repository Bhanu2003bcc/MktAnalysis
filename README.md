# 📈 Financial Analysis Agent Crew

> A production-grade, multi-agent AI system that collaborates to generate comprehensive, professional stock analysis reports — powered by **Google Gemini** and **LangGraph**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-purple?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-1.5--Pro-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🏗️ Architecture Overview

```
User Input (Ticker + Date Range)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│              LangGraph Orchestrator                  │
│                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────┐   ┌──────────────┐  │
│  │   Market     │───▶│    Data      │───▶│    Report     │──▶│     PDF      │  │
│  │  Researcher  │    │   Analyst    │    │    Writer     │   │  Generator   │  │
│  │              │    │              │    │  (Gemini LLM) │   │              │  │
│  │ • yfinance   │    │ • KPIs       │    │ • Exec Summary│   │ • fpdf2      │  │
│  │ • RSS news   │    │ • 5 charts   │    │ • Rec. report │   │ • 7 pages    │  │
│  │ • VADER NLP  │    │ • Technicals │    │ • Risk/Cats.  │   │ • Branded    │  │
│  └──────────────┘    └──────────────┘    └───────────────┘   └──────────────┘  │
└─────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────┐
  │   Outputs                   │
  │  • Streamlit Dashboard      │
  │  • PDF Report (7 pages)     │
  │  • Agent Conversation Log   │
  └─────────────────────────────┘
```

---

## 🤖 The Three Agents

### 1. 🔍 Market Researcher Agent
- Fetches real-time stock data via **Yahoo Finance** (price, volume, fundamentals)
- Scrapes **news articles** from multiple RSS feeds (Yahoo Finance, Reuters, etc.)
- Runs **VADER sentiment analysis** on all headlines
- Extracts key market themes (Earnings, Macro, Analyst Ratings, etc.)
- Summarises balance sheet, income statement, and cash flow data

### 2. 🧮 Data Analyst Agent
- Computes **19 fundamental KPIs** with benchmark signals (P/E, ROE, D/E, etc.)
- Generates **5 professional charts** (price history, KPIs, returns, sentiment, financials)
- Calculates moving averages (20-day, 50-day), RSI, annualised volatility
- Determines price trend (Uptrend / Downtrend / Sideways) and volatility regime
- Assigns an overall **Fundamental Health Score (0–100)**

### 3. ✍️ Report Writer Agent (Gemini)
- Uses **Gemini 1.5 Pro** to write 4 professional sections
- Synthesises all data into an executive summary, research narrative, and data analysis
- Outputs a structured **BUY / HOLD / SELL recommendation** with confidence score, 12-month target price, risk factors, and catalysts
- Includes retry logic with exponential backoff

---

## 📁 Project Structure

```
financial_agent_crew/
│
├── agents/
│   ├── market_researcher.py   # Agent 1: data + news gathering
│   ├── data_analyst.py        # Agent 2: KPIs, charts, insights
│   └── report_writer.py       # Agent 3: Gemini LLM report writing
│
├── graph/
│   ├── state.py               # LangGraph shared state schema
│   └── workflow.py            # Graph assembly + routing logic
│
├── tools/
│   ├── financial_tools.py     # yfinance wrappers + KPI computation
│   ├── news_tools.py          # RSS scraping + VADER sentiment
│   └── chart_tools.py         # Matplotlib chart generators
│
├── report/
│   └── pdf_generator.py       # 7-page branded PDF (fpdf2)
│
├── ui/
│   └── app.py                 # Streamlit dashboard
│
├── reports/                   # Auto-created: PDF + chart outputs
│   └── charts/
│
├── config.py                  # Central config (paths, models, constants)
├── run.py                     # CLI runner
├── requirements.txt
├── .env.example               # Template — copy to .env
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- A **Google Gemini API key** — free at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

---

### Step 1: Clone / Download

```bash
git clone https://github.com/yourname/financial-agent-crew.git
cd financial-agent-crew
```

Or if you have the zip file:
```bash
unzip financial_agent_crew.zip
cd financial_agent_crew
```

---

### Step 2: Create a Virtual Environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `langgraph`, `langchain-google-genai` — orchestration
- `google-generativeai` — Gemini API
- `yfinance`, `pandas`, `numpy` — financial data
- `matplotlib`, `plotly` — charting
- `fpdf2`, `Pillow` — PDF generation
- `feedparser`, `beautifulsoup4`, `vaderSentiment` — news + NLP
- `streamlit` — web dashboard

---

### Step 4: Configure API Key

```bash
cp .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

---

### Step 5: Run

#### Option A — Streamlit Dashboard (Recommended)

```bash
streamlit run ui/app.py
```

Opens at `http://localhost:8501`. Enter your API key (if not in `.env`), ticker, and date range.

#### Option B — Command Line

```bash
python run.py --ticker AAPL --start 2024-01-01 --end 2025-01-01
```

```bash
# Full options
python run.py \
  --ticker NVDA \
  --name "NVIDIA Corporation" \
  --start 2024-01-01 \
  --end 2025-01-01 \
  --api-key YOUR_KEY_HERE
```

---

## 🖥️ Streamlit Dashboard Features

| Tab | Contents |
|-----|----------|
| **📈 Price & Charts** | Interactive Plotly price chart with MA overlays, volume bars, returns histogram |
| **🧮 KPI Analysis** | Full KPI table with signal indicators, health score gauge, financials bar chart |
| **📰 News & Sentiment** | Sentiment gauge, article breakdown pie chart, themed headlines feed |
| **📝 Full Report** | Complete Gemini-written report with download button for PDF |
| **🤖 Agent Log** | Live colour-coded inter-agent conversation feed |

---

## 📊 Sample Output

### PDF Report (7 pages)
1. **Cover Page** — company overview, quick stats, recommendation badge
2. **Executive Summary** — Gemini-written overview, key metrics table
3. **Price Charts** — price history, KPI chart, returns distribution
4. **KPI Table** — all 19 metrics with colour-coded signals
5. **Market Research** — news sentiment, top headlines
6. **Recommendation** — BUY/HOLD/SELL with target price, risk factors, catalysts
7. **Agent Log** — full inter-agent conversation transcript

### CLI Output
```
============================================================
  ANALYSIS COMPLETE: Apple Inc. (AAPL)
============================================================
  Recommendation : BUY
  Confidence     : 72%
  Target Price   : $215
  Fund. Score    : 68/100
  Price Trend    : Uptrend 📈
  Sentiment      : Bullish 🟢
  PDF Report     : ./reports/AAPL_analysis_20250601_143022.pdf
============================================================
```

---

## ⚙️ Configuration Options

All settings live in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *(required)* | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-1.5-pro` | Model to use |
| `GEMINI_TEMPERATURE` | `0.2` | LLM temperature (0-1) |
| `REPORT_OUTPUT_DIR` | `./reports` | Where PDFs are saved |
| `CHART_OUTPUT_DIR` | `./reports/charts` | Where chart PNGs are saved |
| `MAX_RETRIES` | `3` | Gemini API retry attempts |
| `AGENT_TIMEOUT_SECONDS` | `120` | Per-agent timeout |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## 🔑 Getting a Free Gemini API Key

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API key"**
4. Copy the key and paste it into your `.env` file or the Streamlit sidebar

> The free tier allows ~15 requests/minute and 1 million tokens/day — more than enough for this project.

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `langgraph` | Multi-agent workflow orchestration |
| `langchain-google-genai` | Gemini integration for LangChain |
| `google-generativeai` | Direct Gemini API access |
| `yfinance` | Real-time stock & financial data |
| `vaderSentiment` | Financial news sentiment scoring |
| `feedparser` | RSS news feed parsing |
| `matplotlib` | Static chart generation (PNG for PDF) |
| `plotly` | Interactive charts in Streamlit |
| `fpdf2` | Professional PDF generation |
| `streamlit` | Interactive web dashboard |
| `tenacity` | Retry logic for API calls |
| `loguru` | Structured logging |

---

## 🛠️ Extending the Project

### Add a new data source
Create a function in `tools/financial_tools.py` or `tools/news_tools.py` and call it from the appropriate agent.

### Add a new chart
Add a function to `tools/chart_tools.py` and call it in `agents/data_analyst.py`. The PDF generator picks up all chart paths automatically.

### Change the LLM prompt
Edit `agents/report_writer.py`. Each section has its own prompt string — easy to customise.

### Add a new agent
1. Create `agents/new_agent.py` with a `_node(state) -> state` function
2. Add it to `graph/workflow.py`:
```python
graph.add_node("new_agent", new_agent_node)
graph.add_edge("report_writer", "new_agent")
graph.add_edge("new_agent", "pdf_generator")
```

### Use a different LLM
Replace `google-generativeai` with any LangChain-compatible LLM in `agents/report_writer.py`. The rest of the pipeline is model-agnostic.

---

## ⚠️ Disclaimer

> This tool is for **educational and informational purposes only**. It does not constitute financial advice. The AI-generated recommendations should not be used as the sole basis for investment decisions. Always consult a qualified financial advisor before making investment decisions.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph) — agent orchestration
- [yfinance](https://github.com/ranaroussi/yfinance) — financial data
- [Google Gemini](https://aistudio.google.com) — LLM reasoning
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) — NLP sentiment
- [fpdf2](https://py-fpdf2.readthedocs.io) — PDF generation
# MktAnalysis
