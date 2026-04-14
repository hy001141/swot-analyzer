"""
SWOT Analysis Web App
Flask backend with polling-based streaming (works on Render free tier).
"""

import os
import sys
import json
import threading
import time
import uuid

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from flask import Flask, render_template, request, jsonify
import yfinance as yf
import anthropic

app = Flask(__name__)

# ── In-memory stores ─────────────────────────────────────────────────────

# Active analysis jobs: job_id -> { status, text, meta, error, done }
jobs: dict[str, dict] = {}
# Conversations for follow-up Q&A: session_id -> [messages]
conversations: dict[str, list] = {}


# ── Data Gathering ───────────────────────────────────────────────────────

def fetch_company_data(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    data = {}

    try:
        data["info"] = stock.info
    except Exception:
        data["info"] = {}

    for name, attr in [
        ("income_stmt", "income_stmt"),
        ("balance_sheet", "balance_sheet"),
        ("cashflow", "cashflow"),
        ("quarterly_income", "quarterly_income_stmt"),
        ("quarterly_balance", "quarterly_balance_sheet"),
        ("quarterly_cashflow", "quarterly_cashflow"),
    ]:
        try:
            df = getattr(stock, attr)
            data[name] = df.to_dict() if (df is not None and not df.empty) else {}
        except Exception:
            data[name] = {}

    try:
        rec = stock.recommendations
        data["recommendations"] = rec.tail(20).to_dict() if (rec is not None and not rec.empty) else {}
    except Exception:
        data["recommendations"] = {}

    try:
        inst = stock.institutional_holders
        data["institutional_holders"] = inst.head(10).to_dict() if (inst is not None and not inst.empty) else {}
    except Exception:
        data["institutional_holders"] = {}

    try:
        news = stock.news
        data["news"] = news[:10] if news else []
    except Exception:
        data["news"] = []

    try:
        hist = stock.history(period="1y")
        if hist is not None and not hist.empty:
            data["price_history_summary"] = {
                "current_price": round(float(hist["Close"].iloc[-1]), 2),
                "52w_high": round(float(hist["Close"].max()), 2),
                "52w_low": round(float(hist["Close"].min()), 2),
                "avg_volume": int(hist["Volume"].mean()),
                "price_change_1y_pct": round(
                    (hist["Close"].iloc[-1] - hist["Close"].iloc[0])
                    / hist["Close"].iloc[0] * 100, 2
                ),
            }
        else:
            data["price_history_summary"] = {}
    except Exception:
        data["price_history_summary"] = {}

    return data


def build_data_summary(data: dict) -> str:
    info = data.get("info", {})
    lines = []

    lines.append(f"COMPANY: {info.get('longName', 'N/A')} ({info.get('symbol', 'N/A')})")
    lines.append(f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}")
    lines.append(f"Country: {info.get('country', 'N/A')} | Employees: {info.get('fullTimeEmployees', 'N/A')}")

    if info.get("longBusinessSummary"):
        lines.append(f"\nBUSINESS SUMMARY:\n{info['longBusinessSummary']}")

    lines.append("\nKEY FINANCIAL METRICS:")
    metrics = [
        ("Market Cap", "marketCap"), ("Enterprise Value", "enterpriseValue"),
        ("Revenue (TTM)", "totalRevenue"), ("Net Income", "netIncomeToCommon"),
        ("EBITDA", "ebitda"), ("Gross Margin", "grossMargins"),
        ("Operating Margin", "operatingMargins"), ("Profit Margin", "profitMargins"),
        ("ROE", "returnOnEquity"), ("ROA", "returnOnAssets"),
        ("Revenue Growth", "revenueGrowth"), ("Earnings Growth", "earningsGrowth"),
        ("Total Cash", "totalCash"), ("Total Debt", "totalDebt"),
        ("Debt/Equity", "debtToEquity"), ("Current Ratio", "currentRatio"),
        ("Free Cash Flow", "freeCashflow"), ("P/E (Trailing)", "trailingPE"),
        ("P/E (Forward)", "forwardPE"), ("Beta", "beta"),
        ("Dividend Yield", "dividendYield"), ("EV/EBITDA", "enterpriseToEbitda"),
    ]
    for label, key in metrics:
        val = info.get(key)
        if val is not None:
            if isinstance(val, (int, float)) and abs(val) > 1_000_000:
                lines.append(f"  {label}: ${val:,.0f}")
            else:
                lines.append(f"  {label}: {val}")

    ph = data.get("price_history_summary", {})
    if ph:
        lines.append("\nPRICE HISTORY (1Y):")
        for k, v in ph.items():
            lines.append(f"  {k}: {v}")

    news = data.get("news", [])
    if news:
        lines.append("\nRECENT NEWS:")
        for item in news:
            if isinstance(item, dict):
                title = item.get("title", item.get("content", {}).get("title", ""))
                if title:
                    lines.append(f"  - {title}")

    return "\n".join(lines)


SWOT_SYSTEM_PROMPT = """You are a senior equity research analyst and strategic consultant.
Produce rigorous, data-driven SWOT analyses grounded in the financial data provided.

Structure your response EXACTLY as follows using these markdown headers:

## Strengths
- [4-6 bullet points, data-backed, internal factors]

## Weaknesses
- [4-6 bullet points, data-backed, internal factors]

## Opportunities
- [4-6 bullet points, external factors]

## Threats
- [4-6 bullet points, external factors]

## Strategic Fit Assessment
[2-3 paragraphs on how internal strengths align with external opportunities]

## TOWS Matrix
**SO Strategy (Maximize Strengths + Opportunities):** ...
**WO Strategy (Minimize Weaknesses, Maximize Opportunities):** ...
**ST Strategy (Use Strengths to Counter Threats):** ...
**WT Strategy (Minimize Weaknesses and Threats):** ...

## Recommendations
1. [Specific, actionable recommendation]
2. ...
3. ...

## Three Key Questions
1. [Most important question for further investigation]
2. ...
3. ...

Be specific and cite actual numbers (margins, growth rates, ratios) from the data provided."""


# ── Background worker ────────────────────────────────────────────────────

def run_analysis_worker(job_id: str, ticker: str, session_id: str):
    """Run the full analysis in a background thread."""
    job = jobs[job_id]
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    try:
        # Step 1: Fetch data
        job["status"] = "Fetching financial data from Yahoo Finance..."
        data = fetch_company_data(ticker)

        info = data.get("info", {})
        has_data = (info.get("longName") or info.get("shortName") or
                    info.get("symbol") or info.get("regularMarketPrice") or
                    data.get("price_history_summary"))

        if not has_data:
            job["error"] = f"Could not fetch data for '{ticker}'. Check the ticker symbol."
            job["done"] = True
            return

        if not info.get("longName"):
            info["longName"] = info.get("shortName") or ticker

        # Send meta — try multiple price fields as fallback
        price = (info.get("currentPrice") or info.get("regularMarketPrice")
                 or info.get("regularMarketOpen") or info.get("previousClose"))
        ph = data.get("price_history_summary", {})
        if not price and ph.get("current_price"):
            price = ph["current_price"]

        job["meta"] = {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "marketCap": info.get("marketCap"),
            "price": price,
            "priceChange": ph.get("price_change_1y_pct"),
        }

        # Step 2: Build summary
        job["status"] = "Building financial summary..."
        summary = build_data_summary(data)

        # Step 3: Generate SWOT with Claude
        job["status"] = "Generating SWOT analysis with Claude AI..."
        client = anthropic.Anthropic(api_key=api_key)

        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            system=SWOT_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Produce a comprehensive SWOT analysis for {ticker} based on this data:\n\n{summary}"
            }]
        ) as stream:
            for text_chunk in stream.text_stream:
                job["text"] += text_chunk

        # Store conversation for follow-up
        conversations[session_id] = [
            {"role": "user", "content": f"Produce a SWOT analysis for {ticker}. Data:\n\n{summary}"},
            {"role": "assistant", "content": job["text"]},
        ]

    except Exception as e:
        print(f"[ERROR] Analysis failed for {ticker}: {e}", flush=True)
        job["error"] = str(e)
    finally:
        # Always mark done so frontend stops polling
        # Store partial conversation even on error so chat can still work
        if job["text"] and session_id not in conversations:
            conversations[session_id] = [
                {"role": "user", "content": f"Produce a SWOT analysis for {ticker}."},
                {"role": "assistant", "content": job["text"]},
            ]
        job["status"] = ""
        job["done"] = True


def run_chat_worker(job_id: str, session_id: str, question: str):
    """Run a follow-up chat in a background thread."""
    job = jobs[job_id]
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    history = conversations.get(session_id, [])
    if not history:
        job["error"] = "No analysis found. Run an analysis first."
        job["done"] = True
        return

    history.append({"role": "user", "content": question})

    try:
        client = anthropic.Anthropic(api_key=api_key)

        CHAT_PROMPT = """You are a sharp, direct equity research analyst answering follow-up questions.
Rules:
- Be direct and concise. No fluff, no filler.
- Write like a human analyst would in a quick Slack message, not a research report.
- Use short paragraphs (2-3 sentences max each).
- Total response should be 150-300 words max.
- Use numbers and data when relevant but don't over-explain.
- No headers or bullet points unless truly necessary.
- Never start with "Great question" or similar."""

        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            system=CHAT_PROMPT,
            messages=history
        ) as stream:
            for chunk in stream.text_stream:
                job["text"] += chunk

        history.append({"role": "assistant", "content": job["text"]})
        conversations[session_id] = history
        job["done"] = True

    except Exception as e:
        job["error"] = str(e)
        job["done"] = True


# ── Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Start an analysis job in background, return job_id for polling."""
    body = request.get_json()
    ticker = body.get("ticker", "").strip().upper()
    session_id = body.get("session_id", ticker)

    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 500

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "Starting...",
        "text": "",
        "meta": None,
        "error": None,
        "done": False,
        "cursor": 0,  # tracks how much text the client has received
    }

    thread = threading.Thread(target=run_analysis_worker, args=(job_id, ticker, session_id), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/poll/<job_id>")
def poll(job_id):
    """Poll for new text from a running job. Returns only new text since last poll."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    cursor = int(request.args.get("cursor", 0))
    new_text = job["text"][cursor:]

    return jsonify({
        "status": job["status"],
        "text": new_text,
        "cursor": len(job["text"]),
        "meta": job["meta"],
        "error": job["error"],
        "done": job["done"],
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """Start a chat job in background, return job_id for polling."""
    body = request.get_json()
    session_id = body.get("session_id", "")
    question = body.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 500

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "Thinking...",
        "text": "",
        "meta": None,
        "error": None,
        "done": False,
    }

    thread = threading.Thread(target=run_chat_worker, args=(job_id, session_id, question), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting SWOT Analyzer at http://localhost:{port}")
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
