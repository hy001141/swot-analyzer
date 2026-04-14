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
from sec_fetcher import fetch_sec_data, build_sec_summary
from research_agent import run_full_research, build_full_context

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

    # fast_info is more reliable on cloud servers
    try:
        fi = stock.fast_info
        data["fast_info"] = {}
        try: data["fast_info"]["marketCap"] = fi.market_cap
        except: pass
        try: data["fast_info"]["shares"] = fi.shares
        except: pass
        try: data["fast_info"]["lastPrice"] = fi.last_price
        except: pass
        try: data["fast_info"]["previousClose"] = fi.previous_close
        except: pass
    except Exception:
        data["fast_info"] = {}

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


SWOT_SYSTEM_PROMPT = """You are a senior long/short equity analyst at a top-tier fundamental hedge fund. You have been given a comprehensive research package assembled from up to 16 independent data sources. Your job: produce the kind of SWOT analysis that would survive a PM's scrutiny in a Monday morning pitch meeting.

YOU HAVE ACCESS TO THESE DATA SOURCES — USE ALL OF THEM:
1. Yahoo Finance: financials, valuation ratios, margins, growth rates, price history
2. SEC 10-K filing: full text including Business Overview, Risk Factors, MD&A, Financial Statements
3. SEC 8-K: latest earnings press release with quarterly results and guidance
4. Insider transactions: recent buys/sells by officers and directors — are insiders buying or dumping?
5. Short interest & options: short % of float, put/call ratio — how is the market positioned?
6. Analyst estimates: consensus EPS, price targets, recommendation trends, estimate revisions
7. Earnings surprise history: does management consistently beat or miss? By how much?
17. COMPETITOR COMP TABLE: side-by-side valuation and financial metrics vs key peers
18. COMPETITOR 10-K FILINGS: excerpts from competitors' annual filings — what do THEY say about the competitive landscape, their own risks, and their strategy? This reveals dynamics the target company's 10-K won't tell you.
8. Institutional ownership: who are the top holders? Are they adding or trimming?
9. FRED macro data: fed funds rate, CPI, GDP, unemployment, VIX, yield curve — what's the macro backdrop?
10. App Store rankings: for consumer tech companies — user ratings, review velocity
11-16. Web research: company blog posts, patent filings, job postings, congressional trading, competitor moves, supply chain signals (via targeted web scraping)

FRAMEWORK — Think like a PM sizing a position:

## Strengths = What protects earnings durability + supports the multiple
- Mine the 10-K Business section and MD&A for competitive moats management describes
- Cross-reference with financial data: are the margins/growth actually supporting what management claims?
- Check insider transactions: are insiders buying? That's conviction
- Check institutional ownership: are smart money holders adding?
- Check earnings history: consistent beats = management sandbagging (bullish)
- Reference specific 10-K language and financial metrics

## Weaknesses = What threatens earnings + could compress the multiple
- Mine 10-K Risk Factors for what management is legally required to disclose as material risks
- Check short interest: high short % = market skepticism you need to understand
- Check analyst estimate trends: are estimates being revised down?
- Check insider selling patterns: heavy insider sales = red flag
- Look at web research for competitive threats, regulatory actions, supply chain issues
- Identify which risks are MATERIAL based on the numbers, not just disclosed boilerplate

## Opportunities = Catalysts + variant perception (where consensus is wrong)
- From MD&A and earnings release: what growth initiatives is management investing in?
- From web research: what non-obvious signals (patents, hiring, blog posts) suggest upcoming catalysts?
- From analyst estimates: where might consensus be too conservative?
- Check app store data for consumer engagement trends
- Check macro backdrop: does the rate environment favor this company?
- REQUIRED: Include at least 1 clear VARIANT PERCEPTION with evidence — something specific the market is getting wrong

## Threats = Risk events + what kills the bull case
- From 10-K Risk Factors: specific, time-bound risk events (antitrust rulings, patent expirations, regulatory deadlines)
- From web research: competitor launches, regulatory actions, geopolitical exposure
- From macro data: how sensitive is this business to rate changes, recession?
- From short interest/options: what is the bearish positioning telling you?
- REQUIRED: Include 1 clear THESIS KILLER — the single thing that would make you close the position

## Strategic Fit Assessment
How do internal capabilities align with external opportunities? Where is the strategic tension?

## TOWS Matrix
SO/WO/ST/WT strategies — each one sentence, specific and actionable.

## Recommendations — "What should a PM be watching?"
For each of 3-4 recommendations:
- The specific LEADING INDICATOR to monitor (not "watch revenue" but "track data center revenue as % of total, specifically the inference vs training mix shift disclosed in 10-K segment reporting")
- What data point in the NEXT earnings call would confirm or disconfirm
- The specific trigger that changes your view
- Reference which of the 16 data sources informs this recommendation

## Three Key Questions
3 non-obvious, specific questions answerable through further research (channel checks, expert calls, data analysis). Each should reference specific data from the research package that prompted the question.

RULES:
- CITE YOUR SOURCES: "per the 10-K", "insider transactions show", "short interest at X% suggests", "web research indicates", "analyst consensus expects", "competitor X's 10-K states". Every claim needs a source.
- COMPETITIVE CONTEXT IS MANDATORY: reference the comp table and competitor filings throughout. "Trading at 16x EV/EBITDA vs peer median of 22x" is the kind of relative valuation context every point needs. "Competitor X's 10-K flags [risk] as a key strategic priority, which directly threatens [company]'s position in [segment]" — this is the depth that separates real analysis from surface level.
- Cross-reference sources: if insiders are buying but short interest is rising, call that out. If the company claims AI leadership but a competitor's patent filings suggest otherwise, flag it. If management guidance is aggressive but analyst estimates are flat, explain the disconnect.
- Be specific to THIS company. No generic points.
- Write like you're briefing a PM who will challenge every point. Be direct, be specific, defend your reasoning.
- This analysis should be IMPOSSIBLE to produce from Yahoo Finance alone. If it reads like a Yahoo Finance summary, you've failed."""


# ── Background worker ────────────────────────────────────────────────────

def run_analysis_worker(job_id: str, ticker: str, session_id: str):
    """Run the full analysis in a background thread."""
    job = jobs[job_id]
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    try:
        # Step 1: Run full research pipeline (all sources in parallel)
        def research_status(msg):
            job["status"] = msg

        research = run_full_research(ticker, status_callback=research_status)

        # Check if we got any data at all
        meta = research.get("meta", {})
        if not meta.get("name") or meta["name"] == ticker:
            # Try basic yfinance as fallback
            try:
                stock = yf.Ticker(ticker)
                info = stock.info or {}
                if not info.get("longName"):
                    job["error"] = f"Could not fetch data for '{ticker}'. Check the ticker symbol."
                    job["done"] = True
                    return
                meta["name"] = info.get("longName", ticker)
                meta["sector"] = info.get("sector", "")
                meta["industry"] = info.get("industry", "")
            except:
                job["error"] = f"Could not fetch data for '{ticker}'. Check the ticker symbol."
                job["done"] = True
                return

        # Add comp table and sources to meta for frontend
        meta["comp_table"] = research.get("comp_table", "")
        meta["sources"] = research.get("sources_succeeded", [])
        job["meta"] = meta

        # Build the full context — Haiku digests raw sources into clean briefs
        full_context = build_full_context(research, status_callback=research_status)
        sources = research.get("sources_succeeded", [])
        failed = research.get("sources_failed", [])
        print(f"[RESEARCH] {ticker}: {len(sources)} sources OK ({', '.join(sources)}), {len(failed)} failed ({', '.join(failed)}), context={len(full_context)} chars", flush=True)

        # Step 2: Generate SWOT with Claude (retry on overload)
        job["status"] = f"Analyzing with QuantumIQ AI ({len(sources)} sources)..."
        client = anthropic.Anthropic(api_key=api_key)
        max_api_retries = 3

        for attempt in range(max_api_retries):
            try:
                job["text"] = ""  # reset on retry
                with client.messages.stream(
                    model="claude-opus-4-20250514",
                    max_tokens=16000,
                    system=SWOT_SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": f"Produce a comprehensive SWOT analysis for {ticker} ({meta.get('name', '')}).\n\nThe following research has been gathered from {len(sources)} sources: {', '.join(sources)}.\n\n{full_context}"
                }]
                ) as stream:
                    for text_chunk in stream.text_stream:
                        job["text"] += text_chunk
                break  # success, exit retry loop

            except Exception as api_err:
                is_overloaded = "overloaded" in str(api_err).lower() or "529" in str(api_err)
                if is_overloaded and attempt < max_api_retries - 1:
                    import time
                    job["status"] = f"Server busy, retrying ({attempt + 2}/{max_api_retries})..."
                    time.sleep(3)
                    continue
                else:
                    raise api_err

        # Store conversation with FULL context for follow-up questions
        # This lets the chat answer detailed questions about specific 10-K sections, financials, etc.
        conversations[session_id] = [
            {"role": "user", "content": f"Here is the complete research package for {ticker}:\n\n{full_context}\n\nBased on this data, produce a SWOT analysis."},
            {"role": "assistant", "content": job["text"]},
        ]

    except Exception as e:
        print(f"[ERROR] Analysis failed for {ticker}: {e}", flush=True)
        # Clean up error message for user
        err_str = str(e)
        if "overloaded" in err_str.lower():
            job["error"] = "AI servers are temporarily overloaded. Please try again in a few seconds."
        else:
            job["error"] = "Analysis failed. Please try again."
    finally:
        # Always store conversation if we have any text (even partial)
        if job["text"]:
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

    base_history = conversations.get(session_id, [])
    if not base_history:
        job["error"] = "No analysis found. Run an analysis first."
        job["done"] = True
        return

    # Build messages for this request WITHOUT mutating the stored conversation
    messages = list(base_history) + [{"role": "user", "content": question}]

    try:
        client = anthropic.Anthropic(api_key=api_key)

        CHAT_PROMPT = """You are a sharp, direct equity research analyst answering follow-up questions about the company you just analyzed.
Rules:
- Answer ANY question about the company: leadership, CEO, management, products, history, competitors, financials, strategy, news, culture, anything.
- Use your general knowledge about the company freely — you're not limited to just the data provided.
- Only refuse if the question is completely unrelated to business/finance (e.g. "write me a poem", "what's 2+2").
- Be direct and concise. No fluff, no filler.
- Write like a human analyst in a quick Slack message, not a research report.
- Use short paragraphs (2-4 sentences each).
- Total response should be 200-400 words.
- No headers or bullet points unless truly necessary.
- Never start with "Great question" or similar."""

        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            system=CHAT_PROMPT,
            messages=messages
        ) as stream:
            for chunk in stream.text_stream:
                job["text"] += chunk

        # Only update conversation AFTER success
        messages.append({"role": "assistant", "content": job["text"]})
        conversations[session_id] = messages
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
