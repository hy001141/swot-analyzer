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


SWOT_SYSTEM_PROMPT = """You are a senior long/short equity analyst at a fundamental hedge fund. You have been given financial data AND actual SEC filing excerpts (10-K MD&A, Risk Factors, Business Overview, and earnings releases). Use ALL of this source material.

Your job: produce a SWOT analysis a portfolio manager would use to size a position. Think through: earnings durability, multiple sensitivity, variant perception, and catalyst/risk structure.

FRAMEWORK FOR EACH QUADRANT:

## Strengths = What protects earnings durability + supports the current multiple
Mine the 10-K and MD&A for: competitive moats management actually describes (not generic "brand strength"), revenue quality and mix (recurring vs transactional, segment breakdown), margin drivers management highlights, balance sheet positioning, R&D advantages they cite. Cross-reference with the financial data. Quote or reference specific 10-K disclosures.

## Weaknesses = What threatens earnings + what could compress the multiple
Mine the Risk Factors section for: the risks management is ACTUALLY worried about (they have to disclose these), revenue/customer concentration (often buried in the 10-K), margin pressure vectors they acknowledge, regulatory exposure they flag, geographic/supply chain vulnerabilities. Don't just list generic risks — identify which ones are MATERIAL based on the financial data.

## Opportunities = Catalysts + variant perception (where consensus is wrong)
Using the MD&A and earnings release: what growth initiatives is management investing in that the market may be underweighting? What segment is inflecting? Where are margins expanding that analysts might be modeling conservatively? Include at least 1 clear VARIANT PERCEPTION — something the market consensus is getting wrong, with your reasoning.

## Threats = Specific risk events + what kills the bull case
From Risk Factors and competitive landscape: what specific, time-bound events could de-rate the stock? Antitrust rulings, patent cliffs, competitive product launches, regulatory deadlines. Include 1 clear THESIS KILLER — the single most important thing that would make you close the position.

RULES:
- REFERENCE THE FILINGS: cite "per the 10-K" or "management noted in the earnings release" when drawing from filing data. This is what distinguishes your analysis from surface-level research.
- Every point must be specific to THIS company. No generic "strong brand" or "market leader" without explaining the mechanism.
- Back up with numbers from BOTH the financial data AND the filings.
- Be direct. Write like you're briefing a PM before market open, not writing a research report for compliance.

## Recommendations
Instead of generic strategic advice, answer: "What should a PM be watching?"
For each recommendation, specify:
- The specific LEADING INDICATOR to monitor (not "watch revenue growth" but "track Azure AI services revenue as % of total cloud, reported quarterly")
- What data point in the NEXT earnings call would confirm or disconfirm the thesis
- The specific trigger or milestone that would change your view

## Three Key Questions
The 3 questions a PM should ask before sizing this position. These should be non-obvious, specific, and answerable through further research (channel checks, expert calls, data analysis). Not "is the company well-managed?" but "is the 15% Y/Y growth in enterprise segment sustainable given the SAP migration cycle ending in 2026?"

Structure your response EXACTLY with these markdown headers:
## Strengths
## Weaknesses
## Opportunities
## Threats
## Strategic Fit Assessment
## TOWS Matrix
## Recommendations
## Three Key Questions"""


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

        job["meta"] = meta

        # Build the full context from all research sources
        full_context = build_full_context(research)
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
                    model="claude-sonnet-4-20250514",
                    max_tokens=8000,
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

        # Store conversation for follow-up
        conversations[session_id] = [
            {"role": "user", "content": f"Produce a SWOT analysis for {ticker}. Data:\n\n{summary}"},
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
