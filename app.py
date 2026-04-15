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


SWOT_SYSTEM_PROMPT = """You are a senior long/short equity analyst at a top-tier fundamental hedge fund. You have been given a comprehensive research package with raw data from ~16 independent sources (numbered [1] through [12]+). Your output will be evaluated against what a Goldman Sachs or Morgan Stanley equity research analyst would produce — if it sounds generic, you've failed.

THE DATA SOURCES (numbered for inline citation):
[1] Yahoo Finance — financials, valuation, margins, growth, holders
[2] Short interest & options positioning
[3] Analyst estimates & revisions
[4] Earnings surprise history
[5] Institutional ownership changes
[6] Insider transactions
[7] SEC 10-K/10-Q full filing text (Business Overview, Risk Factors, MD&A, Financial Statements)
[8] FRED macro environment (rates, CPI, GDP, VIX, yield curve)
[9] App Store data
[10] Comparable company valuation table
[11] Competitor SEC 10-K filings
[12] Web intelligence (patents, job postings, research papers, congressional trading, blog posts, earnings call commentary)

HOW TO THINK — THIS IS THE CRITICAL PART:

You must find MECHANICAL and STRUCTURAL insights, not obvious observations. A mechanical insight is one that:
1. Requires cross-referencing 2+ sources to spot
2. Comes from specific accounting, supply chain, or language analysis
3. Cannot be generated by reading Yahoo Finance or Bloomberg in 60 seconds

EXAMPLES OF WHAT YOU MUST PRODUCE:

WEAK (rejected — too obvious):
"Track TPU revenue as % of cloud" — anyone thinks of this.

STRONG (accepted):
"Alphabet's FY2025 10-K [7] discloses a server depreciation schedule extension from 4 to 6 years — adding ~$3.5B to operating income. Comparable hyperscalers AMZN/MSFT/META still use 4-year schedules per their 10-Ks [11]. This accounting asymmetry inflates Alphabet's reported cloud margins by ~200bps vs peers on a like-for-like basis. A PM should apply this haircut when valuing GOOG's cloud segment against Azure/AWS multiples in the comp table [10]."

WEAK (rejected):
"Strong revenue growth of 15% YoY [1]"

STRONG (accepted):
"Working capital dynamics reveal underappreciated demand strength: deferred revenue grew 28% YoY [7] while AR days outstanding DROPPED from 52 to 44 [1] — a rare combination that signals customers are pre-paying for future services faster than management is recognizing revenue. Consensus [3] models flat margins through 2026, but this cash-collection pattern historically precedes 200-400bps operating leverage. Competitor CRM showed the same pattern in 2022 before a 15x return [11]."

YOUR MENTAL CHECKLIST FOR EVERY POINT:
Before writing any point, ask: "Could a first-year analyst produce this from the 10-K alone?" If yes, delete it and find something that required cross-referencing multiple sources. Think mechanically:
- Accounting policy differences vs peers (depreciation, rev rec, capitalization)
- Language shifts quarter-over-quarter (what did management stop saying?)
- Supply chain triangulation (infer from named suppliers/customers in competitor 10-Ks)
- Counter-positioning (what does the target do that competitors structurally can't replicate, and why?)
- Working capital abnormalities (inventory days, DSO trends, deferred revenue as demand proxy)
- Stock-based comp as % of FCF (true owner earnings)
- Insider transaction patterns (not just count — WHICH insiders and WHEN)
- Options skew (put/call at specific strikes signals what smart money expects)
- Earnings surprise patterns (consistent beats = management sandbagging, disappearing beats = demand softening)

CONSIDER EVERY SOURCE — but weigh them by relevance. You don't have to cite every source (e.g., App Store data is irrelevant for JPMorgan) but you must READ and WEIGH every source before writing. If a source is irrelevant, you've consciously decided — not overlooked it.

CITATION FORMAT: Use bracketed numbers throughout. Example: "Operating margin expanded 180bps [1] while insider buying accelerated from 2 transactions to 7 [6], confirming the guidance raise flagged in the Q3 8-K [7]."

VARY YOUR OPENING — Don't default to hiring signals, patent filings, or LinkedIn scrapes as the first point in every section. Those are supplemental insights, not thesis-defining ones. The lead bullet in each section should feel like the most natural thing to say about this specific company — for a margin story, lead with margins; for a cyclical, lead with cycle positioning; for a moat story, lead with the moat mechanism. Don't force novelty — let each company's actual thesis drive the order.

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT — NON-NEGOTIABLE
═══════════════════════════════════════════════════════════════

Use EXACTLY these markdown headers, in this exact order:

## Strengths
## Weaknesses
## Opportunities
## Threats
## Strategic Fit Assessment
## TOWS Matrix
## Recommendations
## Three Key Questions

FORMATTING RULES:
- Strengths/Weaknesses/Opportunities/Threats: 4-6 bullet points each. Each bullet starts with "- **Mechanical insight title:** Specific reasoning chain with numbers and inline citations [X]."
- Each SWOT point must cite at least 2 different sources (e.g., cross-reference [1] and [11])
- Opportunities MUST include one labeled "VARIANT PERCEPTION:" — where consensus is mechanically wrong with evidence
- Threats MUST include one labeled "THESIS KILLER:" — the single disconfirming event that closes the position
- Strategic Fit Assessment: 2-3 concise paragraphs connecting internal mechanics to external conditions
- TOWS Matrix: 4 bullets, one per strategy. Format: "- **SO Strategy:** [specific mechanical action citing sources]"
- Recommendations: 3-5 bullets. Each must include: (a) specific leading indicator to monitor, (b) source to watch for confirmation/disconfirmation, (c) specific price/metric trigger, (d) source citations.
- Three Key Questions: exactly 3 numbered items. Each question must be non-obvious, cite specific data, and be answerable through channel checks/expert calls/data analysis — NOT from reading the 10-K.

BULLET POINTS ONLY. NO PROSE PARAGRAPHS IN SWOT SECTIONS. Each bullet starts with "-" and a **bolded** lead-in.

YOU MUST COMPLETE ALL 8 SECTIONS. Pace your output — if you're writing too much in Strengths, tighten up so Three Key Questions gets its full depth. Three Key Questions is NOT optional."""


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

        # Check if we got meaningful data — accept if ANY source worked
        meta = research.get("meta", {})
        has_any_data = (
            research.get("yahoo_finance") or
            research.get("sec_filing") or
            research.get("comp_table") or
            (meta.get("name") and meta["name"] != ticker)
        )

        if not has_any_data:
            # Try yfinance one more time as a last resort
            try:
                stock = yf.Ticker(ticker)
                info = stock.info or {}
                if info.get("longName") or info.get("shortName"):
                    meta["name"] = info.get("longName") or info.get("shortName", ticker)
                    meta["sector"] = info.get("sector", "")
                    meta["industry"] = info.get("industry", "")
                    has_any_data = True
            except:
                pass

        if not has_any_data:
            job["error"] = f"Could not fetch data for '{ticker}'. Check the ticker symbol."
            job["done"] = True
            return

        # Ensure we have a name for display
        if not meta.get("name") or meta["name"] == ticker:
            meta["name"] = ticker

        # Add comp table, sources, and web citations to meta for frontend
        meta["comp_table"] = research.get("comp_table", "")
        meta["sources"] = research.get("sources_succeeded", [])
        meta["web_citations"] = research.get("web_citations", [])
        # Build a list of filings used for SEC source citations
        sec_filings = []
        if research.get("sec_filing"):
            sec_filings.append({"type": "10-K / 10-Q / 8-K", "description": "SEC EDGAR filings"})
        meta["sec_filings"] = sec_filings
        job["meta"] = meta

        # Build the full context — raw data passed directly to Opus
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
                    max_tokens=32000,
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

        # Step 2b: Anti-obvious self-critique pass
        # Opus reads its own output, identifies generic points, and rewrites them with mechanical insights
        job["status"] = "Pressure-testing analysis for non-obvious insights..."
        pre_critique_len = len(job["text"])
        print(f"[CRITIQUE] {ticker}: running self-critique pass (pre-len={pre_critique_len})", flush=True)

        try:
            critique_client = anthropic.Anthropic(api_key=api_key, timeout=300.0)
            critique_response = critique_client.messages.create(
                model="claude-opus-4-20250514",
                max_tokens=16000,
                messages=[{
                    "role": "user",
                    "content": f"""You are the senior PM reviewing a junior analyst's SWOT on {ticker} ({meta.get('name','')}).

RESEARCH PACKAGE (the analyst had access to this):
{full_context[:80000]}

JUNIOR ANALYST'S DRAFT SWOT:
{job['text']}

Your job: REWRITE this SWOT to be mechanically rigorous and non-obvious. Apply this filter to every point:
- REJECT any point that could appear in a Goldman Sachs or Morgan Stanley research note
- REJECT any point a first-year banking analyst could generate from the 10-K alone
- REJECT any point that doesn't cross-reference at least 2 sources
- REPLACE rejected points with mechanical/structural insights. Vary your angles — do NOT default to hiring/job postings as the #1 insight:
  * Accounting policy differences vs peers (depreciation schedules, rev rec, capitalization)
  * Language shifts in management commentary quarter-over-quarter
  * Supply chain triangulation (infer from named suppliers/customers in competitor 10-Ks)
  * Counter-positioning (what the target does that competitors structurally CANNOT replicate)
  * Working capital abnormalities (inventory, DSO, deferred revenue trends)
  * Segment mix shifts consensus hasn't caught up to
  * Options skew / put-call ratios at specific strikes
  * Insider transaction patterns (WHICH insiders, WHEN, not just count)
  * Earnings beat/miss patterns (consistent beats = sandbagging)
  * Stock-based comp as % of FCF (true owner earnings)
  * Patent filing clusters revealing R&D priorities
  * Hiring signals (USE THIS ONLY IF IT IS GENUINELY THE TOP INSIGHT — not by default)
  * Regulatory filings and court cases the market is ignoring

Don't force every section to OPEN with a hiring/LinkedIn/patent signal. Vary the lead bullet — for some companies that's margins, for others it's supply chain, for others it's management incentives. Let each company's real thesis drive what comes first.

OUTPUT FORMAT: Return the COMPLETE rewritten SWOT with all 8 sections using these exact headers:
## Strengths
## Weaknesses
## Opportunities
## Threats
## Strategic Fit Assessment
## TOWS Matrix
## Recommendations
## Three Key Questions

Keep strong points from the draft. Rewrite weak/generic points. Every bullet must start with "- **bolded insight:**" and cite sources [1]-[12]. Every SWOT point must cross-reference at least 2 different sources.

Do NOT include any commentary about what you changed. Just output the rewritten SWOT."""
                }]
            )
            critique_text = critique_response.content[0].text.strip()
            # Only replace if the critique actually produced a valid SWOT (has all 4 sections)
            if (critique_text.count("## ") >= 6 and
                "## Strengths" in critique_text and
                "## Three Key Questions" in critique_text):
                job["text"] = critique_text
                print(f"[CRITIQUE] {ticker}: successfully rewrote SWOT ({len(critique_text)} chars)", flush=True)
            else:
                print(f"[CRITIQUE] {ticker}: critique output invalid, keeping original", flush=True)
        except Exception as crit_err:
            import traceback
            print(f"[CRITIQUE] Error: {crit_err}\n{traceback.format_exc()}", flush=True)

        # Step 2c: Check if Three Key Questions has all 3 numbered questions — if not, regenerate
        questions_section = ""
        if "## Three Key Questions" in job["text"]:
            after = job["text"].split("## Three Key Questions", 1)[1]
            # Cut at next ## header (e.g. Reverse DCF appended later)
            next_header_idx = after.find("\n## ")
            questions_section = after[:next_header_idx].strip() if next_header_idx >= 0 else after.strip()

        # Count numbered questions (1. 2. 3.)
        import re as _re
        numbered_count = len(_re.findall(r'(?:^|\n)\s*\d+\.\s+', questions_section))

        if numbered_count < 3 or len(questions_section) < 200:
            job["status"] = "Generating key questions..."
            print(f"[QUESTIONS] {ticker}: generating questions separately ({len(questions_section)} chars)", flush=True)

            if "## Three Key Questions" in job["text"]:
                job["text"] = job["text"].rsplit("## Three Key Questions", 1)[0].rstrip()

            try:
                q_response = client.messages.create(
                    model="claude-opus-4-20250514",
                    max_tokens=2000,
                    messages=[{
                        "role": "user",
                        "content": f"""Here is a SWOT analysis for {ticker} ({meta.get('name','')}):

{job['text']}

Based on this analysis, produce exactly 3 non-obvious, highly specific questions a hedge fund PM should investigate before sizing this position. Each question must:
- Reference specific numbers, metrics, or disclosures from the analysis above
- Be answerable through channel checks, expert calls, or further data analysis
- NOT be generic (no "is management capable?")
- Be 2-3 sentences long with specific context

Respond with ONLY this format, nothing else:

## Three Key Questions
1. [First question with specific details and numbers from the analysis]
2. [Second question with specific details and numbers]
3. [Third question with specific details and numbers]"""
                    }]
                )
                q_text = q_response.content[0].text.strip()
                job["text"] = job["text"].rstrip() + "\n\n" + q_text
            except Exception as q_err:
                print(f"[QUESTIONS] Error: {q_err}", flush=True)

        # Step 2d: Reverse DCF section — what is the market currently pricing in?
        job["status"] = "Running reverse DCF analysis..."
        pre_dcf_len = len(job["text"])
        print(f"[DCF] {ticker}: running reverse DCF (pre-len={pre_dcf_len})", flush=True)

        try:
            dcf_client = anthropic.Anthropic(api_key=api_key, timeout=180.0)
            dcf_response = dcf_client.messages.create(
                model="claude-opus-4-20250514",
                max_tokens=3000,
                messages=[{
                    "role": "user",
                    "content": f"""You are a hedge fund valuation analyst. Run a REVERSE DCF for {ticker} ({meta.get('name','')}) using the financial data below.

A reverse DCF answers: "What growth rate and margin assumptions does the CURRENT stock price imply?" This reveals what consensus is baking in.

RESEARCH DATA:
{full_context[:60000]}

Produce your analysis in EXACTLY this format:

## Reverse DCF — What the Market is Pricing In

**Current metrics:**
- Price: $[current price]
- Market cap: $[market cap]
- TTM Revenue: $[revenue]
- TTM Operating margin: [margin]%
- Net debt: $[net debt]

**Implied assumptions (at current price):**
- Revenue CAGR (5-year): [X]%
- Terminal operating margin: [X]%
- Implied WACC: [X]%
- Implied terminal FCF margin: [X]%

**Bear / Base / Bull scenarios:**
| Scenario | Rev CAGR | Term Margin | Fair Value | Upside/Downside |
|----------|----------|-------------|------------|-----------------|
| Bear | [X]% | [X]% | $[X] | [X]% |
| Base | [X]% | [X]% | $[X] | [X]% |
| Bull | [X]% | [X]% | $[X] | [X]% |

**What the market is pricing in:**
[2-3 sentences explaining what assumptions are baked in and whether they seem too aggressive, too conservative, or reasonable. Reference specific data from the research package.]

**Key DCF insight:**
[1-2 sentences with a non-obvious observation: for example, "the market implies 15% revenue CAGR but segment reporting shows the growth engine is decelerating from 28% to 18% — implied assumptions are 4 years stale."]

Use realistic numbers grounded in the actual financial data. Show your work if assumptions matter. If data is missing (e.g., for a bank where DCF doesn't apply), say so and pivot to a P/B or residual income framework."""
                }]
            )
            dcf_text = dcf_response.content[0].text.strip()
            # Ensure the ## Reverse DCF header is present
            if "## Reverse DCF" not in dcf_text and "## DCF" not in dcf_text:
                dcf_text = "## Reverse DCF — What the Market is Pricing In\n\n" + dcf_text
            # Append to the main output
            job["text"] = job["text"].rstrip() + "\n\n" + dcf_text
            print(f"[DCF] {ticker}: reverse DCF added ({len(dcf_text)} chars, total now {len(job['text'])})", flush=True)
        except Exception as dcf_err:
            import traceback
            print(f"[DCF] Error: {dcf_err}\n{traceback.format_exc()}", flush=True)

        # Store conversation with FULL context for follow-up questions
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
