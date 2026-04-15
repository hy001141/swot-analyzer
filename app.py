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

# Heavy imports (yfinance, anthropic, beautifulsoup, lxml) are deferred to first request
# to avoid Render's port-scan timeout. Flask binds the port instantly, then loads on demand.
yf = None
anthropic = None
fetch_sec_data = None
build_sec_summary = None
run_full_research = None
build_full_context = None
LookupTool = None
TOOL_DEFINITIONS = None
execute_tool = None

def _ensure_imports():
    """Load heavy dependencies on first use."""
    global yf, anthropic, fetch_sec_data, build_sec_summary
    global run_full_research, build_full_context
    global LookupTool, TOOL_DEFINITIONS, execute_tool
    if yf is None:
        import yfinance as _yf
        import anthropic as _anthropic
        from sec_fetcher import fetch_sec_data as _fsd, build_sec_summary as _bss
        from research_agent import run_full_research as _rfr, build_full_context as _bfc
        from lookup_tool import LookupTool as _LT, TOOL_DEFINITIONS as _TD, execute_tool as _et
        yf = _yf
        anthropic = _anthropic
        fetch_sec_data = _fsd
        build_sec_summary = _bss
        run_full_research = _rfr
        build_full_context = _bfc
        LookupTool = _LT
        TOOL_DEFINITIONS = _TD
        execute_tool = _et

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


SWOT_SYSTEM_PROMPT = """You are a senior long/short equity analyst at a top-tier fundamental hedge fund. You have access to lookup tools that return REAL data from SEC XBRL filings, ClinicalTrials.gov, 10-K text, competitor filings, and web research. Your output will be evaluated against what a Goldman Sachs or Morgan Stanley equity research analyst would produce.

═══════════════════════════════════════════════════════════════
GROUNDED GENERATION — TOOLS FOR FACTS, JUDGMENT FOR INTERPRETATION
═══════════════════════════════════════════════════════════════

CRITICAL DISTINCTION:

**SPECIFIC FACTS need tool calls** (you cannot invent these):
- Numbers, percentages, dollar amounts, dates, NCT IDs, executive names
- Quarterly/annual financial line items
- Specific trial phases, completion dates
- Direct quotes from filings
For these, you MUST call a lookup tool first. If the tool returns "not found", drop the specific claim and use qualitative framing.

**ANALYTICAL INTERPRETATION is encouraged** (you SHOULD make these inferences):
- "Concentration in renal disease programs creates portfolio risk if any single program fails"
- "The competitive moat depends on switching costs in enterprise contracts"
- "Management's commentary suggests they're positioning for an inflection point"
- Cross-referencing what management says vs what competitors are doing
- Connecting dots between sources to identify non-obvious risks or opportunities
- Standard equity research interpretation that any analyst would make

A senior analyst is PAID to make analytical judgments that go beyond literally restating data. Pattern recognition, risk assessment, thesis construction — these are NOT hallucination. They are the job.

The line is: **specific facts** must be tool-verified, but **analytical interpretation** of those facts is what makes a real analyst valuable. Do not refuse to interpret. Refuse to invent specific numbers.

Workflow:
1. CALL TOOLS to gather specific facts you'll cite (10-20 tool calls)
2. THINK about what those facts mean — what patterns, risks, opportunities they reveal
3. Write the SWOT with specific facts (citations [N]) AND analytical interpretation (no citation needed for inference)

═══════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════
ABSOLUTE RULE — ZERO TOLERANCE FOR FABRICATION
═══════════════════════════════════════════════════════════════

You are FORBIDDEN from inventing specific numbers, metrics, dates, trigger thresholds, or data points that are not explicitly present in the research package below. This includes:

- DO NOT invent specific statistics like "3 ATC activations per month" if that number is not in the research
- DO NOT invent specific dates like "expiring March 2026" if not in the research
- DO NOT invent specific thresholds like ">$1B run-rate" if not grounded in actual disclosed data
- DO NOT invent specific percentages like "82.1% SWE-bench" if not in the research
- DO NOT invent "ClinicalTrials.gov shows..." or "Per the LinkedIn postings..." unless you actually see that content in the sources provided
- DO NOT cite a source [N] for a claim that didn't come from that source

WHAT YOU CAN DO:
- Cite actual numbers from the research package. Source [13] (XBRL) has REAL quarterly/annual financial line items straight from SEC XBRL — use these for any revenue, margin, expense, or balance sheet figure. This is ground truth.
- Source [14] has REAL clinical trial registrations with NCT IDs, phases, statuses, and completion dates. Use these for any clinical trial claim.
- Source [7] has the actual 10-K text — quote directly from it for qualitative claims.
- Source [12] has real scraped web content — if a specific stat appears in [12.X], cite [12.X] and the PM can click through to verify.
- Use general industry knowledge clearly marked as such ("industry typically sees 60-70% gross margins for SaaS")
- Present frameworks without fake precision ("watch for margin compression" rather than "watch for >3% margin compression")
- Write conditionally when data is missing ("if the next 10-Q discloses X, that would indicate Y")

IF THE DATA ISN'T IN THE RESEARCH PACKAGE, YOU DO NOT HAVE IT. Say so, or use a general framework instead. Better to say "monitor segment margin trajectory" than to invent "watch for >180bps compression in Q3."

BEFORE CITING ANY SPECIFIC NUMBER, ASK YOURSELF: "Is this number actually in source [1]-[14] above?" If you cannot point to it, DO NOT write it.

Hallucinated data points with fake citations will cause a PM to lose money and lose their job. This is a firing offense. If you are unsure whether a specific number exists in the research, DO NOT use it — use a qualitative description instead.

═══════════════════════════════════════════════════════════════

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
[13] SEC XBRL structured facts — REAL quarterly/annual financial line items (revenue, R&D, operating income, etc.) extracted directly from XBRL filings. When you want a specific financial number, USE THIS — it is the ground truth.
[14] Clinical Trials pipeline (ClinicalTrials.gov) — for pharma/biotech, real trial registrations with phases, status, completion dates, and primary endpoints. When citing clinical trial info, USE THIS — do NOT invent trial results.

HOW TO THINK — THIS IS THE CRITICAL PART:

You must find MECHANICAL and STRUCTURAL insights, not obvious observations. A mechanical insight is one that:
1. Requires cross-referencing 2+ sources to spot
2. Comes from specific accounting, supply chain, or language analysis
3. Cannot be generated by reading Yahoo Finance or Bloomberg in 60 seconds

⚠️ DO NOT COPY EXAMPLES FROM THIS PROMPT AS IF THEY WERE REAL FACTS ABOUT THE COMPANY. Any illustrative example in these instructions is hypothetical — the only real data is in sources [1] through [14] below. If you cannot verify a claim against one of those sources, DO NOT make the claim.

TYPES OF MECHANICAL INSIGHTS TO LOOK FOR (apply these frameworks to the actual data, don't assume any specific example applies to this company):
- Accounting policy details disclosed in THIS company's 10-K [7] (depreciation schedules, rev rec, capitalization) — compared to what THIS company's competitors [11] actually disclose
- Language shifts in THIS management's actual commentary quarter-over-quarter (what did they stop saying?)
- Supply chain details actually named in filings [7] [11] — do NOT invent relationships
- Counter-positioning: what does THIS company do that is mechanically hard for its specific competitors to replicate?
- Working capital trends visible in the XBRL data [13] — only cite numbers that are actually in [13]
- Stock-based comp as % of operating cash flow — only using numbers in [13]
- Insider transaction patterns — only claims supported by what's actually in [6]
- Short interest / options positioning — only using what's actually in [2]
- Earnings surprise patterns — only using what's actually in [4]
- For pharma: specific trial NCT IDs and phases — only from [14]

The ONLY sources of truth are the data blocks [1] through [14] below. Every claim must be grounded in what those sources actually contain for THIS specific company. Do not import insights from other companies you know about, do not recycle patterns from prior analyses, and do not use any concrete example you might have seen in instructions.

CONSIDER EVERY SOURCE — but weigh them by relevance. You don't have to cite every source (e.g., App Store data is irrelevant for JPMorgan) but you must READ and WEIGH every source before writing. If a source is irrelevant, you've consciously decided — not overlooked it.

CITATION FORMAT: Use bracketed numbers throughout. After any claim, put the source number in brackets, e.g. [1] [6] [7]. If a claim cross-references multiple sources, include all of them. Do NOT use any specific numbers or company names from this instruction as if they were real data — use ONLY numbers that are actually present in sources [1]-[14] below.

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
## Key Questions

FORMATTING RULES:
- Strengths/Weaknesses/Opportunities/Threats: 4-6 bullet points each. Each bullet starts with "- **Mechanical insight title:** Specific reasoning chain with numbers and inline citations [X]."
- Each SWOT point must cite at least 2 different sources (e.g., cross-reference [1] and [11])
- Opportunities MUST include one labeled "VARIANT PERCEPTION:" — where consensus is mechanically wrong with evidence
- Threats MUST include one labeled "THESIS KILLER:" — the single disconfirming event that closes the position
- Strategic Fit Assessment: 2-3 concise paragraphs connecting internal mechanics to external conditions
- TOWS Matrix: 4 bullets, one per strategy. Format: "- **SO Strategy:** [specific mechanical action citing sources]"
- Recommendations: 3-5 bullets. These are NON-OBVIOUS SIGNALS TO MONITOR, not portfolio construction advice. Do NOT write "build long position", "pair-trade", "sell puts", or any specific position-sizing — that's the PM's job. Your job is to surface the EARLY WARNING SIGNS or HIDDEN CONNECTIONS that would change someone's view of the company before it becomes consensus. Format each bullet as "**[Signal title]:** [what specifically to watch for, where to find it, and what it would mean if you saw it]". Examples of what TO write: "**Watch for Apple TAC disclosure timing in Q1 calls:** Any change in language around the maintenance of current ~36% rev share would signal renegotiation; a 90-day delay in commentary historically precedes deals." or "**Cross-reference Broadcom's largest customer commentary:** AVGO's next 10-Q will disclose customer concentration; any sequential acceleration in 'networking and custom silicon' confirms NVDA losing share to in-house alternatives." Do NOT recommend specific trades. Do NOT mention portfolio weights. Surface non-obvious signals and connections only.
- Key Questions: 3-5 numbered items. Each question must be non-obvious, cite specific data, and be answerable through channel checks/expert calls/data analysis — NOT from reading the 10-K. Only write as many as you actually have meaningful questions for — don't pad.

BULLET POINTS ONLY. NO PROSE PARAGRAPHS IN SWOT SECTIONS. Each bullet starts with "-" and a **bolded** lead-in.

YOU MUST COMPLETE ALL 8 SECTIONS. Pace your output — if you're writing too much in Strengths, tighten up so Key Questions gets its full depth. Key Questions is NOT optional."""


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

        # Step 2: Grounded SWOT generation via tool_use loop
        # Opus calls lookup tools to retrieve REAL data — cannot fabricate.
        job["status"] = f"Analyzing with QuantumIQ AI ({len(sources)} sources via grounded lookup)..."
        client = anthropic.Anthropic(api_key=api_key, timeout=600.0)

        # Build the lookup tool with the research data
        lookup = LookupTool(research)

        # Initial user message — concise, since Opus will fetch what it needs via tools
        initial_message = f"""Produce a comprehensive SWOT analysis for {ticker} ({meta.get('name', '')}).

You have {len(sources)} data sources available via lookup tools. CALL THE TOOLS to retrieve specific data — every claim in your analysis MUST come from a tool call result. Do not write any specific number, percentage, date, or trial ID that you have not retrieved via a tool.

Recommended workflow:
1. Start by calling lookup_yahoo_finance() and lookup_financial_metric() for key metrics (Revenue, Operating Income, R&D Expense, Net Income) to ground yourself
2. Call lookup_10k_passage() with specific topics relevant to this company's business (segments, competitive moats, risk factors)
3. Call lookup_competitor_10k() for at least 2 named competitors to cross-reference positioning
4. Call lookup_clinical_trials() if this is a pharma/biotech company
5. Call lookup_insider_transactions(), lookup_short_interest(), lookup_analyst_estimates() for positioning data
6. Call lookup_web_intelligence() with specific queries (patents, hiring, congressional, earnings calls)
7. Call lookup_comp_table() to get peer valuation context
8. THEN write the SWOT, citing every claim with the source tag returned by the tool (e.g. [13] for XBRL, [14] for trials, [12.X] for web sources)

ABSOLUTE RULE: If you cannot point to a specific tool call result for a claim, do not make the claim. Use qualitative framing instead.

Make 8-15 tool calls before writing the analysis. Be thorough."""

        messages = [{"role": "user", "content": initial_message}]
        max_tool_iterations = 25
        tool_call_count = 0
        final_text = ""

        for iteration in range(max_tool_iterations):
            try:
                response = client.messages.create(
                    model="claude-opus-4-20250514",
                    max_tokens=16000,
                    system=SWOT_SYSTEM_PROMPT,
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )
            except Exception as api_err:
                is_overloaded = "overloaded" in str(api_err).lower() or "529" in str(api_err)
                if is_overloaded:
                    import time
                    job["status"] = f"Server busy, retrying..."
                    time.sleep(3)
                    continue
                raise api_err

            # Check stop reason
            stop_reason = response.stop_reason

            # Append the assistant's response to messages
            assistant_blocks = []
            for block in response.content:
                assistant_blocks.append(block.model_dump() if hasattr(block, 'model_dump') else block.__dict__)

            messages.append({"role": "assistant", "content": [
                {k: v for k, v in b.items() if k in ("type", "text", "id", "name", "input")}
                for b in assistant_blocks
            ]})

            # If Opus is calling tools, execute them and continue
            if stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if hasattr(block, 'type') and block.type == "tool_use":
                        tool_call_count += 1
                        tool_name = block.name
                        tool_input = block.input or {}
                        job["status"] = f"Verifying data (call #{tool_call_count})... {tool_name}"
                        print(f"[TOOL] {ticker}: {tool_name}({tool_input})", flush=True)

                        result = execute_tool(tool_name, tool_input, lookup)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})
                continue

            # If Opus stopped (end_turn), extract the final text
            if stop_reason in ("end_turn", "stop_sequence", "max_tokens"):
                for block in response.content:
                    if hasattr(block, 'type') and block.type == "text":
                        final_text += block.text
                break

        job["text"] = final_text
        print(f"[GROUNDED] {ticker}: {tool_call_count} tool calls, {len(final_text)} chars output", flush=True)

        # NOTE: Self-critique pass removed. It caused visual flickering (replacing job.text mid-stream)
        # and removed comp table / downstream sections. The main SWOT prompt's anti-fabrication
        # rules are sufficient — we trust the first Opus pass and don't rewrite it.

        # Step 2c: Check if Key Questions has at least a couple numbered items — if not, regenerate
        questions_section = ""
        if "## Key Questions" in job["text"]:
            after = job["text"].split("## Key Questions", 1)[1]
            # Cut at next ## header (e.g. Reverse DCF appended later)
            next_header_idx = after.find("\n## ")
            questions_section = after[:next_header_idx].strip() if next_header_idx >= 0 else after.strip()

        # Count numbered questions
        import re as _re
        numbered_count = len(_re.findall(r'(?:^|\n)\s*\d+\.\s+', questions_section))

        # Regenerate if fewer than 2 questions or too short overall
        if numbered_count < 2 or len(questions_section) < 150:
            job["status"] = "Generating key questions..."
            print(f"[QUESTIONS] {ticker}: generating questions separately ({len(questions_section)} chars, {numbered_count} numbered)", flush=True)

            if "## Key Questions" in job["text"]:
                job["text"] = job["text"].rsplit("## Key Questions", 1)[0].rstrip()

            try:
                q_response = client.messages.create(
                    model="claude-opus-4-20250514",
                    max_tokens=2500,
                    messages=[{
                        "role": "user",
                        "content": f"""Here is a SWOT analysis for {ticker} ({meta.get('name','')}):

{job['text']}

CRITICAL: Do NOT fabricate numbers, dates, thresholds, or specific data points. Only reference figures that actually appear in the analysis above. If you don't have specific data, use qualitative framing instead. Inventing fake precision is a firing offense.

Produce 3-5 non-obvious questions a hedge fund PM should investigate. Each question must:
- Only reference numbers/metrics that actually appear in the analysis above
- Be answerable through channel checks, expert calls, or further data analysis
- NOT be generic ("is management capable?") but also NOT pretend to know data you don't have
- Be 2-3 sentences long with specific context
- If you want precision, use conditional language: "if the next 10-Q discloses..." instead of fake numbers

Write as many questions as you have genuinely meaningful ones — aim for 3-5, don't pad.

Respond with ONLY this format, nothing else:

## Key Questions
1. [Question grounded in actual data or qualitative framing]
2. ...
3. ..."""
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

CRITICAL — ZERO FABRICATION: Use ONLY numbers that actually appear in the research data below. Current price, market cap, revenue, margins, net debt — these must come from the actual data. If a specific figure isn't present, say "N/A" or use the closest disclosed metric. Do NOT invent precision. Your scenario assumptions (rev CAGR, terminal margin) should be reasonable given actual historical data. Show your work: if you assume 12% rev CAGR, base it on actual 3-year trailing growth from the data.

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
[1-2 sentences with a non-obvious observation grounded in the actual data — e.g., a gap between implied assumptions and actual segment trends from the research package. Do NOT invent numbers; only use what is actually in the data above.]

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

        CHAT_PROMPT = """You are a senior buy-side analyst pinging back on Slack. Direct, punchy, no fluff.

VOICE:
- Lead with the conclusion in sentence 1. No setup, no "great question", no preamble.
- 100-200 words MAX. If you can't say it in 200 words you're padding.
- Use short, declarative sentences. Cut every word that doesn't add information.
- Talk like a person who's already decided what they think, not a textbook explaining options.
- One angle per response, not five.

FACTS vs INTERPRETATION:
- Specific numbers/dates/IDs → only if you can quote them directly from the research package above. If you don't have a verified number, say "don't have that exact number" and move on.
- Interpretation, risk calls, judgment → expected, but ONLY if grounded in actual data points. Don't make sweeping qualitative claims like "structurally behind X" or "losing share" without specific evidence in the research.

🚨 ZERO-FABRICATION RULES 🚨

1. Before making ANY factual claim about competitive position, market share, or scale comparisons:
   - Search the research package for the specific data point
   - If you can't find it, say "I'd need to verify that against [specific data]" — DON'T just assert
   - Words like "structurally behind", "market leader", "losing share" require evidence — not vibes

2. Do not do percentage math on raw numbers without verifying units. If shares are listed as "12,102,000,000" don't decide that's wrong and add a decimal — that IS twelve billion.

3. If you previously gave a number in this conversation that you can't verify, OWN IT — say "I can't verify that figure, here's what I CAN see." Don't invent unit conversion errors to retrofit your previous claim.

4. The phrase "wrong inference on my part" should be RARE. Don't make claims you'll have to retract — verify first, claim second. A senior analyst doesn't constantly self-correct; they get it right the first time by being careful about what they actually know.

5. If a user asks about something not in the research, the answer is "I don't have that data" — NOT a guess dressed up as analysis.

WHAT TO AVOID:
- Bullet point lists (use prose)
- Headers / bolded sections
- "Great question" / "That's a good point" / any preamble
- Hedging language ("it's possible that", "one could argue")
- Restating the question
- Multi-paragraph "on one hand / on the other hand" structure
- Apologizing for making reasonable inferences
- Long disclaimers about what you don't have
- Doing math you can't verify

Think: a senior PM asks you a question in person. You have 30 seconds. What's your take? That's the response."""

        with client.messages.stream(
            model="claude-opus-4-20250514",
            max_tokens=600,
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
    _ensure_imports()
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
        "cursor": 0,
    }

    thread = threading.Thread(target=run_analysis_worker, args=(job_id, ticker, session_id), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/poll/<job_id>")
def poll(job_id):
    """Poll for new text from a running job. Returns delta from cursor."""
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
    _ensure_imports()
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
