"""
SWOT Analysis Web App
Flask backend with real-time SSE streaming.
"""

import os
import sys
import json
import queue
import threading

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from flask import Flask, render_template, request, Response, jsonify, stream_with_context
import yfinance as yf
import anthropic

app = Flask(__name__)

# ── Reuse data-gathering logic from swot_analyzer ──────────────────────────

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


# ── Conversation store (in-memory, keyed by session) ─────────────────────

conversations: dict[str, list] = {}


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/company-info")
def company_info():
    """Quick lookup of company name/sector before full analysis."""
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        name = info.get("longName") or info.get("shortName", "")
        if not name:
            return jsonify({"error": f"Ticker '{ticker}' not found"}), 404
        return jsonify({
            "name": name,
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "country": info.get("country", ""),
            "marketCap": info.get("marketCap"),
            "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze")
def analyze():
    """Stream SWOT analysis via Server-Sent Events."""
    ticker = request.args.get("ticker", "").strip().upper()
    session_id = request.args.get("session_id", ticker)

    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 500

    def generate():
        # Step 1: Fetch data
        yield f"data: {json.dumps({'type': 'status', 'message': 'Fetching financial data from Yahoo Finance...'})}\n\n"

        try:
            data = fetch_company_data(ticker)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        info = data.get("info", {})
        # Accept if we got any meaningful info at all
        has_data = info.get("longName") or info.get("shortName") or info.get("symbol") or info.get("regularMarketPrice") or data.get("price_history_summary")
        if not has_data:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Could not fetch data for {ticker!r}. Check the ticker symbol and try again.'})}\n\n"
            return
        # Fill in name fallbacks
        if not info.get("longName"):
            info["longName"] = info.get("shortName") or ticker

        yield f"data: {json.dumps({'type': 'status', 'message': 'Building financial summary...'})}\n\n"
        summary = build_data_summary(data)

        # Send company metadata to frontend
        yield f"data: {json.dumps({'type': 'meta', 'name': info.get('longName', ticker), 'sector': info.get('sector',''), 'industry': info.get('industry',''), 'marketCap': info.get('marketCap'), 'price': info.get('currentPrice') or info.get('regularMarketPrice'), 'priceChange': data.get('price_history_summary', {}).get('price_change_1y_pct')})}\n\n"

        yield f"data: {json.dumps({'type': 'status', 'message': 'Generating SWOT analysis with Claude AI...'})}\n\n"

        # Step 2: Stream Claude response
        client = anthropic.Anthropic(api_key=api_key)
        full_text = ""

        try:
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
                    full_text += text_chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'text': text_chunk})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        # Store conversation for follow-up Q&A
        conversations[session_id] = [
            {"role": "user", "content": f"Produce a SWOT analysis for {ticker}. Data:\n\n{summary}"},
            {"role": "assistant", "content": full_text},
        ]

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle follow-up questions."""
    body = request.get_json()
    session_id = body.get("session_id", "")
    question = body.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    history = conversations.get(session_id, [])
    if not history:
        return jsonify({"error": "No analysis found. Run an analysis first."}), 400

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 500

    history.append({"role": "user", "content": question})

    def generate():
        client = anthropic.Anthropic(api_key=api_key)
        full_answer = ""
        try:
            with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=3000,
                system=SWOT_SYSTEM_PROMPT,
                messages=history
            ) as stream:
                for chunk in stream.text_stream:
                    full_answer += chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        history.append({"role": "assistant", "content": full_answer})
        conversations[session_id] = history
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting SWOT Analyzer at http://localhost:{port}")
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
