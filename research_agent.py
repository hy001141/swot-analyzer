"""
Research Agent
==============
Level 3 research pipeline:
1. Gather structured data (Yahoo Finance, SEC, FRED, App Store, insider transactions)
2. Ask Haiku what company-specific sources to investigate
3. Run targeted Brave searches in parallel
4. Scrape top results
5. Package everything for Opus analysis

Every source is independent — any can fail without breaking the pipeline.
"""

import os
import json
import time
import requests
import concurrent.futures
from typing import Callable

import yfinance as yf
import anthropic

from sec_fetcher import fetch_sec_data, build_sec_summary

# ── Configuration ────────────────────────────────────────────────────────

BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
WEB_TIMEOUT = 10  # seconds per web request
MAX_BRAVE_QUERIES = 8


# ── Data Sources (all independent, all have try/except) ──────────────────

def _build_yahoo_summary(stock, info: dict) -> str:
    """Build a financial summary string from yfinance data."""
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

    try:
        hist = stock.history(period="1y")
        if hist is not None and not hist.empty:
            lines.append("\nPRICE HISTORY (1Y):")
            lines.append(f"  Current: ${hist['Close'].iloc[-1]:.2f}")
            lines.append(f"  52w High: ${hist['Close'].max():.2f}")
            lines.append(f"  52w Low: ${hist['Close'].min():.2f}")
            lines.append(f"  Avg Volume: {int(hist['Volume'].mean()):,}")
    except Exception:
        pass

    try:
        news = stock.news
        if news:
            lines.append("\nRECENT NEWS:")
            for item in news[:10]:
                if isinstance(item, dict):
                    title = item.get("title", item.get("content", {}).get("title", ""))
                    if title:
                        lines.append(f"  - {title}")
    except Exception:
        pass

    return "\n".join(lines)


def fetch_insider_transactions(ticker: str) -> str:
    """Get insider buy/sell activity from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        ins = stock.insider_transactions
        if ins is None or ins.empty:
            return ""

        lines = ["INSIDER TRANSACTIONS (recent):"]
        for _, row in ins.head(15).iterrows():
            name = row.get("Insider", "Unknown")
            pos = row.get("Position", "")
            text = row.get("Text", "")
            date = str(row.get("Start Date", ""))[:10]
            value = row.get("Value", 0)
            val_str = f"${value:,.0f}" if value and value > 0 else ""
            lines.append(f"  {date} | {name} ({pos}): {text} {val_str}")

        # Add summary
        buys = ins[ins["Text"].str.contains("Purchase|Buy", case=False, na=False)]
        sells = ins[ins["Text"].str.contains("Sale|Sell", case=False, na=False)]
        lines.append(f"\n  Summary: {len(buys)} purchases, {len(sells)} sales in recent history")

        return "\n".join(lines)
    except Exception as e:
        print(f"[INSIDER] Error: {e}")
        return ""


def fetch_short_interest_and_options(ticker: str) -> str:
    """Get short interest data and put/call ratio."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        lines = ["SHORT INTEREST & OPTIONS POSITIONING:"]

        short_ratio = info.get("shortRatio")
        short_pct = info.get("shortPercentOfFloat")
        shares_short = info.get("sharesShort")
        shares_short_prior = info.get("sharesShortPriorMonth")

        if short_ratio is not None:
            lines.append(f"  Short ratio (days to cover): {short_ratio}")
        if short_pct is not None:
            lines.append(f"  Short % of float: {short_pct:.2%}")
        if shares_short is not None:
            lines.append(f"  Shares short: {shares_short:,}")
        if shares_short_prior is not None and shares_short is not None:
            chg = shares_short - shares_short_prior
            direction = "increased" if chg > 0 else "decreased"
            lines.append(f"  Shares short prior month: {shares_short_prior:,} ({direction} by {abs(chg):,})")

        # Put/Call ratio from nearest expiry
        try:
            opts = stock.options
            if opts:
                chain = stock.option_chain(opts[0])
                calls_vol = chain.calls["volume"].sum()
                puts_vol = chain.puts["volume"].sum()
                if calls_vol and calls_vol > 0:
                    pcr = puts_vol / calls_vol
                    lines.append(f"  Put/Call volume ratio (nearest expiry): {pcr:.2f}")
                    if pcr > 1.0:
                        lines.append(f"  Signal: Elevated put activity — bearish positioning or hedging")
                    elif pcr < 0.5:
                        lines.append(f"  Signal: Low put activity — bullish positioning")
        except Exception:
            pass

        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception as e:
        print(f"[SHORT] Error: {e}")
        return ""


def fetch_analyst_estimates(ticker: str) -> str:
    """Get analyst price targets, estimate revisions, and recommendation trends."""
    try:
        stock = yf.Ticker(ticker)
        lines = ["ANALYST ESTIMATES & REVISIONS:"]

        # Price targets
        try:
            targets = stock.analyst_price_targets
            if targets:
                lines.append(f"  Price target — Mean: ${targets.get('mean', 0):.2f}, "
                             f"High: ${targets.get('high', 0):.2f}, "
                             f"Low: ${targets.get('low', 0):.2f}, "
                             f"Current: ${targets.get('current', 0):.2f}")
        except Exception:
            pass

        # Recommendation trends
        try:
            rec = stock.recommendations_summary
            if rec is not None and not rec.empty:
                current = rec.iloc[0]
                prior = rec.iloc[1] if len(rec) > 1 else None
                lines.append(f"  Current ratings — Strong Buy: {current.get('strongBuy', 0)}, "
                             f"Buy: {current.get('buy', 0)}, "
                             f"Hold: {current.get('hold', 0)}, "
                             f"Sell: {current.get('sell', 0)}")
                if prior is not None:
                    sb_chg = current.get('strongBuy', 0) - prior.get('strongBuy', 0)
                    b_chg = current.get('buy', 0) - prior.get('buy', 0)
                    if sb_chg + b_chg > 0:
                        lines.append(f"  Trend: Net {sb_chg + b_chg} upgrades vs prior month")
                    elif sb_chg + b_chg < 0:
                        lines.append(f"  Trend: Net {abs(sb_chg + b_chg)} downgrades vs prior month")
        except Exception:
            pass

        # Earnings estimates
        try:
            ee = stock.earnings_estimate
            if ee is not None and not ee.empty:
                for idx, row in ee.iterrows():
                    growth = row.get("growth")
                    avg = row.get("avg")
                    yago = row.get("yearAgoEps")
                    analysts = row.get("numberOfAnalysts")
                    if avg is not None:
                        growth_str = f" ({growth:+.1%} Y/Y)" if growth is not None else ""
                        lines.append(f"  EPS estimate ({idx}): ${avg:.2f}{growth_str} — {int(analysts) if analysts else '?'} analysts")
        except Exception:
            pass

        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception as e:
        print(f"[ANALYST] Error: {e}")
        return ""


def fetch_earnings_history(ticker: str) -> str:
    """Get earnings beat/miss history."""
    try:
        stock = yf.Ticker(ticker)
        eh = stock.earnings_history
        if eh is None or eh.empty:
            return ""

        lines = ["EARNINGS SURPRISE HISTORY:"]
        beats = 0
        total = 0
        for idx, row in eh.iterrows():
            actual = row.get("epsActual")
            estimate = row.get("epsEstimate")
            surprise = row.get("surprisePercent")
            if actual is not None and estimate is not None:
                total += 1
                beat = actual > estimate
                if beat:
                    beats += 1
                icon = "BEAT" if beat else "MISS"
                surprise_str = f" ({surprise:+.1%})" if surprise is not None else ""
                lines.append(f"  {idx}: Actual ${actual:.2f} vs Est ${estimate:.2f} — {icon}{surprise_str}")

        if total > 0:
            lines.append(f"\n  Track record: {beats}/{total} beats ({beats/total:.0%} hit rate)")
            if beats == total:
                lines.append(f"  Signal: Consistent beater — management likely sandbagging guidance")

        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception as e:
        print(f"[EARNINGS_HIST] Error: {e}")
        return ""


def fetch_institutional_changes(ticker: str) -> str:
    """Get institutional ownership with changes."""
    try:
        stock = yf.Ticker(ticker)
        ih = stock.institutional_holders
        if ih is None or ih.empty:
            return ""

        lines = ["INSTITUTIONAL OWNERSHIP (top holders + changes):"]
        for _, row in ih.head(10).iterrows():
            holder = row.get("Holder", "Unknown")
            pct = row.get("pctHeld", 0)
            shares = row.get("Shares", 0)
            value = row.get("Value", 0)
            chg = row.get("pctChange", 0)

            chg_str = ""
            if chg is not None and chg != 0:
                direction = "added" if chg > 0 else "reduced"
                chg_str = f" — {direction} {abs(chg):.1%}"

            val_str = f"${value/1e9:.1f}B" if value and value > 1e9 else f"${value/1e6:.0f}M" if value else ""
            lines.append(f"  {holder}: {pct:.1%} ({shares:,.0f} shares, {val_str}){chg_str}")

        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception as e:
        print(f"[INSTITUTIONAL] Error: {e}")
        return ""


def fetch_macro_data() -> str:
    """Get key macro indicators from FRED (no API key needed)."""
    try:
        indicators = {
            "FEDFUNDS": "Fed Funds Rate",
            "CPIAUCSL": "CPI (Consumer Price Index)",
            "GDP": "GDP",
            "UNRATE": "Unemployment Rate",
            "T10Y2Y": "10Y-2Y Treasury Spread",
            "VIXCLS": "VIX (Volatility Index)",
        }

        lines = ["MACRO ENVIRONMENT (from FRED):"]
        for series_id, label in indicators.items():
            try:
                r = requests.get(
                    f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=2025-01-01",
                    timeout=WEB_TIMEOUT
                )
                if r.status_code == 200:
                    rows = r.text.strip().split("\n")
                    if len(rows) >= 2:
                        latest = rows[-1].split(",")
                        if len(latest) == 2 and latest[1] != ".":
                            lines.append(f"  {label}: {latest[1]} (as of {latest[0]})")
            except Exception:
                continue

        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception as e:
        print(f"[MACRO] Error: {e}")
        return ""


def fetch_app_store_data(company_name: str) -> str:
    """Get app ratings and review counts from iTunes API."""
    try:
        r = requests.get(
            f"https://itunes.apple.com/search?term={requests.utils.quote(company_name)}&entity=software&country=us&limit=10",
            timeout=WEB_TIMEOUT
        )
        if r.status_code != 200:
            return ""

        data = r.json()
        results = data.get("results", [])
        if not results:
            return ""

        lines = ["APP STORE DATA (iOS):"]
        for app in results[:8]:
            name = app.get("trackName", "")
            rating = app.get("averageUserRating", 0)
            reviews = app.get("userRatingCount", 0)
            if reviews > 1000:  # Only show significant apps
                lines.append(f"  {name}: {rating:.1f}/5 ({reviews:,} reviews)")

        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception as e:
        print(f"[APPSTORE] Error: {e}")
        return ""


# ── Brave Search ─────────────────────────────────────────────────────────

def brave_search(query: str, count: int = 5) -> list[dict]:
    """Run a single Brave Search query. Returns list of {title, url, description}."""
    if not BRAVE_API_KEY:
        return []
    try:
        r = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": count, "freshness": "py"},  # past year
            headers={"X-Subscription-Token": BRAVE_API_KEY},
            timeout=WEB_TIMEOUT
        )
        if r.status_code != 200:
            return []

        results = r.json().get("web", {}).get("results", [])
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "description": r.get("description", ""),
            }
            for r in results
        ]
    except Exception as e:
        print(f"[BRAVE] Search error for '{query}': {e}")
        return []


def scrape_url(url: str, max_chars: int = 5000) -> str:
    """Scrape text content from a URL."""
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; QuantumIQ/1.0)"},
            timeout=WEB_TIMEOUT
        )
        if r.status_code != 200:
            return ""

        from bs4 import BeautifulSoup
        import warnings
        from bs4 import XMLParsedAsHTMLWarning
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

        soup = BeautifulSoup(r.text[:200000], "lxml")
        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        # Clean
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()

        return text[:max_chars] if text else ""
    except Exception:
        return ""


def run_brave_research(queries: list[str], status_callback: Callable = None) -> str:
    """Run multiple Brave searches and scrape top results."""
    if not BRAVE_API_KEY:
        return ""

    all_results = []

    # Run searches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_query = {
            executor.submit(brave_search, q, 3): q for q in queries[:MAX_BRAVE_QUERIES]
        }
        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                results = future.result()
                for r in results:
                    r["query"] = query
                    all_results.append(r)
            except Exception:
                continue

    if not all_results:
        return ""

    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            unique_results.append(r)

    # Scrape top results in parallel (limit to 8 to avoid being slow)
    if status_callback:
        status_callback(f"Analyzing {len(unique_results[:8])} web sources...")

    scraped = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_result = {
            executor.submit(scrape_url, r["url"], 3000): r
            for r in unique_results[:8]
        }
        for future in concurrent.futures.as_completed(future_to_result):
            result = future_to_result[future]
            try:
                text = future.result()
                if text and len(text) > 200:
                    scraped.append({
                        "title": result["title"],
                        "url": result["url"],
                        "query": result["query"],
                        "content": text,
                    })
            except Exception:
                continue

    if not scraped:
        return ""

    # Format for Claude
    lines = [f"WEB RESEARCH ({len(scraped)} sources scraped):"]
    for s in scraped:
        lines.append(f"\n--- Source: {s['title']} ---")
        lines.append(f"URL: {s['url']}")
        lines.append(f"Search query: {s['query']}")
        lines.append(s["content"])

    return "\n".join(lines)


# ── Competitor Analysis ───────────────────────────────────────────────────

def identify_competitors(ticker: str, company_name: str, sector: str, industry: str) -> list[str]:
    """Ask Haiku to identify 4-5 closest public company competitors."""
    if not ANTHROPIC_API_KEY:
        return []

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": f"""For {company_name} ({ticker}), sector: {sector}, industry: {industry}.

List the 4-5 closest PUBLIC company competitors by revenue overlap and market positioning. Only US-listed tickers.

Return ONLY a JSON array of ticker strings. No explanation. Example: ["META", "SNAP", "PINS", "TTD"]"""
            }]
        )
        text = response.content[0].text.strip()
        if "[" in text:
            json_str = text[text.index("["):text.rindex("]") + 1]
            tickers = json.loads(json_str)
            # Remove self if included
            tickers = [t for t in tickers if t.upper() != ticker.upper()]
            return tickers[:5]
    except Exception as e:
        print(f"[COMPETITORS] Error identifying competitors: {e}")

    return []


def fetch_competitor_data(comp_ticker: str) -> dict:
    """Fetch financial data + abbreviated 10-K for one competitor."""
    result = {
        "ticker": comp_ticker,
        "financials": {},
        "filing_excerpt": "",
    }

    try:
        stock = yf.Ticker(comp_ticker)
        info = stock.info or {}

        result["financials"] = {
            "name": info.get("longName") or info.get("shortName") or comp_ticker,
            "marketCap": info.get("marketCap"),
            "pe_trailing": info.get("trailingPE"),
            "pe_forward": info.get("forwardPE"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "revenue": info.get("totalRevenue"),
            "revenueGrowth": info.get("revenueGrowth"),
            "grossMargin": info.get("grossMargins"),
            "operatingMargin": info.get("operatingMargins"),
            "profitMargin": info.get("profitMargins"),
            "roe": info.get("returnOnEquity"),
            "debtToEquity": info.get("debtToEquity"),
            "freeCashflow": info.get("freeCashflow"),
            "beta": info.get("beta"),
            "dividendYield": info.get("dividendYield"),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
        }
    except Exception as e:
        print(f"[COMP] Error fetching {comp_ticker} financials: {e}")

    # Abbreviated 10-K (15K chars — business overview + risk factors)
    try:
        from sec_fetcher import fetch_sec_data
        sec = fetch_sec_data(comp_ticker)
        if sec.get("annual_filing"):
            # Take less for competitors — just enough for Claude to understand their positioning
            result["filing_excerpt"] = sec["annual_filing"][:15000]
    except Exception as e:
        print(f"[COMP] Error fetching {comp_ticker} 10-K: {e}")

    return result


def run_competitor_analysis(ticker: str, company_name: str, sector: str, industry: str,
                            status_callback: Callable = None) -> dict:
    """Identify competitors and pull their data in parallel."""
    result = {
        "competitors": [],
        "comp_table": "",
        "comp_filings": "",
    }

    if status_callback:
        status_callback("Identifying key competitors...")

    comp_tickers = identify_competitors(ticker, company_name, sector, industry)
    if not comp_tickers:
        return result

    if status_callback:
        status_callback(f"Analyzing {len(comp_tickers)} competitors: {', '.join(comp_tickers)}...")

    # Fetch all competitor data in parallel
    comp_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_competitor_data, t): t for t in comp_tickers}
        for future in concurrent.futures.as_completed(futures):
            try:
                data = future.result(timeout=45)
                if data.get("financials", {}).get("name"):
                    comp_data.append(data)
            except Exception:
                continue

    if not comp_data:
        return result

    result["competitors"] = comp_data

    # Build comp table
    result["comp_table"] = _build_comp_table(comp_data)

    # Build competitor filing summaries
    filing_lines = []
    for comp in comp_data:
        if comp.get("filing_excerpt"):
            filing_lines.append(f"\n{'=' * 40}")
            filing_lines.append(f"COMPETITOR 10-K: {comp['financials']['name']} ({comp['ticker']})")
            filing_lines.append("=" * 40)
            filing_lines.append(comp["filing_excerpt"])
    result["comp_filings"] = "\n".join(filing_lines)

    return result


def _build_comp_table(comp_data: list[dict]) -> str:
    """Build a formatted comparison table."""
    def fmt_val(val, fmt_type="number"):
        if val is None:
            return "N/A"
        if fmt_type == "cap":
            if val >= 1e12: return f"${val/1e12:.1f}T"
            if val >= 1e9: return f"${val/1e9:.0f}B"
            if val >= 1e6: return f"${val/1e6:.0f}M"
            return f"${val:,.0f}"
        if fmt_type == "pct":
            return f"{val:.1%}" if abs(val) < 10 else f"{val:.0f}%"
        if fmt_type == "ratio":
            return f"{val:.1f}x"
        return f"{val}"

    lines = ["COMPARABLE COMPANY ANALYSIS:"]
    lines.append("")

    # Header
    tickers = [c["ticker"] for c in comp_data]
    header = f"{'Metric':<22}" + "".join(f"{t:>12}" for t in tickers)
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    metrics = [
        ("Market Cap", "marketCap", "cap"),
        ("P/E (Trailing)", "pe_trailing", "ratio"),
        ("P/E (Forward)", "pe_forward", "ratio"),
        ("EV/EBITDA", "ev_ebitda", "ratio"),
        ("Revenue Growth", "revenueGrowth", "pct"),
        ("Gross Margin", "grossMargin", "pct"),
        ("Operating Margin", "operatingMargin", "pct"),
        ("Profit Margin", "profitMargin", "pct"),
        ("ROE", "roe", "pct"),
        ("Debt/Equity", "debtToEquity", "ratio"),
        ("Beta", "beta", "ratio"),
    ]

    for label, key, fmt_type in metrics:
        row = f"{label:<22}"
        for c in comp_data:
            val = c.get("financials", {}).get(key)
            row += f"{fmt_val(val, fmt_type):>12}"
        lines.append(row)

    return "\n".join(lines)


# ── Haiku Source Recommender ─────────────────────────────────────────────

def get_research_queries(ticker: str, company_name: str, sector: str) -> list[str]:
    """Ask Haiku what to search for this specific company."""
    if not ANTHROPIC_API_KEY:
        return _default_queries(ticker, company_name)

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"""You are a hedge fund research analyst. For {company_name} ({ticker}), sector: {sector}.

What 6 specific web searches would find non-obvious, investment-relevant intelligence that ISN'T in standard filings or Yahoo Finance?

Think: company blog posts, engineering publications, patent filings, regulatory submissions, supply chain signals, hiring trends, competitor moves, industry-specific data.

Return ONLY a JSON array of 6 search query strings. No explanation. Example:
["NVIDIA blog AI inference optimization 2025", "TSMC capacity allocation NVIDIA vs AMD"]"""
            }]
        )
        text = response.content[0].text.strip()
        # Parse JSON array
        if "[" in text:
            json_str = text[text.index("["):text.rindex("]") + 1]
            queries = json.loads(json_str)
            if isinstance(queries, list) and len(queries) > 0:
                return queries[:8]
    except Exception as e:
        print(f"[HAIKU] Error getting research queries: {e}")

    return _default_queries(ticker, company_name)


def _default_queries(ticker: str, company_name: str) -> list[str]:
    """Fallback search queries if Haiku fails."""
    return [
        f'"{company_name}" blog announcement 2025 2026',
        f'"{company_name}" patent filing recent',
        f"{ticker} earnings analysis bull bear case",
        f'"{company_name}" competitive threat disruption',
        f"{ticker} insider trading congressional",
        f'"{company_name}" hiring engineering jobs signal',
    ]


# ── Main Research Pipeline ───────────────────────────────────────────────

def run_full_research(ticker: str, status_callback: Callable = None) -> dict:
    """
    Run the complete Level 3 research pipeline.
    All sources are independent — any can fail without breaking others.
    Returns dict with all gathered intelligence.
    """
    results = {
        "yahoo_finance": "",
        "sec_filing": "",
        "insider_transactions": "",
        "short_interest": "",
        "analyst_estimates": "",
        "earnings_history": "",
        "institutional_changes": "",
        "macro_data": "",
        "app_store": "",
        "web_research": "",
        "sources_succeeded": [],
        "sources_failed": [],
        "meta": {},
    }

    def _status(msg):
        if status_callback:
            status_callback(msg)

    # Step 1: Yahoo Finance (blocking — we need company name for other queries)
    _status("Fetching financial data...")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        # fast_info for reliable market data
        fi_data = {}
        try:
            fi = stock.fast_info
            try: fi_data["marketCap"] = float(fi.market_cap)
            except: pass
            try: fi_data["lastPrice"] = fi.last_price
            except: pass
            try: fi_data["previousClose"] = fi.previous_close
            except: pass
        except:
            pass

        company_name = info.get("longName") or info.get("shortName") or ticker
        sector = info.get("sector", "")
        industry = info.get("industry", "")

        results["meta"] = {
            "name": company_name,
            "sector": sector,
            "industry": industry,
            "marketCap": fi_data.get("marketCap") or info.get("marketCap"),
            "price": (fi_data.get("lastPrice") or fi_data.get("previousClose") or
                      info.get("currentPrice") or info.get("regularMarketPrice")),
        }

        # Build financial summary using yfinance data we already have
        results["yahoo_finance"] = _build_yahoo_summary(stock, info)

        # Get price history for meta
        try:
            hist = stock.history(period="1y")
            if hist is not None and not hist.empty:
                price_chg = round(
                    (hist["Close"].iloc[-1] - hist["Close"].iloc[0])
                    / hist["Close"].iloc[0] * 100, 2
                )
                results["meta"]["priceChange"] = float(price_chg)
                if not results["meta"].get("price"):
                    results["meta"]["price"] = float(hist["Close"].iloc[-1])
        except Exception:
            pass

        results["sources_succeeded"].append("Yahoo Finance")
    except Exception as e:
        print(f"[YAHOO] Error: {e}")
        results["sources_failed"].append("Yahoo Finance")
        company_name = ticker
        sector = ""

    # Step 2: Run independent sources in parallel
    _status("Gathering intelligence from multiple sources...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # SEC filings
        sec_future = executor.submit(
            fetch_sec_data, ticker, lambda m: _status(m)
        )

        # Insider transactions
        insider_future = executor.submit(fetch_insider_transactions, ticker)

        # Short interest + options
        short_future = executor.submit(fetch_short_interest_and_options, ticker)

        # Analyst estimates
        analyst_future = executor.submit(fetch_analyst_estimates, ticker)

        # Earnings surprise history
        earnings_hist_future = executor.submit(fetch_earnings_history, ticker)

        # Institutional ownership changes
        institutional_future = executor.submit(fetch_institutional_changes, ticker)

        # Macro data
        macro_future = executor.submit(fetch_macro_data)

        # App store (only for consumer-facing companies)
        app_future = executor.submit(fetch_app_store_data, company_name)

        # Haiku research queries (needs company name)
        _status("Planning targeted research...")
        queries_future = executor.submit(
            get_research_queries, ticker, company_name, sector
        )

        # Collect results as they complete
        try:
            sec_data = sec_future.result(timeout=60)
            results["sec_filing"] = build_sec_summary(sec_data)
            if results["sec_filing"]:
                results["sources_succeeded"].append(f"SEC {sec_data.get('annual_type', 'Filing')}")
        except Exception as e:
            print(f"[SEC] Error: {e}")
            results["sources_failed"].append("SEC EDGAR")

        try:
            results["insider_transactions"] = insider_future.result(timeout=15)
            if results["insider_transactions"]:
                results["sources_succeeded"].append("Insider Transactions")
        except Exception:
            results["sources_failed"].append("Insider Transactions")

        try:
            results["short_interest"] = short_future.result(timeout=15)
            if results["short_interest"]:
                results["sources_succeeded"].append("Short Interest & Options")
        except Exception:
            pass

        try:
            results["analyst_estimates"] = analyst_future.result(timeout=15)
            if results["analyst_estimates"]:
                results["sources_succeeded"].append("Analyst Estimates")
        except Exception:
            pass

        try:
            results["earnings_history"] = earnings_hist_future.result(timeout=15)
            if results["earnings_history"]:
                results["sources_succeeded"].append("Earnings History")
        except Exception:
            pass

        try:
            results["institutional_changes"] = institutional_future.result(timeout=15)
            if results["institutional_changes"]:
                results["sources_succeeded"].append("Institutional Ownership")
        except Exception:
            pass

        try:
            results["macro_data"] = macro_future.result(timeout=15)
            if results["macro_data"]:
                results["sources_succeeded"].append("FRED Macro Data")
        except Exception:
            results["sources_failed"].append("FRED Macro Data")

        try:
            app_data = app_future.result(timeout=15)
            if app_data:
                results["app_store"] = app_data
                results["sources_succeeded"].append("App Store Rankings")
        except Exception:
            pass  # Not critical, don't report as failure

        # Get Haiku's search recommendations
        try:
            search_queries = queries_future.result(timeout=20)
        except Exception:
            search_queries = _default_queries(ticker, company_name)

    # Step 3: Run Brave web research and competitor analysis in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Brave web research
        brave_future = None
        if BRAVE_API_KEY:
            _status(f"Searching {len(search_queries)} targeted queries...")
            brave_future = executor.submit(
                run_brave_research, search_queries, _status
            )

        # Competitor analysis
        _status("Analyzing competitors...")
        comp_future = executor.submit(
            run_competitor_analysis, ticker, company_name, sector,
            info.get("industry", ""), _status
        )

        if brave_future:
            try:
                results["web_research"] = brave_future.result(timeout=60)
                if results["web_research"]:
                    results["sources_succeeded"].append("Web Research (Brave)")
            except Exception as e:
                print(f"[BRAVE] Error: {e}")
                results["sources_failed"].append("Web Research")

        try:
            comp_results = comp_future.result(timeout=90)
            results["comp_table"] = comp_results.get("comp_table", "")
            results["comp_filings"] = comp_results.get("comp_filings", "")
            results["competitors"] = comp_results.get("competitors", [])
            if results["comp_table"]:
                comp_names = [c["financials"]["name"] for c in results["competitors"]]
                results["sources_succeeded"].append(f"Competitor Analysis ({', '.join(comp_names)})")
        except Exception as e:
            print(f"[COMP] Error: {e}")
            results["sources_failed"].append("Competitor Analysis")

    return results


def build_full_context(results: dict) -> str:
    """Combine all research results into a single context string for Claude."""
    sections = []

    # Sources summary
    succeeded = results.get("sources_succeeded", [])
    failed = results.get("sources_failed", [])
    sections.append(f"RESEARCH SOURCES USED: {', '.join(succeeded)}")
    if failed:
        sections.append(f"SOURCES UNAVAILABLE: {', '.join(failed)}")
    sections.append("")

    # Financial data
    if results.get("yahoo_finance"):
        sections.append("=" * 60)
        sections.append("FINANCIAL DATA (Yahoo Finance)")
        sections.append("=" * 60)
        sections.append(results["yahoo_finance"])

    # SEC filing
    if results.get("sec_filing"):
        sections.append("\n" + results["sec_filing"])

    # Insider transactions
    if results.get("insider_transactions"):
        sections.append("\n" + "=" * 60)
        sections.append(results["insider_transactions"])

    # Short interest & options
    if results.get("short_interest"):
        sections.append("\n" + "=" * 60)
        sections.append(results["short_interest"])

    # Analyst estimates
    if results.get("analyst_estimates"):
        sections.append("\n" + "=" * 60)
        sections.append(results["analyst_estimates"])

    # Earnings history
    if results.get("earnings_history"):
        sections.append("\n" + "=" * 60)
        sections.append(results["earnings_history"])

    # Institutional ownership
    if results.get("institutional_changes"):
        sections.append("\n" + "=" * 60)
        sections.append(results["institutional_changes"])

    # Macro
    if results.get("macro_data"):
        sections.append("\n" + "=" * 60)
        sections.append(results["macro_data"])

    # App store
    if results.get("app_store"):
        sections.append("\n" + "=" * 60)
        sections.append(results["app_store"])

    # Competitor comp table
    if results.get("comp_table"):
        sections.append("\n" + "=" * 60)
        sections.append(results["comp_table"])

    # Competitor 10-K excerpts
    if results.get("comp_filings"):
        sections.append("\n" + "=" * 60)
        sections.append("COMPETITOR SEC FILINGS (10-K excerpts):")
        sections.append("Use these to understand how competitors describe THEIR competitive position,")
        sections.append("what risks THEY flag, and where THEY see opportunities — this reveals the")
        sections.append("competitive dynamics that the target company's own 10-K won't tell you.")
        sections.append("=" * 60)
        sections.append(results["comp_filings"])

    # Web research
    if results.get("web_research"):
        sections.append("\n" + "=" * 60)
        sections.append(results["web_research"])

    return "\n".join(sections)
