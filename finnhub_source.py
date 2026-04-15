"""
Finnhub data source — Yahoo Finance fallback.

Free tier: 60 API calls/minute, no credit card required.
Sign up at https://finnhub.io
Set FINNHUB_API_KEY env variable to enable.

Provides:
- Quote (real-time price, change, day range)
- Company profile (name, sector, industry, market cap)
- Financial statements (income, balance, cash flow)
- Insider transactions
- Recommendation trends
- News
"""

import os
import requests
from typing import Optional

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
BASE_URL = "https://finnhub.io/api/v1"


def _get(endpoint: str, params: dict = None, timeout: int = 10) -> Optional[dict]:
    """GET request to Finnhub API with key."""
    if not FINNHUB_API_KEY:
        return None
    params = params or {}
    params["token"] = FINNHUB_API_KEY
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception as e:
        print(f"[FINNHUB] {endpoint} error: {e}")
        return None


def fetch_finnhub_profile(ticker: str) -> dict:
    """Get company profile: name, sector, market cap, share count."""
    data = _get("/stock/profile2", {"symbol": ticker})
    if not data:
        return {}
    return {
        "name": data.get("name", ""),
        "exchange": data.get("exchange", ""),
        "country": data.get("country", ""),
        "currency": data.get("currency", ""),
        "industry": data.get("finnhubIndustry", ""),
        "ipo": data.get("ipo", ""),
        "marketCap": (data.get("marketCapitalization", 0) or 0) * 1_000_000,  # they report in millions
        "sharesOutstanding": (data.get("shareOutstanding", 0) or 0) * 1_000_000,
        "weburl": data.get("weburl", ""),
        "logo": data.get("logo", ""),
        "phone": data.get("phone", ""),
    }


def fetch_finnhub_quote(ticker: str) -> dict:
    """Get real-time quote: current price, change, day range."""
    data = _get("/quote", {"symbol": ticker})
    if not data:
        return {}
    return {
        "current": data.get("c", 0),
        "change": data.get("d", 0),
        "change_pct": data.get("dp", 0),
        "high": data.get("h", 0),
        "low": data.get("l", 0),
        "open": data.get("o", 0),
        "previousClose": data.get("pc", 0),
    }


def fetch_finnhub_metrics(ticker: str) -> dict:
    """Get key financial metrics: P/E, P/B, ROE, margins, etc."""
    data = _get("/stock/metric", {"symbol": ticker, "metric": "all"})
    if not data:
        return {}
    metrics = data.get("metric", {})
    return metrics or {}


def fetch_finnhub_insider_transactions(ticker: str) -> list:
    """Get recent insider buy/sell transactions."""
    data = _get("/stock/insider-transactions", {"symbol": ticker})
    if not data:
        return []
    return data.get("data", [])[:25]


def fetch_finnhub_recommendation_trends(ticker: str) -> list:
    """Get analyst recommendation history (buy/hold/sell counts over time)."""
    data = _get("/stock/recommendation", {"symbol": ticker})
    if not data:
        return []
    return data[:6] if isinstance(data, list) else []


def fetch_finnhub_earnings(ticker: str) -> list:
    """Get historical EPS surprises (actual vs estimate)."""
    data = _get("/stock/earnings", {"symbol": ticker, "limit": 8})
    if not data:
        return []
    return data if isinstance(data, list) else []


def fetch_finnhub_news(ticker: str, days_back: int = 30) -> list:
    """Get recent company news."""
    from datetime import datetime, timedelta
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    data = _get("/company-news", {"symbol": ticker, "from": start, "to": end})
    if not data:
        return []
    return data[:15] if isinstance(data, list) else []


def build_finnhub_summary(ticker: str) -> dict:
    """
    Pull everything Finnhub offers and return a structured summary.
    Used as a Yahoo Finance fallback.
    """
    if not FINNHUB_API_KEY:
        return {"available": False, "reason": "FINNHUB_API_KEY not set"}

    profile = fetch_finnhub_profile(ticker)
    quote = fetch_finnhub_quote(ticker)
    metrics = fetch_finnhub_metrics(ticker)
    insiders = fetch_finnhub_insider_transactions(ticker)
    recs = fetch_finnhub_recommendation_trends(ticker)
    earnings = fetch_finnhub_earnings(ticker)
    news = fetch_finnhub_news(ticker)

    has_data = any([profile.get("name"), quote.get("current"), metrics])

    return {
        "available": has_data,
        "profile": profile,
        "quote": quote,
        "metrics": metrics,
        "insiders": insiders,
        "recommendations": recs,
        "earnings": earnings,
        "news": news,
    }


def format_finnhub_for_llm(data: dict) -> str:
    """Format Finnhub data into text for the LLM context."""
    if not data.get("available"):
        return ""

    lines = ["FINANCIAL DATA (Finnhub):"]
    profile = data.get("profile", {})
    quote = data.get("quote", {})
    metrics = data.get("metrics", {})

    if profile.get("name"):
        lines.append(f"COMPANY: {profile['name']}")
        lines.append(f"Industry: {profile.get('industry', 'N/A')}")
        lines.append(f"Country: {profile.get('country', 'N/A')}")
        if profile.get("marketCap"):
            mc = profile["marketCap"]
            mc_str = f"${mc/1e12:.2f}T" if mc >= 1e12 else f"${mc/1e9:.1f}B" if mc >= 1e9 else f"${mc/1e6:.0f}M"
            lines.append(f"Market Cap: {mc_str}")
        if profile.get("sharesOutstanding"):
            lines.append(f"Shares Outstanding: {profile['sharesOutstanding']:,.0f}")
        lines.append(f"IPO Date: {profile.get('ipo', 'N/A')}")

    if quote.get("current"):
        lines.append(f"\nPRICE:")
        lines.append(f"  Current: ${quote['current']:.2f}")
        lines.append(f"  Change: ${quote.get('change', 0):.2f} ({quote.get('change_pct', 0):.2f}%)")
        lines.append(f"  Day Range: ${quote.get('low', 0):.2f} - ${quote.get('high', 0):.2f}")
        lines.append(f"  Previous Close: ${quote.get('previousClose', 0):.2f}")

    if metrics:
        lines.append(f"\nKEY METRICS:")
        key_metrics = [
            ("peTTM", "P/E (TTM)"),
            ("peExclExtraTTM", "P/E (Excl Extra)"),
            ("pbAnnual", "P/B"),
            ("psAnnual", "P/S"),
            ("evToEbitdaTTM", "EV/EBITDA"),
            ("currentRatioAnnual", "Current Ratio"),
            ("totalDebtToEquityAnnual", "D/E"),
            ("roeTTM", "ROE TTM"),
            ("roaTTM", "ROA TTM"),
            ("netProfitMarginTTM", "Net Margin TTM"),
            ("grossMarginTTM", "Gross Margin TTM"),
            ("operatingMarginTTM", "Op Margin TTM"),
            ("revenueGrowthTTMYoy", "Revenue Growth TTM YoY"),
            ("epsGrowthTTMYoy", "EPS Growth TTM YoY"),
            ("dividendYieldIndicatedAnnual", "Dividend Yield"),
            ("beta", "Beta"),
            ("52WeekHigh", "52W High"),
            ("52WeekLow", "52W Low"),
        ]
        for key, label in key_metrics:
            val = metrics.get(key)
            if val is not None:
                if isinstance(val, (int, float)):
                    if abs(val) < 10 and "Growth" in label or "Yield" in label or "Margin" in label or "ROE" in label or "ROA" in label:
                        lines.append(f"  {label}: {val:.2%}")
                    else:
                        lines.append(f"  {label}: {val:.2f}")
                else:
                    lines.append(f"  {label}: {val}")

    insiders = data.get("insiders", [])
    if insiders:
        lines.append(f"\nINSIDER TRANSACTIONS (recent {len(insiders)}):")
        for ins in insiders[:15]:
            name = ins.get("name", "Unknown")
            change = ins.get("change", 0)
            shares = ins.get("share", 0)
            tx_price = ins.get("transactionPrice", 0)
            tx_date = ins.get("transactionDate", "")
            direction = "BUY" if change > 0 else "SELL"
            lines.append(f"  {tx_date} | {name} | {direction} {abs(change):,} shares @ ${tx_price:.2f}")

    recs = data.get("recommendations", [])
    if recs:
        lines.append(f"\nANALYST RECOMMENDATION TRENDS (last {len(recs)} months):")
        for r in recs:
            period = r.get("period", "")
            sb = r.get("strongBuy", 0)
            b = r.get("buy", 0)
            h = r.get("hold", 0)
            s = r.get("sell", 0)
            ss = r.get("strongSell", 0)
            lines.append(f"  {period}: SB={sb} B={b} H={h} S={s} SS={ss}")

    earnings = data.get("earnings", [])
    if earnings:
        lines.append(f"\nEARNINGS SURPRISE HISTORY:")
        for e in earnings:
            period = e.get("period", "")
            actual = e.get("actual")
            estimate = e.get("estimate")
            surprise_pct = e.get("surprisePercent")
            if actual is not None and estimate is not None:
                beat = "BEAT" if actual > estimate else "MISS"
                lines.append(f"  {period}: ${actual} actual vs ${estimate} est — {beat} ({surprise_pct}%)")

    news = data.get("news", [])
    if news:
        lines.append(f"\nRECENT NEWS HEADLINES:")
        for n in news[:10]:
            headline = n.get("headline", "")
            if headline:
                lines.append(f"  - {headline}")

    return "\n".join(lines)
