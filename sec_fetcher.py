"""
SEC EDGAR Data Fetcher
Pulls 10-K, 10-Q, and 8-K filings from SEC EDGAR.
Extracts key sections: MD&A, Risk Factors, Business Overview, Earnings Releases.
"""

import requests
import re
import warnings

HEADERS = {"User-Agent": "QuantumIQ research@quantumiq.com"}


def html_to_text(html: str) -> str:
    """Convert HTML/XBRL to plain text using BeautifulSoup."""
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

    soup = BeautifulSoup(html, "lxml")
    # Remove script/style elements
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'[�]', '', text)  # Remove replacement chars
    return text.strip()


def get_cik(ticker: str) -> str | None:
    """Get CIK number from ticker symbol."""
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=HEADERS, timeout=10
        )
        data = r.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker") == ticker_upper:
                return str(entry["cik_str"]).zfill(10)
    except Exception:
        pass
    return None


def get_company_info_from_sec(ticker: str) -> dict:
    """Get company name + CIK from SEC EDGAR (free, no rate limit)."""
    result = {"cik": None, "name": None}
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=HEADERS, timeout=10
        )
        data = r.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker") == ticker_upper:
                result["cik"] = str(entry["cik_str"]).zfill(10)
                result["name"] = entry.get("title", "")
                return result
    except Exception:
        pass
    return result


def get_filing_urls(cik: str, form_types: list[str] = None, count: int = 5) -> list[dict]:
    """Get recent filing URLs from EDGAR submissions API."""
    if form_types is None:
        form_types = ["10-K", "10-Q", "8-K"]

    try:
        r = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=HEADERS, timeout=10
        )
        data = r.json()
        recent = data["filings"]["recent"]

        filings = []
        for i in range(len(recent["form"])):
            if recent["form"][i] in form_types and len(filings) < count:
                acc = recent["accessionNumber"][i].replace("-", "")
                cik_num = cik.lstrip("0")
                doc = recent["primaryDocument"][i]
                filings.append({
                    "form": recent["form"][i],
                    "date": recent["filingDate"][i],
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc}/{doc}",
                    "items": recent.get("items", [""])[i] if i < len(recent.get("items", [])) else "",
                })
        return filings
    except Exception as e:
        print(f"[SEC] Error getting filings: {e}")
        return []


def download_filing(url: str, max_size: int = 2_000_000) -> str:
    """Download a filing and convert to text."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return ""
        # Truncate very large filings
        html = r.text[:max_size]
        return html_to_text(html)
    except Exception as e:
        print(f"[SEC] Error downloading {url}: {e}")
        return ""


def smart_truncate(text: str, max_chars: int = 50000) -> str:
    """
    Truncate filing text smartly.
    Skip the table of contents and boilerplate at the start,
    then take up to max_chars of the actual content.
    """
    # Try to skip past the TOC by finding where "Part I" or "PART I" content starts
    # (not the TOC reference, but the actual section)
    part_i_matches = list(re.finditer(r'\n\s*(?:PART|Part)\s+I\b[^I]', text))

    # Use the LAST "Part I" match (usually the real content, not TOC)
    if len(part_i_matches) >= 2:
        start = part_i_matches[-1].start()
    elif part_i_matches:
        start = part_i_matches[0].start()
    else:
        # No Part I found — skip first 5% (usually cover page / TOC)
        start = len(text) // 20

    content = text[start:start + max_chars]
    if len(text[start:]) > max_chars:
        content += "\n\n[...filing truncated, additional sections available in full 10-K]"
    return content


def fetch_sec_data(ticker: str, status_callback=None) -> dict:
    """
    Fetch SEC filings for a ticker.
    Returns raw filing text — Claude handles the analysis.
    No regex section extraction = works for any company's format.
    """
    result = {
        "annual_filing": "",      # Full 10-K or 10-Q text (truncated)
        "earnings_release": "",   # Latest 8-K earnings
        "filings_found": [],
        "annual_type": "",        # "10-K" or "10-Q"
        "annual_date": "",
    }

    if status_callback:
        status_callback("Looking up SEC filings...")

    cik = get_cik(ticker)
    if not cik:
        print(f"[SEC] Could not find CIK for {ticker}")
        return result

    # Get recent filings
    filings = get_filing_urls(cik, ["10-K", "10-Q", "8-K"], count=8)
    result["filings_found"] = [
        f"{f['form']} ({f['date']})" for f in filings
    ]

    # Find the latest 10-K AND latest 10-Q (pull both, not just one)
    ten_k = next((f for f in filings if f["form"] == "10-K"), None)
    ten_q = next((f for f in filings if f["form"] == "10-Q"), None)

    # Find earnings-related 8-Ks (item 2.02 = earnings release)
    earnings_8k = next(
        (f for f in filings if f["form"] == "8-K" and "2.02" in str(f.get("items", ""))),
        None
    )

    # Download 10-K
    if ten_k:
        if status_callback:
            status_callback(f"Downloading 10-K ({ten_k['date']})...")
        text = download_filing(ten_k["url"])
        if text:
            result["annual_filing"] = smart_truncate(text, max_chars=80000)
            result["annual_type"] = "10-K"
            result["annual_date"] = ten_k["date"]

    # Download 10-Q (most recent quarterly — has the freshest numbers)
    if ten_q:
        if status_callback:
            status_callback(f"Downloading 10-Q ({ten_q['date']})...")
        text = download_filing(ten_q["url"])
        if text:
            result["quarterly_filing"] = smart_truncate(text, max_chars=30000)
            result["quarterly_date"] = ten_q["date"]

    # Fall back: if no 10-K available, use 10-Q as the annual slot
    if not result.get("annual_filing") and result.get("quarterly_filing"):
        result["annual_filing"] = result["quarterly_filing"]
        result["annual_type"] = "10-Q"
        result["annual_date"] = result.get("quarterly_date", "")

    # Download latest earnings 8-K
    if earnings_8k:
        if status_callback:
            status_callback("Downloading latest earnings release...")
        earnings_text = download_filing(earnings_8k["url"])
        if earnings_text:
            result["earnings_release"] = earnings_text[:10000]

    return result


def build_sec_summary(sec_data: dict) -> str:
    """Build a text summary from SEC data for the LLM."""
    sections = []

    filings = sec_data.get("filings_found", [])
    if filings:
        sections.append(f"SEC FILINGS FOUND: {', '.join(filings)}")

    if sec_data.get("annual_filing"):
        filing_type = sec_data.get("annual_type", "10-K")
        filing_date = sec_data.get("annual_date", "")
        sections.append("=" * 60)
        sections.append(f"FULL {filing_type} FILING TEXT (filed {filing_date}):")
        sections.append("Raw text from the company's annual filing: Business Overview, Risk Factors, MD&A, Financial Statements.")
        sections.append("=" * 60)
        sections.append(sec_data["annual_filing"])

    if sec_data.get("quarterly_filing") and sec_data.get("annual_type") != "10-Q":
        quarterly_date = sec_data.get("quarterly_date", "")
        sections.append("\n" + "=" * 60)
        sections.append(f"FULL 10-Q QUARTERLY FILING TEXT (filed {quarterly_date}):")
        sections.append("Raw text from the latest quarterly filing — most recent numbers and management commentary.")
        sections.append("=" * 60)
        sections.append(sec_data["quarterly_filing"])

    if sec_data.get("earnings_release"):
        sections.append("\n" + "=" * 60)
        sections.append("LATEST EARNINGS RELEASE (8-K):")
        sections.append("=" * 60)
        sections.append(sec_data["earnings_release"])

    return "\n\n".join(sections) if sections else ""
