"""
Grounded data lookup tool for Opus.

Instead of dumping raw text and hoping the model uses it correctly, we expose
a structured `lookup` function the model can call via tool_use. Every value
returned is traceable back to a real source — the model cannot fabricate
numbers that didn't come from an actual lookup call.
"""

import json
import re
from typing import Any


def _find_in_text(text: str, query: str, window: int = 400) -> list[str]:
    """Find passages in a large text block that contain the query terms."""
    if not text or not query:
        return []
    # Look for exact phrase first
    matches = []
    text_lower = text.lower()
    query_lower = query.lower()

    idx = 0
    while True:
        found = text_lower.find(query_lower, idx)
        if found < 0:
            break
        start = max(0, found - window // 2)
        end = min(len(text), found + len(query) + window // 2)
        matches.append(text[start:end].strip())
        idx = found + len(query)
        if len(matches) >= 3:
            break

    # Fallback: search for each word
    if not matches:
        words = [w for w in query_lower.split() if len(w) > 3]
        if words:
            for word in words[:3]:
                found = text_lower.find(word)
                if found >= 0:
                    start = max(0, found - window // 2)
                    end = min(len(text), found + len(word) + window // 2)
                    matches.append(text[start:end].strip())
                    if len(matches) >= 3:
                        break

    return matches


class LookupTool:
    """
    Grounded data source access for Opus. Holds all the research data and
    exposes structured lookup functions. Every response includes a citation
    string that the model is instructed to quote in its output.
    """

    def __init__(self, research: dict):
        self.research = research
        self._xbrl_cache: dict[str, dict] = {}
        self._parse_xbrl()

    def _parse_xbrl(self):
        """Parse the XBRL text blob into a lookup dict keyed by line item name."""
        xbrl_text = self.research.get("xbrl_facts", "")
        if not xbrl_text:
            return

        current_item = None
        for line in xbrl_text.split("\n"):
            # Header line like "Revenue:" or "R&D Expense:"
            if line and not line.startswith(" ") and line.endswith(":") and "=" not in line:
                current_item = line.rstrip(":").strip()
                self._xbrl_cache[current_item] = []
                self._xbrl_cache[current_item.lower()] = self._xbrl_cache[current_item]
                continue
            # Value line like "  FY 2025: $12.00B" or "  Q3 2025-09-30: $8.81B"
            if current_item and line.strip().startswith(("FY", "Q1", "Q2", "Q3", "Q4")):
                m = re.match(r'\s*(FY|Q[1-4])\s*([\d-]+):\s*(.*)', line)
                if m:
                    period, end, value = m.groups()
                    self._xbrl_cache[current_item].append({
                        "period": period.strip(),
                        "end": end.strip(),
                        "value": value.strip(),
                    })

    # ── Tool methods ──────────────────────────────────────────────────────

    def lookup_financial_metric(self, metric_name: str, period: str = "latest") -> dict:
        """
        Look up a specific financial line item from SEC XBRL data.
        metric_name examples: "Revenue", "Operating Income", "R&D Expense",
            "Net Income", "Cash", "Total Assets", "Stockholders Equity"
        period: "latest" (default), "FY" (all annual), "quarterly" (all Q), or a specific year like "2025"
        """
        if not self._xbrl_cache:
            return {"found": False, "reason": "No XBRL data available for this ticker", "source": "[13]"}

        # Fuzzy match the metric name
        keys = [k for k in self._xbrl_cache.keys() if not isinstance(self._xbrl_cache.get(k), list) or k == k.lower()]
        # Try exact match first
        exact = None
        for k in self._xbrl_cache.keys():
            if k.lower() == metric_name.lower():
                exact = k
                break
        if not exact:
            # Fuzzy match
            for k in self._xbrl_cache.keys():
                if metric_name.lower() in k.lower() or k.lower() in metric_name.lower():
                    exact = k
                    break

        if not exact:
            return {
                "found": False,
                "reason": f"Metric '{metric_name}' not found in XBRL data",
                "available_metrics": sorted(set(k for k in self._xbrl_cache.keys() if k[0].isupper())),
                "source": "[13]",
            }

        values = self._xbrl_cache[exact]
        if not values:
            return {"found": False, "reason": f"No values for {exact}", "source": "[13]"}

        # Filter by period
        if period == "latest":
            # Most recent FY + most recent Q
            annual = [v for v in values if v["period"] == "FY"]
            quarterly = [v for v in values if v["period"].startswith("Q")]
            annual = sorted(annual, key=lambda x: x["end"], reverse=True)[:1]
            quarterly = sorted(quarterly, key=lambda x: x["end"], reverse=True)[:1]
            result_values = annual + quarterly
        elif period == "FY":
            result_values = sorted([v for v in values if v["period"] == "FY"],
                                   key=lambda x: x["end"], reverse=True)[:5]
        elif period == "quarterly":
            result_values = sorted([v for v in values if v["period"].startswith("Q")],
                                   key=lambda x: x["end"], reverse=True)[:6]
        else:
            result_values = [v for v in values if period in v["end"]]

        return {
            "found": True,
            "metric": exact,
            "values": result_values,
            "source": "[13] SEC XBRL structured facts",
        }

    def lookup_clinical_trials(self, phase: str = None, status: str = None) -> dict:
        """
        Look up clinical trials for this company.
        phase: "1", "2", "3", or None for all
        status: "RECRUITING", "COMPLETED", "ACTIVE", or None for all
        Returns real trial IDs, titles, phases, statuses from ClinicalTrials.gov.
        """
        trials_text = self.research.get("clinical_trials", "")
        if not trials_text:
            return {
                "found": False,
                "reason": "No clinical trial data available (company has no trials as sponsor or not biotech/pharma)",
                "source": "[14]",
            }

        # Extract trial entries
        lines = trials_text.split("\n")
        trials = []
        current = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("NCT"):
                # Trial line: "NCT12345678 | STATUS | conditions"
                parts = [p.strip() for p in stripped.split("|")]
                if len(parts) >= 2:
                    current = {"nct": parts[0], "status": parts[1]}
                    if len(parts) > 2:
                        current["conditions"] = parts[2]
                    trials.append(current)
            elif current and stripped and not stripped.startswith("Phase") and "completion" not in stripped.lower():
                current["title"] = stripped
            elif current and "completion" in stripped.lower():
                current["completion"] = stripped

        # Try to extract phase from the section headers
        # The text groups trials by phase, so we can tag them
        current_phase = None
        for i, line in enumerate(lines):
            if "PHASE3" in line:
                current_phase = "3"
            elif "PHASE2" in line:
                current_phase = "2"
            elif "PHASE1" in line:
                current_phase = "1"
            elif current_phase and "NCT" in line:
                # Tag the most recently added trial
                for t in trials:
                    if t.get("nct") in line and "phase" not in t:
                        t["phase"] = current_phase

        # Filter
        filtered = trials
        if phase:
            filtered = [t for t in filtered if t.get("phase") == str(phase)]
        if status:
            filtered = [t for t in filtered if status.upper() in t.get("status", "").upper()]

        return {
            "found": True,
            "total_trials": len(trials),
            "filtered_trials": filtered[:10],
            "source": "[14] ClinicalTrials.gov pipeline",
        }

    def lookup_insider_transactions(self) -> dict:
        """Get recent insider buy/sell transactions with names, positions, and amounts."""
        text = self.research.get("insider_transactions", "")
        if not text:
            return {"found": False, "reason": "No insider transaction data", "source": "[6]"}
        return {"found": True, "raw_text": text, "source": "[6] Insider transactions (yfinance)"}

    def lookup_short_interest(self) -> dict:
        """Get short interest and options positioning data."""
        text = self.research.get("short_interest", "")
        if not text:
            return {"found": False, "reason": "No short interest data", "source": "[2]"}
        return {"found": True, "raw_text": text, "source": "[2] Short interest & options"}

    def lookup_analyst_estimates(self) -> dict:
        """Get analyst price targets, EPS estimates, recommendation trends."""
        text = self.research.get("analyst_estimates", "")
        if not text:
            return {"found": False, "reason": "No analyst data", "source": "[3]"}
        return {"found": True, "raw_text": text, "source": "[3] Analyst estimates & revisions"}

    def lookup_earnings_history(self) -> dict:
        """Get earnings beat/miss history (actual vs estimate)."""
        text = self.research.get("earnings_history", "")
        if not text:
            return {"found": False, "reason": "No earnings history data", "source": "[4]"}
        return {"found": True, "raw_text": text, "source": "[4] Earnings surprise history"}

    def lookup_institutional_holders(self) -> dict:
        """Get top institutional holders and their position changes."""
        text = self.research.get("institutional_changes", "")
        if not text:
            return {"found": False, "reason": "No institutional holder data", "source": "[5]"}
        return {"found": True, "raw_text": text, "source": "[5] Institutional ownership"}

    def lookup_10k_passage(self, query: str) -> dict:
        """
        Search the company's 10-K / 10-Q filing for a specific topic.
        Returns actual passages from the filing text.
        query: a topic or phrase like "revenue recognition", "R&D", "segment", "risk factors"
        """
        text = self.research.get("sec_filing", "")
        if not text:
            return {"found": False, "reason": "No 10-K available", "source": "[7]"}

        passages = _find_in_text(text, query, window=600)
        if not passages:
            return {"found": False, "reason": f"'{query}' not found in 10-K", "source": "[7]"}

        return {
            "found": True,
            "query": query,
            "passages": passages,
            "source": "[7] SEC 10-K / 10-Q filing",
        }

    def lookup_competitor_10k(self, competitor_ticker: str, query: str = None) -> dict:
        """Search a competitor's 10-K for a specific topic."""
        text = self.research.get("comp_filings", "")
        if not text:
            return {"found": False, "reason": "No competitor filings", "source": "[11]"}

        # Find the competitor's section
        marker = f"({competitor_ticker.upper()})"
        idx = text.find(marker)
        if idx < 0:
            return {
                "found": False,
                "reason": f"Competitor {competitor_ticker} not in research package",
                "available": self._extract_competitor_tickers(),
                "source": "[11]",
            }

        # Take the section from this competitor to the next "=" separator or end
        next_sep = text.find("====", idx + len(marker))
        section = text[idx:next_sep if next_sep > 0 else idx + 15000]

        if query:
            passages = _find_in_text(section, query, window=500)
            if not passages:
                return {
                    "found": False,
                    "reason": f"'{query}' not found in {competitor_ticker}'s 10-K",
                    "source": "[11]",
                }
            return {
                "found": True,
                "competitor": competitor_ticker,
                "query": query,
                "passages": passages,
                "source": f"[11] {competitor_ticker} 10-K",
            }

        return {
            "found": True,
            "competitor": competitor_ticker,
            "raw_section": section[:5000],
            "source": f"[11] {competitor_ticker} 10-K",
        }

    def _extract_competitor_tickers(self) -> list[str]:
        """Extract list of competitor tickers from comp_filings."""
        text = self.research.get("comp_filings", "")
        matches = re.findall(r'\(([A-Z]{1,5})\)', text)
        return list(set(matches))

    def lookup_comp_table(self) -> dict:
        """Get the comparable company valuation table."""
        text = self.research.get("comp_table", "")
        if not text:
            return {"found": False, "reason": "No comp table", "source": "[10]"}
        return {"found": True, "table": text, "source": "[10] Comparable companies"}

    def lookup_web_intelligence(self, query: str) -> dict:
        """Search the web research package for specific information."""
        text = self.research.get("web_research", "")
        if not text:
            return {"found": False, "reason": "No web research", "source": "[12]"}

        passages = _find_in_text(text, query, window=800)
        if not passages:
            return {"found": False, "reason": f"'{query}' not found in web research", "source": "[12]"}

        # Also find any [12.X] citations in the matched passages
        return {
            "found": True,
            "query": query,
            "passages": passages,
            "source": "[12] Web intelligence",
        }

    def lookup_yahoo_finance(self) -> dict:
        """Get the Yahoo Finance data summary (financials, ratios, holders, news)."""
        text = self.research.get("yahoo_finance", "")
        if not text:
            return {"found": False, "reason": "No Yahoo Finance data", "source": "[1]"}
        return {"found": True, "raw_text": text, "source": "[1] Yahoo Finance"}

    def lookup_macro(self) -> dict:
        """Get current macro environment data from FRED."""
        text = self.research.get("macro_data", "")
        if not text:
            return {"found": False, "reason": "No macro data", "source": "[8]"}
        return {"found": True, "raw_text": text, "source": "[8] FRED macro"}


# ── Anthropic tool schema ────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "lookup_financial_metric",
        "description": "Look up a specific financial line item from SEC XBRL structured data. Returns REAL quarterly/annual values. Use this for any revenue, margin, expense, or balance sheet number.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metric_name": {
                    "type": "string",
                    "description": "The financial line item name, e.g. 'Revenue', 'Operating Income', 'R&D Expense', 'Net Income', 'Cash', 'Total Assets'"
                },
                "period": {
                    "type": "string",
                    "description": "'latest' for most recent FY + Q, 'FY' for all annual, 'quarterly' for all quarters, or a year like '2025'",
                    "default": "latest"
                }
            },
            "required": ["metric_name"]
        }
    },
    {
        "name": "lookup_clinical_trials",
        "description": "Look up clinical trials for this company from ClinicalTrials.gov. Returns real NCT IDs, phases, statuses. ONLY works for pharma/biotech companies with registered trials.",
        "input_schema": {
            "type": "object",
            "properties": {
                "phase": {
                    "type": "string",
                    "description": "'1', '2', '3', or omit for all phases"
                },
                "status": {
                    "type": "string",
                    "description": "'RECRUITING', 'COMPLETED', 'ACTIVE', or omit for all statuses"
                }
            }
        }
    },
    {
        "name": "lookup_insider_transactions",
        "description": "Get recent insider buy/sell transactions with names, positions, and dollar amounts.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "lookup_short_interest",
        "description": "Get short interest, short % of float, put/call ratio, and options positioning.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "lookup_analyst_estimates",
        "description": "Get consensus analyst price targets, EPS estimates, recommendation trends, and recent revisions.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "lookup_earnings_history",
        "description": "Get earnings beat/miss history — actual EPS vs estimate for recent quarters.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "lookup_institutional_holders",
        "description": "Get top institutional holders and their recent position changes.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "lookup_10k_passage",
        "description": "Search the company's 10-K/10-Q filing for a specific topic. Returns actual passages from the filing text. Use this to verify any claim you want to make about management commentary, risk factors, or business disclosures.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic or phrase to search for in the 10-K, e.g. 'revenue recognition', 'customer concentration', 'Gemini', 'Azure', 'depreciation'"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "lookup_competitor_10k",
        "description": "Search a specific competitor's 10-K filing for a topic. Use to cross-reference how competitors describe the competitive landscape.",
        "input_schema": {
            "type": "object",
            "properties": {
                "competitor_ticker": {
                    "type": "string",
                    "description": "Competitor's ticker symbol, e.g. 'META', 'AMD'"
                },
                "query": {
                    "type": "string",
                    "description": "Optional: topic to search for in the competitor's 10-K. Omit to get a general excerpt."
                }
            },
            "required": ["competitor_ticker"]
        }
    },
    {
        "name": "lookup_comp_table",
        "description": "Get the comparable company valuation table (P/E, EV/EBITDA, margins, etc. for peers and target).",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "lookup_web_intelligence",
        "description": "Search the web research package for specific information (patents, job postings, earnings call commentary, congressional trading, research papers). Returns actual scraped passages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic to search for, e.g. 'patent filing', 'congressional trading', 'earnings call', 'hiring'"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "lookup_yahoo_finance",
        "description": "Get the Yahoo Finance summary: key financial metrics, ratios, price history, holders, news headlines.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "lookup_macro",
        "description": "Get current macro environment data: fed funds rate, CPI, GDP, unemployment, VIX, yield curve.",
        "input_schema": {"type": "object", "properties": {}}
    },
]


def execute_tool(tool_name: str, tool_input: dict, lookup: LookupTool) -> str:
    """Execute a tool call and return a formatted string for Opus."""
    try:
        method = getattr(lookup, tool_name, None)
        if not method:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        result = method(**tool_input) if tool_input else method()
        return json.dumps(result, indent=2, default=str)[:8000]  # cap size
    except Exception as e:
        return json.dumps({"error": str(e), "tool": tool_name})
