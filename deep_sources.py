"""
Deep data sources — ClinicalTrials.gov + SEC XBRL structured facts.
These give Opus REAL primary-source numbers to cite, eliminating
the need to fabricate specificity.
"""

import requests

SEC_HEADERS = {"User-Agent": "QuantumIQ research@quantumiq.com"}


def fetch_clinical_trials(company_name: str, max_trials: int = 15) -> str:
    """
    Pull recent clinical trials for a pharma/biotech company.
    Returns formatted text for the LLM context.
    Returns empty string if company has no trials (not a biotech).
    """
    try:
        r = requests.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params={
                "query.spons": company_name,
                "pageSize": max_trials,
                "countTotal": "true",
                "sort": "LastUpdatePostDate:desc",
            },
            timeout=15,
        )
        if r.status_code != 200:
            return ""

        data = r.json()
        total = data.get("totalCount", 0)
        studies = data.get("studies", [])

        if total == 0 or not studies:
            return ""

        lines = [f"CLINICAL TRIALS PIPELINE ({total} total trials with {company_name} as sponsor):"]
        lines.append("Data from ClinicalTrials.gov — real trial registrations, phases, and statuses.")
        lines.append("")

        # Group by phase for a pipeline view
        by_phase = {"PHASE3": [], "PHASE2": [], "PHASE1": [], "OTHER": []}

        for s in studies:
            proto = s.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            cond = proto.get("conditionsModule", {})

            nct_id = ident.get("nctId", "")
            title = ident.get("briefTitle", "")[:120]
            phases = design.get("phases", [])
            phase = phases[0] if phases else "OTHER"
            overall_status = status.get("overallStatus", "")
            primary_completion = status.get("primaryCompletionDateStruct", {}).get("date", "")
            conditions = ", ".join(cond.get("conditions", [])[:3])

            entry = {
                "nct": nct_id,
                "title": title,
                "phase": phase,
                "status": overall_status,
                "completion": primary_completion,
                "conditions": conditions,
            }
            if phase in by_phase:
                by_phase[phase].append(entry)
            else:
                by_phase["OTHER"].append(entry)

        for phase_label, entries in [("PHASE3", "Phase 3 (late-stage):"), ("PHASE2", "Phase 2 (mid-stage):"),
                                      ("PHASE1", "Phase 1 (early):"), ("OTHER", "Other/Unspecified:")]:
            trials = by_phase.get(phase_label.replace(" (late-stage):", "").replace(" (mid-stage):", "").replace(" (early):", "").replace("/Unspecified:", ""), [])
            if trials:
                lines.append(f"\n{phase_label}")
                for t in trials[:5]:
                    lines.append(f"  {t['nct']} | {t['status']} | {t['conditions']}")
                    lines.append(f"    {t['title']}")
                    if t['completion']:
                        lines.append(f"    Primary completion: {t['completion']}")

        return "\n".join(lines)
    except Exception as e:
        print(f"[CLINICAL TRIALS] Error: {e}")
        return ""


def fetch_sec_xbrl_facts(cik: str, ticker: str) -> str:
    """
    Pull structured numerical facts from SEC XBRL company facts API.
    Returns formatted text with real financial line items for the LLM.
    """
    if not cik:
        return ""
    try:
        cik_padded = str(cik).zfill(10)
        r = requests.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json",
            headers=SEC_HEADERS,
            timeout=15,
        )
        if r.status_code != 200:
            return ""

        data = r.json()
        entity = data.get("entityName", ticker)
        us_gaap = data.get("facts", {}).get("us-gaap", {})

        if not us_gaap:
            return ""

        # Key line items to extract (in priority order)
        key_metrics = [
            # Income statement
            ("Revenues", "Revenue"),
            ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenue (ASC 606)"),
            ("CostOfRevenue", "Cost of Revenue"),
            ("GrossProfit", "Gross Profit"),
            ("ResearchAndDevelopmentExpense", "R&D Expense"),
            ("SellingGeneralAndAdministrativeExpense", "SG&A"),
            ("OperatingIncomeLoss", "Operating Income"),
            ("NetIncomeLoss", "Net Income"),
            ("EarningsPerShareDiluted", "Diluted EPS"),
            # Balance sheet
            ("Assets", "Total Assets"),
            ("AssetsCurrent", "Current Assets"),
            ("CashAndCashEquivalentsAtCarryingValue", "Cash"),
            ("Liabilities", "Total Liabilities"),
            ("LongTermDebt", "Long-term Debt"),
            ("StockholdersEquity", "Stockholders Equity"),
            ("InventoryNet", "Inventory"),
            ("AccountsReceivableNetCurrent", "Accounts Receivable"),
            # Cash flow
            ("NetCashProvidedByUsedInOperatingActivities", "Operating Cash Flow"),
            ("PaymentsToAcquirePropertyPlantAndEquipment", "CapEx"),
            ("PaymentsForRepurchaseOfCommonStock", "Share Buybacks"),
            ("PaymentsOfDividends", "Dividends Paid"),
            # Shares
            ("CommonStockSharesOutstanding", "Shares Outstanding"),
            ("WeightedAverageNumberOfDilutedSharesOutstanding", "Diluted Share Count"),
        ]

        lines = [f"SEC XBRL STRUCTURED FACTS for {entity}:"]
        lines.append("Real numerical data extracted directly from XBRL filings.")
        lines.append("")

        found_any = False
        for xbrl_key, display_name in key_metrics:
            if xbrl_key not in us_gaap:
                continue
            found_any = True

            fact = us_gaap[xbrl_key]
            units_dict = fact.get("units", {})
            # Prefer USD, then shares, then any unit
            unit_data = units_dict.get("USD") or units_dict.get("USD/shares") or units_dict.get("shares")
            if not unit_data:
                for k, v in units_dict.items():
                    unit_data = v
                    break
            if not unit_data:
                continue

            # Get the most recent annual (FY) and 4 most recent quarterly
            annual = [u for u in unit_data if u.get("fp") == "FY"]
            quarterly = [u for u in unit_data if u.get("fp") in ("Q1", "Q2", "Q3", "Q4")]

            # Sort by end date desc
            annual = sorted(annual, key=lambda x: x.get("end", ""), reverse=True)[:3]
            quarterly = sorted(quarterly, key=lambda x: x.get("end", ""), reverse=True)[:4]

            if annual or quarterly:
                lines.append(f"{display_name}:")
                for a in annual:
                    val = a.get("val", 0)
                    end = a.get("end", "")
                    if abs(val) >= 1e9:
                        val_str = f"${val/1e9:.2f}B"
                    elif abs(val) >= 1e6:
                        val_str = f"${val/1e6:.1f}M"
                    elif abs(val) >= 1e3:
                        val_str = f"${val/1e3:.0f}K"
                    else:
                        val_str = f"{val:,}"
                    lines.append(f"  FY {end[:4]}: {val_str}")
                for q in quarterly:
                    val = q.get("val", 0)
                    end = q.get("end", "")
                    fp = q.get("fp", "")
                    if abs(val) >= 1e9:
                        val_str = f"${val/1e9:.2f}B"
                    elif abs(val) >= 1e6:
                        val_str = f"${val/1e6:.1f}M"
                    else:
                        val_str = f"{val:,}"
                    lines.append(f"  {fp} {end}: {val_str}")
                lines.append("")

        if not found_any:
            return ""

        return "\n".join(lines)
    except Exception as e:
        print(f"[XBRL] Error: {e}")
        return ""
