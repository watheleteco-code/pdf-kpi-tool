import os
import re
import json
import tempfile
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pdfplumber
import pandas as pd
import camelot

# Optional dependency (recommended): pip install openai
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================================================
# Streamlit setup
# ============================================================
st.set_page_config(page_title="PDF KPI Tool (AI Row Mapping)", layout="wide")
st.title("PDF Financial Statement Analyzer (AI Row Mapping)")

uploaded_file = st.file_uploader("Upload a financial statement PDF", type=["pdf"])


# ============================================================
# Helpers
# ============================================================
HEADER_KEYWORDS = [
    "years ended", "year ended", "for the year ended",
    "years ended december", "year ended december",
    "unaudited", "in thousands", "in millions"
]
NUM_PATTERN = r"\(?-?\d[\d,]*\.?\d*\)?"


def clean_number(x):
    """Convert PDF-extracted numeric strings to float; handle commas and parentheses negatives."""
    if isinstance(x, str):
        x = x.replace(",", "").replace("$", "").strip()
        if x.startswith("(") and x.endswith(")"):
            x = "-" + x[1:-1]
        if x in ["-", "", "–", "—"]:
            return None
    try:
        return float(x)
    except Exception:
        return None


def safe_div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure first col is label, others are values."""
    df = df.copy()
    df.columns = range(df.shape[1])
    df.rename(columns={0: "label"}, inplace=True)
    return df


def is_text_based_pdf(pdf_path: str) -> bool:
    """Heuristic: text-based PDFs have a meaningful number of text chars."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_chars = sum(len(page.chars) for page in pdf.pages)
        return total_chars > 200
    except Exception:
        return False


def extract_tables_with_pdfplumber(pdf_path: str, max_pages: int = 10):
    """Extract tables with pdfplumber (works best for bordered/line-detected tables)."""
    extracted = []
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = min(len(pdf.pages), max_pages)
        for page_idx in range(num_pages):
            page = pdf.pages[page_idx]
            tables = page.extract_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "intersection_tolerance": 5,
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "edge_min_length": 20,
                    "min_words_vertical": 2,
                    "min_words_horizontal": 1,
                    "text_tolerance": 3,
                }
            )
            for t_idx, table in enumerate(tables):
                df = pd.DataFrame(table)
                df = df.dropna(how="all").dropna(axis=1, how="all")
                if df.shape[0] < 2 or df.shape[1] < 2:
                    continue
                extracted.append({"page": page_idx + 1, "table_index": t_idx + 1, "df": df})
    return extracted


def extract_tables_with_camelot(pdf_path: str, max_pages: int = 10):
    """
    Try Camelot in both stream and lattice modes.
    Some PDFs work only with one of them.
    """
    results = []
    pages = ",".join(str(i) for i in range(1, max_pages + 1))

    for flavor in ["stream", "lattice"]:
        try:
            camelot_tables = camelot.read_pdf(pdf_path, pages=pages, flavor=flavor)
            for i, table in enumerate(camelot_tables):
                df = table.df
                df = df.dropna(how="all").dropna(axis=1, how="all")
                if df.shape[0] < 2 or df.shape[1] < 2:
                    continue
                results.append({"page": f"{flavor}", "table_index": i + 1, "df": df})
            if len(results) > 0:
                return results
        except Exception as e:
            st.warning(f"Camelot ({flavor}) failed: {e}")

    return results


def extract_lines(pdf_path: str, max_pages: int = 10):
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:max_pages]:
            text = page.extract_text() or ""
            for line in text.splitlines():
                line = line.strip()
                if line:
                    lines.append(line)
    return lines


def detect_years(lines):
    for line in lines[:25]:
        years = re.findall(r"\b(?:19|20)\d{2}\b", line)
        if len(years) >= 2:
            return years[:2]
    return None


def parse_text_aligned_statement(lines, years=None):
    rows = []
    year_set = set(years or [])

    for line in lines:
        lower = line.lower()

        if any(k in lower for k in HEADER_KEYWORDS):
            continue

        nums = re.findall(NUM_PATTERN, line)
        if len(nums) < 2:
            continue

        filtered = []
        for tok in nums:
            tok_clean = tok.strip("()").replace(",", "")
            if tok_clean in year_set:
                continue
            filtered.append(tok)

        if len(filtered) < 2:
            continue

        v1 = clean_number(filtered[0])
        v2 = clean_number(filtered[1])

        if v1 is not None and 1 <= v1 <= 31 and (v2 is None or abs(v2) < 1000):
            continue

        first = filtered[0]
        idx = line.find(first)
        label = line[:idx].strip(" .:-\t")
        if len(label) < 2:
            continue

        label = re.sub(r"[.\u2026]+", " ", label)   # remove dotted leaders
        label = label.replace("$", "").strip()
        label = re.sub(r"\s+", " ", label)

        rows.append([label, v1, v2])

    if not rows:
        return None

    if years and len(years) == 2:
        cols = ["label", years[0], years[1]]
    else:
        cols = ["label", "value_1", "value_2"]

    return pd.DataFrame(rows, columns=cols)


def to_row_payload(df: pd.DataFrame, value_col: str, max_rows: int = 250) -> List[dict]:
    """
    Create a compact payload for the model:
    [{"label": "...", "value": 123.0}, ...]
    """
    rows = []
    for _, r in df.iterrows():
        lab = str(r.get("label", "")).strip()
        if not lab:
            continue
        val = r.get(value_col, None)
        val = None if (isinstance(val, float) and pd.isna(val)) else val
        # include even if val is None; some labels still help context, but keep small
        rows.append({"label": lab[:200], "value": val})
        if len(rows) >= max_rows:
            break
    return rows


# ============================================================
# AI mapping layer (Structured Outputs via JSON schema)
# ============================================================
BALANCE_SHEET_FIELDS = [
    {"name": "total_assets", "description": "Total Assets"},
    {"name": "total_liabilities", "description": "Total Liabilities"},
    {"name": "total_equity", "description": "Total Equity / Shareholders' Equity"},
    {"name": "current_assets", "description": "Total Current Assets (if present)"},
    {"name": "current_liabilities", "description": "Total Current Liabilities (if present)"},
    {"name": "cash_and_equivalents", "description": "Cash and cash equivalents (if present)"},
    {"name": "short_term_debt", "description": "Short-term debt / current portion of LT debt (if present)"},
    {"name": "long_term_debt", "description": "Long-term debt (if present)"},
]

INCOME_STATEMENT_FIELDS = [
    {"name": "revenue", "description": "Total revenue / sales / turnover"},
    {"name": "cogs", "description": "Cost of goods sold / cost of sales"},
    {"name": "operating_income", "description": "Operating income / EBIT"},
    {"name": "net_income", "description": "Net income / profit for the period"},
    {"name": "income_tax", "description": "Income tax expense / provision for taxes (if present)"},
    {"name": "interest_expense", "description": "Interest expense / net finance costs (if present)"},
]


def build_mapping_schema(canonical_fields: List[dict]) -> dict:
    """
    JSON schema for structured output. Must be an object at the top-level.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "statement_type": {"type": "string"},
            "mappings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "canonical_field": {"type": "string"},
                        "source_row_label": {"type": "string"},
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["canonical_field", "source_row_label", "confidence", "reason"],
                },
            },
            "notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["statement_type", "mappings", "notes"],
    }


def ai_map_rows(
    df: pd.DataFrame,
    statement_type: str,
    period_col: str,
    model: str,
    api_key: Optional[str],
    constraints: Optional[List[str]] = None,
) -> Dict:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    if not api_key:
        raise RuntimeError("No API key provided. Set OPENAI_API_KEY or enter it in the sidebar.")

    client = OpenAI(api_key=api_key)

    if statement_type == "Balance Sheet":
        fields = BALANCE_SHEET_FIELDS
    elif statement_type == "Income Statement":
        fields = INCOME_STATEMENT_FIELDS
    else:
        raise ValueError("AI mapping currently implemented for Balance Sheet and Income Statement only.")

    row_payload = to_row_payload(df, period_col)

    schema = build_mapping_schema(fields)

    canonical_field_list = [{"name": f["name"], "description": f["description"]} for f in fields]
    constraint_text = "\n".join([f"- {c}" for c in (constraints or [])])

    instructions = f"""
You are mapping extracted financial statement rows to canonical fields.

Return ONLY JSON that matches the provided schema.

Task:
- For EACH canonical field, choose the single best matching 'source_row_label' from the provided rows.
- Prefer totals/subtotals (e.g., "Total assets") when available.
- If a field is truly not present, map it to an empty string and set confidence to 0.0.
- Confidence is 0.0 to 1.0.
- 'reason' should be short and specific (e.g., "Exact match to 'Total assets' row").

Canonical fields:
{json.dumps(canonical_field_list, indent=2)}

Additional constraints (if any):
{constraint_text if constraint_text else "- (none)"}

Rows (label + value):
{json.dumps(row_payload, indent=2)}
""".strip()

    # Structured output request (JSON schema enforced)
    resp = client.responses.create(
        model=model,
        input=instructions,
        text={
            "format": {
                "type": "json_schema",
                "name": "statement_row_mapping",
                "strict": True,
                "schema": schema,
            }
        },
    )

    # The SDK returns text; parse JSON.
    # This is intentionally strict: if it isn't valid JSON, we fail fast.
    out_text = resp.output_text
    return json.loads(out_text)


def apply_mapping(df: pd.DataFrame, period_col: str, mapping_json: Dict) -> Tuple[Dict[str, Optional[float]], Dict[str, dict]]:
    """
    Convert mapping into canonical numeric values.
    Returns:
      values: {canonical_field: number or None}
      meta:   {canonical_field: {"label": ..., "confidence": ..., "reason": ...}}
    """
    label_to_value = {
        str(r["label"]).strip(): r.get(period_col, None)
        for _, r in df.iterrows()
        if str(r.get("label", "")).strip()
    }

    values = {}
    meta = {}

    for m in mapping_json.get("mappings", []):
        cf = m["canonical_field"]
        src = (m["source_row_label"] or "").strip()
        conf = float(m.get("confidence", 0.0))
        reason = m.get("reason", "")

        meta[cf] = {"label": src, "confidence": conf, "reason": reason}

        if not src:
            values[cf] = None
            continue

        # Robust retrieval: direct match first, then case-insensitive match
        if src in label_to_value:
            values[cf] = label_to_value.get(src)
        else:
            src_lower = src.lower()
            hit = None
            for lab, val in label_to_value.items():
                if lab.lower() == src_lower:
                    hit = val
                    break
            values[cf] = hit

    return values, meta


# ============================================================
# Sanity checks (no human-in-loop gating)
# ============================================================
def balance_sheet_checks(vals: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
    """
    Returns (ok, failures).
    """
    failures = []

    a = vals.get("total_assets")
    l = vals.get("total_liabilities")
    e = vals.get("total_equity")

    # Core identity: A ≈ L + E
    if a is not None and l is not None and e is not None:
        denom = max(abs(a), 1.0)
        diff = abs(a - (l + e)) / denom
        if diff > 0.02:  # 2% tolerance
            failures.append(f"Balance check failed: total_assets ({a}) != liabilities+equity ({l+e}). Δ={diff:.2%}")
    else:
        failures.append("Balance check skipped: missing one of total_assets / total_liabilities / total_equity.")

    # Hierarchy: current assets <= total assets
    ca = vals.get("current_assets")
    if ca is not None and a is not None and ca > a * 1.05:
        failures.append("Hierarchy check failed: current_assets appears greater than total_assets.")

    cl = vals.get("current_liabilities")
    if cl is not None and l is not None and cl > l * 1.05:
        failures.append("Hierarchy check failed: current_liabilities appears greater than total_liabilities.")

    ok = len([f for f in failures if "skipped" not in f.lower()]) == 0
    return ok, failures


# ============================================================
# KPI calculators (deterministic)
# ============================================================
def compute_income_kpis(vals: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    revenue = vals.get("revenue")
    op_inc = vals.get("operating_income")
    net_inc = vals.get("net_income")
    tax = vals.get("income_tax")
    interest = vals.get("interest_expense")

    # Simple EBT estimate: operating income - interest (best-effort)
    ebt = None
    if op_inc is not None:
        ebt = op_inc - (interest or 0)

    return {
        "Operating Margin": safe_div(op_inc, revenue),
        "Net Margin": safe_div(net_inc, revenue),
        "Effective Tax Rate": safe_div(tax, ebt),
    }


def compute_balance_kpis(vals: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    cash = vals.get("cash_and_equivalents")
    ca = vals.get("current_assets")
    cl = vals.get("current_liabilities")
    a = vals.get("total_assets")
    l = vals.get("total_liabilities")
    e = vals.get("total_equity")
    st_debt = vals.get("short_term_debt") or 0
    lt_debt = vals.get("long_term_debt") or 0
    total_debt = None if (vals.get("short_term_debt") is None and vals.get("long_term_debt") is None) else (st_debt + lt_debt)

    return {
        "Current Ratio": safe_div(ca, cl),
        "Cash Ratio": safe_div(cash, cl),
        "Debt to Equity": safe_div(total_debt, e),
        "Debt Ratio": safe_div(l, a),
        "Equity Ratio": safe_div(e, a),
    }


# ============================================================
# Main app flow
# ============================================================
if uploaded_file is None:
    st.stop()

# Sidebar: AI config
st.sidebar.header("AI Mapping Settings")
api_key = st.sidebar.text_input("OpenAI API Key (optional if env var set)", type="password")
api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
model = st.sidebar.text_input("Model", value="gpt-4o-mini")

st.success("PDF uploaded successfully")
st.write(f"Filename: {uploaded_file.name}")

# Save uploaded file to a temp path (Camelot is much happier with paths)
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_file.getbuffer())
    pdf_path = tmp.name

if not is_text_based_pdf(pdf_path):
    st.error("Scanned PDF detected")
    st.warning("This PDF appears to be scanned (image-based). Extraction may not work without OCR.")
    st.stop()

st.success("Text-based PDF detected")

max_pages = st.slider("Pages to scan (from start)", min_value=1, max_value=50, value=15)

# Try extractors
with st.spinner("Extracting tables from the PDF..."):
    tables = extract_tables_with_pdfplumber(pdf_path, max_pages=max_pages)

if len(tables) == 0:
    st.info("No tables found with pdfplumber. Trying Camelot...")
    tables = extract_tables_with_camelot(pdf_path, max_pages=max_pages)

st.write(f"Tables found: **{len(tables)}**")

statement_df = None

# If still nothing, do text-line parsing fallback
if len(tables) == 0:
    st.info("No tables extracted. Parsing as text-aligned statement...")
    lines = extract_lines(pdf_path, max_pages=max_pages)
    years = detect_years(lines)
    df_text = parse_text_aligned_statement(lines, years=years)

    if df_text is None:
        st.error("Could not parse statement from text.")
        st.stop()

    st.subheader("Parsed statement (text-based)")
    st.dataframe(df_text, use_container_width=True)
    statement_df = df_text

else:
    options = [
        f"Page {t['page']} — Table {t['table_index']} ({t['df'].shape[0]}x{t['df'].shape[1]})"
        for t in tables
    ]
    selected = st.selectbox("Select a table to preview", options=options, index=0)
    selected_idx = options.index(selected)

    st.subheader("Selected table preview")
    st.dataframe(tables[selected_idx]["df"], use_container_width=True)

    raw_df = tables[selected_idx]["df"]
    df = normalize_table(raw_df)

    for col in df.columns[1:]:
        df[col] = df[col].apply(clean_number)

    st.subheader("Normalized table")
    st.dataframe(df, use_container_width=True)

    statement_df = df

st.divider()
st.subheader("KPI Engine")

statement_type = st.radio(
    "What kind of statement is this?",
    ["Income Statement", "Balance Sheet", "Cash Flow Statement"],
    index=0,
    key="statement_type",
)

cols = [c for c in statement_df.columns if c != "label"]
if not cols:
    st.error("No numeric columns detected.")
    st.stop()

period_col = st.selectbox("Select period/column for analysis", cols, index=0)

if statement_type == "Cash Flow Statement":
    st.info("Cash Flow KPIs not implemented in this version.")
    st.stop()

# AI mapping + KPIs
st.subheader("AI Row Mapping → Deterministic KPIs")

if OpenAI is None:
    st.error("Missing dependency: openai. Install with: pip install openai")
    st.stop()

if not api_key:
    st.warning("No API key set. Add OPENAI_API_KEY to your environment or paste it in the sidebar.")
    st.stop()

run_ai = st.button("Run AI Mapping + KPIs", type="primary")

if run_ai:
    with st.spinner("Calling AI to map rows..."):
        # First pass
        mapping = ai_map_rows(
            df=statement_df,
            statement_type=statement_type,
            period_col=period_col,
            model=model,
            api_key=api_key,
            constraints=None,
        )

        vals, meta = apply_mapping(statement_df, period_col, mapping)

        # Sanity checks (balance sheet only) + one retry with constraints
        if statement_type == "Balance Sheet":
            ok, failures = balance_sheet_checks(vals)
            if not ok:
                with st.spinner("Sanity checks failed. Retrying mapping with constraints..."):
                    mapping2 = ai_map_rows(
                        df=statement_df,
                        statement_type=statement_type,
                        period_col=period_col,
                        model=model,
                        api_key=api_key,
                        constraints=failures,
                    )
                    vals2, meta2 = apply_mapping(statement_df, period_col, mapping2)
                    ok2, failures2 = balance_sheet_checks(vals2)

                    # Keep the better one (prefer passing checks)
                    if ok2:
                        mapping, vals, meta = mapping2, vals2, meta2
                        failures = failures2
                    else:
                        failures = failures2

            with st.expander("Sanity check results"):
                st.write({"passed": ok, "failures": failures})

        # Show mapping transparency (even if you don't “review,” it is useful for debugging)
        with st.expander("AI mapping details (canonical → detected row)"):
            show = []
            for k, v in meta.items():
                show.append({
                    "canonical_field": k,
                    "source_row_label": v.get("label", ""),
                    "confidence": v.get("confidence", 0.0),
                    "reason": v.get("reason", ""),
                    "value": vals.get(k, None),
                })
            st.dataframe(pd.DataFrame(show), use_container_width=True)

        # Compute KPIs
        if statement_type == "Income Statement":
            kpis = compute_income_kpis(vals)
        else:
            kpis = compute_balance_kpis(vals)

        st.subheader("KPI Results")
        for name, value in kpis.items():
            if value is None:
                st.write(f"{name}: N/A")
            else:
                # Ratios that look like percentages vs scalars:
                if "Margin" in name or "Rate" in name or "Ratio" in name and name.endswith("Ratio"):
                    # Keep margins/rates in %
                    if "Margin" in name or "Rate" in name:
                        st.write(f"{name}: {value:.2%}")
                    else:
                        st.write(f"{name}: {value:.2f}")
                else:
                    # default ratio formatting
                    if abs(value) < 3:
                        st.write(f"{name}: {value:.2f}")
                    else:
                        st.write(f"{name}: {value:.2%}")

        # Optional: model notes
        if mapping.get("notes"):
            with st.expander("Model notes"):
                for n in mapping["notes"]:
                    st.write(f"- {n}")
