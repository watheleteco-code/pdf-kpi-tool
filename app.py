import re
from difflib import SequenceMatcher

import streamlit as st
import pdfplumber
import pandas as pd
import camelot

st.set_page_config(page_title="PDF KPI Tool", layout="wide")
st.title("PDF Financial Statement Analyzer")

uploaded_file = st.file_uploader("Upload a financial statement PDF", type=["pdf"])


# -------------------------
# Helpers
# -------------------------

def is_text_based_pdf(file) -> bool:
    try:
        with pdfplumber.open(file) as pdf:
            total_chars = sum(len(page.chars) for page in pdf.pages)
        return total_chars > 200
    except Exception:
        return False


def clean_number(x):
    if isinstance(x, str):
        x = x.replace(",", "").replace("$", "").strip()
        if x.startswith("(") and x.endswith(")"):
            x = "-" + x[1:-1]
        if x in ["-", "", "–"]:
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
    df = df.copy()
    df.columns = range(df.shape[1])
    df.rename(columns={0: "label"}, inplace=True)
    return df


def extract_tables_with_pdfplumber(file, max_pages: int = 10):
    extracted = []
    with pdfplumber.open(file) as pdf:
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


def extract_tables_with_camelot(file, max_pages: int = 10):
    results = []
    pages = ",".join(str(i) for i in range(1, max_pages + 1))

    for flavor in ["stream", "lattice"]:
        try:
            camelot_tables = camelot.read_pdf(file, pages=pages, flavor=flavor)

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


# ---------- Text-line fallback (aligned text statements) ----------

HEADER_KEYWORDS = [
    "years ended", "year ended", "for the year ended",
    "years ended december", "year ended december",
    "unaudited", "in thousands", "in millions"
]
NUM_PATTERN = r"\(?-?\d[\d,]*\.?\d*\)?"


def extract_lines(file, max_pages: int = 10):
    lines = []
    with pdfplumber.open(file) as pdf:
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

        # drop date-like lines
        if v1 is not None and 1 <= v1 <= 31 and (v2 is None or abs(v2) < 1000):
            continue

        first = filtered[0]
        idx = line.find(first)
        label = line[:idx].strip(" .:-\t")
        if len(label) < 2:
            continue

        # label normalization
        label = re.sub(r"[.\u2026]+", " ", label)
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


def get_value_contains(df, label_contains: str, col: str):
    hit = df[df["label"].str.contains(label_contains, case=False, na=False)]
    if hit.empty:
        return None
    return hit.iloc[0][col]


def get_value_exact(df, label: str, col: str):
    hit = df[df["label"] == label]
    if hit.empty:
        return None
    return hit.iloc[0][col]


# -------------------------
# Auto-mapping (preselect dropdowns)
# -------------------------

CANONICAL_MATCHERS = {
    "Cash": ["cash", "cash and equivalents"],
    "Current Assets": ["total current assets", "current assets"],
    "Current Liabilities": ["total current liabilities", "current liabilities"],
    "Total Assets": ["total assets"],
    "Total Liabilities": ["total liabilities"],
    "Equity": ["total equity", "equity", "shareholders equity", "stockholders equity"],
    "Total Debt": ["total debt", "long-term debt", "borrowings", "notes payable", "debt"],
}


def normalize_label(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def auto_map_labels(statement_labels, matchers, threshold: float = 0.65):
    mapping = {}
    norm_labels = {lbl: normalize_label(lbl) for lbl in statement_labels}

    for item, keywords in matchers.items():
        best_match = None
        best_score = 0.0

        for lbl, norm_lbl in norm_labels.items():
            for kw in keywords:
                score = SequenceMatcher(None, norm_lbl, normalize_label(kw)).ratio()
                if score > best_score:
                    best_score = score
                    best_match = lbl

        if best_score >= threshold and best_match is not None:
            mapping[item] = best_match
        else:
            mapping[item] = "(not mapped)"

    return mapping


# -------------------------
# KPI rendering
# -------------------------

def render_income_statement_kpis(statement_df: pd.DataFrame):
    st.subheader("KPI selection (Income Statement)")

    cols = [c for c in statement_df.columns if c != "label"]
    year_col = st.selectbox("Select period/column", cols, key="is_year")

    kpi_options = ["Operating Margin", "Net Margin", "Expense Ratio", "Effective Tax Rate"]
    selected_kpis = st.multiselect(
        "Select KPIs", kpi_options, default=["Operating Margin", "Net Margin"], key="is_kpis"
    )

    total_sales = get_value_contains(statement_df, "Total Sales", year_col)
    total_expenses = get_value_contains(statement_df, "Total Expenses", year_col)
    operating_income = get_value_contains(statement_df, "Operating Income", year_col)
    taxes = get_value_contains(statement_df, "Provision for Income Taxes", year_col)
    net_income = get_value_contains(statement_df, "Net Income", year_col)
    nonop = get_value_contains(statement_df, "Total Non-Operating", year_col)

    st.subheader("KPI Results")

    if "Operating Margin" in selected_kpis:
        v = safe_div(operating_income, total_sales)
        st.write("Operating Margin:", "N/A" if v is None else f"{v:.2%}")

    if "Net Margin" in selected_kpis:
        v = safe_div(net_income, total_sales)
        st.write("Net Margin:", "N/A" if v is None else f"{v:.2%}")

    if "Expense Ratio" in selected_kpis:
        v = safe_div(total_expenses, total_sales)
        st.write("Expense Ratio:", "N/A" if v is None else f"{v:.2%}")

    if "Effective Tax Rate" in selected_kpis:
        ebt = None if operating_income is None else operating_income + (nonop or 0)
        v = safe_div(taxes, ebt)
        st.write("Effective Tax Rate:", "N/A" if v is None else f"{v:.2%}")


def render_balance_sheet_kpis(statement_df: pd.DataFrame):
    st.subheader("KPI selection (Balance Sheet)")

    # choose year/period column
    cols = [c for c in statement_df.columns if c != "label"]
    year_col = st.selectbox("Select period/column", cols, key="bs_year")

    # helper to get values by "contains"
    def v(contains: str):
        return get_value_contains(statement_df, contains, year_col)

    # --- Auto-extract core inputs from THIS type of balance sheet ---
    cash = v("Cash")
    current_assets = v("Total current assets")
    current_liabilities = v("Total current liabilities")
    total_assets = v("Total assets")
    equity = v("Total owners")  # matches "Total owners' equity"

    # debt components (often split)
    note_payable = v("Note payable")
    long_term_debt = v("Long-term debt")

    # Derived metrics
    total_debt = None
    if note_payable is not None or long_term_debt is not None:
        total_debt = (note_payable or 0) + (long_term_debt or 0)

    total_liabilities = None
    if current_liabilities is not None or long_term_debt is not None:
        total_liabilities = (current_liabilities or 0) + (long_term_debt or 0)

    # If totals missing, try to derive liabilities from Assets - Equity
    if total_liabilities is None and total_assets is not None and equity is not None:
        total_liabilities = total_assets - equity

    # KPIs
    balance_kpis = [
        "Current Ratio",
        "Cash Ratio",
        "Debt to Equity",
        "Debt Ratio",
        "Equity Ratio",
    ]
    selected_kpis = st.multiselect(
        "Select KPIs",
        balance_kpis,
        default=["Current Ratio", "Debt Ratio"],
        key="bs_kpis",
    )

    # Optional transparency (not interactive)
    with st.expander("Show detected values (auto)"):
        st.write({
            "cash": cash,
            "current_assets": current_assets,
            "current_liabilities": current_liabilities,
            "total_assets": total_assets,
            "equity": equity,
            "note_payable": note_payable,
            "long_term_debt": long_term_debt,
            "total_debt (derived)": total_debt,
            "total_liabilities (derived)": total_liabilities,
        })

    st.subheader("KPI Results")

    if "Current Ratio" in selected_kpis:
        r = safe_div(current_assets, current_liabilities)
        st.write("Current Ratio:", "N/A" if r is None else f"{r:.2f}")

    if "Cash Ratio" in selected_kpis:
        r = safe_div(cash, current_liabilities)
        st.write("Cash Ratio:", "N/A" if r is None else f"{r:.2f}")

    if "Debt to Equity" in selected_kpis:
        r = safe_div(total_debt, equity)
        st.write("Debt to Equity:", "N/A" if r is None else f"{r:.2f}")

    if "Debt Ratio" in selected_kpis:
        r = safe_div(total_liabilities, total_assets)
        st.write("Debt Ratio:", "N/A" if r is None else f"{r:.2%}")

    if "Equity Ratio" in selected_kpis:
        r = safe_div(equity, total_assets)
        st.write("Equity Ratio:", "N/A" if r is None else f"{r:.2%}")



# -------------------------
# Main app flow
# -------------------------

if uploaded_file is None:
    st.stop()

st.success("PDF uploaded successfully")
st.write(f"Filename: {uploaded_file.name}")

if not is_text_based_pdf(uploaded_file):
    st.error("Scanned PDF detected ⚠️")
    st.warning("This PDF appears to be scanned (image-based). Automatic extraction may not work without OCR.")
    st.stop()

st.success("Text-based PDF detected ✅")

max_pages = st.slider("Pages to scan (from start)", min_value=1, max_value=50, value=15)

with st.spinner("Extracting tables from the PDF..."):
    tables = extract_tables_with_pdfplumber(uploaded_file, max_pages=max_pages)

if len(tables) == 0:
    st.info("No tables found with pdfplumber. Trying Camelot...")
    tables = extract_tables_with_camelot(uploaded_file, max_pages=max_pages)

st.write(f"Tables found: **{len(tables)}**")

statement_df = None

# --- If no tables, parse as aligned text ---
if len(tables) == 0:
    st.info("No tables extracted. Parsing as text-aligned statement...")
    lines = extract_lines(uploaded_file, max_pages=max_pages)
    years = detect_years(lines)
    df_text = parse_text_aligned_statement(lines, years=years)

    if df_text is None:
        st.error("Could not parse statement from text.")
        st.stop()

    st.subheader("Parsed statement (text-based)")
    st.dataframe(df_text, use_container_width=True)

    statement_df = df_text

# --- If tables exist, user selects and we normalize ---
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

# --- KPI section (works for BOTH extraction paths) ---
st.divider()
st.subheader("KPI Engine")

statement_type = st.radio(
    "What kind of statement is this?",
    ["Income Statement", "Balance Sheet", "Cash Flow Statement"],
    index=0,
    key="statement_type",
)

if statement_type == "Income Statement":
    render_income_statement_kpis(statement_df)
elif statement_type == "Balance Sheet":
    render_balance_sheet_kpis(statement_df)
else:
    st.info("Cash Flow KPIs not implemented yet. Next step: add Operating Cash Flow metrics.")

