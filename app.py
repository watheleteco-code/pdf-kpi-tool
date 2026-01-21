import re
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
    """Heuristic: text-based PDFs have a meaningful number of text chars."""
    try:
        with pdfplumber.open(file) as pdf:
            total_chars = sum(len(page.chars) for page in pdf.pages)
        return total_chars > 200
    except Exception:
        return False


def clean_number(x):
    """Convert PDF-extracted numeric strings to float; handle commas and parentheses negatives."""
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


def normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure first col is label, others are values."""
    df = df.copy()
    df.columns = range(df.shape[1])
    df.rename(columns={0: "label"}, inplace=True)
    return df


def extract_tables_with_pdfplumber(file, max_pages: int = 10):
    """Extract tables with pdfplumber (works best for bordered/line-detected tables)."""
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
    """
    Try Camelot in both stream and lattice modes.
    Some PDFs work only with one of them.
    """
    results = []
    pages = ",".join(str(i) for i in range(1, max_pages + 1))

    for flavor in ["stream", "lattice"]:
        try:
            camelot_tables = camelot.read_pdf(
                file,
                pages=pages,
                flavor=flavor
            )

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


# ---------- Text-line fallback (works for “aligned text” statements) ----------

HEADER_KEYWORDS = [
    "years ended", "year ended", "for the year ended",
    "years ended december", "year ended december",
    "unaudited", "in thousands", "in millions"
]

NUM_PATTERN = r"\(?-?\d[\d,]*\.?\d*\)?"

BALANCE_SHEET_ITEMS = [
    "Cash",
    "Current Assets",
    "Current Liabilities",
    "Total Assets",
    "Total Liabilities",
    "Equity",
    "Total Debt"
]

balance_kpis = [
    "Current Ratio",
    "Cash Ratio",
    "Debt to Equity",
    "Debt Ratio"
]

# --- Choose which period/column to use ---
cols = [c for c in df_text.columns if c != "label"]
year_col = st.selectbox("Select period/column", cols, key="bs_year_col")

labels = df_text["label"].tolist()

st.subheader("Map balance sheet rows to financial concepts")

# --- Mapping UI ---
mapping = {}
for item in BALANCE_SHEET_ITEMS:
    mapping[item] = st.selectbox(
        f"{item}",
        options=["(not mapped)"] + labels,
        index=0,
        key=f"map_bs_{item}"
    )

# --- Helpers ---
def get_value_exact(df, label: str, col: str):
    hit = df[df["label"] == label]
    if hit.empty:
        return None
    return hit.iloc[0][col]

def safe_div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b

# --- Build values dict (THIS is what you are missing) ---
values = {}
for item, chosen_label in mapping.items():
    if chosen_label == "(not mapped)":
        values[item] = None
    else:
        values[item] = get_value_exact(df_text, chosen_label, year_col)

# --- KPI selection ---
selected_kpis = st.multiselect("Select KPIs", balance_kpis, default=["Current Ratio", "Debt Ratio"], key="bs_kpis")

st.subheader("KPI Results")

if "Current Ratio" in selected_kpis:
    v = safe_div(values["Current Assets"], values["Current Liabilities"])
    st.write("Current Ratio:", "N/A" if v is None else f"{v:.2f}")

if "Cash Ratio" in selected_kpis:
    v = safe_div(values["Cash"], values["Current Liabilities"])
    st.write("Cash Ratio:", "N/A" if v is None else f"{v:.2f}")

if "Debt to Equity" in selected_kpis:
    v = safe_div(values["Total Debt"], values["Equity"])
    st.write("Debt to Equity:", "N/A" if v is None else f"{v:.2f}")

if "Debt Ratio" in selected_kpis:
    v = safe_div(values["Total Liabilities"], values["Total Assets"])
    st.write("Debt Ratio:", "N/A" if v is None else f"{v:.2%}")


def extract_lines(file, max_pages: int = 10):
    lines = []
    with pdfplumber.open(file) as pdf:
        for page_idx, page in enumerate(pdf.pages[:max_pages]):
            text = page.extract_text() or ""
            for line in text.splitlines():
                line = line.strip()
                if line:
                    lines.append(line)
    return lines


def detect_years(lines):
    """
    Try to detect year headers like: '2003 2004' near the top of the statement.
    If not found, returns None.
    """
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

        # 1) Skip header / meta lines
        if any(k in lower for k in HEADER_KEYWORDS):
            continue

        # 2) Find numeric tokens
        nums = re.findall(NUM_PATTERN, line)
        if len(nums) < 2:
            continue

        # 3) Remove pure year tokens from candidates (e.g., 2003, 2004)
        filtered = []
        for tok in nums:
            tok_clean = tok.strip("()").replace(",", "")
            if tok_clean in year_set:
                continue
            filtered.append(tok)

        if len(filtered) < 2:
            continue

        # 4) Convert first two values
        v1 = clean_number(filtered[0])
        v2 = clean_number(filtered[1])

        # Extra guard: drop date-like lines (day number + small/empty value)
        if v1 is not None and 1 <= v1 <= 31 and (v2 is None or abs(v2) < 1000):
            continue

        # 5) Label is before the first remaining numeric token
        first = filtered[0]
        idx = line.find(first)
        label = line[:idx].strip(" .:-\t")
        if len(label) < 2:
            continue

        # ---- LABEL NORMALIZATION ----
        label = re.sub(r"[.\u2026]+", " ", label)   # remove dotted leaders
        label = label.replace("$", "").strip()      # remove currency symbol
        label = re.sub(r"\s+", " ", label)          # normalize whitespace
        # ----------------------------

        rows.append([label, v1, v2])

    # IMPORTANT: this is OUTSIDE the for-loop
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


def safe_div(a, b):
    if a is None or b in (None, 0):
        return None
    return a / b


# -------------------------
# Main app flow
# -------------------------

if uploaded_file is None:
    st.stop()

st.success("PDF uploaded successfully")
st.write(f"Filename: {uploaded_file.name}")

if not is_text_based_pdf(uploaded_file):
    st.error("Scanned PDF detected ⚠️")
    st.warning(
        "This PDF appears to be scanned (image-based). "
        "Automatic extraction may not work without OCR."
    )
    st.stop()

st.success("Text-based PDF detected ✅")

max_pages = st.slider("Pages to scan (from start)", min_value=1, max_value=50, value=15)

# Try extractors
with st.spinner("Extracting tables from the PDF..."):
    tables = extract_tables_with_pdfplumber(uploaded_file, max_pages=max_pages)

if len(tables) == 0:
    st.info("No tables found with pdfplumber. Trying Camelot...")
    tables = extract_tables_with_camelot(uploaded_file, max_pages=max_pages)

st.write(f"Tables found: **{len(tables)}**")

# If still nothing, do text-line parsing fallback
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

    # KPI selection UI for text-parsed income statement-like PDFs
    st.subheader("KPI selection (text-parsed statements)")
    cols = [c for c in df_text.columns if c != "label"]
    year_col = st.selectbox("Select period/column", cols)

    kpi_options = ["Operating Margin", "Net Margin", "Expense Ratio", "Effective Tax Rate"]
    selected_kpis = st.multiselect("Select KPIs", kpi_options, default=["Operating Margin", "Net Margin"])

    total_sales = get_value_contains(df_text, "Total Sales", year_col)
    total_expenses = get_value_contains(df_text, "Total Expenses", year_col)
    operating_income = get_value_contains(df_text, "Operating Income", year_col)
    taxes = get_value_contains(df_text, "Provision for Income Taxes", year_col)
    net_income = get_value_contains(df_text, "Net Income", year_col)
    nonop = get_value_contains(df_text, "Total Non-Operating", year_col)

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

    st.stop()

# If we have tables, show table picker + statement type + normalization
options = [
    f"Page {t['page']} — Table {t['table_index']} ({t['df'].shape[0]}x{t['df'].shape[1]})"
    for t in tables
]
selected = st.selectbox("Select a table to preview", options=options, index=0)
selected_idx = options.index(selected)

st.subheader("Selected table preview")
st.dataframe(tables[selected_idx]["df"], use_container_width=True)

st.subheader("Statement type")
statement_type = st.radio(
    "What kind of statement is this table?",
    ["Income Statement", "Balance Sheet", "Cash Flow Statement"]
)

raw_df = tables[selected_idx]["df"]
df = normalize_table(raw_df)

for col in df.columns[1:]:
    df[col] = df[col].apply(clean_number)

st.subheader("Normalized table")
st.dataframe(df, use_container_width=True)
