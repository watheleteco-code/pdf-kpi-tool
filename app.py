import streamlit as st
import pdfplumber
import pandas as pd

st.set_page_config(page_title="PDF KPI Tool", layout="wide")
st.title("PDF Financial Statement Analyzer")

uploaded_file = st.file_uploader("Upload a financial statement PDF", type=["pdf"])


def is_text_based_pdf(file) -> bool:
    try:
        with pdfplumber.open(file) as pdf:
            total_chars = sum(len(page.chars) for page in pdf.pages)
        return total_chars > 200
    except Exception:
        return False

def clean_number(x):
    if isinstance(x, str):
        x = x.replace(",", "").strip()
        if x.startswith("(") and x.endswith(")"):
            x = "-" + x[1:-1]
        if x in ["-", "", "–"]:
            return None
    try:
        return float(x)
    except Exception:
        return None

        
def normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures:
    - first column = labels (string)
    - other columns = numeric values
    """
    df = df.copy()
    df.columns = range(df.shape[1])

    # Rename first column
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

                df = df.dropna(how="all")
                df = df.dropna(axis=1, how="all")

                if df.shape[0] < 2 or df.shape[1] < 2:
                    continue

                extracted.append({"page": page_idx + 1, "table_index": t_idx + 1, "df": df})

    return extracted


if uploaded_file is not None:
    st.success("PDF uploaded successfully")
    st.write(f"Filename: {uploaded_file.name}")

    if not is_text_based_pdf(uploaded_file):
        st.error("Scanned PDF detected ⚠️")
        st.warning("This PDF appears to be scanned (image-based). Extraction may not work without OCR.")
        st.stop()

    st.success("Text-based PDF detected ✅")

    max_pages = st.slider("Pages to scan (from start)", min_value=1, max_value=50, value=15)

    with st.spinner("Extracting tables from the PDF..."):
        tables = extract_tables_with_pdfplumber(uploaded_file, max_pages=max_pages)

    st.write(f"Tables found: **{len(tables)}**")

    if len(tables) == 0:
        st.warning("No tables extracted. Next we can add a Camelot fallback extractor.")
        st.stop()

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

