import streamlit as st
import pdfplumber
import pandas as pd


st.set_page_config(page_title="PDF KPI Tool", layout="wide")
st.title("PDF Financial Statement Analyzer")

uploaded_file = st.file_uploader(
    "Upload a financial statement PDF",
    type=["pdf"]
)


def is_text_based_pdf(file) -> bool:
    try:
        with pdfplumber.open(file) as pdf:
            total_chars = sum(len(page.chars) for page in pdf.pages)
        return total_chars > 200
    except Exception:
        return False


def extract_tables_with_pdfplumber(file, max_pages: int = 10):
    """
    Extract tables from a text-based PDF using pdfplumber.
    Returns a list of DataFrames with metadata (page number, table index).
    """
    extracted = []
    with pdfplumber.open(file) as pdf:
        num_pages = min(len(pdf.pages), max_pages)

        for page_idx in range(num_pages):
            page = pdf.pages[page_idx]
            tables = page.extract_tables()

            for t_idx, table in enumerate(tables):
                # table is a list of rows; each row is a list of cells
                df = pd.DataFrame(table)

                # Drop fully empty rows/cols
                df = df.dropna(how="all")
                df = df.dropna(axis=1, how="all")

                # Skip tiny tables (often noise)
                if df.shape[0] < 2 or df.shape[1] < 2:
                    continue

                extracted.append(
                    {
                        "page": page_idx + 1,
                        "table_index": t_idx + 1,
                        "df": df
                    }
                )

    return extracted


if uploaded_file is not None:
    st.success("PDF uploaded successfully")
    st.write(f"Filename: {uploaded_file.name}")

    if not is_text_based_pdf(uploaded_file):
        st.error("Scanned PDF detected ⚠️")
        st.warning(
            "This PDF appears to be scanned (image-based). "
            "Automatic table extraction may not work without OCR."
        )
        st.stop()

    st.success("Text-based PDF detected ✅")

    with st.spinner("Extracting tables from the PDF..."):
        tables = extract_tables_with_pdfplumber(uploaded_file, max_pages=15)

    st.write(f"Tables found: **{len(tables)}**")

    if len(tables) == 0:
        st.warning(
            "No tables were extracted. This can happen if the PDF layout is complex "
            "or tables are not recognized. Next, we can add an alternative extractor."
        )
        st.stop()

    # Build labels for the dropdown
    options = [
        f"Page {t['page']} — Table {t['table_index']} ({t['df'].shape[0]}x{t['df'].shape[1]})"
        for t in tables
    ]

    selected = st.selectbox("Select a table to preview", options=options, index=0)
    selected_idx = options.index(selected)

    st.subheader("Selected table preview")
    st.dataframe(tables[selected_idx]["df"], use_container_width=True)
