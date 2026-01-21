import streamlit as st
import pdfplumber

st.set_page_config(page_title="PDF KPI Tool", layout="wide")

st.title("PDF Financial Statement Analyzer")

uploaded_file = st.file_uploader(
    "Upload a financial statement PDF",
    type=["pdf"]
)


def is_text_based_pdf(file) -> bool:
    """
    Returns True if the PDF contains a meaningful amount of text.
    """
    try:
        with pdfplumber.open(file) as pdf:
            total_chars = sum(len(page.chars) for page in pdf.pages)
        return total_chars > 200
    except Exception:
        return False


if uploaded_file is not None:
    st.success("PDF uploaded successfully")
    st.write(f"Filename: {uploaded_file.name}")

    if is_text_based_pdf(uploaded_file):
        st.success("Text-based PDF detected ✅")
        st.info("Table extraction should be possible.")
    else:
        st.error("Scanned PDF detected ⚠️")
        st.warning(
            "This PDF appears to be scanned (image-based). "
            "Automatic table extraction may not work without OCR."
        )
