import streamlit as st

st.set_page_config(page_title="PDF KPI Tool", layout="wide")

st.title("PDF Financial Statement Analyzer")

uploaded_file = st.file_uploader(
    "Upload a financial statement PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    st.success("PDF uploaded successfully!")
    st.write(f"Filename: {uploaded_file.name}")
