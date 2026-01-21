# PDF KPI Tool (Student Project)

This project is a personal learning tool built with **Streamlit** to analyze financial statements from **PDF files** and compute selected financial KPIs.

The goal is to:
- Upload a financial statement PDF
- Extract tables from the document
- Map extracted rows to standardized financial line items
- Select which KPIs to compute
- Display and export the results

This project is **not commercial** and is intended for learning purposes only.

---

## Features (current and planned)

### Implemented
- Streamlit web interface
- PDF upload
- Basic project structure

### Planned
- Detection of text-based vs scanned PDFs
- Table extraction from PDFs
- Manual mapping of extracted rows to financial items
- KPI selection (profitability, liquidity, leverage, efficiency)
- KPI computation and visualization
- Export of KPI results

---

## Tech Stack
- Python
- Streamlit
- pandas
- pdfplumber (PDF text detection and extraction)

---

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
