# Document AI: PDF Invoice Extraction (OCR + Claude Sonnet)

A Streamlit app that extracts structured invoice fields from PDF documents using:

- **PDF text layer** extraction (PyMuPDF)
- **OCR fallback** for scanned PDFs (Tesseract)
- **Claude Sonnet** (Anthropic) with **tool-calling** via LangGraph
- **Evidence snippets** + **heuristic confidence scores** per field
- **ISO date normalization** for downstream systems

> ✅ Designed for prototyping and internal tools.  
> ⚠️ Not production-hardened (see “Production notes”).

---

## Demo Output (Example)

The extraction returns a JSON object like:

- invoice_number
- purchase_order_number
- invoice_date + invoice_date_iso
- due_date + due_date_iso
- vendor_name
- customer_name
- tax
- total
- balance_due

Each field includes:
- `evidence`: short snippet copied from the document text
- `confidence`: 0.0–1.0 (heuristic rubric)
- `extraction_method`: short explanation of how it was found

---

## How It Works

1. User uploads a PDF in Streamlit
2. App extracts text per page:
   - Prefer **PDF text layer**
   - If missing/low quality → run **Tesseract OCR**
3. The full extracted text is cached
4. Claude Sonnet agent is instructed to:
   - **Call tools** to read the document (`get_full_text`, `get_page_text`, `ocr_page`)
   - Return **JSON only** matching the schema
5. App post-processes:
   - Normalizes dates to ISO (`YYYY-MM-DD`)
   - Coerces numeric values
   - Shows output + provides JSON download

---

## Tech Stack

- Python
- Streamlit
- PyMuPDF (fitz)
- Tesseract OCR (pytesseract)
- LangChain tools
- LangGraph prebuilt ReAct agent
- Anthropic Claude Sonnet via `langchain-anthropic`

---

## Setup

### 1) System dependency: Tesseract

macOS:
```bash
brew install tesseract
# document-ai-invoice-extractor
Invoice extraction from PDFs using OCR + Claude Sonnet with evidence, confidence scores, and normalized dates.
