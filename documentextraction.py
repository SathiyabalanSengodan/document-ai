# documentextraction.py
# Streamlit: PDF text-layer + OCR fallback + Claude Sonnet tool-calling extraction (LangGraph)
# Adds: normalized dates (ISO), evidence snippets, confidence + extraction_method per field
#
# Install:
#   pip install streamlit pymupdf pillow pytesseract langchain langgraph langchain-anthropic
# System deps:
#   macOS: brew install tesseract
#
# Key:
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   OR create .streamlit/secrets.toml with:
#      ANTHROPIC_API_KEY="sk-ant-..."

import io
import json
import os
import re
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import fitz  # PyMuPDF
import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import pytesseract

from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage


# ----------------------------
# GLOBAL DOC STORE (tool-safe)
# ----------------------------
DOC_STORE: Dict[str, Any] = {
    "doc_id": None,
    "pages": None,      # list[{page, used_ocr, text}]
    "images": None,     # list[PIL.Image]
    "full_text": None,  # str
}


# ----------------------------
# Streamlit UI setup
# ----------------------------
st.set_page_config(page_title="PDF OCR + Claude Sonnet Extraction", layout="wide")
st.title("PDF OCR + Agent Tooling (Tesseract + Claude Sonnet)")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
dpi = st.slider("Render DPI (quality vs speed)", 150, 300, 200, 25)

st.sidebar.header("Claude Settings")
CLAUDE_SONNET_MODEL = "claude-sonnet-4-5"
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
st.sidebar.caption(f"Model fixed to: {CLAUDE_SONNET_MODEL}")

# Optional: set tesseract path explicitly if needed (macOS Homebrew)
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


# ----------------------------
# Helpers
# ----------------------------
def extract_json(text: str) -> str:
    """Extract JSON from a string that may include ```json ... ``` fences."""
    if not text:
        return text
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def compute_doc_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


def page_to_image(page: fitz.Page, dpi: int = 200) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Light preprocessing to improve OCR robustness."""
    img = ImageOps.exif_transpose(img)
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.point(lambda x: 0 if x < 160 else 255, "1")  # binarize
    return img


def extract_text_from_page(page: fitz.Page, img: Image.Image) -> Tuple[str, bool]:
    """Prefer PDF text layer; fallback to OCR."""
    text_layer = (page.get_text("text") or "").strip()
    if len(text_layer) >= 30:
        return text_layer, False
    ocr_img = preprocess_for_ocr(img)
    text = pytesseract.image_to_string(ocr_img, config="--psm 6")
    return text, True


def get_anthropic_key() -> Optional[str]:
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return None


def normalize_date_to_iso(date_str: Optional[str]) -> Optional[str]:
    """
    Convert common invoice date formats to ISO yyyy-mm-dd.
    Supported: MM/DD/YY, MM/DD/YYYY, YYYY-MM-DD
    """
    if not date_str or not isinstance(date_str, str):
        return None

    s = date_str.strip()
    for fmt in ("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def coerce_number(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        s = re.sub(r"^\$", "", s)
        try:
            return float(s)
        except ValueError:
            return None
    return None


# ----------------------------
# Tools (read from DOC_STORE)
# ----------------------------
@tool
def get_full_text() -> str:
    """Return concatenated text of entire uploaded PDF (all pages)."""
    full_text = DOC_STORE.get("full_text")
    if full_text and str(full_text).strip():
        return full_text
    return "ERROR: No document text loaded. Upload and process a PDF first."


@tool
def get_page_text(page_number: int) -> str:
    """Return extracted text for 1-based page_number (1 = first page)."""
    pages = DOC_STORE.get("pages")
    if not pages:
        return "ERROR: No pages loaded. Upload and process a PDF first."
    if page_number < 1 or page_number > len(pages):
        return f"ERROR: page_number out of range. Must be 1..{len(pages)}"
    return pages[page_number - 1]["text"]


@tool
def ocr_page(page_number: int) -> str:
    """Force OCR for a 1-based page_number and return OCR text."""
    images = DOC_STORE.get("images")
    if not images:
        return "ERROR: No document images loaded. Upload and process a PDF first."
    if page_number < 1 or page_number > len(images):
        return f"ERROR: page_number out of range. Must be 1..{len(images)}"
    img = images[page_number - 1]
    ocr_img = preprocess_for_ocr(img)
    return pytesseract.image_to_string(ocr_img, config="--psm 6")


def build_agent():
    api_key = get_anthropic_key()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Set it as an environment variable or in .streamlit/secrets.toml"
        )
    llm = ChatAnthropic(model=CLAUDE_SONNET_MODEL, temperature=temperature)
    tools = [get_full_text, get_page_text, ocr_page]
    return create_react_agent(llm, tools)


# ----------------------------
# Main app
# ----------------------------
if uploaded:
    pdf_bytes = uploaded.getvalue()
    doc_id = compute_doc_id(pdf_bytes)

    if DOC_STORE.get("doc_id") != doc_id or DOC_STORE.get("pages") is None:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        pages: List[Dict[str, Any]] = []
        images: List[Image.Image] = []
        full_text_parts: List[str] = []

        for i in range(len(doc)):
            page = doc[i]
            img = page_to_image(page, dpi=dpi)
            text, used_ocr = extract_text_from_page(page, img)

            images.append(img)
            pages.append({"page": i + 1, "used_ocr": used_ocr, "text": text})
            full_text_parts.append(text)

        DOC_STORE["doc_id"] = doc_id
        DOC_STORE["pages"] = pages
        DOC_STORE["images"] = images
        DOC_STORE["full_text"] = "\n\n".join(full_text_parts)

    st.subheader("Per-page preview")
    pages = DOC_STORE["pages"]
    images = DOC_STORE["images"]

    for p in pages:
        i = p["page"] - 1
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(images[i], caption=f"Page {p['page']}", width="stretch")
        with col2:
            st.markdown(f"**Text (used OCR: {p['used_ocr']})**")
            st.text_area(f"Extracted Text Page {p['page']}", p["text"], height=260)

    st.divider()
    st.subheader("Claude Sonnet: structured extraction (with evidence + confidence)")

    default_task = "Extract invoice fields from the document."
    task = st.text_area("Extraction task (prompt)", value=default_task, height=120)
    run = st.button("Run Claude Sonnet Extraction")

    extracted_fields: Dict[str, Any] = {}
    agent_output_text: Optional[str] = None
    error_msg: Optional[str] = None

    if run:
        try:
            agent = build_agent()

            strict_task = f"""
You are an extraction engine. You MUST use tools.

MANDATORY: Call get_full_text() first. If it returns "ERROR:", return ONLY this JSON:
{{
  "error": "no_document_text",
  "details": "<paste tool output>",
  "document_type": "invoice",
  "fields": {{}}
}}

Otherwise, from the document text, return ONLY valid JSON matching EXACTLY this schema:

{{
  "document_type": "invoice",
  "fields": {{
    "invoice_number": {{
      "value": <string|null>,
      "evidence": <string>,
      "confidence": <number>,
      "extraction_method": <string>
    }},
    "purchase_order_number": {{
      "value": <string|null>,
      "evidence": <string>,
      "confidence": <number>,
      "extraction_method": <string>
    }},
    "invoice_date": {{
      "value": <string|null>,
      "value_iso": <string|null>,
      "evidence": <string>,
      "confidence": <number>,
      "extraction_method": <string>
    }},
    "due_date": {{
      "value": <string|null>,
      "value_iso": <string|null>,
      "evidence": <string>,
      "confidence": <number>,
      "extraction_method": <string>
    }},
    "vendor_name": {{
      "value": <string|null>,
      "evidence": <string>,
      "confidence": <number>,
      "extraction_method": <string>
    }},
    "customer_name": {{
      "value": <string|null>,
      "evidence": <string>,
      "confidence": <number>,
      "extraction_method": <string>
    }},
    "tax": {{
      "value": <number|null>,
      "evidence": <string>,
      "confidence": <number>,
      "extraction_method": <string>
    }},
    "total": {{
      "value": <number|null>,
      "evidence": <string>,
      "confidence": <number>,
      "extraction_method": <string>
    }},
    "balance_due": {{
      "value": <number|null>,
      "evidence": <string>,
      "confidence": <number>,
      "extraction_method": <string>
    }}
  }}
}}

Rules:
- Output ONLY JSON. No markdown. No code fences.
- Evidence must be a SHORT snippet copied from the document text that supports the value (e.g., a line containing the label/value).
- confidence must be a number between 0.0 and 1.0.
  Use this rubric:
  - 0.95–1.00: exact label + value clearly present (e.g., "BALANCE DUE $ 186.51")
  - 0.75–0.94: value present but label proximity is weaker or formatting messy
  - 0.40–0.74: plausible but ambiguous (multiple candidates)
  - 0.00–0.39: mostly guess / uncertain (avoid unless you must)
- extraction_method must be a SHORT explanation (1 sentence) like:
  "Matched label 'Invoice Date' and read the next date token."
  Do NOT provide chain-of-thought; just the method.
- invoice_date.value_iso and due_date.value_iso MUST be ISO yyyy-mm-dd if possible; otherwise null.
- If tax is not present, set tax.value = null and confidence low (<=0.4) with evidence like "Tax not found".
- Prefer BALANCE DUE / AMOUNT DUE for balance_due. If total not explicitly shown, set total = balance_due.

User request:
{task}
""".strip()

            resp = agent.invoke({"messages": [HumanMessage(content=strict_task)]})
            agent_output_text = resp["messages"][-1].content
            extracted_fields = json.loads(extract_json(agent_output_text))

            # --- Post-normalization / sanity fixes (non-destructive) ---
            # Ensure ISO dates exist (if model missed)
            f = extracted_fields.get("fields", {})
            inv = f.get("invoice_date", {})
            due = f.get("due_date", {})

            if inv and inv.get("value") and not inv.get("value_iso"):
                inv["value_iso"] = normalize_date_to_iso(inv.get("value"))
            if due and due.get("value") and not due.get("value_iso"):
                due["value_iso"] = normalize_date_to_iso(due.get("value"))

            # Coerce numeric fields to float or null
            for k in ("tax", "total", "balance_due"):
                node = f.get(k)
                if isinstance(node, dict):
                    node["value"] = coerce_number(node.get("value"))

            extracted_fields["fields"] = f

        except json.JSONDecodeError:
            error_msg = "Claude did not return valid JSON. See agent_output_raw for details."
        except Exception as e:
            error_msg = str(e)

    # Final payload
    result = {
        "text": DOC_STORE.get("full_text", ""),
        "pages": DOC_STORE.get("pages", []),
        "extraction": extracted_fields,
        "agent_output_raw": agent_output_text,
        "error": error_msg,
        "llm": {"provider": "anthropic", "model": CLAUDE_SONNET_MODEL, "temperature": float(temperature)},
    }

    st.subheader("Structured Output (JSON)")
    st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")

    st.download_button(
        "Download JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="ocr_claude_sonnet_result_with_evidence.json",
        mime="application/json",
    )

else:
    st.info("Upload a PDF to begin.")
