# app.py
import streamlit as st
import os
import tempfile
import shutil
from io import BytesIO
import pandas as pd

# ---- Try available PDF backends (PyMuPDF -> pdfplumber -> PyPDF2) ----
BACKEND = None
try:
    import fitz  # PyMuPDF
    BACKEND = "pymupdf"
except Exception:
    try:
        import pdfplumber
        BACKEND = "pdfplumber"
    except Exception:
        try:
            from PyPDF2 import PdfReader
            BACKEND = "pypdf2"
        except Exception:
            BACKEND = None

st.set_page_config(page_title="PDF → TXT Converter", layout="wide")
st.title("📄 Convert PDFs to TXT and Save in One Folder")

if BACKEND is None:
    st.error(
        "No PDF backend found. Please install one of: "
        "`pip install pymupdf` (recommended) or `pip install pdfplumber` or `pip install PyPDF2`."
    )
    st.stop()

st.caption(f"Active text-extraction backend: **{BACKEND}**")

# ---- Prepare a persistent temp output folder for this session ----
if "txt_out_dir" not in st.session_state:
    st.session_state.txt_out_dir = tempfile.mkdtemp(prefix="pdf_txt_")

out_dir = st.session_state.txt_out_dir
st.info(f"All .txt files will be saved to this temporary folder:\n{out_dir}")

# ---- Uploader ----
uploaded = st.file_uploader(
    "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
)

# ---- Extractor ----
def extract_text(pdf_bytes: bytes) -> str:
    """Extract text using the first available backend."""
    if BACKEND == "pymupdf":
        import fitz
        text = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text).strip()

    elif BACKEND == "pdfplumber":
        import pdfplumber
        text = []
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text.append(page_text)
        return "\n".join(text).strip()

    elif BACKEND == "pypdf2":
        from PyPDF2 import PdfReader
        reader = PdfReader(BytesIO(pdf_bytes))
        text = []
        for page in reader.pages:
            content = page.extract_text() or ""
            text.append(content)
        return "\n".join(text).strip()

    return ""

# ---- Actions row ----
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    run = st.button("🚀 Convert to TXT", use_container_width=True)
with col2:
    clear = st.button("🧹 Clear Temp Folder", type="secondary", use_container_width=True)
with col3:
    st.write("")

# ---- Clear output folder ----
if clear:
    try:
        shutil.rmtree(out_dir, ignore_errors=True)
        st.session_state.txt_out_dir = tempfile.mkdtemp(prefix="pdf_txt_")
        out_dir = st.session_state.txt_out_dir
        st.success(f"Cleared. New temp folder: {out_dir}")
    except Exception as e:
        st.error(f"Failed to clear folder: {e}")

# ---- Convert ----
if run:
    if not uploaded:
        st.warning("Please upload at least one PDF.")
    else:
        records = []
        for uf in uploaded:
            try:
                pdf_bytes = uf.read()
                text = extract_text(pdf_bytes)
                base = os.path.splitext(uf.name)[0]
                out_path = os.path.join(out_dir, f"{base}.txt")

                # Ensure unique filename if duplicates
                i = 1
                unique_path = out_path
                while os.path.exists(unique_path):
                    unique_path = os.path.join(out_dir, f"{base}({i}).txt")
                    i += 1

                with open(unique_path, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(text)

                records.append({"pdf": uf.name, "txt_path": unique_path, "chars": len(text)})
                st.write(f"✅ Saved: **{os.path.basename(unique_path)}**")
            except Exception as e:
                st.error(f"Error converting {uf.name}: {e}")

        if records:
            df = pd.DataFrame(records)
            st.success(f"Done! Converted {len(records)} PDFs.")
            st.dataframe(df, use_container_width=True)

            # Make a zip of the entire output folder
            zip_base = os.path.join(tempfile.gettempdir(), f"txt_export_{os.path.basename(out_dir)}")
            zip_file = shutil.make_archive(zip_base, "zip", out_dir)

            with open(zip_file, "rb") as f:
                st.download_button(
                    "💾 Download all TXT as ZIP",
                    data=f.read(),
                    file_name="pdf_texts.zip",
                    mime="application/zip",
                    use_container_width=True
                )

st.caption("Tip: If you see low/empty text for scanned PDFs, try installing `pytesseract` + `ocrmypdf` for OCR, or switch to PyMuPDF which is generally stronger.")
