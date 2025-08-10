# app.py
import streamlit as st
import os
import re
import tempfile
import shutil
from io import BytesIO
from pathlib import Path
import pandas as pd

# ===================== SETUP =====================
st.set_page_config(page_title="PDF → TXT → Headings DataFrame", layout="wide")
st.title("📄 Convert PDFs to TXT → 🧾 Consolidate by Headings")

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

if BACKEND is None:
    st.error(
        "No PDF backend found. Please install one of:\n"
        "• pip install pymupdf  (recommended)\n"
        "• pip install pdfplumber\n"
        "• pip install PyPDF2"
    )
    st.stop()

st.caption(f"Active text-extraction backend: **{BACKEND}**")

# ---- Persistent temp output folder for this session ----
if "txt_out_dir" not in st.session_state:
    st.session_state.txt_out_dir = tempfile.mkdtemp(prefix="pdf_txt_")
out_dir = st.session_state.txt_out_dir
st.info(f"All .txt files will be saved to this temporary folder:\n{out_dir}")

# ===================== UPLOAD & CONVERT =====================
uploaded = st.file_uploader(
    "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
)

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

col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
with col1:
    run_convert = st.button("🚀 Convert to TXT", use_container_width=True)
with col2:
    clear = st.button("🧹 Clear Temp Folder", type="secondary", use_container_width=True)
with col3:
    consolidate = st.button("🧾 Consolidate by Headings", use_container_width=True)
with col4:
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

# ---- Convert PDFs → TXT ----
if run_convert:
    if not uploaded:
        st.warning("Please upload at least one PDF.")
    else:
        records = []
        prog = st.progress(0)
        for i, uf in enumerate(uploaded, 1):
            try:
                pdf_bytes = uf.read()
                text = extract_text(pdf_bytes)
                base = os.path.splitext(uf.name)[0]
                out_path = os.path.join(out_dir, f"{base}.txt")

                # Ensure unique filename if duplicates
                i_suffix = 1
                unique_path = out_path
                while os.path.exists(unique_path):
                    unique_path = os.path.join(out_dir, f"{base}({i_suffix}).txt")
                    i_suffix += 1

                with open(unique_path, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(text)

                records.append({"pdf": uf.name, "txt_path": unique_path, "chars": len(text)})
                st.write(f"✅ Saved: **{os.path.basename(unique_path)}**")
            except Exception as e:
                st.error(f"Error converting {uf.name}: {e}")
            finally:
                prog.progress(i / len(uploaded))

        if records:
            df_conv = pd.DataFrame(records)
            st.success(f"Done! Converted {len(df_conv)} PDFs.")
            st.dataframe(df_conv, use_container_width=True)

            # ZIP of the entire output folder
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

st.divider()

# ===================== CONSOLIDATE TXT BY HEADINGS =====================
st.subheader("🧾 Consolidate TXT → Per-Heading Columns (one row per .txt file)")
txt_root = Path(out_dir)

# Canonical headings & patterns (case-insensitive; heading on its own line)
HEADINGS = [
    ("context",        r"^\s*Update on the context and situation of children\s*$"),
    ("contributions",  r"^\s*Major contributions and drivers of results\s*$"),
    ("collaborations", r"^\s*UN Collaboration and Other Partnerships\s*$"),
    ("innovations",    r"^\s*Lessons Learned and Innovations\s*$"),
]
heading_keys = [k for k, _ in HEADINGS]
heading_patterns = [re.compile(p, re.IGNORECASE) for _, p in HEADINGS]

with st.expander("Options", expanded=True):
    lowercase = st.checkbox("lowercase all extracted text", value=False)
    strip_newlines = st.checkbox("remove newlines (join as one paragraph)", value=True)
    include_paths = st.checkbox("include full file path column", value=False)
    colA, colB, colC, colD = st.columns(4)
    max_context = colA.number_input("Max chars: context (0 = no limit)", min_value=0, value=0, step=1000)
    max_contrib = colB.number_input("Max chars: contributions (0 = no limit)", min_value=0, value=0, step=1000)
    max_collab  = colC.number_input("Max chars: collaborations (0 = no limit)", min_value=0, value=0, step=1000)
    max_innov   = colD.number_input("Max chars: innovations (0 = no limit)", min_value=0, value=0, step=1000)

def extract_sections_from_txt(file_path: Path) -> dict:
    """Return dict with filename (and optional filepath) + text for each heading."""
    result = {k: "" for k in heading_keys}
    result["filename"] = file_path.name
    if include_paths:
        result["filepath"] = str(file_path)

    current = None
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            # Check if this line is a heading
            matched = None
            for key, pattern in zip(heading_keys, heading_patterns):
                if pattern.match(line.strip()):
                    matched = key
                    break
            if matched is not None:
                current = matched
                continue
            # Accumulate into the current section
            if current:
                result[current] += raw  # keep original spacing
    return result

def clean_and_truncate(df: pd.DataFrame) -> pd.DataFrame:
    trunc = {"context": max_context, "contributions": max_contrib,
             "collaborations": max_collab, "innovations": max_innov}
    for col in heading_keys:
        if lowercase:
            df[col] = df[col].apply(lambda x: x.lower() if isinstance(x, str) else x)
        if strip_newlines:
            df[col] = df[col].str.replace("\n", "", regex=False)
        if trunc[col] and trunc[col] > 0:
            df[col] = df[col].apply(lambda x: x[:trunc[col]] if isinstance(x, str) else x)
    return df

if consolidate:
    if not txt_root.exists():
        st.warning("Temp folder not found. Convert PDFs to TXT first.")
    else:
        # Find ALL .txt files recursively
        txt_files = sorted([p for p in txt_root.rglob("*.txt")])
        if not txt_files:
            st.info("No .txt files found in the temp folder.")
        else:
            st.write(f"Found **{len(txt_files)}** TXT files.")
            rows = []
            prog = st.progress(0)
            for i, p in enumerate(txt_files, 1):
                rows.append(extract_sections_from_txt(p))
                prog.progress(i / len(txt_files))

            cols = ["filename"] + (["filepath"] if include_paths else []) + heading_keys
            df = pd.DataFrame(rows, columns=cols)
            df = clean_and_truncate(df)

            st.success(f"Consolidated **{len(df)}** files.")
            st.dataframe(df, use_container_width=True)

            st.download_button(
                "💾 Download Consolidated CSV",
                df.to_csv(index=False),
                file_name="coar_sections.csv",
                mime="text/csv",
                use_container_width=True
            )

st.caption(
    "Tip: If a PDF is scanned (no text layer), consider OCR (e.g., `ocrmypdf`) before conversion, "
    "or use PyMuPDF which tends to perform better for text extraction."
)
