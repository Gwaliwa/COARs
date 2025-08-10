
# coar_extractor_failsafe.py
# Streamlit app to extract COAR sections with multiple extractors + optional OCR

import io, re, hashlib, unicodedata, shutil, subprocess
from io import StringIO
from pathlib import Path
import pandas as pd
import streamlit as st

# ---------- Extractor backends ----------
# PDFMiner
try:
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from pdfminer.pdfpage import PDFPage
    PDFMINER_OK = True
except Exception:
    PDFMINER_OK = False

# PyMuPDF
try:
    import fitz
except Exception:
    fitz = None

# pypdf (secondary fallback)
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# DOCX
try:
    import docx
except Exception:
    docx = None

# OCR (optional): pytesseract + pillow
try:
    import pytesseract
    from PIL import Image
    PYTESS_OK = True
except Exception:
    pytesseract = None
    Image = None
    PYTESS_OK = False

# ---------------- App Setup ----------------
st.set_page_config(page_title="COAR Section Extractor → CSV (failsafe)", layout="wide")
st.title("📄 Extract COAR Sections → CSV (failsafe)")

# ---------------- Session ----------------
if "batches" not in st.session_state:
    st.session_state.batches = []
if "batch_counter" not in st.session_state:
    st.session_state.batch_counter = 1
if "consolidated_df" not in st.session_state:
    st.session_state.consolidated_df = None

COLS_ORDER = ["h_update_context", "h_major_results", "h_lessons_constraints"]
COLS_LABELS = {
    "h_update_context": "Part 1 / Situation update",
    "h_major_results": "Part 2 / Major results vs CPD",
    "h_lessons_constraints": "Part 3 / Lessons learned & constraints",
}

# ---------------- Matching ----------------
TARGET_PATTERNS = {
    "h_update_context": r"part\s*1\s*[:\-]",
    "h_major_results": r"part\s*2\s*[:\-]",
    "h_lessons_constraints": r"part\s*3\s*[:\-]",
}
COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in TARGET_PATTERNS.items()}

# ---------------- Helpers ----------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00a0", " ").replace("\u200b", "")
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)  # dehyphenate across linebreaks
    s = re.sub(r"[ \t]+", " ", s)
    return s

def extract_text_pdfminer(file_bytes: bytes) -> str:
    if not PDFMINER_OK:
        return ""
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    device = TextConverter(rsrcmgr, retstr, laparams=LAParams())
    try:
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.get_pages(io.BytesIO(file_bytes)):
            interpreter.process_page(page)
        return retstr.getvalue()
    except Exception:
        return ""
    finally:
        device.close()
        retstr.close()

def extract_text_fitz(file_bytes: bytes) -> str:
    if fitz is None:
        return ""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        txt = "\n".join(page.get_text() or "" for page in doc)
        if txt.strip():
            return txt
        # fallback to blocks
        blocks = []
        for p in doc:
            for b in p.get_text("blocks") or []:
                if isinstance(b, tuple) and len(b) >= 5:
                    blocks.append(b[4])
        return "\n".join(blocks)
    except Exception:
        return ""

def extract_text_pypdf(file_bytes: bytes) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        return ""

def extract_text_ocr(file_bytes: bytes, dpi=300) -> str:
    if not (PYTESS_OK and pytesseract and Image):
        return ""
    # Requires local Tesseract installed (binary)
    if not shutil.which("tesseract"):
        return ""
    try:
        # Render each page to image using PyMuPDF when available; else fallback raises
        if fitz is None:
            return ""
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        txts = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            t = pytesseract.image_to_string(img)
            txts.append(t)
        return "\n".join(txts)
    except Exception:
        return ""

def extract_text_router(name: str, file_bytes: bytes, backend="auto", allow_ocr=False) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".docx":
        if docx:
            try:
                d = docx.Document(io.BytesIO(file_bytes))
                return "\n".join(p.text or "" for p in d.paragraphs)
            except Exception:
                return ""
        return ""
    if suffix != ".pdf":
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return file_bytes.decode(enc, errors="ignore")
            except Exception:
                continue
        return ""

    # PDF
    order = []
    if backend == "pdfminer":
        order = [extract_text_pdfminer, extract_text_fitz, extract_text_pypdf]
    elif backend == "pymupdf":
        order = [extract_text_fitz, extract_text_pdfminer, extract_text_pypdf]
    else:  # auto: prefer PDFMiner to mirror the notebook behavior
        order = [extract_text_pdfminer, extract_text_fitz, extract_text_pypdf]

    for fn in order:
        txt = fn(file_bytes)
        if txt and len(txt.strip()) > 0:
            return txt

    if allow_ocr:
        ocr_txt = extract_text_ocr(file_bytes)
        if ocr_txt.strip():
            return ocr_txt

    return ""

def find_all_headings(text_clean: str):
    hits = []
    for key, pat in COMPILED.items():
        for m in pat.finditer(text_clean):
            hits.append((key, m.start(), m.end()))
    hits.sort(key=lambda x: x[1])
    return hits

def split_by_headings(text: str):
    clean = normalize_text(text)
    hits = find_all_headings(clean)
    out = {k: "" for k in COLS_ORDER}
    if not hits:
        return out, {"method": "none", "hit_count": 0, "text_chars": len(clean)}
    for i, (key, s, e) in enumerate(hits):
        next_s = hits[i+1][1] if i+1 < len(hits) else len(clean)
        content = clean[e:next_s].strip()
        out[key] = (out[key] + "\n\n" + content).strip() if out[key] else content
    return out, {"method": "regex", "hit_count": len(hits), "text_chars": len(clean), "hits": hits}

# ---------------- Controls ----------------
colx, coly, colz = st.columns([1,1,2])
with colx:
    backend = st.selectbox("Extractor backend", ["auto", "pdfminer", "pymupdf"], index=0,
                           help="Use 'pdfminer' to mirror the working notebook.")
with coly:
    allow_ocr = st.checkbox("Enable OCR fallback (needs Tesseract)", value=False,
                             help="Requires local Tesseract + pytesseract + pillow.")
with colz:
    st.caption(f"PDFMiner available: {PDFMINER_OK} | PyMuPDF available: {bool(fitz)} | pypdf available: {bool(PdfReader)} | OCR ready: {allow_ocr and PYTESS_OK and shutil.which('tesseract') is not None}")

# ---------------- Sidebar: batches ----------------
st.sidebar.header("Batches")
if st.session_state.batches:
    for i, b in enumerate(st.session_state.batches, start=1):
        st.sidebar.write(f"**Batch {i}:** {b['label']} — {len(b['files'])} file(s)")
    if st.sidebar.button("🗑️ Clear all"):
        st.session_state.batches = []
        st.session_state.batch_counter = 1
        st.session_state.consolidated_df = None
        st.rerun()

# ---------------- Upload ----------------
st.subheader("Add a new batch")
c1, c2 = st.columns([2, 3])
with c1:
    batch_label = st.text_input("Batch label (optional)", value=f"batch_{st.session_state.batch_counter}")
with c2:
    new_files = st.file_uploader(
        "Select multiple files (repeat for different folders)",
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.batch_counter}",
    )
if new_files:
    saved = []
    for f in new_files:
        data = f.read()
        saved.append({"name": f.name, "size": f.size, "data": data, "sha256": sha256_bytes(data)})
    st.session_state.batches.append({"label": (batch_label or '').strip() or f"batch_{st.session_state.batch_counter}",
                                     "files": saved})
    st.session_state.batch_counter += 1
    st.success(f"Added {len(saved)} file(s) to **{batch_label}**")
    st.rerun()

# ---------------- Extraction form ----------------
st.subheader("Extract sections & consolidate")
with st.form("extract_form", clear_on_submit=False):
    st.markdown("**Per-heading truncation (0 = no limit):**")
    col_a, col_b = st.columns(2)
    with col_a:
        max_update = st.number_input("Max chars: Part 1 / Situation update", min_value=0, value=0, step=250)
        debug = st.checkbox("Debug: show diagnostics", value=True)
    with col_b:
        max_major = st.number_input("Max chars: Part 2 / Major results vs CPD", min_value=0, value=0, step=250)
        max_lessons = st.number_input("Max chars: Part 3 / Lessons learned & constraints", min_value=0, value=0, step=250)
    submit = st.form_submit_button("🧩 Extract & Consolidate to CSV")

# ---------------- Run extraction ----------------
if submit:
    if not st.session_state.batches:
        st.warning("Please upload files first.")
    else:
        per_heading_limits = {
            "h_update_context": max_update,
            "h_major_results": max_major,
            "h_lessons_constraints": max_lessons,
        }
        rows = []
        total = sum(len(b["files"]) for b in st.session_state.batches)
        prog = st.progress(0.0, text="Extracting sections…")
        done = 0

        for batch in st.session_state.batches:
            for f in batch["files"]:
                raw_text = extract_text_router(f["name"], f["data"], backend=backend, allow_ocr=allow_ocr)
                sections, meta = split_by_headings(raw_text)

                if debug:
                    st.markdown(f"**Diagnostics for:** `{Path(f['name']).name}`")
                    st.caption(f"backend={backend} | text_chars={meta.get('text_chars')} | heading_hits={meta.get('hit_count')} | OCR={'on' if allow_ocr else 'off'}")
                    if (meta.get('text_chars', 0) or 0) <= 0:
                        st.error("No text extracted. Install/enable chosen backend or turn on OCR.")
                    elif meta.get("hit_count", 0) == 0:
                        st.warning("Text extracted but no headings matched. Showing first 200 chars:")
                        st.code(normalize_text(raw_text)[:200].replace("\n","⏎"))
                    else:
                        clean = normalize_text(raw_text)
                        for (key, s, e) in (meta.get("hits") or [])[:6]:
                            snippet = clean[max(0, s-60):min(len(clean), e+120)].replace("\n","⏎")
                            st.code(f"{key} [{s}:{e}] …{snippet}…")

                row = {"batch": batch["label"], "filename": Path(f["name"]).name}
                for k in COLS_ORDER:
                    cell = sections.get(k, "")
                    limit = per_heading_limits[k]
                    row[k] = (cell[:limit] if limit else cell)
                rows.append(row)

                done += 1
                prog.progress(done / max(1, total), text=f"Processed {done}/{total}")

        st.session_state.consolidated_df = pd.DataFrame(rows)
        prog.empty()
        st.success(f"Extracted sections from {done} file(s).")

# ---------------- Preview & Download ----------------
if st.session_state.consolidated_df is not None:
    st.subheader("Preview")
    st.dataframe(st.session_state.consolidated_df.head(50), use_container_width=True)
    buf = io.StringIO()
    st.session_state.consolidated_df.to_csv(buf, index=False)
    st.download_button(
        "⬇️ Download sections.csv",
        data=buf.getvalue().encode("utf-8"),
        file_name="sections.csv",
        mime="text/csv",
    )

with st.expander("ℹ️ Tips & setup"):
    st.write(
        "- If nothing extracts, try backend='pdfminer' to match the working notebook.\n"
        "- Install: `pip install pdfminer.six pymupdf pypdf python-docx pillow pytesseract`\n"
        "- For OCR, also install Tesseract locally and enable the checkbox.\n"
        "- Click '🗑️ Clear all' in the sidebar after changing backends to reset state.\n"
    )
