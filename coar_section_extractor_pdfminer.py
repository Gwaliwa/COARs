# coar_folder_extractor_app.py
# Streamlit UI: browse folders (no tkinter), scan PDFs/DOCX recursively, extract "Part 1/2/3", export CSV

import io
import os
import re
import unicodedata
from io import StringIO
from pathlib import Path
import pandas as pd
import streamlit as st

# -------- Dependencies --------
# pip install streamlit pdfminer.six python-docx

# PDFMiner (primary)
try:
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from pdfminer.pdfpage import PDFPage
    PDFMINER_OK = True
except Exception:
    PDFMINER_OK = False

# DOCX (optional)
try:
    import docx
except Exception:
    docx = None

# ----- Part matchers (tolerant) -----
PART1 = re.compile(r"part\s*1\s*[:\-]", re.IGNORECASE)
PART2 = re.compile(r"part\s*2\s*[:\-]", re.IGNORECASE)
PART3 = re.compile(r"part\s*3\s*[:\-]", re.IGNORECASE)

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00a0", " ").replace("\u200b", "")
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)  # join hyphenated linebreaks
    s = re.sub(r"[ \t]+", " ", s)
    return s

def pdf_to_text_pdfminer(file_bytes: bytes) -> str:
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

def docx_to_text(file_bytes: bytes) -> str:
    if docx is None:
        return ""
    try:
        d = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text or "" for p in d.paragraphs)
    except Exception:
        return ""

def extract_text(path: Path) -> str:
    data = path.read_bytes()
    suf = path.suffix.lower()
    if suf == ".pdf":
        return pdf_to_text_pdfminer(data)
    if suf == ".docx":
        return docx_to_text(data)
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def split_three_parts(text: str) -> dict:
    clean = normalize_text(text)
    out = {"h_update_context": "", "h_major_results": "", "h_lessons_constraints": ""}
    # find the first occurrence of each part
    hits = []
    for key, pat in (("h_update_context", PART1), ("h_major_results", PART2), ("h_lessons_constraints", PART3)):
        m = pat.search(clean)
        if m:
            hits.append((key, m.start(), m.end()))
    if not hits:
        return out
    hits.sort(key=lambda x: x[1])
    for i, (key, s, e) in enumerate(hits):
        next_s = hits[i+1][1] if i+1 < len(hits) else len(clean)
        out[key] = clean[e:next_s].strip()
    return out

def collect_files(root: Path, include_docx=True) -> list[Path]:
    exts = {".pdf"}
    if include_docx:
        exts.add(".docx")
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in exts:
                files.append(p)
    return files

# ---------- Web-style folder picker (no Tk) ----------
def list_dir(path: Path):
    dirs, files = [], []
    try:
        for entry in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if entry.is_dir():
                dirs.append(entry)
            else:
                files.append(entry)
    except PermissionError:
        pass
    return dirs, files

def render_folder_picker():
    st.subheader("Choose a folder (local file system)")
    # state
    if "nav_path" not in st.session_state:
        st.session_state.nav_path = Path.home()
    # manual path entry
    manual = st.text_input("Path", value=str(st.session_state.nav_path))
    if manual and Path(manual).exists():
        st.session_state.nav_path = Path(manual)
    # nav buttons
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⬆️ Up one"):
            st.session_state.nav_path = st.session_state.nav_path.parent
    with c2:
        if st.button("🏠 Home"):
            st.session_state.nav_path = Path.home()
    with c3:
        select_here = st.button("✅ Use this folder")

    dirs, files = list_dir(st.session_state.nav_path)

    st.caption(f"Current: {st.session_state.nav_path}")
    st.write("Folders:")
    cols = st.columns(3)
    for i, d in enumerate(dirs):
        if cols[i % 3].button(f"📁 {d.name}", key=f"dir_{d}_{i}"):
            st.session_state.nav_path = d

    with st.expander("Show some files in this folder"):
        st.code("\n".join(p.name for p in files[:30]) or "(no files)")

    return (select_here, st.session_state.nav_path)

# ---------------------- UI ----------------------
st.set_page_config(page_title="COAR Folder Extractor → CSV", layout="wide")
st.title("📁 COAR Folder Extractor → CSV (no tkinter)")

include_docx = st.checkbox("Include .docx files", value=True)
limit = st.number_input("Max files to process (0 = no limit)", min_value=0, value=0, step=50)

selected, root = render_folder_picker()

if selected:
    if not root.exists() or not root.is_dir():
        st.error("That path is not a folder.")
    else:
        files = collect_files(root, include_docx=include_docx)
        if limit:
            files = files[:limit]
        st.success(f"Selected: {root}")
        st.write(f"Found **{len(files)}** file(s).")
        if not files:
            st.warning("No PDFs/DOCX found here.")
        else:
            st.caption("Sample files:")
            st.code("\n".join(str(p) for p in files[:10]))
            if st.button("🚀 Extract"):
                rows = []
                prog = st.progress(0.0, text="Extracting…")
                for i, f in enumerate(files, start=1):
                    try:
                        txt = extract_text(f)
                        parts = split_three_parts(txt)
                        rows.append({
                            "filename": f.name,
                            "path": str(f),
                            "h_update_context": parts["h_update_context"],
                            "h_major_results": parts["h_major_results"],
                            "h_lessons_constraints": parts["h_lessons_constraints"],
                        })
                    except Exception as e:
                        rows.append({
                            "filename": f.name, "path": str(f),
                            "h_update_context": "", "h_major_results": "",
                            "h_lessons_constraints": f"ERROR: {e}",
                        })
                    prog.progress(i / max(1, len(files)), text=f"Processed {i}/{len(files)}")

                df = pd.DataFrame(rows)
                st.subheader("Preview")
                st.dataframe(df.head(50), use_container_width=True)
                buf = io.StringIO()
                df.to_csv(buf, index=False)
                st.download_button("⬇️ Download sections.csv", data=buf.getvalue().encode("utf-8"),
                                   file_name="sections.csv", mime="text/csv")

st.info("No native dialogs used. If you need a specific path, type it in the Path box, press Enter, then click '✅ Use this folder'.")
st.caption("If PDFs are image-only, PDFMiner will return empty text. For OCR we could add a Tesseract fallback later.")
