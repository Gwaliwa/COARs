
import io
import re
import hashlib
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import streamlit as st

try:
    from rapidfuzz import fuzz, process as rf_process
except Exception:
    fuzz = None
    rf_process = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import docx
except Exception:
    docx = None

st.set_page_config(page_title="Section Extractor → CSV (robust matcher)", layout="wide")
st.title("📄 Extract COAR Sections by Headings → CSV (robust matcher)")

if "batches" not in st.session_state:
    st.session_state.batches = []
if "batch_counter" not in st.session_state:
    st.session_state.batch_counter = 1
if "consolidated_df" not in st.session_state:
    st.session_state.consolidated_df = None

CANON = {
    "h_update_context": "Part 1 / Situation update",
    "h_major_results": "Part 2 / Major results vs CPD",
    "h_lessons_constraints": "Part 3 / Lessons learned & constraints",
}

TARGET_PATTERNS = {
    "h_update_context": r"""
        (?:^|\n)\s*
        (?:
            (?:\d+(?:\.\d+)*|[IVXLC]+)?\s*[.)-]?\s*
            (?:update\s+on\s+the\s+)?context\s+(?:and\s+)?situation\s+of\s+children\s*:?
          |
            part\s*1\s*[:\-]
        )
    """,
    "h_major_results": r"""
        (?:^|\n)\s*
        (?:
            (?:\d+(?:\.\d+)*|[IVXLC]+)?\s*[.)-]?\s*
            major\s+results .*?(?:humanitarian|emergency).*?(?:gender).*?(?:country\s+programme\s+document(?:s)?|cpd(?:s)?)\s*:?
          |
            part\s*2\s*[:\-]
        )
    """,
    "h_lessons_constraints": r"""
        (?:^|\n)\s*
        (?:
            (?:\d+(?:\.\d+)*|[IVXLC]+)?\s*[.)-]?\s*
            lessons?\s+learn(?:ed|t)\s*(?:and|&)\s*constraints?\s*:?
          |
            part\s*3\s*[:\-]
        )
    """,
}
COMPILED = {k: re.compile(v, re.IGNORECASE | re.DOTALL | re.VERBOSE) for k, v in TARGET_PATTERNS.items()}
COLS_ORDER = ["h_update_context", "h_major_results", "h_lessons_constraints"]

def infer_folder_from_name(name: str) -> str:
    parts = Path(name).parts
    return str(Path(*parts[:-1])) if len(parts) > 1 else "(none)"

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00a0", " ").replace("\u200b", "")
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    # Correct de-hyphenation (use \1\2, not \\1\\2)
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if fitz is not None:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            txt = "\n".join(page.get_text() or "" for page in doc)
            if txt.strip():
                return txt
            blocks = []
            for p in doc:
                for b in p.get_text("blocks") or []:
                    if isinstance(b, tuple) and len(b) >= 5:
                        blocks.append(b[4])
            return "\n".join(blocks)
        except Exception:
            pass
    if PdfReader is not None:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception:
            pass
    return ""

def extract_text_from_docx(file_bytes: bytes) -> str:
    if docx is None:
        return ""
    try:
        d = docx.Document(io.BytesIO(file_bytes))
        lines = []
        for p in d.paragraphs:
            lines.append(p.text or "")
        return "\n".join(lines)
    except Exception:
        return ""

def extract_text_router(name: str, file_bytes: bytes) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_bytes)
    if suffix == ".docx":
        return extract_text_from_docx(file_bytes)
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return file_bytes.decode(enc, errors="ignore")
        except Exception:
            continue
    return ""

def find_all_headings_regex(text_clean: str):
    hits = []
    for key, pat in COMPILED.items():
        for m in pat.finditer(text_clean):
            hits.append((key, m.start(), m.end(), m.group(0)))
    hits.sort(key=lambda x: x[1])
    return hits

def split_by_headings(text: str):
    clean = normalize_text(text)
    hits = find_all_headings_regex(clean)
    out = {k: "" for k in COLS_ORDER}
    if hits:
        for i, (key, start, end, _) in enumerate(hits):
            next_start = hits[i+1][1] if i+1 < len(hits) else len(clean)
            content = clean[end:next_start].strip()
            out[key] = (out[key] + "\n\n" + content).strip() if out[key] else content
        return out, {"method": "regex", "hits": hits, "text_used": clean}

    # Minimal fallback (keyword windows)
    paras = [p.strip() for p in clean.split("\n") if p.strip()]
    KEYBOX = {
        "h_update_context": ["part 1", "situation", "context", "children"],
        "h_major_results": ["part 2", "major results", "humanitarian", "gender", "cpd"],
        "h_lessons_constraints": ["part 3", "lessons", "constraints"],
    }
    for k, keys in KEYBOX.items():
        bag = []
        for idx, p in enumerate(paras):
            hits_kw = sum(1 for kw in keys if kw in p.lower())
            if hits_kw >= 2:
                start = max(0, idx - 2)
                end = min(len(paras), idx + 8)
                bag.append("\n".join(paras[start:end]))
        out[k] = "\n\n".join(bag).strip()
    return out, {"method": "keyword_box", "hits": [], "text_used": clean}

def apply_limit(text: str, limit: int) -> str:
    return text[:limit] if limit and limit > 0 else text

# ---- Sidebar: batches ----
st.sidebar.header("Batches")
if st.session_state.batches:
    for i, b in enumerate(st.session_state.batches, start=1):
        st.sidebar.write(f"**Batch {i}:** {b['label']} — {len(b['files'])} file(s)")
    if st.sidebar.button("🗑️ Clear all"):
        st.session_state.batches = []
        st.session_state.batch_counter = 1
        st.session_state.consolidated_df = None
        st.rerun()

# ---- Add batch ----
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
        saved.append({
            "name": f.name,
            "size": f.size,
            "data": data,
            "inferred_folder": infer_folder_from_name(f.name),
            "sha256": sha256_bytes(data),
        })
    st.session_state.batches.append({
        "label": (batch_label or "").strip() or f"batch_{st.session_state.batch_counter}",
        "files": saved
    })
    st.session_state.batch_counter += 1
    st.success(f"Added {len(saved)} file(s) to **{batch_label}**")
    st.rerun()

# ---- Form: extraction ----
st.subheader("Extract sections & consolidate")
with st.form("extract_form", clear_on_submit=False):
    st.markdown("**Per-heading truncation (0 = no limit):**")
    col_a, col_b = st.columns(2)
    with col_a:
        max_update = st.number_input("Max chars: Update on the context…", min_value=0, value=0, step=250)
        debug = st.checkbox("Debug: show detected headings/snippets", value=False)
    with col_b:
        max_major = st.number_input("Max chars: Major Results… vs CPD", min_value=0, value=0, step=250)
        max_lessons = st.number_input("Max chars: Lessons learned & constraints", min_value=0, value=0, step=250)
    submit = st.form_submit_button("🧩 Extract & Consolidate to CSV")

# ---- Run extraction ----
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
                raw_text = extract_text_router(f["name"], f["data"])
                sections, meta = split_by_headings(raw_text)

                if debug:
                    st.markdown(f"**Debug for:** `{Path(f['name']).name}`")
                    st.caption(f"Matching method: {meta.get('method')}")
                    clean = meta.get("text_used", "")
                    hits = meta.get("hits", [])
                    if hits:
                        for rec in hits:
                            key, s, e, *_ = rec
                            snippet = clean[max(0, s-80):min(len(clean), e+120)].replace("\n", "⏎")
                            st.code(f"{key} [{s}:{e}] …{snippet}…")
                    else:
                        st.info("No headings detected; used keyword-box fallback.")

                row = {
                    "batch": batch["label"],
                    "inferred_folder": f["inferred_folder"],
                    "filename": Path(f["name"]).name,
                }
                for k in COLS_ORDER:
                    row[k] = apply_limit(sections.get(k, ""), per_heading_limits[k])
                rows.append(row)

                done += 1
                prog.progress(done / max(1, total), text=f"Processed {done}/{total}")

        st.session_state.consolidated_df = pd.DataFrame(rows)
        prog.empty()
        st.success(f"Extracted sections from {done} file(s).")

# ---- Preview & Download ----
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

with st.expander("ℹ️ Tips & dependencies"):
    st.write(
        "- Ensure real newlines are used. This build uses '\\n' (not literal '\\\\n').\n"
        "- If columns are blank, enable **Debug** to see matched ranges.\n"
        "- Regex accepts both classic headings and bare 'Part 1/2/3:' lines."
    )
