# Home.py
# One-file Streamlit app:
# - PDF -> TXT (temp folder)
# - Consolidate by headings + auto language detect & (optional) translate to English
# - Region overrides UX (Unknowns don't disappear; always refresh)
# - Private Chatroom over consolidated DF or temp .txt (BM25/TF-IDF + optional SBERT fallback)
# - Composer mode: Docked (fixed bottom) OR Inline (collapsible) with dynamic spacing to avoid collisions
# - No external data sharing

import os
import re
import glob
import shutil
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional retrieval libs
BM25_AVAILABLE = True
try:
    from rank_bm25 import BM25Okapi  # pip install rank-bm25
except Exception:
    BM25_AVAILABLE = False

# Optional SBERT (we'll fall back gracefully)
SBERT_IMPORTABLE = True
try:
    from sentence_transformers import SentenceTransformer  # may need wheels; see notes
except Exception:
    SBERT_IMPORTABLE = False

# Optional language tools
LANG_DETECT_AVAILABLE = True
try:
    from langdetect import detect as _detect_lang
except Exception:
    LANG_DETECT_AVAILABLE = False

ARGOS_AVAILABLE = True
try:
    import argostranslate.translate as argos_translate
except Exception:
    ARGOS_AVAILABLE = False

# ===================== PAGE SETUP =====================
st.set_page_config(page_title="PDF → TXT → Consolidate → Chat", layout="wide")
st.title("UNICEF COARs:  📄 NLP 💬 Generative AI")

# ===================== PDF BACKEND =====================
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
        "No PDF backend found. Install one of:\n"
        "• pip install pymupdf  (recommended)\n"
        "• pip install pdfplumber\n"
        "• pip install PyPDF2"
    )
    st.stop()

# ===================== TEMP OUTPUT FOLDER =====================
if "txt_out_dir" not in st.session_state:
    st.session_state.txt_out_dir = tempfile.mkdtemp(prefix="pdf_txt_")
out_dir = st.session_state.txt_out_dir

# ===================== GLOBAL OPTIONS STATE (defaults) =====================
if "opts" not in st.session_state:
    st.session_state.opts = {
        "lowercase": False,
        "strip_newlines": True,
        "include_paths": False,
        "max_context": 0,
        "max_contrib": 0,
        "max_collab": 0,
        "max_innov": 0,
        "auto_translate": True,
        "use_sbert": False,    # Default off to avoid sentencepiece build issues
        "alpha": 0.60,
    }

# ===================== NAV TABS =====================
tab_convert, tab_consolidate, tab_chat, tab_options = st.tabs(
    ["① Convert to TXT", "② Consolidate & Normalize", "③ Chatroom (Private)", "⚙️ Options / About"]
)

# ===================== HELPERS: TEXT, LANG, TRANSLATION =====================
def detect_lang(text: str) -> str:
    if not text or not text.strip() or not LANG_DETECT_AVAILABLE:
        return "unknown"
    try:
        return _detect_lang(text[:5000])
    except Exception:
        return "unknown"

def translate_to_en(text: str) -> str:
    """Offline best-effort: Argos only if installed + model available."""
    if not text or not text.strip():
        return text
    src = detect_lang(text)
    if src in ("unknown", "en"):
        return text
    if not ARGOS_AVAILABLE:
        return text
    try:
        langs = argos_translate.get_installed_languages()
        src_lang = next((l for l in langs if l.code == src), None)
        en_lang = next((l for l in langs if l.code == "en"), None)
        if not src_lang or not en_lang:
            return text
        translator = src_lang.get_translation(en_lang)
        return translator.translate(text)
    except Exception:
        return text

def sent_split(x: str):
    x = re.sub(r"\s+", " ", str(x).strip())
    if not x:
        return []
    return re.split(r"(?<=[\.\?\!])\s+", x)

def chunk_text(text: str, max_words: int = 140, overlap: int = 40):
    # sentence-aware chunks with word overlap to avoid boundary misses
    sents = sent_split(text)
    chunks, cur, count = [], [], 0
    for s in sents:
        w = len(s.split())
        if count + w > max_words and cur:
            chunks.append(" ".join(cur))
            # overlap
            ov, ccount = [], 0
            for rs in reversed(cur):
                cw = len(rs.split())
                if ccount + cw >= overlap:
                    ov.insert(0, rs)
                    break
                ov.insert(0, rs)
                ccount += cw
            cur = ov + [s]
            count = len(" ".join(cur).split())
        else:
            cur.append(s)
            count += w
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if c.strip()]

# ===================== COUNTRY / REGION HELPERS =====================
def _normalize_dashes_spaces(s: str) -> str:
    s = re.sub(r"[\u2010-\u2015\-]+", " ", s)  # all dashes → space
    s = re.sub(r"[_\s]+", " ", s).strip()
    return s

COUNTRY_ALIASES = {
    r"\bbolivia\b.*": "Bolivia",
    r"\bbosnia[\s\-–—]*and[\s\-–—]*herzegovina\b": "Bosnia and Herzegovina",
    r"\bcote[\s\-–—]*d[\s\-–—]*ivoire\b": "Côte d’Ivoire",
    r"\btimor[\s\-–—]*leste\b": "Timor-Leste",
    r"\bdemocratic\s*republic\s*of\s*the\s*congo\b": "Democratic Republic of the Congo",
    r"\brepublic\s*of\s*korea\b": "ROK",
    r"\bnorth\s*korea\b|^DPRK$": "DPRK",
    r"\blao[\s\-–—]*pdr\b": "Lao PDR",
    r"\bstate\s*of\s*palestine\b|^palestine$": "State of Palestine",
    r"\bviet[\s\-–—]*nam\b": "Viet Nam",
}
COUNTRY_TO_REGION = {
    # ROSA
    "Afghanistan": "ROSA", "Pakistan": "ROSA", "India": "ROSA", "Bangladesh": "ROSA",
    "Nepal": "ROSA", "Bhutan": "ROSA", "Sri Lanka": "ROSA", "Maldives": "ROSA",
    # ECA
    "Albania": "ECA", "Bosnia and Herzegovina": "ECA", "Armenia": "ECA", "Azerbaijan": "ECA",
    "Georgia": "ECA", "Ukraine": "ECA", "Belarus": "ECA", "Moldova": "ECA",
    "Kazakhstan": "ECA", "Kyrgyzstan": "ECA", "Tajikistan": "ECA", "Turkmenistan": "ECA",
    "Uzbekistan": "ECA", "Serbia": "ECA", "North Macedonia": "ECA", "Montenegro": "ECA",
    "Kosovo": "ECA", "Türkiye": "ECA", "Turkey": "ECA", "Romania": "ECA", "Bulgaria": "ECA",
    "Greece": "ECA", "Croatia": "ECA", "Slovenia": "ECA",
    # LAC
    "Argentina": "LAC", "Bolivia": "LAC", "Brazil": "LAC", "Chile": "LAC", "Uruguay": "LAC",
    "Paraguay": "LAC", "Peru": "LAC", "Ecuador": "LAC", "Colombia": "LAC", "Venezuela": "LAC",
    "Mexico": "LAC", "Guatemala": "LAC", "El Salvador": "LAC", "Honduras": "LAC",
    "Nicaragua": "LAC", "Costa Rica": "LAC", "Panama": "LAC", "Cuba": "LAC",
    "Dominican Republic": "LAC", "Haiti": "LAC", "Belize": "LAC", "Jamaica": "LAC",
    "Barbados": "LAC", "Guyana": "LAC", "Suriname": "LAC", "Trinidad and Tobago": "LAC",
    # MENA
    "Algeria": "MENA", "Egypt": "MENA", "Morocco": "MENA", "Tunisia": "MENA", "Libya": "MENA",
    "Sudan": "MENA", "Djibouti": "MENA", "Yemen": "MENA", "Syria": "MENA", "Iraq": "MENA",
    "Jordan": "MENA", "Lebanon": "MENA", "State of Palestine": "MENA", "Iran": "MENA",
    # EAP
    "China": "EAP", "Mongolia": "EAP", "DPRK": "EAP", "ROK": "EAP", "Japan": "EAP",
    "Philippines": "EAP", "Indonesia": "EAP", "Malaysia": "EAP", "Thailand": "EAP",
    "Viet Nam": "EAP", "Vietnam": "EAP", "Cambodia": "EAP", "Lao PDR": "EAP", "Laos": "EAP",
    "Myanmar": "EAP", "Timor-Leste": "EAP", "Papua New Guinea": "EAP", "Solomon Islands": "EAP",
    "Fiji": "EAP", "Samoa": "EAP", "Tonga": "EAP", "Vanuatu": "EAP", "Kiribati": "EAP",
    "Micronesia": "EAP",
    # ESA
    "Angola": "ESA", "Ethiopia": "ESA", "Kenya": "ESA", "Tanzania": "ESA", "Uganda": "ESA",
    "Rwanda": "ESA", "Burundi": "ESA", "Somalia": "ESA", "South Sudan": "ESA", "Eritrea": "ESA",
    "Botswana": "ESA", "Lesotho": "ESA", "Namibia": "ESA", "South Africa": "ESA",
    "Eswatini": "ESA", "Zimbabwe": "ESA", "Zambia": "ESA", "Malawi": "ESA", "Mozambique": "ESA",
    "Madagascar": "ESA", "Comoros": "ESA", "Seychelles": "ESA", "Mauritius": "ESA",
    # WCA
    "Nigeria": "WCA", "Ghana": "WCA", "Côte d’Ivoire": "WCA", "Cote d Ivoire": "WCA",
    "Cote d'Ivoire": "WCA", "Senegal": "WCA", "Mali": "WCA", "Niger": "WCA", "Chad": "WCA",
    "Cameroon": "WCA", "Central African Republic": "WCA", "CAR": "WCA",
    "Sierra Leone": "WCA", "Liberia": "WCA", "Guinea": "WCA", "Guinea-Bissau": "WCA",
    "Togo": "WCA", "Benin": "WCA", "Burkina Faso": "WCA", "Gambia": "WCA", "Congo": "WCA",
    "DRC": "WCA", "Democratic Republic of the Congo": "WCA", "Gabon": "WCA",
    "Equatorial Guinea": "WCA", "Mauritania": "WCA", "Cape Verde": "WCA",
}

def normalize_country_name(raw: str) -> str:
    if not raw:
        return ""
    raw = re.sub(r"\([^)]*\)$", "", raw)
    raw = re.sub(r"\b(coar|annual report|unicef|report)\b.*$", "", raw, flags=re.IGNORECASE)
    cleaned = _normalize_dashes_spaces(raw)
    for pat, canon in COUNTRY_ALIASES.items():
        if re.search(pat, cleaned, flags=re.IGNORECASE):
            cleaned = canon
            break
    keep_caps = {"DRC", "ROK", "DPRK"}
    parts = [tok.upper() if tok.upper() in keep_caps else tok.capitalize() for tok in cleaned.split()]
    country = " ".join(parts)
    country = country.replace("Cote D Ivoire", "Côte d’Ivoire").replace("Timor Leste", "Timor-Leste")
    return country.strip()

def parse_country_year_from_filename(name: str):
    base = os.path.splitext(os.path.basename(name))[0]
    base = _normalize_dashes_spaces(base)
    m = re.search(r"(?P<country>.+?)\s+(?P<year>19\d{2}|20\d{2})\b", base)
    if not m:
        m = re.search(r"(?P<year>19\d{2}|20\d{2})\s+(?P<country>.+)", base)
    if not m:
        return (None, None)
    raw_country = re.sub(r"\s*(coar|annual report|unicef|report).*$", "", m.group("country"), flags=re.IGNORECASE)
    year = int(m.group("year"))
    return (normalize_country_name(raw_country) or None, year)

def country_to_region(country: str | None) -> str:
    if not country:
        return "Unknown"
    if country in COUNTRY_TO_REGION:
        return COUNTRY_TO_REGION[country]
    for k, v in COUNTRY_TO_REGION.items():
        if k.lower() == country.lower():
            return v
    low = country.lower()
    for k, v in COUNTRY_TO_REGION.items():
        if k.lower() in low or low in k.lower():
            return v
    return "Unknown"

# ===================== EXTRACT TEXT FROM PDF =====================
def extract_text(pdf_bytes: bytes) -> str:
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

# ===================== TAB ①: CONVERT =====================
with tab_convert:
    uploaded = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        run_convert = st.button("🚀 Convert to TXT", use_container_width=True)
    with c2:
        clear = st.button("🧹 Clear Temp Folder", type="secondary", use_container_width=True)
    with c3:
        show_folder = st.button("📁 Show Temp Folder", use_container_width=True)
    with c4:
        pass

    if clear:
        try:
            shutil.rmtree(out_dir, ignore_errors=True)
            st.session_state.txt_out_dir = tempfile.mkdtemp(prefix="pdf_txt_")
            out_dir = st.session_state.txt_out_dir
            st.success(f"Cleared. New temp folder: {out_dir}")
        except Exception as e:
            st.error(f"Failed to clear folder: {e}")

    if show_folder:
        st.info(out_dir)

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
                    # unique filename if duplicates
                    uniq = out_path
                    counter = 1
                    while os.path.exists(uniq):
                        uniq = os.path.join(out_dir, f"{base}({counter}).txt")
                        counter += 1
                    with open(uniq, "w", encoding="utf-8", errors="ignore") as f:
                        f.write(text)
                    records.append({"pdf": uf.name, "txt_path": uniq, "chars": len(text)})
                    st.write(f"✅ Saved: **{os.path.basename(uniq)}**")
                except Exception as e:
                    st.error(f"Error converting {uf.name}: {e}")
                finally:
                    prog.progress(i / len(uploaded))
            if records:
                df_conv = pd.DataFrame(records)
                st.success(f"Done! Converted {len(df_conv)} PDFs.")
                st.dataframe(df_conv, use_container_width=True)
                # zip of the entire output folder
                zip_base = os.path.join(tempfile.gettempdir(), f"txt_export_{os.path.basename(out_dir)}")
                zip_file = shutil.make_archive(zip_base, "zip", out_dir)
                with open(zip_file, "rb") as f:
                    st.download_button("💾 Download all TXT as ZIP", data=f.read(),
                                       file_name="pdf_texts.zip", mime="application/zip",
                                       use_container_width=True)

# ===================== SECTION PATTERNS =====================
SECTION_ALIASES = {
    "context": [
        r"Part\s*1\s*[:\-–]\s*Situation\s+update\s+in\s+the\s+country",
        r"Update\s+on\s+the\s+context\s+and\s+situation\s+of\s+children",
        r"Situation\s+update\s+in\s+the\s+country",
    ],
    "contributions": [
        r"Major\s+contributions\s+and\s+drivers\s+of\s+results",
        r"Part\s*2\s*[:\-–].{0,80}\bresults?\b",
        r"Results\s+achieved",
    ],
    "collaborations": [
        r"UN\s+Collaboration\s+and\s+Other\s+Partnerships?",
        r"UN\s+cooperation|UN\s+collaboration",
        r"Partnerships?",
    ],
    "innovations": [
        r"Lessons\s+Learn(?:ed|t)\s+and\s+(?:Innovations?|good\s+practices)",
        r"Lessons\s+learn(?:ed|t)",
        r"Innovation[s]?",
    ],
}
SECTION_ORDER = ["context", "contributions", "collaborations", "innovations"]

def compile_section_patterns():
    compiled = {}
    for key, pats in SECTION_ALIASES.items():
        joined = "|".join(f"(?:{p})" for p in pats)
        compiled[key] = re.compile(joined, flags=re.IGNORECASE | re.DOTALL)
    return compiled

SECTION_PATTERNS = compile_section_patterns()

# ===================== CONSOLIDATE HELPERS =====================
def extract_sections_from_txt(file_path: Path, include_paths: bool) -> dict:
    result = {k: "" for k in SECTION_ORDER}
    result["filename"] = file_path.name
    if include_paths:
        result["filepath"] = str(file_path)
    country, year = parse_country_year_from_filename(file_path.name)
    result["country"] = country
    result["year"] = year
    result["unicef_region"] = country_to_region(country)
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return result
    hits = []
    for key in SECTION_ORDER:
        m = SECTION_PATTERNS[key].search(text)
        if m:
            start = m.start()
            heading_end = text.find("\n", m.end())
            if heading_end == -1:
                heading_end = m.end()
            hits.append((key, start, max(heading_end, m.end())))
    hits.sort(key=lambda x: x[1])
    for idx, (key, _, content_start) in enumerate(hits):
        content_end = len(text) if idx == len(hits) - 1 else hits[idx + 1][1]
        result[key] = text[content_start:content_end].strip()
    return result

def clean_and_truncate(df: pd.DataFrame, lowercase: bool, strip_newlines: bool,
                       max_context: int, max_contrib: int, max_collab: int, max_innov: int) -> pd.DataFrame:
    trunc = {"context": max_context, "contributions": max_contrib, "collaborations": max_collab, "innovations": max_innov}
    for col in SECTION_ORDER:
        if lowercase:
            df[col] = df[col].apply(lambda x: x.lower() if isinstance(x, str) else x)
        if strip_newlines:
            df[col] = df[col].str.replace("\n", " ", regex=False)
            df[col] = df[col].str.replace(r"\s{2,}", " ", regex=True)
        if trunc[col] and trunc[col] > 0:
            df[col] = df[col].apply(lambda x: x[:trunc[col]] if isinstance(x, str) else x)
    return df

def translate_sections(df: pd.DataFrame, enable: bool) -> pd.DataFrame:
    if not enable:
        return df
    for col in SECTION_ORDER:
        def _t(x):
            if not isinstance(x, str) or not x.strip():
                return x
            lang = detect_lang(x)
            if lang not in ("en", "unknown"):
                return translate_to_en(x)
            return x
        df[col] = df[col].apply(_t)
    return df

# ===================== TAB ②: CONSOLIDATE (MERGED & FIXED) =====================
with tab_consolidate:
    st.subheader("🧾 Consolidate TXT → Per-Heading Columns")
    txt_root = Path(out_dir)

    # Equal-width buttons
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        btn_consolidate = st.button("🧾 Run Consolidation", use_container_width=True)
    with c2:
        show_unknown_only = st.toggle("Show Unknowns Only", value=False)
    with c3:
        btn_refresh = st.button("🔄 Refresh View", use_container_width=True)
    with c4:
        btn_clear_picks = st.button("🧽 Clear Region Picks", use_container_width=True)

    opts = st.session_state.opts

    # --- helpers we call multiple times ---
    def _load_txt_rows():
        txt_files = sorted([p for p in Path(out_dir).rglob("*.txt")])
        rows, found_dbg = [], []
        prog = st.progress(0)
        for i, p in enumerate(txt_files, 1):
            row = extract_sections_from_txt(p, opts["include_paths"])
            rows.append(row)
            found_dbg.append({"filename": row["filename"], **{k: (len(row.get(k, "")) > 0) for k in SECTION_ORDER}})
            prog.progress(i / max(1, len(txt_files)))
        return rows, found_dbg

    def _apply_overrides(df_in: pd.DataFrame) -> pd.DataFrame:
        df2 = df_in.copy()
        def _pick(r):
            fset = st.session_state["region_overrides_by_file"].get(r.get("filename"))
            if fset: return fset
            cset = st.session_state["region_overrides_by_country"].get(r.get("country"))
            if cset: return cset
            return r.get("unicef_region")
        df2["unicef_region"] = df2.apply(_pick, axis=1)
        return df2

    def _recompute_and_save():
        """Build df from TXT -> clean -> translate -> overrides -> save to session + CSV."""
        if not Path(out_dir).exists():
            return None, None
        txts = sorted([p for p in Path(out_dir).rglob("*.txt")])
        if not txts:
            return None, None

        rows, found_dbg = _load_txt_rows()
        base_cols = ["filename"] + (["filepath"] if opts["include_paths"] else [])
        meta_cols = ["country", "year", "unicef_region"]
        cols = base_cols + meta_cols + SECTION_ORDER
        df = pd.DataFrame(rows, columns=cols)

        df = clean_and_truncate(
            df,
            opts["lowercase"], opts["strip_newlines"],
            opts["max_context"], opts["max_contrib"], opts["max_collab"], opts["max_innov"]
        )
        df = translate_sections(df, opts["auto_translate"])

        # ensure override dicts exist
        st.session_state.setdefault("region_overrides_by_file", {})
        st.session_state.setdefault("region_overrides_by_country", {})

        df = _apply_overrides(df)

        # save
        st.session_state["consolidated_df"] = df
        csv_path = os.path.join(out_dir, "coar_sections.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        st.session_state["consolidated_csv_path"] = csv_path
        return df, pd.DataFrame(found_dbg)

    # init override stores + clear picks button
    st.session_state.setdefault("region_overrides_by_file", {})
    st.session_state.setdefault("region_overrides_by_country", {})
    if btn_clear_picks:
        st.session_state["region_overrides_by_file"].clear()
        st.session_state["region_overrides_by_country"].clear()
        st.success("Cleared region overrides.")
        st.rerun()

    # recompute on demand
    df, found_dbg_df = (None, None)
    if btn_consolidate or btn_refresh:
        if not txt_root.exists():
            st.warning("Temp folder not found. Convert PDFs to TXT first.")
        else:
            df, found_dbg_df = _recompute_and_save()

    # show current state if already consolidated earlier
    if df is None:
        df = st.session_state.get("consolidated_df")
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            # Best effort rebuild of found_dbg table
            _, found_dbg_df = _load_txt_rows()

    if df is None or (not isinstance(df, pd.DataFrame)) or df.empty:
        st.info("Run consolidation to view results.")
    else:
        st.caption("Detected headings per file (True = section found)")
        if found_dbg_df is not None:
            st.dataframe(found_dbg_df, use_container_width=True)

        # ---- Unknowns panel (always computed fresh from current df) ----
        current_unknowns = df[df["unicef_region"].eq("Unknown")].copy()

        with st.expander("Assign UNICEF region (Unknowns)", expanded=not current_unknowns.empty):
            REGIONS = ["", "ROSA", "ECA", "LAC", "MENA", "EAP", "ESA", "WCA"]
            if current_unknowns.empty:
                st.success("No unknown regions 🎉")
            else:
                st.warning(f"{len(current_unknowns)} file(s) have Unknown UNICEF region. Assign and click Apply.")
                unknown_filenames = list(current_unknowns["filename"])  # stable this round

                with st.form("assign_unknowns_form"):
                    for fname in unknown_filenames:
                        row = current_unknowns[current_unknowns["filename"] == fname].iloc[0]
                        country_val = row.get("country") or "(unparsed)"
                        default = st.session_state["region_overrides_by_file"].get(fname, "")
                        c1_, c2_, c3_ = st.columns([5, 3, 2])
                        with c1_:
                            st.write(f"**{fname}**")
                            st.caption(f"country: {country_val}")
                        with c2_:
                            _ = st.selectbox(
                                "Region",
                                REGIONS,
                                index=(REGIONS.index(default) if default in REGIONS else 0),
                                key=f"reg_pick_file_{fname}",
                                label_visibility="collapsed",
                            )
                        with c3_:
                            st.write("")

                    submitted = st.form_submit_button("Apply assignments ✅", use_container_width=True)

                if submitted:
                    # apply only non-empty selections
                    changed = 0
                    for fname in unknown_filenames:
                        key = f"reg_pick_file_{fname}"
                        pick = st.session_state.get(key, "")
                        if pick:
                            st.session_state["region_overrides_by_file"][fname] = pick
                            changed += 1
                        # clear the temporary widget state so stale values don't linger
                        if key in st.session_state:
                            del st.session_state[key]

                    # Recompute + persist, then rerun to refresh the view
                    _ = _recompute_and_save()
                    if changed > 0:
                        st.success(f"Applied {changed} assignment(s).")
                    st.rerun()

            st.markdown("---")
            st.markdown("**Bulk by country (optional)** – applies to all rows of that country (unless a per-file override exists).")
            unk_countries = sorted((current_unknowns["country"].dropna()).unique().tolist() + (["(unparsed)"] if current_unknowns["country"].isna().any() else []))
            bc1, bc2, bc3 = st.columns([3, 2, 2])
            with bc1:
                bulk_country = st.selectbox("Pick a country to assign", [""] + unk_countries, index=0, key="bulk_country")
            with bc2:
                bulk_region = st.selectbox("Assign region", ["", "ROSA", "ECA", "LAC", "MENA", "EAP", "ESA", "WCA"], index=0, key="bulk_region")
            with bc3:
                if st.button("Apply bulk", use_container_width=True):
                    if bulk_country and bulk_region:
                        if bulk_country != "(unparsed)":
                            st.session_state["region_overrides_by_country"][bulk_country] = bulk_region
                            _ = _recompute_and_save()
                            st.success(f"Assigned {bulk_region} to '{bulk_country}'.")
                            st.rerun()
                        else:
                            st.info("Cannot bulk-assign '(unparsed)'. Use per-file assignment above.")

        # Main table (live view)
        view_df = df[df["unicef_region"].eq("Unknown")] if show_unknown_only else df
        st.success(f"Consolidated **{len(df)}** files. Added: country, year, unicef_region.")
        st.dataframe(view_df, use_container_width=True)
        st.download_button(
            "💾 Download Consolidated CSV",
            df.to_csv(index=False),
            file_name="coar_sections.csv",
            mime="text/csv",
            use_container_width=True
        )

# ===================== TAB ③: CHATROOM (PRIVATE, DOCKED or INLINE-COLLAPSIBLE) =====================
with tab_chat:
    st.subheader("🔒 Chatroom (Private, local-only)")

    # --- Source: prefer consolidated_df, else temp .txt ---
    df = st.session_state.get("consolidated_df", None)

    def load_txt_folder(folder_path: str) -> pd.DataFrame:
        files = glob.glob(os.path.join(folder_path, "*.txt"))
        recs = []
        for path in files:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                recs.append({"filename": os.path.basename(path), "text": text})
            except Exception as e:
                st.warning(f"Failed to read {path}: {e}")
        return pd.DataFrame(recs)

    used_source = None
    if isinstance(df, pd.DataFrame) and not df.empty:
        used_source = f"Session DataFrame (‘consolidated_df’): {df.shape[0]} rows"
    else:
        txt_df = load_txt_folder(out_dir)
        if not txt_df.empty:
            df = txt_df
            used_source = f"Text files: {len(txt_df)} docs from {out_dir}"
        else:
            st.error("No data found (no consolidated_df in session and no .txt files in temp folder).")
            st.stop()

    st.success(f"Loaded: {used_source}")

    # --- Build corpus columns ---
    if "text" in df.columns and "filename" in df.columns:
        available_cols = list(df.columns)
        default_cols = ["text"]
    else:
        available_cols = list(df.columns)
        default_cols = [c for c in ["context", "contributions", "collaborations", "innovations"] if c in available_cols]
        if not default_cols:
            default_cols = [c for c in available_cols if c.lower() not in ("filename", "id")][:4]

    cols_for_chat = st.multiselect(
        "Columns to include for Q&A grounding",
        options=available_cols,
        default=default_cols
    )
    if not cols_for_chat:
        st.warning("Select at least one column to include in the chat corpus.")
        st.stop()

    # --- Build docs ---
    def _coalesce_cols(row, cols):
        parts = []
        for c in cols:
            if c in row and pd.notna(row[c]):
                v = str(row[c]).strip()
                if v: parts.append(v)
        return " ".join(parts).strip()

    if "text" in df.columns and cols_for_chat == ["text"]:
        docs_df = pd.DataFrame({
            "filename": df["filename"] if "filename" in df.columns else np.arange(len(df)).astype(str),
            "text": df["text"].astype(str)
        })
    else:
        docs_df = pd.DataFrame({
            "filename": df["filename"] if "filename" in df.columns else np.arange(len(df)).astype(str),
            "text": df.apply(lambda r: _coalesce_cols(r, cols_for_chat), axis=1).astype(str)
        })

    docs_df["text"] = docs_df["text"].fillna("").astype(str)
    docs_df = docs_df[docs_df["text"].str.strip().ne("")]
    if docs_df.empty:
        st.error("All selected columns are empty. Pick different columns.")
        st.stop()

    # --- Chunk into passages ---
    passages = []
    for i, row in docs_df.iterrows():
        for j, ch in enumerate(chunk_text(row["text"], max_words=140, overlap=40)):
            passages.append({"pid": f"{i}-{j}", "filename": row["filename"], "chunk_index": j, "text": ch})
    passages_df = pd.DataFrame(passages)
    if passages_df.empty:
        st.error("No text found after chunking. Pick different columns.")
        st.stop()

    # --- Sidebar: chat settings + COMPOSER MODE ---
    with st.sidebar:
        st.header("Chat settings")
        top_k = st.slider("Top-k passages", 3, 20, 8, 1)
        sim_gate = st.slider("Answer gate (refuse if below)", 0.02, 0.60, 0.12, 0.01)
        use_sbert = st.checkbox(
            "Use SBERT semantic reranking (local)",
            value=st.session_state.opts.get("use_sbert", False),
            help="If not available locally, we’ll fall back automatically."
        )
        alpha = st.slider("Hybrid weight (SBERT vs BM25)", 0.0, 1.0, st.session_state.opts.get("alpha", 0.60), 0.05)
        rebuild = st.button("🔁 Rebuild Index", use_container_width=True)
        clear_chat_btn = st.button("🧹 Clear chat history", use_container_width=True)

        st.markdown("---")
        composer_mode = st.radio(
            "Composer mode",
            ["Docked (fixed bottom)", "Inline (collapsible)"],
            index=0,
            help="Use Inline to avoid collisions when the sidebar/navigation is expanded."
        )
        inline_expanded_default = st.checkbox("Expand inline composer by default", value=False)

    if clear_chat_btn:
        st.session_state.chat_history = []

    # --- Build retrievers (rebuildable) ---
    @st.cache_resource(show_spinner=False)
    def build_indices(pass_df: pd.DataFrame):
        bm25 = None
        if BM25_AVAILABLE:
            tokens = [re.findall(r"[A-Za-z0-9]+", t.lower()) for t in pass_df["text"].tolist()]
            bm25 = BM25Okapi(tokens)
        stop_words = {
            "the","and","to","of","in","for","on","with","a","an","is","are","was","were","be","by","as","that","this","it",
            "from","or","at","we","our","their","they","these","those","has","have","had","not","but","can","will","may"
        }
        vec = TfidfVectorizer(stop_words=sorted(stop_words), max_df=0.95, ngram_range=(1,2))
        X = vec.fit_transform(pass_df["text"].values)
        return bm25, vec, X

    if rebuild:
        passages_df = passages_df.sample(frac=1.0, random_state=0).sort_values(["filename","chunk_index"]).reset_index(drop=True)

    bm25, tfidf_vec, X_tfidf = build_indices(passages_df)

    sbert = None
    if use_sbert and SBERT_IMPORTABLE:
        try:
            DEFAULT_MODEL = "all-MiniLM-L6-v2"
            sbert = SentenceTransformer(DEFAULT_MODEL)
        except Exception:
            sbert = None
            st.info("SBERT model not available locally; using BM25/TF-IDF only.")

    def retrieve(query: str, k: int = 8):
        # 1) Lexical
        if bm25 is not None:
            q_tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
            bm25_scores = np.array(bm25.get_scores(q_tokens))
            top_lex_idx = np.argsort(-bm25_scores)[: min(5000, len(bm25_scores))]
            cand_idx = top_lex_idx
            lex_scores = bm25_scores[top_lex_idx]
            lex_norm = (lex_scores - lex_scores.min()) / (np.ptp(lex_scores) + 1e-12) if len(lex_scores) else lex_scores
        else:
            q_vec = tfidf_vec.transform([query])
            sims = cosine_similarity(q_vec, X_tfidf).ravel()
            top_lex_idx = np.argsort(-sims)[: min(5000, len(sims))]
            cand_idx = top_lex_idx
            lex_norm = (sims[top_lex_idx] - sims[top_lex_idx].min()) / (np.ptp(sims[top_lex_idx]) + 1e-12)

        # 2) Optional semantic rerank
        if sbert is not None and len(cand_idx):
            cand_texts = passages_df.iloc[cand_idx]["text"].tolist()
            q_emb = sbert.encode([query], normalize_embeddings=True)
            c_emb = sbert.encode(cand_texts, normalize_embeddings=True)
            sem = (q_emb @ c_emb.T).ravel()
            sem_norm = (sem - sem.min()) / (np.ptp(sem) + 1e-12)
            final = alpha * sem_norm + (1 - alpha) * lex_norm
        else:
            final = lex_norm

        order = np.argsort(-final)[:k]
        chosen_idx = np.array(cand_idx)[order]
        chosen = passages_df.iloc[chosen_idx].copy()
        chosen["score"] = final[order]
        return chosen.reset_index(drop=True)

    def synthesize(query: str, results_df: pd.DataFrame, max_snippets: int = 4, max_chars: int = 900):
        out_sents, used = [], set()
        key_terms = [w for w in re.findall(r"[A-Za-z]{3,}", query.lower())]
        for _, row in results_df.head(max_snippets).iterrows():
            sents = sent_split(row["text"])
            scored = []
            for s in sents:
                low = s.lower()
                hits = sum(1 for t in key_terms if t in low)
                if hits: scored.append((hits, len(s), s))
            if not scored and sents:
                scored = [(0, len(sents[0]), sents[0])]
            picked = [x[2] for x in sorted(scored, key=lambda x: (-x[0], x[1]))[:2]]
            for p in picked:
                if p not in used:
                    out_sents.append(p); used.add(p)
        answer = " ".join(out_sents).strip()
        if len(answer) > max_chars:
            answer = answer[:max_chars].rsplit(" ", 1)[0] + "…"
        sources = list(dict.fromkeys(results_df["filename"].tolist()))
        return answer, sources

    def max_confidence(results_df: pd.DataFrame) -> float:
        return float(results_df["score"].max()) if not results_df.empty else 0.0

    # --- UI layout & CSS depending on composer mode (with dynamic bottom padding) ---
    INPUT_BAR_H = 116   # fallback height
    MAX_COL_W   = 900
    EXTRA_GAP   = 28    # breathing room above the input

    if composer_mode.startswith("Docked"):
        # Docked: fixed input + dynamic bottom padding so content never hides behind it
        st.markdown(f"""
        <style>
        .chat-container {{
          max-width: {MAX_COL_W}px; width: calc(100% - 2rem); margin: 0 auto;
        }}
        /* Fallback uses INPUT_BAR_H; JS below overrides via --chat-input-h */
        [data-testid="stAppViewContainer"] > .main {{
          padding-bottom: calc(var(--chat-input-h, {INPUT_BAR_H}px) + {EXTRA_GAP}px) !important;
        }}
        div[data-testid="stChatInput"] {{
          position: fixed !important; left: 50%; transform: translateX(-50%);
          bottom: 0; max-width: {MAX_COL_W}px; width: calc(100% - 2rem);
          z-index: 9999; background: var(--background-color);
          border-top: 1px solid rgba(49,51,63,0.2);
          padding: .25rem 0 .5rem 0;
        }}
        </style>
        """, unsafe_allow_html=True)

        # Dynamically measure the input height and set --chat-input-h
        components.html("""
        <script>
        (function(){
          const doc = parent.document;
          const main = doc.querySelector('[data-testid="stAppViewContainer"] > .main');
          const input = doc.querySelector('div[data-testid="stChatInput"]');
          if(!main || !input) return;
          const update = () => {
            const h = input.offsetHeight || 0;
            main.style.setProperty('--chat-input-h', h + 'px');
          };
          new ResizeObserver(update).observe(input);
          window.addEventListener('resize', update);
          update();
        })();
        </script>
        """, height=0)
    else:
        # Inline: hide built-in chat_input, no bottom padding
        st.markdown(f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{ padding-bottom: 0 !important; }}
        div[data-testid="stChatInput"] {{ display: none !important; }}
        .chat-container {{ max-width: {MAX_COL_W}px; width: calc(100% - 2rem); margin: 0 auto; }}
        </style>
        """, unsafe_allow_html=True)

    # --- Chat history & messages (centered column) ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown("### Chat")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Auto-scroll to bottom
    st.markdown("<div id='chat-end' style='height:1px;'></div>", unsafe_allow_html=True)
    components.html(
        """
        <script>
          const s = parent.document.scrollingElement || parent.document.documentElement;
          s.scrollTop = s.scrollHeight;
        </script>
        """,
        height=0,
    )

    # Close centered container BEFORE input (docked) or before inline expander
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Composer handling ---
    user_q = None

    if composer_mode.startswith("Docked"):
        # Use built-in fixed chat input
        user_q = st.chat_input("Ask a question about the loaded documents…")
    else:
        # Inline collapsible composer (no collision with sidebar)
        with st.expander("💬 Compose message", expanded=inline_expanded_default):
            st.text_area(
                "Type your message",
                key="inline_composer",
                label_visibility="collapsed",
                height=120,
                placeholder="Ask a question about the loaded documents…"
            )
            send = st.button("Send", use_container_width=True, key="send_inline")
        if send:
            q = (st.session_state.get("inline_composer") or "").strip()
            if q:
                user_q = q
                st.session_state["inline_composer"] = ""  # clear after send

    # --- Process query ---
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        lower_q = user_q.lower()
        blocked = [
            "send", "email", "upload", "share", "post", "publish", "external", "internet", "api", "slack", "teams",
            "full dataset", "entire dataset", "raw data", "all rows", "download everything"
        ]
        if any(p in lower_q for p in blocked):
            reply = "I can’t share or export data. I can only answer questions grounded in the locally loaded documents."
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
        else:
            hits = retrieve(user_q, k=top_k)
            if max_confidence(hits) < sim_gate:
                tip = "Try lowering the answer gate in the sidebar, or include specific keywords from the document."
                reply = f"That seems outside the scope of the loaded documents. {tip}"
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.markdown(reply)
            else:
                answer, sources = synthesize(user_q, hits)
                if not answer:
                    answer = "I found relevant passages but couldn’t synthesize a concise answer. Try asking more specifically."
                final = f"{answer}\n\n**Sources (filenames):** {', '.join(sources[:5])}"
                st.session_state.chat_history.append({"role": "assistant", "content": final})
                with st.chat_message("assistant"):
                    st.markdown(final)

# ===================== TAB ⚙️: OPTIONS / ABOUT =====================
with tab_options:
    st.markdown("### Processing Options")
    o = st.session_state.opts
    colA, colB, colC = st.columns(3)
    with colA:
        o["lowercase"] = st.checkbox("lowercase all extracted text", value=o["lowercase"])
        o["strip_newlines"] = st.checkbox("remove newlines (join as one paragraph)", value=o["strip_newlines"])
        o["include_paths"] = st.checkbox("include full file path column", value=o["include_paths"])
        o["auto_translate"] = st.checkbox("🈺 auto-detect language & translate sections to English (Argos offline)", value=o["auto_translate"],
                                          help="Requires Argos Translate models installed for your languages → English. If missing, text stays as-is.")
    with colB:
        o["max_context"] = st.number_input("Max chars: context (0 = no limit)", min_value=0, value=o["max_context"], step=1000)
        o["max_contrib"] = st.number_input("Max chars: contributions (0 = no limit)", min_value=0, value=o["max_contrib"], step=1000)
        o["max_collab"]  = st.number_input("Max chars: collaborations (0 = no limit)", min_value=0, value=o["max_collab"], step=1000)
        o["max_innov"]   = st.number_input("Max chars: innovations (0 = no limit)", min_value=0, value=o["max_innov"], step=1000)
    with colC:
        st.markdown("### Chat semantics")
        o["use_sbert"] = st.checkbox("Use SBERT semantic reranking (local)", value=o["use_sbert"],
                                     help="If the model isn't installed locally (and you’re offline), the app falls back to BM25/TF-IDF.")
        o["alpha"] = st.slider("Hybrid weight (SBERT vs BM25)", 0.0, 1.0, o["alpha"], 0.05)

    st.markdown("---")
    st.markdown("**Notes:**")
    st.markdown("- This app never calls external APIs; everything is local/private.")
    st.markdown("- For best PDF extraction, install **PyMuPDF** (`pip install pymupdf`).")
    st.markdown("- For offline translation, install **Argos Translate** and language models (e.g. es→en).")
    st.markdown("- If you’re on Python 3.13, **SBERT may require build tools**. Keep it off or use Python 3.11 for ready-made wheels.")
