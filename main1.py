# main1.py
import streamlit as st
import os
import re
import tempfile
import shutil
from io import BytesIO
from pathlib import Path
import pandas as pd

# ===================== APP SETUP =====================
st.set_page_config(page_title="PDF → TXT → Headings DataFrame", layout="wide")
st.title("📄 Convert PDFs to TXT → 🧾 Consolidate by Headings")

st.sidebar.markdown("### Flow")
st.sidebar.write("1) Upload PDFs → Convert to TXT\n2) Consolidate → Analyze")

# ===================== PICK A PDF BACKEND =====================
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

# ===================== TEMP OUTPUT FOLDER =====================
if "txt_out_dir" not in st.session_state:
    st.session_state.txt_out_dir = tempfile.mkdtemp(prefix="pdf_txt_")
out_dir = st.session_state.txt_out_dir
st.info(f"All .txt files will be saved to this temporary folder:\n{out_dir}")

# ===================== UPLOAD & CONVERT =====================
uploaded = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

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

# ---- Flexible section aliases (covers COAR variants) ----
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

# ===================== COUNTRY / YEAR / REGION HELPERS =====================
# Optional normalization via pycountry (makes parser tolerant to official/common names)
try:
    import pycountry
except Exception:
    pycountry = None

# No inline (?i); use flags in search/sub
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

def _normalize_dashes_spaces(s: str) -> str:
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\-]+", " ", s)  # all dashes → space
    s = re.sub(r"[_\s]+", " ", s).strip()
    return s

def normalize_country_name(raw: str) -> str:
    if not raw:
        return ""
    raw = re.sub(r"\([^)]*\)$", "", raw)  # remove "(1)" suffixes
    raw = re.sub(r"\b(coar|annual report|unicef|report)\b.*$", "", raw, flags=re.IGNORECASE)
    cleaned = _normalize_dashes_spaces(raw)

    # alias regexes first (case-insensitive)
    for pat, canon in COUNTRY_ALIASES.items():
        if re.search(pat, cleaned, flags=re.IGNORECASE):
            cleaned = canon
            break

    # pycountry official/common matching (loose)
    try:
        import pycountry  # local import keeps optional dependency optional
        for c in pycountry.countries:
            names = {c.name}
            if hasattr(c, "official_name"): names.add(c.official_name)
            if hasattr(c, "common_name"):   names.add(c.common_name)
            lowers = {n.lower() for n in names}
            if cleaned.lower() in lowers:
                return next(iter(names))
            # punctuation-free compare
            if re.sub(r"[^a-z]", "", cleaned.lower()) in {re.sub(r"[^a-z]", "", n) for n in lowers}:
                return next(iter(names))
    except Exception:
        pass

    # Title-case tokens except acronyms
    keep_caps = {"DRC", "ROK", "DPRK"}
    parts = [tok.upper() if tok.upper() in keep_caps else tok.capitalize() for tok in cleaned.split()]
    country = " ".join(parts)

    # final tweaks
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
    # exact
    if country in COUNTRY_TO_REGION:
        return COUNTRY_TO_REGION[country]
    # case-insensitive
    for k, v in COUNTRY_TO_REGION.items():
        if k.lower() == country.lower():
            return v
    # substring fallback (helps with odd spacing/diacritics)
    low = country.lower()
    for k, v in COUNTRY_TO_REGION.items():
        if k.lower() in low or low in k.lower():
            return v
    return "Unknown"

# ===================== OPTIONS =====================
with st.expander("Options", expanded=True):
    lowercase = st.checkbox("lowercase all extracted text", value=False)
    strip_newlines = st.checkbox("remove newlines (join as one paragraph)", value=True)
    include_paths = st.checkbox("include full file path column", value=False)
    colA, colB, colC, colD = st.columns(4)
    max_context = colA.number_input("Max chars: context (0 = no limit)", min_value=0, value=0, step=1000)
    max_contrib = colB.number_input("Max chars: contributions (0 = no limit)", min_value=0, value=0, step=1000)
    max_collab  = colC.number_input("Max chars: collaborations (0 = no limit)", min_value=0, value=0, step=1000)
    max_innov   = colD.number_input("Max chars: innovations (0 = no limit)", min_value=0, value=0, step=1000)

# ===================== SECTION EXTRACTION =====================
def extract_sections_from_txt(file_path: Path) -> dict:
    """
    Read the entire text and carve sections by locating the start of each section
    (using flexible aliases). Content runs from a section start to the next section start.
    """
    result = {k: "" for k in SECTION_ORDER}
    result["filename"] = file_path.name
    if include_paths:
        result["filepath"] = str(file_path)

    # meta from filename
    country, year = parse_country_year_from_filename(file_path.name)
    result["country"] = country
    result["year"] = year
    result["unicef_region"] = country_to_region(country)

    # read whole file
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return result

    # find section starts
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

    # slice content
    for idx, (key, _, content_start) in enumerate(hits):
        content_end = len(text) if idx == len(hits) - 1 else hits[idx + 1][1]
        result[key] = text[content_start:content_end].strip()

    return result

def clean_and_truncate(df: pd.DataFrame) -> pd.DataFrame:
    trunc = {
        "context": max_context,
        "contributions": max_contrib,
        "collaborations": max_collab,
        "innovations": max_innov,
    }
    for col in SECTION_ORDER:
        if lowercase:
            df[col] = df[col].apply(lambda x: x.lower() if isinstance(x, str) else x)
        if strip_newlines:
            df[col] = df[col].str.replace("\n", " ", regex=False)
            df[col] = df[col].str.replace(r"\s{2,}", " ", regex=True)
        if trunc[col] and trunc[col] > 0:
            df[col] = df[col].apply(lambda x: x[:trunc[col]] if isinstance(x, str) else x)
    return df

# ===================== CONSOLIDATE =====================
if consolidate:
    if not txt_root.exists():
        st.warning("Temp folder not found. Convert PDFs to TXT first.")
    else:
        txt_files = sorted([p for p in txt_root.rglob("*.txt")])
        if not txt_files:
            st.info("No .txt files found in the temp folder.")
        else:
            st.write(f"Found **{len(txt_files)}** TXT files.")
            rows = []
            found_debug = []
            prog = st.progress(0)
            for i, p in enumerate(txt_files, 1):
                row = extract_sections_from_txt(p)
                rows.append(row)
                found = {k: (len(row.get(k, "")) > 0) for k in SECTION_ORDER}
                found_debug.append({"filename": row["filename"], **found})
                prog.progress(i / len(txt_files))

            base_cols = ["filename"] + (["filepath"] if include_paths else [])
            meta_cols = ["country", "year", "unicef_region"]
            cols = base_cols + meta_cols + SECTION_ORDER
            df = pd.DataFrame(rows, columns=cols)
            df = clean_and_truncate(df)

            st.caption("Detected headings per file (True = section found)")
            st.dataframe(pd.DataFrame(found_debug), use_container_width=True)

            # ===================== OVERRIDE UI FOR UNKNOWN REGIONS =====================
            # Two kinds of overrides: per-file (strongest), and per-country (bulk).
            if "region_overrides_by_file" not in st.session_state:
                st.session_state["region_overrides_by_file"] = {}   # {filename: region}
            if "region_overrides_by_country" not in st.session_state:
                st.session_state["region_overrides_by_country"] = {}  # {country: region}

            def apply_overrides(df_in: pd.DataFrame) -> pd.DataFrame:
                df2 = df_in.copy()
                df2["unicef_region"] = df2.apply(
                    lambda r: st.session_state["region_overrides_by_file"].get(
                        r.get("filename"),
                        st.session_state["region_overrides_by_country"].get(r.get("country"), r.get("unicef_region"))
                    ),
                    axis=1
                )
                return df2

            # Apply any existing overrides before listing unknowns
            df = apply_overrides(df)

            unknowns = df[df["unicef_region"].eq("Unknown")].copy()
            if not unknowns.empty:
                st.warning(f"{len(unknowns)} file(s) have Unknown UNICEF region. Assign below and click Apply.")
                with st.expander("Assign UNICEF region", expanded=True):

                    # ---- A) Per-file assignments (recommended; always available) ----
                    st.markdown("**Per-file assignment** (takes precedence)")
                    REGIONS = ["", "ROSA", "ECA", "LAC", "MENA", "EAP", "ESA", "WCA"]

                    with st.form("assign_unknowns_form"):
                        for _, row in unknowns.iterrows():
                            fname = row["filename"]
                            country_val = row.get("country") or "(unparsed)"
                            default = st.session_state["region_overrides_by_file"].get(fname, "")
                            col1, col2, col3 = st.columns([4, 3, 2])
                            with col1:
                                st.write(f"**{fname}**")
                                st.caption(f"country: {country_val}")
                            with col2:
                                sel = st.selectbox(
                                    "Region",
                                    REGIONS,
                                    index=(REGIONS.index(default) if default in REGIONS else 0),
                                    key=f"reg_pick_file_{fname}",
                                    label_visibility="collapsed",
                                )
                            with col3:
                                st.write("")  # spacer

                        submitted = st.form_submit_button("Apply assignments ✅")
                        if submitted:
                            # Persist selections to session
                            for _, row in unknowns.iterrows():
                                fname = row["filename"]
                                pick = st.session_state.get(f"reg_pick_file_{fname}", "")
                                if pick:
                                    st.session_state["region_overrides_by_file"][fname] = pick

                            # Re-apply and show remaining unknowns
                            df = apply_overrides(df)
                            still_unknown = df[df["unicef_region"].eq("Unknown")]
                            if still_unknown.empty:
                                st.success("All unknown regions resolved.")
                            else:
                                st.info(f"{len(still_unknown)} file(s) still Unknown (likely missing selections).")

                    # ---- B) Optional: bulk per-country assignment ----
                    st.markdown("---")
                    st.markdown("**Bulk by country (optional)** – applies to all rows of that country (unless a per-file override exists).")
                    unk_countries = sorted(unknowns["country"].fillna("(unparsed)").unique())
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        bulk_country = st.selectbox("Pick a country to assign", [""] + unk_countries, index=0)
                    with c2:
                        bulk_region = st.selectbox("Assign region to that country", REGIONS, index=0)

                    if st.button("Apply bulk country assignment"):
                        if bulk_country and bulk_region:
                            if bulk_country != "(unparsed)":
                                st.session_state["region_overrides_by_country"][bulk_country] = bulk_region
                                df = apply_overrides(df)
                                st.success(f"Assigned {bulk_region} to country '{bulk_country}'.")
                            else:
                                st.info("Cannot bulk-assign '(unparsed)'. Use per-file assignment above.")

                # Show a quick view after overrides
                df = apply_overrides(df)
                st.info("Updated regions (after overrides):")
                st.dataframe(df[["filename", "country", "year", "unicef_region"]], use_container_width=True)
            else:
                st.success("No unknown regions 🎉")

            # ===================== SAVE & EXPORT =====================
            # Save to session for other pages
            st.session_state["consolidated_df"] = df

            # Persist CSV in temp as well
            csv_path = os.path.join(out_dir, "coar_sections.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8")
            st.session_state["consolidated_csv_path"] = csv_path

            st.success(f"Consolidated **{len(df)}** files. Added: country, year, unicef_region.")
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
