
# coar_simple_extractor.py
# Minimal CLI: extract "Part 1/2/3" sections from PDFs (and DOCX) to CSV.
# Usage:
#   pip install pdfminer.six python-docx
#   python coar_simple_extractor.py /path/to/folder -o sections.csv

import argparse, io, re, unicodedata, sys
from io import StringIO
from pathlib import Path
import csv

# ---- PDFMiner (only) ----
try:
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from pdfminer.pdfpage import PDFPage
    PDFMINER_OK = True
except Exception:
    PDFMINER_OK = False

# ---- DOCX (optional) ----
try:
    import docx
except Exception:
    docx = None

PART1 = re.compile(r"part\s*1\s*[:\-]", re.IGNORECASE)
PART2 = re.compile(r"part\s*2\s*[:\-]", re.IGNORECASE)
PART3 = re.compile(r"part\s*3\s*[:\-]", re.IGNORECASE)

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00a0", " ").replace("\u200b", "")
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)  # join hyphenated line breaks
    s = re.sub(r"[ \t]+", " ", s)
    return s

def pdf_to_text_pdfminer(file_bytes: bytes) -> str:
    if not PDFMINER_OK:
        return ""
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
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
    suf = path.suffix.lower()
    data = path.read_bytes()
    if suf == ".pdf":
        return pdf_to_text_pdfminer(data)
    elif suf == ".docx":
        return docx_to_text(data)
    else:
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""

def split_three_parts(text: str):
    clean = normalize_text(text)
    # find headings anywhere
    hits = []
    for key, pat in (("h_update_context", PART1), ("h_major_results", PART2), ("h_lessons_constraints", PART3)):
        m = pat.search(clean)
        if m:
            hits.append((key, m.start(), m.end()))
    if not hits:
        return {"h_update_context": "", "h_major_results": "", "h_lessons_constraints": ""}
    # ensure sorted by position
    hits.sort(key=lambda x: x[1])
    out = {"h_update_context": "", "h_major_results": "", "h_lessons_constraints": ""}
    for i, (key, s, e) in enumerate(hits):
        next_s = hits[i+1][1] if i+1 < len(hits) else len(clean)
        out[key] = clean[e:next_s].strip()
    return out

def collect_files(inputs):
    files = []
    for p in inputs:
        path = Path(p)
        if path.is_file():
            if path.suffix.lower() in (".pdf", ".docx", ".txt"):
                files.append(path)
        elif path.is_dir():
            for f in path.rglob("*"):
                if f.suffix.lower() in (".pdf", ".docx", ".txt"):
                    files.append(f)
    return files

def main():
    ap = argparse.ArgumentParser(description="Extract COAR parts (Part 1/2/3) to CSV.")
    ap.add_argument("inputs", nargs="+", help="PDF/DOCX files or folder(s)")
    ap.add_argument("-o", "--output", default="sections.csv", help="Output CSV (default: sections.csv)")
    args = ap.parse_args()

    if not PDFMINER_OK:
        print("Error: pdfminer.six is not installed. Run: pip install pdfminer.six", file=sys.stderr)
        sys.exit(1)

    files = collect_files(args.inputs)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for f in sorted(files):
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
                "filename": f.name,
                "path": str(f),
                "h_update_context": "",
                "h_major_results": "",
                "h_lessons_constraints": f"ERROR: {e}",
            })

    with open(args.output, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["filename", "path", "h_update_context", "h_major_results", "h_lessons_constraints"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")

if __name__ == "__main__":
    main()
