# pages/chatroom.py
# 🔒 Chatroom — Private Q&A on your data (session DataFrame OR temp .txt folder)
# Smarter retrieval:
#   - BM25 lexical search (rank_bm25)
#   - Optional local SBERT semantic search (SentenceTransformer) with hybrid reranking
#   - Extractive synthesis with diverse, high-match sentences
# Privacy:
#   - No external network calls; uses only local files and local/cached models
#   - Politely refuses out-of-scope questions and blocks "share/export" requests

import os
import re
import glob
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------- Optional smarter retrievers (pure local if models are cached) -------
BM25_AVAILABLE = True
try:
    from rank_bm25 import BM25Okapi  # pip install rank-bm25
except Exception:
    BM25_AVAILABLE = False

SBERT_AVAILABLE = True
try:
    from sentence_transformers import SentenceTransformer
    import torch
except Exception:
    SBERT_AVAILABLE = False

st.set_page_config(page_title="🔒 Chatroom (Private Q&A)", layout="wide")
st.title("🔒 Chatroom")
st.caption("Answers are strictly based on locally loaded data. No external services are used.")

# ---------------- Configuration ----------------
SESSION_KEY = "consolidated_df"
TEMP_TXT_FOLDER = "/var/folders/8j/lqm_252j4499p5syx4w4n0_80000gn/T/pdf_txt_eqelfgr4"  # your temp folder of .txt
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # local/cached SBERT model name

# ---------------- Source selection (session first, else temp .txt) ----------------
def load_txt_folder(folder_path: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(folder_path, "*.txt"))
    records = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            records.append({"filename": os.path.basename(path), "text": text})
        except Exception as e:
            st.warning(f"Failed to read {path}: {e}")
    return pd.DataFrame(records)

def coalesce_cols(row: pd.Series, cols) -> str:
    parts = []
    for c in cols:
        if c in row and pd.notna(row[c]):
            v = str(row[c]).strip()
            if v:
                parts.append(v)
    return " ".join(parts).strip()

# Try session first
df = st.session_state.get(SESSION_KEY, None)
used_source = None
if isinstance(df, pd.DataFrame) and not df.empty:
    used_source = f"Session DataFrame (‘{SESSION_KEY}’): {df.shape[0]} rows"
else:
    txt_df = load_txt_folder(TEMP_TXT_FOLDER)
    if not txt_df.empty:
        used_source = f"Text files: {len(txt_df)} documents from {TEMP_TXT_FOLDER}"
        df = txt_df
    else:
        st.error(
            "No data found.\n\n"
            f"- Session key '{SESSION_KEY}' not set or empty, and\n"
            f"- No .txt files found in: {TEMP_TXT_FOLDER}\n\n"
            "Populate the session DataFrame in your main page OR ensure .txt files exist in the temp folder."
        )
        st.stop()

st.success(f"Loaded: {used_source}")

# ---------------- Build chat corpus ----------------
st.markdown("#### Build chat corpus (select the columns to include)")
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

if "text" in df.columns and cols_for_chat == ["text"]:
    docs_df = pd.DataFrame({
        "filename": df["filename"] if "filename" in df.columns else np.arange(len(df)).astype(str),
        "text": df["text"].astype(str)
    })
else:
    docs_df = pd.DataFrame({
        "filename": df["filename"] if "filename" in df.columns else np.arange(len(df)).astype(str),
        "text": df.apply(lambda r: coalesce_cols(r, cols_for_chat), axis=1).astype(str)
    })

# ---------------- Chunking with overlap ----------------
def sent_split(x: str):
    x = re.sub(r"\s+", " ", str(x).strip())
    if not x:
        return []
    return re.split(r"(?<=[\.\?\!])\s+", x)

def chunk_text(text: str, max_words: int = 140, overlap: int = 40):
    sents = sent_split(text)
    chunks, cur, count = [], [], 0
    for s in sents:
        w = len(s.split())
        if count + w > max_words and cur:
            chunks.append(" ".join(cur))
            ov = []
            ccount = 0
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

passages = []
for i, row in docs_df.iterrows():
    chs = chunk_text(row["text"], max_words=140, overlap=40)
    for j, ch in enumerate(chs):
        passages.append({
            "pid": f"{i}-{j}",
            "filename": row["filename"],
            "chunk_index": j,
            "text": ch
        })
passages_df = pd.DataFrame(passages)
if passages_df.empty:
    st.error("No text found in the selected columns to build the chat corpus.")
    st.stop()

# ---------------- Build retrievers ----------------
with st.sidebar:
    st.header("Chat settings")
    top_k = st.slider("Top-k passages", 3, 20, 8, 1)
    sim_gate = st.slider("Answer gate (refuse if below)", 0.05, 0.80, 0.20, 0.01)
    use_sbert = st.checkbox("Use SBERT semantic reranking (local)", value=True)
    alpha = st.slider("Hybrid weight (SBERT vs BM25)", 0.0, 1.0, 0.60, 0.05)
    def clear_chat(): st.session_state.chat_history = []
    st.button("🧹 Clear chat", on_click=clear_chat)

def _tokenize(text):
    return re.findall(r"[A-Za-z0-9]+", text.lower())

bm25 = None
if BM25_AVAILABLE:
    corpus_tokens = [_tokenize(t) for t in passages_df["text"].tolist()]
    bm25 = BM25Okapi(corpus_tokens)

@st.cache_resource(show_spinner=False)
def build_tfidf(pass_df: pd.DataFrame):
    stop_words = {
        "the","and","to","of","in","for","on","with","a","an","is","are","was","were","be","by","as","that","this","it",
        "from","or","at","we","our","their","they","these","those","has","have","had","not","but","can","will","may"
    }
    vec = TfidfVectorizer(stop_words=sorted(stop_words), max_df=0.95, ngram_range=(1,2))
    X = vec.fit_transform(pass_df["text"].values)
    return vec, X

tfidf_vec, X_tfidf = build_tfidf(passages_df)

@st.cache_resource(show_spinner=False)
def get_sbert_model():
    return SentenceTransformer(DEFAULT_MODEL)

sbert = None
if use_sbert and SBERT_AVAILABLE:
    try:
        sbert = get_sbert_model()
    except Exception:
        sbert = None

# ---------------- Retrieval ----------------
def retrieve(query: str, k: int = 8):
    cand_idx = []
    if bm25 is not None:
        q_tokens = _tokenize(query)
        bm25_scores = np.array(bm25.get_scores(q_tokens))
        top_lex_idx = np.argsort(-bm25_scores)[: min(5000, len(bm25_scores))]
        cand_idx = top_lex_idx
        lex_scores = bm25_scores[top_lex_idx]
        if np.ptp(lex_scores) > 0:
            lex_norm = (lex_scores - lex_scores.min()) / np.ptp(lex_scores)
        else:
            lex_norm = (lex_scores > 0).astype(float)
    else:
        q_vec = tfidf_vec.transform([query])
        sims = cosine_similarity(q_vec, X_tfidf).ravel()
        top_lex_idx = np.argsort(-sims)[: min(5000, len(sims))]
        cand_idx = top_lex_idx
        lex_norm = sims[top_lex_idx]
        if np.ptp(lex_norm) > 0:
            lex_norm = (lex_norm - lex_norm.min()) / np.ptp(lex_norm)

    if sbert is not None:
        cand_texts = passages_df.iloc[cand_idx]["text"].tolist()
        q_emb = sbert.encode([query], normalize_embeddings=True)
        c_emb = sbert.encode(cand_texts, normalize_embeddings=True)
        sem = (q_emb @ c_emb.T).ravel()
        sem_norm = (sem - sem.min()) / (np.ptp(sem) + 1e-9)
        final = alpha * sem_norm + (1 - alpha) * lex_norm
    else:
        final = lex_norm

    order = np.argsort(-final)[:k]
    chosen_idx = np.array(cand_idx)[order]
    chosen = passages_df.iloc[chosen_idx].copy()
    chosen["score"] = final[order]
    return chosen.reset_index(drop=True)

# ---------------- Answer synthesis ----------------
def synthesize(query: str, results_df: pd.DataFrame, max_snippets: int = 3, max_chars: int = 900):
    out_sents = []
    key_terms = [w for w in re.findall(r"[A-Za-z]{3,}", query.lower())]
    used_spans = set()
    for _, row in results_df.head(max_snippets).iterrows():
        sents = sent_split(row["text"])
        scored = []
        for s in sents:
            low = s.lower()
            hits = sum(1 for t in key_terms if t in low)
            if hits:
                scored.append((hits, len(s), s))
        if not scored and sents:
            scored = [(0, len(sents[0]), sents[0])]
        picked = [x[2] for x in sorted(scored, key=lambda x: (-x[0], x[1]))[:2]]
        for p in picked:
            if p not in used_spans:
                out_sents.append(p)
                used_spans.add(p)

    answer = " ".join(out_sents).strip()
    if len(answer) > max_chars:
        answer = answer[:max_chars].rsplit(" ", 1)[0] + "…"
    sources = list(dict.fromkeys(results_df["filename"].tolist()))
    return answer, sources

def max_confidence(results_df: pd.DataFrame) -> float:
    return float(results_df["score"].max()) if not results_df.empty else 0.0

with st.expander("🔒 Privacy & scope (read-only, local)"):
    st.markdown("""
- Answers are generated only from locally loaded data (session or temp .txt files).
- No external services, models, or APIs are called.
- If a question is unrelated to the loaded content, I will decline.
- I will not export or transmit the data elsewhere.
""")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def add_msg(role, content):
    st.session_state.chat_history.append({"role": role, "content": content})

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_q = st.chat_input("Ask a question about the loaded documents…")
if user_q:
    add_msg("user", user_q)
    with st.chat_message("user"):
        st.write(user_q)

    lower_q = user_q.lower()
    blocked = [
        "send", "email", "upload", "share", "post", "publish", "external", "internet", "api", "slack", "teams",
        "full dataset", "entire dataset", "raw data", "all rows", "download everything"
    ]
    if any(p in lower_q for p in blocked):
        refusal = "I can’t share or export data. I can only answer questions grounded in the locally loaded documents."
        add_msg("assistant", refusal)
        with st.chat_message("assistant"):
            st.write(refusal)
    else:
        hits = retrieve(user_q, k=top_k)
        if max_confidence(hits) < sim_gate:
            out = "That seems outside the scope of the loaded documents. Please ask about information contained in them."
            add_msg("assistant", out)
            with st.chat_message("assistant"):
                st.write(out)
        else:
            answer, sources = synthesize(user_q, hits, max_snippets=min(4, top_k))
            if not answer:
                answer = "I found relevant passages but couldn’t synthesize a concise answer. Try asking more specifically."
            source_str = ", ".join(sources[:5])
            final = f"{answer}\n\n**Sources (filenames):** {source_str}"
            add_msg("assistant", final)
            with st.chat_message("assistant"):
                st.markdown(final)
