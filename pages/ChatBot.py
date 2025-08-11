# pages/ChatBot.py
# 🔒 Private Q&A on your local data — ChatGPT-style layout
# - Composer mode: Docked (fixed bottom) OR Inline (collapsible) with dynamic spacing
# - Sources: session DataFrame ('consolidated_df') OR temp .txt folder (prefers session)
# - Retrieval: BM25 (rank_bm25) + TF-IDF; optional local SBERT rerank (SentenceTransformer)
# - User-friendly error handling; no stack traces shown
# - Privacy: blocks export/share requests, no external network calls

import os
import re
import glob
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Optional local retrievers (robust import) ----------
BM25_AVAILABLE = True
try:
    from rank_bm25 import BM25Okapi  # pip install rank-bm25
except Exception:
    BM25_AVAILABLE = False

SBERT_AVAILABLE = True
try:
    from sentence_transformers import SentenceTransformer  # may need wheels/tooling in some envs
except Exception:
    SBERT_AVAILABLE = False

# ---------- Page ----------
st.set_page_config(page_title="🔒 Chatroom (Private Q&A)", layout="wide")

# ---------- UI constants ----------
INPUT_BAR_H = 116   # fallback height if JS can't measure
MAX_COL_W   = 900   # center column width
EXTRA_GAP   = 28    # extra breathing room above the docked input

# ---------- Header ----------
st.title("🔒 Chatroom")
st.caption("Answers are strictly based on locally loaded data. No external services are used.")

# ---------- Config ----------
SESSION_KEY = "consolidated_df"
DEFAULT_TEMP_DIR = st.session_state.get("txt_out_dir", os.path.join(tempfile.gettempdir(), "pdf_txt_fallback"))
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# ---------- Helpers ----------
def sent_split(x: str):
    x = re.sub(r"\s+", " ", str(x).strip())
    if not x: return []
    return re.split(r"(?<=[\.\?\!])\s+", x)

def chunk_text(text: str, max_words: int = 140, overlap: int = 40):
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
                ov.insert(0, rs); ccount += cw
                if ccount >= overlap: break
            cur = ov + [s]
            count = len(" ".join(cur).split())
        else:
            cur.append(s); count += w
    if cur: chunks.append(" ".join(cur))
    return [c for c in chunks if c.strip()]

def _tokenize(text):
    return re.findall(r"[A-Za-z0-9]+", text.lower())

def coalesce_cols(row: pd.Series, cols) -> str:
    parts = []
    for c in cols:
        if c in row and pd.notna(row[c]):
            v = str(row[c]).strip()
            if v: parts.append(v)
    return " ".join(parts).strip()

def load_txt_folder(folder_path: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(folder_path, "*.txt"))
    records = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            records.append({"filename": os.path.basename(path), "text": text})
        except Exception:
            pass
    return pd.DataFrame(records)

# ---------- Data source (session first, else temp .txt) ----------
df = st.session_state.get(SESSION_KEY, None)
used_source = None
if isinstance(df, pd.DataFrame) and not df.empty:
    used_source = f"Session DataFrame (‘{SESSION_KEY}’): {df.shape[0]} rows"
else:
    txt_df = load_txt_folder(DEFAULT_TEMP_DIR)
    if not txt_df.empty:
        df = txt_df
        used_source = f"Text files: {len(txt_df)} documents from {DEFAULT_TEMP_DIR}"
    else:
        st.error("No data to chat with.")
        st.info("Tip: On **Home** → convert PDFs to TXT → **Run Consolidation**. Then return here.")
        st.stop()
st.success(f"Loaded: {used_source}")

# ---------- Build chat corpus (gentle UX) ----------
with st.expander("Corpus & indexing", expanded=False):
    st.write("Pick which columns the chat can use as context.")
    if "text" in df.columns and "filename" in df.columns:
        available_cols = list(df.columns); default_cols = ["text"]
    else:
        available_cols = list(df.columns)
        default_cols = [c for c in ["context","contributions","collaborations","innovations"] if c in available_cols]
        if not default_cols:
            default_cols = [c for c in available_cols if c.lower() not in ("filename","id")][:4]

    cols_for_chat = st.multiselect(
        "Columns to include",
        options=available_cols,
        default=default_cols
    )

if not cols_for_chat:
    st.warning("Select at least one column for the chat corpus (e.g., ‘text’ or the section columns).")
    st.stop()

# Coalesce text
if "text" in df.columns and cols_for_chat == ["text"]:
    docs_df = pd.DataFrame({
        "filename": df["filename"] if "filename" in df.columns else np.arange(len(df)).astype(str),
        "text": df["text"].astype(str),
    })
else:
    docs_df = pd.DataFrame({
        "filename": df["filename"] if "filename" in df.columns else np.arange(len(df)).astype(str),
        "text": df.apply(lambda r: coalesce_cols(r, cols_for_chat), axis=1).astype(str),
    })

docs_df["text"] = docs_df["text"].fillna("").astype(str)
docs_df = docs_df[docs_df["text"].str.strip().ne("")]
if docs_df.empty:
    st.error("All selected columns are empty.")
    st.info("Tip: Re-run Consolidation on **Home** or pick different columns.")
    st.stop()

# Chunk into passages
passages = []
for i, row in docs_df.iterrows():
    for j, ch in enumerate(chunk_text(row["text"], max_words=140, overlap=40)):
        passages.append({"pid": f"{i}-{j}", "filename": row["filename"], "chunk_index": j, "text": ch})
passages_df = pd.DataFrame(passages)
if passages_df.empty:
    st.error("No text found after chunking.")
    st.info("Tip: Try different columns or ensure your TXT files contain extractable text.")
    st.stop()

# ---------- Sidebar: retrieval settings & status (+ Composer mode) ----------
with st.sidebar:
    st.header("Chat settings")
    top_k = st.slider("Top-k passages", 3, 20, 8, 1)
    sim_gate = st.slider("Answer gate (min score)", 0.02, 0.60, 0.12, 0.01)
    use_sbert = st.checkbox("Use SBERT semantic rerank (local)", value=True)
    alpha = st.slider("Hybrid weight (SBERT vs BM25/TF-IDF)", 0.0, 1.0, 0.60, 0.05)

    st.markdown("---")
    composer_mode = st.radio(
        "Composer mode",
        ["Docked (fixed bottom)", "Inline (collapsible)"],
        index=0,
        help="Use Inline to avoid collisions when the sidebar/navigation is expanded.",
    )
    inline_expanded_default = st.checkbox("Expand inline composer by default", value=False)

    st.markdown("—")
    if BM25_AVAILABLE:
        st.success("BM25: available")
    else:
        st.info("BM25: not installed (using TF-IDF only)")
    if use_sbert and not SBERT_AVAILABLE:
        st.info("SBERT: package not available — falling back gracefully")

    st.markdown("—")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()
    with col_b:
        if st.button("🔁 Rebuild index", use_container_width=True):
            st.session_state["chat_rebuild_nonce"] = st.session_state.get("chat_rebuild_nonce", 0) + 1
            st.success("Index will rebuild now.")
            st.rerun()

# ---------- Build indices (cacheable & robust) ----------
@st.cache_resource(show_spinner=False)
def build_indices(pass_texts: list[str], nonce: int):
    bm25 = None
    if BM25_AVAILABLE:
        tokens = [_tokenize(t) for t in pass_texts]
        bm25 = BM25Okapi(tokens)
    stop_words = {
        "the","and","to","of","in","for","on","with","a","an","is","are","was","were","be","by","as","that","this","it",
        "from","or","at","we","our","their","they","these","those","has","have","had","not","but","can","will","may"
    }
    vec = TfidfVectorizer(stop_words=sorted(stop_words), max_df=0.95, ngram_range=(1,2))
    X = vec.fit_transform(pass_texts)
    return bm25, vec, X

bm25, tfidf_vec, X_tfidf = build_indices(passages_df["text"].tolist(), st.session_state.get("chat_rebuild_nonce", 0))

@st.cache_resource(show_spinner=False)
def get_sbert(model_name: str):
    return SentenceTransformer(model_name)

sbert = None
if use_sbert and SBERT_AVAILABLE:
    try:
        sbert = get_sbert(DEFAULT_MODEL)
    except Exception:
        sbert = None
        st.info("SBERT model not available locally — using BM25/TF-IDF only.")

# ---------- Retrieval ----------
def _safe_minmax_norm(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0: return arr
    rng = np.ptp(arr)
    if rng > 0:
        return (arr - arr.min()) / rng
    return (arr > 0).astype(float)

def retrieve(query: str, k: int = 8):
    try:
        if bm25 is not None:
            q_tokens = _tokenize(query)
            bm25_scores = np.array(bm25.get_scores(q_tokens))
            top_lex_idx = np.argsort(-bm25_scores)[: min(5000, len(bm25_scores))]
            cand_idx = top_lex_idx
            lex_norm = _safe_minmax_norm(bm25_scores[top_lex_idx])
        else:
            q_vec = tfidf_vec.transform([query])
            sims = cosine_similarity(q_vec, X_tfidf).ravel()
            top_lex_idx = np.argsort(-sims)[: min(5000, len(sims))]
            cand_idx = top_lex_idx
            lex_norm = _safe_minmax_norm(sims[top_lex_idx])

        if sbert is not None and len(cand_idx) > 0:
            cand_texts = passages_df.iloc[cand_idx]["text"].tolist()
            q_emb = sbert.encode([query], normalize_embeddings=True)
            c_emb = sbert.encode(cand_texts, normalize_embeddings=True)
            sem = (q_emb @ c_emb.T).ravel()
            sem_norm = _safe_minmax_norm(sem)
            final = alpha * sem_norm + (1 - alpha) * lex_norm
        else:
            final = lex_norm

        if final.size == 0:
            return passages_df.head(0).copy()

        order = np.argsort(-final)[:k]
        chosen_idx = np.array(cand_idx)[order]
        chosen = passages_df.iloc[chosen_idx].copy()
        chosen["score"] = final[order]
        return chosen.reset_index(drop=True)
    except Exception:
        return passages_df.head(0).copy()

# ---------- Answer synthesis ----------
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

# ---------- Privacy note ----------
with st.expander("🔒 Privacy & scope", expanded=False):
    st.markdown("""
- Answers are generated **only** from locally loaded data (session or temp `.txt` files).
- **No** external APIs or services are called.
- If a question is unrelated to the loaded content, I will decline.
- I will not export or transmit the data elsewhere.
""")

# ---------- Composer mode CSS (dynamic) ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Apply mode-specific CSS
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
    # Dynamically measure chat input height and set --chat-input-h
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
    # Inline: hide built-in chat_input; no bottom padding
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{ padding-bottom: 0 !important; }}
    div[data-testid="stChatInput"] {{ display: none !important; }}
    .chat-container {{ max-width: {MAX_COL_W}px; width: calc(100% - 2rem); margin: 0 auto; }}
    </style>
    """, unsafe_allow_html=True)

# ---------- Chat UI (centered column) ----------
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

# Close centered container BEFORE composer
st.markdown("</div>", unsafe_allow_html=True)

# ---------- Composer handling ----------
user_q = None
if composer_mode.startswith("Docked"):
    # Use Streamlit's fixed chat input
    user_q = st.chat_input("Ask a question about the loaded documents…")
else:
    # Inline collapsible composer
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

# ---------- Handle query ----------
if user_q:
    st.session_state.chat_history.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    lower_q = user_q.lower()
    blocked = [
        "send","email","upload","share","post","publish","external","internet","api","slack","teams",
        "full dataset","entire dataset","raw data","all rows","download everything"
    ]
    if any(p in lower_q for p in blocked):
        reply = "I can’t share or export data. I can only answer questions grounded in the locally loaded documents."
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
    else:
        with st.spinner("Thinking…"):
            hits = retrieve(user_q, k=top_k)
        if hits.empty or max_confidence(hits) < sim_gate:
            reply = "That seems outside the scope of the loaded documents. Try adding specific keywords from the files."
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
        else:
            answer, sources = synthesize(user_q, hits, max_snippets=min(4, top_k))
            if not answer:
                answer = "I found relevant passages but couldn’t synthesize a concise answer. Try asking more specifically."
            final = f"{answer}\n\n**Sources (filenames):** {', '.join(sources[:5])}"
            st.session_state.chat_history.append({"role": "assistant", "content": final})
            with st.chat_message("assistant"):
                st.markdown(final)
