# Topic.py
# Multipage-friendly: grabs consolidated DataFrame from st.session_state (Home.py),
# with upload fallback. Includes:
# - Robust VADER sentiment (local NLTK download or vaderSentiment fallback)
# - Corpus builder from chosen columns
# - Keyword windows (±1 sentence)
# - Topic modeling via Sentence-BERT + KMeans (auto/manual k), with top terms per topic
# - Device & batch controls for SBERT (CPU/CUDA, batched encoding with progress)
# - CSV exports
# - Interactive graphs (Plotly): topic sizes, sentiment by topic, t-SNE map, per-topic top-terms
# - PERSISTENCE: saves results to st.session_state so charts stay after reruns

import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st

# --- Optional heavy deps ---
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


# -------------------- Page Config --------------------
st.set_page_config(page_title="🧠 Topics + 😊 Sentiment (per filename)", layout="wide")
st.title("Topic and Sentiment Analysis")
st.subheader("🧠 Topic Modeling (SBERT + KMeans) + 😊 Sentiment (per filename)")


# -------------------- Robust Sentiment Init --------------------
@st.cache_resource(show_spinner=False)
def get_sia():
    """
    Robust VADER init:
    1) Try NLTK VADER; if the lexicon is missing, download it locally to ./nltk_data near this file.
    2) If NLTK route fails, fall back to vaderSentiment package (bundled lexicon).
    """
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer as NLTK_SIA

        base_dir = os.path.dirname(os.path.abspath(__file__))
        nltk_data_dir = os.path.join(base_dir, "nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)

        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.append(nltk_data_dir)

        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", download_dir=nltk_data_dir, quiet=True)

        return NLTK_SIA()

    except Exception:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS_SIA
        return VS_SIA()


# -------------------- Helpers --------------------
def simple_sentence_split(text: str):
    """Lightweight sentence splitter (no NLTK punkt)."""
    text = re.sub(r"\s+", " ", str(text).strip())
    if not text:
        return []
    return re.split(r"(?<=[\.\?\!])\s+", text)

def extract_keyword_windows(text: str, keywords, window: int = 1):
    """Return rows with: keyword, sentence_before, sentence_hit, sentence_after."""
    sents = simple_sentence_split(text)
    rows = []
    if not sents:
        return rows
    for i, s in enumerate(sents):
        s_low = s.lower()
        for kw in keywords:
            k = kw.strip().lower()
            if k and k in s_low:
                before = sents[i - 1] if i - 1 >= 0 else ""
                after = sents[i + 1] if i + 1 < len(sents) else ""
                rows.append(
                    {
                        "keyword": kw.strip(),
                        "sentence_before": before,
                        "sentence_hit": s,
                        "sentence_after": after,
                    }
                )
    return rows

def coalesce_cols(row, cols):
    parts = []
    for c in cols:
        if c in row and pd.notna(row[c]):
            v = str(row[c]).strip()
            if v:
                parts.append(v)
    return " ".join(parts).strip()

def to_download_button(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

def _find_df_in_session():
    """
    Try common keys first; else scan session_state for any DataFrame
    that looks like the consolidated table.
    """
    preferred_keys = [
        "consolidated_df", "corpus_df", "final_consolidated",
        "consolidated", "df_consolidated", "df"
    ]
    expected_any = {"filename", "context", "contributions", "collaborations", "innovations"}

    # 1) Preferred keys
    for k in preferred_keys:
        if k in st.session_state and isinstance(st.session_state[k], pd.DataFrame):
            return st.session_state[k], k

    # 2) Scan any DataFrame in session for expected columns overlap
    best_key = None
    best_overlap = -1
    best_df = None
    for k, v in st.session_state.items():
        if isinstance(v, pd.DataFrame):
            overlap = len(set(v.columns) & expected_any)
            if overlap > best_overlap:
                best_overlap = overlap
                best_df = v
                best_key = k
    if best_df is not None:
        return best_df, best_key

    return None, None


# -------------------- Data In --------------------
st.markdown("#### Data source")
df, found_key = _find_df_in_session()
if df is not None:
    st.success(f"Using DataFrame from session: `{found_key}` ({len(df)} rows).")
else:
    st.info("No consolidated DataFrame found in session. You can upload a CSV as a fallback.")
    uploaded = st.file_uploader(
        "Upload consolidated CSV (columns like: filename, context, contributions, collaborations, innovations, ...)",
        type=["csv"]
    )
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding_errors="ignore")
        st.session_state["consolidated_df"] = df
        st.success(f"Loaded {len(df)} rows into session key 'consolidated_df'.")
    else:
        st.stop()

df = df.copy()


# -------------------- Corpus Builder Controls --------------------
st.markdown("### Filters (optional)")
st.markdown("#### 📚 Build a document corpus (one row per filename)")

available_cols = list(df.columns)
default_cols = [c for c in ["context", "contributions", "collaborations", "innovations"] if c in available_cols]
heading_cols = st.multiselect("Heading columns to concatenate for modeling", options=available_cols, default=default_cols)

colA, colB, colC = st.columns([1,1,1])
with colA:
    min_words = st.number_input("Min words per doc (filter)", min_value=0, value=10, step=1)
with colB:
    ngram_choice = st.selectbox("N-grams (for keyword top-terms only)", options=["1–1", "1–2", "1–3"], index=1)
with colC:
    max_vocab = st.number_input("Max TF-IDF vocab (topic terms)", min_value=0, value=10000, step=500)

extra_stops = st.text_area("Extra stopwords (comma or newline separated, optional)", height=80)

if not heading_cols:
    st.warning("Select at least one heading column to build the corpus.")
    st.stop()

docs_df = pd.DataFrame({
    "filename": df["filename"] if "filename" in df.columns else np.arange(len(df)).astype(str),
    "text": df.apply(lambda r: coalesce_cols(r, heading_cols), axis=1)
})
docs_df["word_count"] = docs_df["text"].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
before = len(docs_df)
docs_df = docs_df[docs_df["word_count"] >= min_words].reset_index(drop=True)
removed = before - len(docs_df)
st.caption(f"Removed {removed} doc(s) under {min_words} words (clean).")


# -------------------- Sentiment --------------------
with st.spinner("Initializing sentiment analyzer..."):
    try:
        sia = get_sia()
    except Exception as e:
        st.error(f"Failed to initialize sentiment analyzer: {e}")
        st.stop()

def compound_score(txt):
    try:
        return float(sia.polarity_scores(str(txt))["compound"])
    except Exception:
        return np.nan

docs_df["sentiment_compound"] = docs_df["text"].apply(compound_score)


# -------------------- Keyword Windows --------------------
st.markdown("### 🔍 Keyword windows (±1 sentence around each keyword, per document)")
kw_input = st.text_input("Keywords (comma-separated; e.g., out of school, learning passport, inclusive education)", "")
collect_rows = []
if kw_input.strip():
    keywords = [k.strip() for k in re.split(r"[,|\n]", kw_input) if k.strip()]
    for _, r in docs_df.iterrows():
        rows = extract_keyword_windows(r["text"], keywords, window=1)
        for item in rows:
            item["filename"] = r["filename"]
            collect_rows.append(item)

    kw_df = pd.DataFrame(collect_rows)
    if not kw_df.empty:
        st.dataframe(kw_df, use_container_width=True)
        to_download_button(kw_df, "⬇️ Download keyword windows (CSV)", "keyword_windows.csv")
    else:
        st.info("No keyword hits found.")


# -------------------- Topic Modeling: SBERT + KMeans --------------------
st.markdown("### 🔁 Topic Modeling (Sentence-BERT + KMeans)")

# Controls for topic modeling
col1, col2, col3 = st.columns([1,1,1])
with col1:
    k_mode = st.selectbox("Number of topics (k)", ["Auto (silhouette)", "Manual"], index=0)
with col2:
    k_manual = st.number_input("Manual k (if selected)", min_value=2, value=10, step=1)
with col3:
    topn = st.number_input("Top words per topic", min_value=5, value=10, step=1)

# Encoder/runtime knobs
col4, col5 = st.columns([1,1])
with col4:
    device_choice = st.selectbox(
        "Encoder device", ["Auto", "CPU", "CUDA"], index=0,
        help="Pick CUDA if you have a GPU; Auto tries CUDA then falls back to CPU."
    )
with col5:
    batch_size = st.number_input("SBERT batch size", min_value=8, max_value=512, value=64, step=8)

# Remember user's n-gram choice for topic terms
st.session_state["_ngram_choice"] = ngram_choice
st.session_state["_extra_stops_text"] = extra_stops

deps_ok = SBERT_AVAILABLE and SKLEARN_AVAILABLE
if not deps_ok:
    st.error("Missing dependencies. Install with: `pip install sentence-transformers scikit-learn`")
if deps_ok and not PLOTLY_AVAILABLE:
    st.error("Plotly not installed. Install with: `pip install plotly`")

def topic_top_terms_with_weights(texts, labels, top_n=10, max_vocab=10000):
    """
    Class-based TF-IDF descriptors.
    Returns:
      - wide: topic -> 'top_terms'
      - long: rows (topic, term, weight)
    """
    df_tmp = pd.DataFrame({"text": texts, "label": labels})
    topics = sorted(df_tmp["label"].unique())
    docs_per_topic = df_tmp.groupby("label")["text"].apply(lambda x: " ".join(map(str, x))).tolist()

    # Stopwords
    stop_extra = set()
    if (extra_text := st.session_state.get("_extra_stops_text")):
        for token in re.split(r"[,|\n]", extra_text):
            tok = token.strip().lower()
            if tok:
                stop_extra.add(tok)

    base_stop = {
        "the","and","to","of","in","for","on","with","a","an","is","are","was","were","be","by","as","that","this","it",
        "from","or","at","we","our","their","they","these","those","has","have","had","not","but","can","will","may",
    }
    stop_words = sorted(base_stop.union(stop_extra))

    ngram_map = {"1–1": (1,1), "1–2": (1,2), "1–3": (1,3)}
    ngram_range = ngram_map.get(st.session_state.get("_ngram_choice", "1–2"), (1,2))

    max_features = None if max_vocab <= 0 else int(max_vocab)
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=0.95, ngram_range=ngram_range, max_features=max_features)
    X = vectorizer.fit_transform(docs_per_topic)
    vocab = np.array(vectorizer.get_feature_names_out())

    rows_wide, rows_long = [], []
    for i, t in enumerate(topics):
        weights = X[i].toarray().ravel()
        top_idx = weights.argsort()[-top_n:][::-1]
        top_terms = vocab[top_idx]
        top_weights = weights[top_idx]
        rows_wide.append({"topic": int(t), "top_terms": ", ".join(top_terms)})
        for term, w in zip(top_terms, top_weights):
            rows_long.append({"topic": int(t), "term": term, "weight": float(w)})

    df_wide = pd.DataFrame(rows_wide).sort_values("topic")
    df_long = pd.DataFrame(rows_long).sort_values(["topic", "weight"], ascending=[True, False])
    return df_wide, df_long

@st.cache_resource(show_spinner=False)
def get_sbert_model(device: str):
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)

def auto_choose_k(embeddings, k_min=5, k_max=20):
    """Pick k using silhouette score in [k_min, k_max]."""
    best_k, best_score = None, -1.0
    n = len(embeddings)
    if n < 3:
        return 2
    k_max_eff = min(k_max, n - 1)
    for k in range(k_min, max(k_min + 1, k_max_eff + 1)):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(embeddings)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_k, best_score = k, score
        except Exception:
            continue
    return best_k or max(2, k_min)

# ---- RUN MODEL BUTTON ----
if deps_ok and PLOTLY_AVAILABLE and st.button("🚀 Run SBERT + KMeans"):
    # Resolve device selection
    dev = "cpu"
    if device_choice == "CUDA":
        dev = "cuda"
    elif device_choice == "Auto":
        try:
            import torch  # noqa
            import torch.cuda  # noqa
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            dev = "cpu"

    with st.spinner(f"Encoding documents on {dev}..."):
        try:
            # 1) SBERT embeddings (batched)
            sbert = get_sbert_model(dev)
            texts = docs_df["text"].tolist()
            total = len(texts)
            if total == 0:
                st.warning("No documents to encode.")
                st.stop()

            bs = int(batch_size)
            first_end = min(bs, total)
            first_emb = sbert.encode(
                texts[:first_end],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=bs
            ).astype("float32")
            emb_dim = first_emb.shape[1]
            embeddings = np.empty((total, emb_dim), dtype="float32")
            embeddings[:first_end] = first_emb
            out_idx = first_end

            prog = st.progress(first_end / total if total else 1.0)
            for start in range(first_end, total, bs):
                end = min(start + bs, total)
                emb = sbert.encode(
                    texts[start:end],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=bs
                ).astype("float32")
                embeddings[out_idx:out_idx + len(emb)] = emb
                out_idx += len(emb)
                prog.progress(end / total if total else 1.0)
            prog.empty()

            # 2) Choose k
            if k_mode.startswith("Auto"):
                k = auto_choose_k(embeddings, k_min=5, k_max=20)
            else:
                k = int(k_manual)

            # 3) Cluster
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(embeddings)
            docs_df["topic"] = labels

            # 4) Topic descriptors
            topic_terms_wide, topic_terms_long = topic_top_terms_with_weights(
                docs_df["text"].tolist(), labels, top_n=int(topn), max_vocab=int(max_vocab)
            )

            # 5) Tables
            topic_sizes = (
                docs_df.groupby("topic")
                       .size()
                       .reset_index(name="doc_count")
                       .sort_values("doc_count", ascending=False)
            )

            # ---- PERSIST RESULTS FOR UI RERUNS ----
            st.session_state["topic_docs_df"] = docs_df.copy()
            st.session_state["topic_terms_long"] = topic_terms_long.copy()
            st.session_state["topic_terms_wide"] = topic_terms_wide.copy()
            st.session_state["topic_sizes"] = topic_sizes.copy()
            st.session_state["topic_embeddings"] = embeddings

            st.success(f"Model run complete. Found k={k} topics.")
        except Exception as e:
            st.error(f"SBERT+KMeans failed: {e}")


# -------------------- RESULTS & INTERACTIVE VISUALS (persisted) --------------------
st.markdown("### 📈 Results & Interactive Visuals")
if not PLOTLY_AVAILABLE:
    st.stop()

if all(k in st.session_state for k in ["topic_docs_df", "topic_terms_wide", "topic_terms_long", "topic_sizes"]):
    docs_df_vis = st.session_state["topic_docs_df"]
    topic_terms_wide = st.session_state["topic_terms_wide"]
    topic_terms_long = st.session_state["topic_terms_long"]
    topic_sizes = st.session_state["topic_sizes"]
    embeddings = st.session_state.get("topic_embeddings", None)

    # Tables
    st.markdown("#### Topic overview (sizes)")
    st.dataframe(topic_sizes, use_container_width=True)

    st.markdown("#### Top terms per topic")
    st.dataframe(topic_terms_wide, use_container_width=True)

    st.markdown("#### Document → Topic assignments (with sentiment)")
    out_cols = ["filename", "topic", "sentiment_compound", "text"]
    st.dataframe(docs_df_vis[out_cols], use_container_width=True, height=500)

    # Downloads
    to_download_button(topic_sizes, "⬇️ Download topic sizes (CSV)", "topic_sizes.csv")
    to_download_button(topic_terms_wide, "⬇️ Download topic terms (CSV)", "topic_terms.csv")
    to_download_button(docs_df_vis[out_cols], "⬇️ Download doc-topic-sentiment (CSV)", "doc_topic_sentiment.csv")

    # Charts
    st.markdown("### 📊 Interactive Charts")
    g1, g2 = st.columns([1,1])

    with g1:
        st.markdown("**Topic sizes**")
        fig_sizes = px.bar(
            topic_sizes, x="topic", y="doc_count", text="doc_count",
            labels={"topic": "Topic", "doc_count": "Documents"},
        )
        fig_sizes.update_traces(textposition="outside")
        fig_sizes.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_sizes, use_container_width=True)

    with g2:
        st.markdown("**Sentiment by topic** (violin + points)")
        fig_sent = px.violin(
            docs_df_vis, x="topic", y="sentiment_compound",
            points="all", box=True, hover_data=["filename"]
        )
        fig_sent.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_sent, use_container_width=True)

    st.markdown("**Document map (t‑SNE on SBERT embeddings)**")
    if embeddings is not None:
        try:
            n_samples = embeddings.shape[0]
            perplexity = max(5, min(30, (n_samples // 3) - 1)) if n_samples > 10 else max(2, n_samples - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
            emb2 = tsne.fit_transform(embeddings)
            plot_df = pd.DataFrame(emb2, columns=["x", "y"])
            plot_df["topic"] = docs_df_vis["topic"].values
            plot_df["filename"] = docs_df_vis["filename"].values
            plot_df["sentiment"] = docs_df_vis["sentiment_compound"].round(3)

            fig_tsne = px.scatter(
                plot_df, x="x", y="y", color="topic",
                hover_data={"filename": True, "sentiment": True, "x": False, "y": False},
                labels={"x": "t-SNE X", "y": "t-SNE Y"},
            )
            fig_tsne.update_layout(margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_tsne, use_container_width=True)
        except Exception as e:
            st.info(f"t‑SNE could not run (samples={n_samples}, perplexity={perplexity}): {e}")
    else:
        st.info("Embeddings not found in session; rerun the model to enable the t‑SNE map.")

    # ---- Top terms by topic (PERSISTENT) ----
    st.markdown("**Top terms by topic**")
    topic_choices = sorted(map(int, docs_df_vis["topic"].unique().tolist()))
    sel_topic = st.selectbox(
        "Select topic to view its top terms",
        options=topic_choices,
        index=0,
        key="sel_topic_top_terms"  # stable key prevents widget clashes
    )
    subset = topic_terms_long[topic_terms_long["topic"] == sel_topic].copy()
    if subset.empty:
        st.info("No terms for the selected topic.")
    else:
        fig_terms = px.bar(
            subset.sort_values("weight", ascending=True),
            x="weight", y="term", orientation="h",
            labels={"weight": "TF‑IDF weight", "term": "Term"},
        )
        fig_terms.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=400)
        st.plotly_chart(fig_terms, use_container_width=True)

    # Quick filter of documents
    st.markdown("**Filter documents by topic**")
    multi_sel = st.multiselect(
        "Topics to show",
        options=topic_choices,
        default=topic_choices[:min(5, len(topic_choices))],
        key="filter_topics_multisel"
    )
    filtered_docs = docs_df_vis[docs_df_vis["topic"].isin(multi_sel)][out_cols]
    st.dataframe(filtered_docs, use_container_width=True, height=350)

else:
    st.info("Run topic modeling (above) to view results and interactive charts.")


# -------------------- Exports: Sentiment-only --------------------
st.markdown("### 📤 Exports")
sent_cols = ["filename", "sentiment_compound", "word_count"]
to_download_button(docs_df[sent_cols], "⬇️ Download sentiment summary (CSV)", "sentiment_summary.csv")

st.success("Done.")
