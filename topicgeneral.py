# pages/Topics.py
import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text as sk_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

nltk.download('punkt', quiet=True)

st.set_page_config(page_title="COAR Modelling & Semantic Analysis", layout="wide")
st.title("🧠 COAR Semantic Analysis & Topic Modelling")

# ---------------- Load consolidated data ----------------
if "consolidated_df" not in st.session_state:
    st.warning("No consolidated data found. Please run extraction first from Home page.")
    st.stop()

df = st.session_state["consolidated_df"].copy()
st.success(f"Loaded {len(df)} rows of consolidated data.")
st.dataframe(df.head(), use_container_width=True)

# ---------------- Column selection ----------------
protected_cols = {"filename", "filepath", "country", "year", "unicef_region"}
text_cols = [c for c in df.columns if c not in protected_cols and df[c].dtype == object]
selected_col = st.selectbox("Select column for analysis", text_cols)

# ---------------- Stopwords ----------------
extra_stops = st.text_area(
    "Extra stopwords (comma or newline separated, optional)",
    value="coar, unicef",
    height=80
)
extra_sw = {w.strip().lower() for chunk in extra_stops.split("\n") for w in chunk.split(",") if w.strip()}
stop_words = list(sk_text.ENGLISH_STOP_WORDS.union(extra_sw))  # ✅ FIX: convert to list

# ---------------- Preprocess text ----------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', str(text))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

df["text_clean"] = df[selected_col].apply(clean_text)
corpus_df = df[df["text_clean"].str.strip() != ""].copy()

# ---------------- TF-IDF ----------------
max_features = st.slider("Max TF-IDF features", 100, 5000, 1000, step=100)
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=max_features)
X = vectorizer.fit_transform(corpus_df["text_clean"])
feature_names = vectorizer.get_feature_names_out()

# ---------------- Topic Modelling ----------------
num_topics = st.slider("Number of topics", 2, 15, 5)
nmf = NMF(n_components=num_topics, random_state=42)
W = nmf.fit_transform(X)
H = nmf.components_

topics = []
for topic_idx, topic in enumerate(H):
    top_terms = [feature_names[i] for i in topic.argsort()[:-11:-1]]
    topics.append({"Topic": topic_idx + 1, "Top Terms": ", ".join(top_terms)})
topics_df = pd.DataFrame(topics)

st.subheader("📌 Top Terms per Topic")
st.dataframe(topics_df, use_container_width=True)
st.download_button(
    "💾 Download Topics CSV",
    topics_df.to_csv(index=False),
    file_name="topics.csv",
    mime="text/csv"
)

# ---------------- Document-topic assignment ----------------
corpus_df["Dominant Topic"] = W.argmax(axis=1) + 1
corpus_df["Topic Score"] = W.max(axis=1)
doc_topics_df = corpus_df[["filename", "country", "unicef_region", "year", selected_col, "Dominant Topic", "Topic Score"]]

st.subheader("📄 Document Dominant Topics")
st.dataframe(doc_topics_df.head(50), use_container_width=True)

# Download with all previous columns
st.download_button(
    "💾 Download Document Topics with All Columns",
    corpus_df.to_csv(index=False),
    file_name="document_topics_full.csv",
    mime="text/csv"
)

# ---------------- Topic distribution plot ----------------
st.subheader("📊 Topic Distribution")
topic_counts = doc_topics_df["Dominant Topic"].value_counts().sort_index()
fig, ax = plt.subplots()
topic_counts.plot(kind="bar", ax=ax)
ax.set_xlabel("Topic")
ax.set_ylabel("Document Count")
st.pyplot(fig)
