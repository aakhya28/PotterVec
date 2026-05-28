import streamlit as st
import gensim
import gensim.models
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Word2Vec Explorer",
    page_icon="🔮",
    layout="wide",
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #e9ecef;
    }
    .score-chip {
        display: inline-block;
        background: #e8f4f8;
        color: #1a6b8a;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Model loading ────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(path: str):
    """Train or load a cached Word2Vec model from a text file."""
    story = []
    with open("data\Harry_Potter_all_books_preprocessed.txt", "r", encoding="utf-8") as f:
        text = f.read()
        
    sentences = sent_tokenize(text)
    for sentence in sentences:
        story.append(simple_preprocess(sentence))

    model = gensim.models.Word2Vec(window=10, min_count=2, vector_size=100)
    model.build_vocab(story)
    model.train(story, total_examples=model.corpus_count, epochs=model.epochs)
    return model, len(story)


# ── Sidebar — file loading ───────────────────────────────────────────────────

with st.sidebar:
    st.title("🔮 Word2Vec Explorer")
    st.caption("Harry Potter corpus · gensim")
    st.divider()

    default_path = r"data\Harry_Potter_all_books_preprocessed.txt"
    data_path = st.text_input("Text file path", value=default_path)

    load_btn = st.button("Load / Train model", type="primary", use_container_width=True)

    if load_btn:
        if not os.path.exists(data_path):
            st.error("File not found. Check the path and try again.")
        else:
            with st.spinner("Training model…"):
                model, n_sentences = load_model(data_path)
                st.session_state["model"] = model
                st.session_state["n_sentences"] = n_sentences
            st.success("Model ready!")

    if "model" in st.session_state:
        m = st.session_state["model"]
        st.divider()
        st.metric("Sentences", f"{st.session_state['n_sentences']:,}")
        st.metric("Vocabulary", f"{len(m.wv):,} words")
        st.metric("Vector size", m.vector_size)
        st.metric("Window", m.window)


# ── Guard — no model yet ─────────────────────────────────────────────────────

if "model" not in st.session_state:
    st.info("👈  Enter the path to your text file and click **Load / Train model** to begin.")
    st.stop()

model: gensim.models.Word2Vec = st.session_state["model"]
wv = model.wv


# ── Helper ───────────────────────────────────────────────────────────────────

def word_exists(word: str) -> bool:
    return word.lower() in wv.key_to_index


def parse_words(raw: str) -> list[str]:
    return [w.strip().lower() for w in raw.split(",") if w.strip()]


# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_sim, tab_similarity, tab_analogy, tab_vocab = st.tabs(
    ["🔍 Most Similar", "↔️ Similarity", "🧮 Analogy", "📚 Vocabulary"]
)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 · Most Similar
# ══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.subheader("Most Similar Words", divider="gray")
    st.caption("Finds words with the closest vectors — optionally with positive / negative vector arithmetic.")

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        sim_word = st.text_input("Word", placeholder="e.g. snape", key="sim_word")
    with col2:
        sim_pos = st.text_input("Additional positive (comma-sep)", placeholder="e.g. magic, wand", key="sim_pos")
    with col3:
        sim_neg = st.text_input("Negative (comma-sep)", placeholder="e.g. dark", key="sim_neg")

    topn = st.slider("Top N results", min_value=3, max_value=30, value=10, key="topn")

    if st.button("Find similar", type="primary", key="btn_sim"):
        if not sim_word:
            st.warning("Enter a word to search.")
        elif not word_exists(sim_word):
            st.error(f'"{sim_word}" is not in the vocabulary.')
        else:
            positive = [sim_word.lower()] + parse_words(sim_pos)
            negative = parse_words(sim_neg)

            missing = [w for w in positive + negative if not word_exists(w)]
            if missing:
                st.error(f"Not in vocabulary: {', '.join(missing)}")
            else:
                results = wv.most_similar(positive=positive, negative=negative, topn=topn)
                df = pd.DataFrame(results, columns=["word", "score"])

                # Bar chart
                fig = px.bar(
                    df,
                    x="score",
                    y="word",
                    orientation="h",
                    color="score",
                    color_continuous_scale="Blues",
                    labels={"score": "Cosine similarity", "word": ""},
                    title=f'Most similar to  "{sim_word}"'
                          + (f' + {", ".join(parse_words(sim_pos))}' if sim_pos else "")
                          + (f' − {", ".join(parse_words(sim_neg))}' if sim_neg else ""),
                )
                fig.update_layout(
                    yaxis=dict(autorange="reversed"),
                    coloraxis_showscale=False,
                    height=max(350, topn * 32),
                    margin=dict(l=0, r=20, t=50, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Table
                df.index += 1
                df["score"] = df["score"].round(4)
                st.dataframe(df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 · Similarity between two words
# ══════════════════════════════════════════════════════════════════════════════
with tab_similarity:
    st.subheader("Word Similarity", divider="gray")
    st.caption("Cosine similarity between two word vectors (−1 to 1; higher = more similar).")

    c1, c2 = st.columns(2)
    with c1:
        w1 = st.text_input("Word 1", placeholder="e.g. harry", key="w1")
    with c2:
        w2 = st.text_input("Word 2", placeholder="e.g. voldemort", key="w2")

    if st.button("Compare", type="primary", key="btn_sim2"):
        missing = [w for w in [w1, w2] if w and not word_exists(w)]
        if not w1 or not w2:
            st.warning("Enter both words.")
        elif missing:
            st.error(f"Not in vocabulary: {', '.join(missing)}")
        else:
            score = wv.similarity(w1.lower(), w2.lower())
            pct = int((score + 1) / 2 * 100)  # map −1..1 → 0..100

            col_score, col_bar = st.columns([1, 3])
            with col_score:
                st.metric("Cosine similarity", f"{score:.4f}")
            with col_bar:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    number={"valueformat": ".4f"},
                    gauge={
                        "axis": {"range": [-1, 1]},
                        "bar": {"color": "#1a6b8a"},
                        "steps": [
                            {"range": [-1, 0], "color": "#fee8e8"},
                            {"range": [0, 0.5], "color": "#e8f4f8"},
                            {"range": [0.5, 1], "color": "#c8e6c9"},
                        ],
                    },
                    domain={"x": [0, 1], "y": [0, 1]},
                ))
                fig.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

            if score > 0.7:
                st.success(f'"{w1}" and "{w2}" are **highly similar** in this corpus.')
            elif score > 0.4:
                st.info(f'"{w1}" and "{w2}" are **moderately related**.')
            elif score > 0:
                st.warning(f'"{w1}" and "{w2}" are **weakly related**.')
            else:
                st.error(f'"{w1}" and "{w2}" are **dissimilar or unrelated**.')


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 · Analogy  (A − B + C = ?)
# ══════════════════════════════════════════════════════════════════════════════
with tab_analogy:
    st.subheader("Word Analogy", divider="gray")
    st.caption("Vector arithmetic: **A − B + C = ?**  &nbsp;·&nbsp; e.g. *king − man + woman = queen*")

    ca, cb, cc = st.columns(3)
    with ca:
        an_a = st.text_input("A  (positive)", placeholder="e.g. king", key="an_a")
    with cb:
        an_b = st.text_input("B  (negative)", placeholder="e.g. man", key="an_b")
    with cc:
        an_c = st.text_input("C  (positive)", placeholder="e.g. woman", key="an_c")

    an_topn = st.slider("Top N answers", 3, 10, 5, key="an_topn")

    if st.button("Solve analogy", type="primary", key="btn_analogy"):
        words = {"A": an_a, "B": an_b, "C": an_c}
        empty = [k for k, v in words.items() if not v.strip()]
        if empty:
            st.warning(f"Fill in: {', '.join(empty)}")
        else:
            missing = [v.lower() for v in words.values() if not word_exists(v)]
            if missing:
                st.error(f"Not in vocabulary: {', '.join(missing)}")
            else:
                results = wv.most_similar(
                    positive=[an_a.lower(), an_c.lower()],
                    negative=[an_b.lower()],
                    topn=an_topn,
                )
                st.markdown(
                    f"### {an_a} − {an_b} + {an_c} = **{results[0][0]}**"
                    f"  &nbsp; `{results[0][1]:.4f}`"
                )
                if len(results) > 1:
                    st.caption("Other candidates")
                    df_an = pd.DataFrame(results[1:], columns=["word", "score"])
                    df_an["score"] = df_an["score"].round(4)
                    df_an.index += 2
                    st.dataframe(df_an, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 · Vocabulary browser
# ══════════════════════════════════════════════════════════════════════════════
with tab_vocab:
    st.subheader("Vocabulary Browser", divider="gray")
    st.caption(f"Model vocabulary · {len(wv):,} words  ·  min_count=2")

    search_q = st.text_input("Filter words", placeholder="type to filter…", key="vocab_q")

    all_words = list(wv.key_to_index.keys())
    if search_q:
        filtered = [w for w in all_words if search_q.lower() in w]
    else:
        filtered = all_words

    st.caption(f"Showing {min(len(filtered), 500):,} of {len(filtered):,} matches")

    # Display as a grid of clickable pills via columns
    cols_per_row = 6
    display_words = filtered[:500]
    rows = [display_words[i:i+cols_per_row] for i in range(0, len(display_words), cols_per_row)]
    for row in rows:
        cols = st.columns(cols_per_row)
        for col, word in zip(cols, row):
            col.code(word)

    st.divider()
    st.caption("Word frequency (top 40 by count in corpus)")
    counts = {w: wv.get_vecattr(w, "count") for w in all_words[:40]}
    df_counts = pd.DataFrame(
        sorted(counts.items(), key=lambda x: -x[1]),
        columns=["word", "count"]
    )
    fig_freq = px.bar(
        df_counts, x="word", y="count",
        color="count", color_continuous_scale="Blues",
        labels={"count": "Frequency", "word": ""},
    )
    fig_freq.update_layout(
        coloraxis_showscale=False,
        height=300,
        margin=dict(l=0, r=0, t=20, b=40),
        xaxis_tickangle=-40,
    )
    st.plotly_chart(fig_freq, use_container_width=True)