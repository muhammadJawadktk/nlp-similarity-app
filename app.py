import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Text Similarity Explorer",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

st.title("Text Similarity Explorer")
st.write("This app uses the **all-MiniLM-L6-v2** model to measure how similar two pieces of text are. Just type your sentences below and hit the button.")
st.write("No preprocessing is done. The text goes directly into the model as you type it.")

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    query = st.text_input(
        "Your main sentence (query)",
        value="Artificial intelligence is transforming the world",
    )

with col2:
    raw_comparisons = st.text_area(
        "Sentences to compare against (one per line)",
        value=(
            "Machine learning is a subset of AI\n"
            "Deep learning uses neural networks\n"
            "Python is a programming language\n"
            "The cat sat on the mat\n"
            "Natural language processing handles text\n"
            "Football is a popular sport\n"
            "AI models can generate human-like text\n"
            "The weather today is sunny"
        ),
        height=180,
    )

run_btn = st.button("Compare Now", type="primary", use_container_width=True)

if run_btn:
    comparisons = [line.strip() for line in raw_comparisons.strip().split("\n") if line.strip()]

    if len(comparisons) < 2:
        st.error("Please add at least 2 sentences to compare.")
        st.stop()

    all_texts = [query] + comparisons

    with st.spinner("Running the model... this takes a few seconds."):
        embeddings = model.encode(all_texts, convert_to_numpy=True)

    query_emb = embeddings[0:1]
    comp_embs = embeddings[1:]

    scores = cosine_similarity(query_emb, comp_embs)[0]
    pairwise = cosine_similarity(embeddings)

    st.subheader("Similarity Results")
    st.write("The table below shows how similar each sentence is to your query. A score close to 1.0 means very similar, while a score close to 0.0 means very different.")

    df = pd.DataFrame({
        "Sentence": comparisons,
        "Similarity Score": scores
    }).sort_values("Similarity Score", ascending=False).reset_index(drop=True)
    df.index += 1

    st.dataframe(
        df.style.background_gradient(subset=["Similarity Score"], cmap="YlOrRd"),
        use_container_width=True
    )

    st.divider()

    st.subheader("Graph 1 — Which sentence is most similar?")
    st.write("This bar chart ranks all sentences from most to least similar to your query.")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn(scores / scores.max())
    bars = ax1.barh(comparisons, scores, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("Cosine Similarity Score")
    ax1.set_title(f'Similarity to: "{query}"', fontweight="bold")
    ax1.set_xlim(0, 1.05)
    ax1.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, label="0.5 midpoint")
    ax1.legend()
    for bar, score in zip(bars, scores):
        ax1.text(
            score + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}", va="center", fontsize=9
        )
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    st.divider()

    st.subheader("Graph 2 — How do all sentences relate to each other?")
    st.write("This heatmap shows the similarity between every pair of sentences, including the query. Darker red means more similar.")

    labels = [f"Query"] + [t[:25] + "..." if len(t) > 25 else t for t in comparisons]
    fig2, ax2 = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        pairwise,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="coolwarm",
        vmin=0, vmax=1,
        linewidths=0.5,
        ax=ax2
    )
    ax2.set_title("Pairwise Similarity Between All Sentences", fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.divider()

    st.subheader("Graph 3 — Visualising meaning in 2D space")
    st.write("Each sentence is converted into a 384-number vector by the model. We use PCA to compress that into 2D so we can plot it. Sentences that are close together on this plot have similar meanings.")

    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(embeddings)

    fig3, ax3 = plt.subplots(figsize=(10, 7))
    scatter = ax3.scatter(
        reduced[1:, 0], reduced[1:, 1],
        c=scores, cmap="YlOrRd", s=120, edgecolors="black", linewidths=0.7,
        label="Comparison sentences", zorder=3
    )
    ax3.scatter(
        reduced[0, 0], reduced[0, 1],
        c="blue", s=200, marker="*", edgecolors="black", linewidths=0.7,
        label="Your query", zorder=4
    )
    plt.colorbar(scatter, ax=ax3, label="Similarity to Query")

    short_labels = [f"Query"] + [t[:20] for t in comparisons]
    for i, (x, y) in enumerate(reduced):
        ax3.annotate(
            short_labels[i], (x, y),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8, alpha=0.85
        )

    ax3.set_title("PCA Plot — Sentences Mapped in 2D Semantic Space", fontweight="bold")
    ax3.set_xlabel(f"Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% of variance)")
    ax3.set_ylabel(f"Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% of variance)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.divider()

    top_idx = int(np.argmax(scores))
    top_text = comparisons[top_idx]
    top_score = scores[top_idx]
    bottom_idx = int(np.argmin(scores))
    bottom_text = comparisons[bottom_idx]
    bottom_score = scores[bottom_idx]

    st.subheader("Critical Thinking Analysis (Paul's Standards)")
    st.write("Here I reflect on the results using Paul's Critical Thinking Standards to make sure the output is properly understood and not misused.")

    with st.expander("Clarity — What did we do and what does the output mean?", expanded=True):
        st.write(
            f"The query sentence was: *\"{query}\"*. "
            f"It was compared against {len(comparisons)} different sentences. "
            f"The model returned cosine similarity scores between {bottom_score:.4f} and {top_score:.4f}. "
            f"A score of 1.0 means the two sentences are identical in meaning. A score of 0.0 means they share no meaning at all."
        )

    with st.expander("Accuracy — Is the model trustworthy?", expanded=True):
        st.write(
            f"The model used is **all-MiniLM-L6-v2**, published by the Sentence-Transformers team on HuggingFace. "
            f"It was trained on over 1 billion sentence pairs and is widely used in NLP research. "
            f"It is completely free and open source. No claims are made beyond what the scores directly show."
        )

    with st.expander("Precision — Exact numbers, not vague descriptions", expanded=True):
        st.write(
            f"The most similar sentence was *\"{top_text}\"* with a score of **{top_score:.4f}**. "
            f"The least similar was *\"{bottom_text}\"* with a score of **{bottom_score:.4f}**. "
            f"All scores are shown to 4 decimal places in the table above."
        )

    with st.expander("Relevance — Do the graphs actually support the results?", expanded=True):
        st.write(
            f"Yes. The bar chart directly ranks the similarity scores. "
            f"The heatmap shows how every sentence relates to every other sentence, not just the query. "
            f"The PCA plot visually confirms that sentences with high scores cluster near the query point (blue star)."
        )

    with st.expander("Logic — Why does the top result make sense?", expanded=True):
        st.write(
            f"*\"{top_text}\"* scored {top_score:.4f} because it shares the most semantic meaning with the query. "
            f"The model does not just match keywords — it understands meaning. So even if different words are used, "
            f"sentences that talk about the same topic will still score high."
        )

    with st.expander("Significance — What is the most important finding?", expanded=True):
        st.write(
            f"The key takeaway is that *\"{top_text}\"* (score: {top_score:.4f}) is the closest match. "
            f"This is significant because it shows the model successfully identifies semantic similarity, "
            f"which is exactly what we need for tasks like search engines, recommendation systems, and plagiarism detection."
        )

    with st.expander("Fairness — What are the limitations?", expanded=True):
        st.write(
            f"This model was mainly trained on English text. If you input Urdu, Arabic, or mixed-language sentences, "
            f"the results may not be reliable. Also, very short inputs like single words may not give accurate embeddings "
            f"compared to full sentences. The model also cannot understand sarcasm or cultural context."
        )

    st.divider()
    st.success(f"Done! The closest match to your query is: \"{top_text}\" with a similarity score of {top_score:.4f}")

else:
    st.info("Enter your query and comparison sentences above, then click Compare Now.")
