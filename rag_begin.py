from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from openai import OpenAI

# =========================
# OpenAI client setup
# =========================


# =========================
# Embedding utilities
# =========================

def _ensure_2d_array(x: np.ndarray) -> np.ndarray:
    """
    Ensure that the input is a 2D numpy array.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


def index_database(frasi: List[str], path: str, model,pwd:str) -> np.ndarray:
    """
    Compute embeddings for a list of sentences and save them to disk.

    Args:
        frasi: List of input sentences.
        path: Base path where embeddings will be saved (without extension).
        model: Embedding model with an .encode(...) method.

    Returns:
        A 2D numpy array of embeddings.
    """
    if not frasi:
        raise ValueError("The input sentence list 'frasi' is empty.")

    # Extract text items only
    texts = [str(f) for f in frasi]
    
    client=OpenAI(pwd)

    # # Generate embeddings
    # embeddings = model.encode(texts)
    # embeddings = _ensure_2d_array(np.asarray(embeddings))

    # # Ensure target directory exists
    # save_path = Path(path).with_suffix(".npy")
    # save_path.parent.mkdir(parents=True, exist_ok=True)

    # # Save embeddings
    # np.save(save_path, embeddings)
    
    response=client.embeddigs.create(model="text-embedding-3-small",  # oppure "text-embedding-3-large"
        input=texts
    )
    
    embeddings = np.array([item.embedding for item in response.data])
    
    # Ensure 2D
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    
    # Save
    save_path = Path(path).with_suffix(".npy")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, embeddings)

    return embeddings


def load_emb(path: str) -> np.ndarray:
    """
    Load embeddings from disk.

    Args:
        path: Base path of the saved embeddings (without extension).

    Returns:
        A 2D numpy array of embeddings.
    """
    load_path = Path(path).with_suffix(".npy")

    if not load_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {load_path}")

    dense_vecs = np.load(load_path)
    return _ensure_2d_array(dense_vecs)

def search(
    query_emb=[],
    embeddings=[],
    frasi=[''],
    top_k=3,
    min_similarity=0.2 
):
    """
    Search for the most similar sentences using cosine similarity.

    Args:
        query_emb: Query embedding, shape (1, d) or (d,).
        embeddings: Corpus embeddings, shape (n, d).
        frasi: Original sentence list, length n.
        top_k: Number of top results to return.
        min_similarity: Minimum similarity threshold. If None, no filtering is applied.

    Returns:
        A list of (sentence, similarity_score) tuples.
    """
    if len(frasi) == 0:
        raise ValueError("'frasi' is empty.")

    embeddings = _ensure_2d_array(np.asarray(embeddings))
    query_emb = _ensure_2d_array(np.asarray(query_emb))
    
    print(embeddings.shape)
    print(query_emb.shape)

    if embeddings.shape[0] != len(frasi):
        raise ValueError(
            f"Mismatch between number of embeddings ({embeddings.shape[0]}) "
            f"and number of sentences ({len(frasi)})."
        )

    if query_emb.shape[1] != embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: query={query_emb.shape[1]}, "
            f"database={embeddings.shape[1]}."
        )

    # Compute cosine similarity between query and corpus
    similarities = cosine_similarity(query_emb, embeddings)[0]

    # Pair each sentence with its similarity score
    pairs = list(zip(frasi, similarities))

    # Apply threshold if requested
    if min_similarity is not None:
        pairs = [(f, s) for f, s in pairs if s >= min_similarity]

    # Sort by similarity descending and keep top_k
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_k]

    return pairs


# =========================
# Visualization utilities
# =========================

def viz(query_emb, embeddings, query: str, frasi: List[str]) -> None:
    """
    Visualize sentence embeddings and the query in 2D using PCA and t-SNE.

    Args:
        query_emb: Query embedding, shape (1, d) or (d,).
        embeddings: Corpus embeddings, shape (n, d).
        query: Input query string.
        frasi: List of corpus sentences.
    """
    embeddings = _ensure_2d_array(np.asarray(embeddings))
    query_emb = _ensure_2d_array(np.asarray(query_emb))

    if embeddings.shape[0] != len(frasi):
        raise ValueError(
            f"Mismatch between embeddings ({embeddings.shape[0]}) and sentences ({len(frasi)})."
        )

    if query_emb.shape[1] != embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: query={query_emb.shape[1]}, "
            f"database={embeddings.shape[1]}."
        )

    # Combine corpus embeddings and query embedding
    all_vecs = np.vstack([embeddings, query_emb])

    # PCA projection
    pca = PCA(n_components=2)
    pca_vecs = pca.fit_transform(all_vecs)

    # t-SNE projection
    # Perplexity must be smaller than the number of samples
    n_samples = all_vecs.shape[0]
    perplexity = max(1, min(5, n_samples - 1))

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto"
    )
    tsne_vecs = tsne.fit_transform(all_vecs)

    # Create a figure with two subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("PCA", "t-SNE"))

    # PCA plot for dataset points
    fig.add_trace(
        go.Scatter(
            x=pca_vecs[:-1, 0],
            y=pca_vecs[:-1, 1],
            mode="markers+text",
            text=frasi,
            textposition="top center",
            marker=dict(color="blue", size=8),
            name="Sentences"
        ),
        row=1,
        col=1
    )

    # PCA plot for query point
    fig.add_trace(
        go.Scatter(
            x=[pca_vecs[-1, 0]],
            y=[pca_vecs[-1, 1]],
            mode="markers+text",
            text=[query],
            textposition="top center",
            marker=dict(color="red", size=10, symbol="diamond"),
            name="Query"
        ),
        row=1,
        col=1
    )

    # t-SNE plot for dataset points
    fig.add_trace(
        go.Scatter(
            x=tsne_vecs[:-1, 0],
            y=tsne_vecs[:-1, 1],
            mode="markers+text",
            text=frasi,
            textposition="top center",
            marker=dict(color="green", size=8),
            name="Sentences"
        ),
        row=1,
        col=2
    )

    # t-SNE plot for query point
    fig.add_trace(
        go.Scatter(
            x=[tsne_vecs[-1, 0]],
            y=[tsne_vecs[-1, 1]],
            mode="markers+text",
            text=[query],
            textposition="top center",
            marker=dict(color="red", size=10, symbol="diamond"),
            name="Query"
        ),
        row=1,
        col=2
    )

    fig.update_layout(
        title="Sentence Embeddings: PCA vs t-SNE",
        showlegend=False,
        height=700,
        width=1400
    )

    return fig


def show_top_k(
    query_emb: np.ndarray,
    embeddings: np.ndarray,
    frasi: List[str],
    top_k: int = 5,
    min_similarity=0.2
) -> pd.DataFrame:
    """
    Display the top-k most similar sentences in a table and bar chart.

    Args:
        query_emb: Query embedding.
        embeddings: Corpus embeddings.
        frasi: List of corpus sentences.
        k: Number of results to display.
        min_similarity: Minimum similarity threshold.

    Returns:
        A pandas DataFrame with top-k results.
    """
    results = search(
        query_emb,
        embeddings,
        frasi,
        top_k,
        min_similarity
     )


    if not results:
        print("No results found above the similarity threshold.")
        return pd.DataFrame(columns=["Sentence", "Similarity"])

    df = pd.DataFrame(
        [{"Sentence": f, "Similarity": round(float(sim), 3)} for f, sim in results]
    )

    print(df)

    fig = px.bar(
        df,
        x="Similarity",
        y="Sentence",
        orientation="h",
        title=f"Top {len(df)} most similar sentences to query",
        text="Similarity"
    )

    # Put the most similar item at the top
    fig.update_yaxes(autorange="reversed")
    fig.show()

    return df


# =========================
# LLM answer generation
# =========================

def Quidditch_gpt_core(
    query: str,
    frasi: List[str],
    embeddings: np.ndarray,
    model_emb,
    model_llm_name: str ,
    top_k: int = 5,
    min_similarity: Optional[float] = 0.2,
    pwd: str= '',
) -> Tuple[str, str, List[Tuple[str, float]]]:
    """
    Answer a query using retrieved context and OpenAI.

    Args:
        query: The user's question.
        frasi: List of context sentences/documents.
        embeddings: Precomputed embeddings for 'frasi'.
        model_emb: Embedding model with an .encode(...) method.
        model_llm_name: OpenAI model name.
        top_k: Number of retrieved chunks.
        min_similarity: Minimum similarity threshold for retrieval.

    Returns:
        A tuple containing:
        - generated answer
        - retrieved context string
        - retrieved (sentence, similarity) pairs
    """
    if not query or not query.strip():
        raise ValueError("Query is empty.")

    if not frasi:
        raise ValueError("'frasi' is empty.")

    embeddings = _ensure_2d_array(np.asarray(embeddings))

    # Encode the query
    query_emb = model_emb.encode([query])
    query_emb = _ensure_2d_array(np.asarray(query_emb))

    # Retrieve the top-k most relevant contexts
    retrieved = search(
        query_emb,
        embeddings,
        frasi,
        top_k,
        min_similarity
    )

    # Fallback if nothing passes the threshold
    if not retrieved:
       retrieved = search(
           query_emb,
           embeddings,
           frasi,
           top_k,
           None
       )

    
    
    texts = [str(f) for f in retrieved]    
    
    embeddings_retr= model_emb.encode(texts)


    context = "\n".join([r[0] for r in retrieved])

    prompt = (
        "Sei un esperto di Quidditch. Rispondi in modo chiaro e preciso, "
        "usando solo le informazioni fornite nel contesto. "
        "Se il contesto non contiene la risposta, dillo esplicitamente.\n\n"
        f"Contesto:\n{context}\n\n"
        
        )

    
    client = OpenAIClient(
        api_key=pwd
        )
    
    agent = Agent(
    name="kb_assistant",
    client=client,
    system_prompt=prompt,
    )    
    
    
    result = agent.run(
    f"Domanda: {query}\n\n"
    "Risposta:")

    answer = result.text

    return answer, context, retrieved, query_emb,embeddings_retr
