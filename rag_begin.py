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


# Ensure numpy array is 2D
def _ensure_2d_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


# Create embeddings with OpenAI API
def get_openai_embeddings(texts: List[str], model: str, api_key: str) -> np.ndarray:
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)
    return _ensure_2d_array(embeddings)


# Index database and save to disk
def index_database(frasi: List[str], path: str, model: str, pwd: str) -> np.ndarray:
    if not frasi:
        raise ValueError("The input sentence list 'frasi' is empty.")

    texts = [str(f) for f in frasi]
    embeddings = get_openai_embeddings(texts, model, pwd)

    save_path = Path(path).with_suffix(".npy")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, embeddings)

    return embeddings


# Load embeddings from disk
def load_emb(path: str) -> np.ndarray:
    load_path = Path(path).with_suffix(".npy")
    if not load_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {load_path}")

    dense_vecs = np.load(load_path)
    return _ensure_2d_array(dense_vecs)


# Search most similar chunks
def search(query_emb, embeddings, frasi, top_k=3, min_similarity=0.2):
    if len(frasi) == 0:
        raise ValueError("'frasi' is empty.")

    embeddings = _ensure_2d_array(np.asarray(embeddings))
    query_emb = _ensure_2d_array(np.asarray(query_emb))

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

    similarities = cosine_similarity(query_emb, embeddings)[0]
    pairs = list(zip(frasi, similarities))

    if min_similarity is not None:
        pairs = [(f, s) for f, s in pairs if s >= min_similarity]

    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_k]
    return pairs


# Generate answer using retrieved context
def Quidditch_gpt_core(
    query: str,
    frasi: List[str],
    embeddings: np.ndarray,
    model_emb: str,
    model_llm_name: str,
    top_k: int = 5,
    min_similarity: Optional[float] = 0.2,
    pwd: str = "",
):
    if not query or not query.strip():
        raise ValueError("Query is empty.")

    if not frasi:
        raise ValueError("'frasi' is empty.")

    embeddings = _ensure_2d_array(np.asarray(embeddings))

    # Create query embedding with OpenAI API
    query_emb = get_openai_embeddings([query], model_emb, pwd)

    retrieved = search(
        query_emb=query_emb,
        embeddings=embeddings,
        frasi=frasi,
        top_k=top_k,
        min_similarity=min_similarity
    )

    # Fallback without threshold
    if not retrieved:
        retrieved = search(
            query_emb=query_emb,
            embeddings=embeddings,
            frasi=frasi,
            top_k=top_k,
            min_similarity=None
        )

    context = "\n".join([r[0] for r in retrieved])

    prompt = (
        "Sei un esperto di Quidditch. Rispondi in modo chiaro e preciso, "
        "usando solo le informazioni fornite nel contesto. "
        "Se il contesto non contiene la risposta, dillo esplicitamente.\n\n"
        f"Contesto:\n{context}\n\n"
    )

    client = OpenAIClient(api_key=pwd)

    agent = Agent(
        name="kb_assistant",
        client=client,
        system_prompt=prompt,
    )

    result = agent.run(
        f"Domanda: {query}\n\n"
        "Risposta:"
    )

    answer = result.text

    return answer, context, retrieved, query_emb
