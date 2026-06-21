import os
from typing import Optional

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
COLLECTION_NAME = "cfpb_complaints"


def stratified_sample(
    df: pd.DataFrame,
    n: int = 12000,
    category_col: str = "product_category",
    seed: int = 42,
) -> pd.DataFrame:
    """Return a stratified sample of n rows, proportional across product categories."""
    fracs = df[category_col].value_counts(normalize=True)
    parts = []
    for cat, frac in fracs.items():
        cat_df = df[df[category_col] == cat]
        k = max(1, round(frac * n))
        k = min(k, len(cat_df))
        parts.append(cat_df.sample(k, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def chunk_dataframe(
    df: pd.DataFrame,
    text_col: str = "clean_narrative",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split each complaint narrative into overlapping character-level chunks.
    Returns a flat list of dicts with text + metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    metadata_cols = [
        c for c in df.columns
        if c not in [text_col, "narrative", "clean_narrative"]
    ]

    records = []
    for _, row in df.iterrows():
        text = row[text_col]
        if not isinstance(text, str) or len(text.strip()) == 0:
            continue
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            meta = {col: str(row[col]) for col in metadata_cols if col in row}
            meta["chunk_index"] = i
            meta["total_chunks"] = len(chunks)
            records.append({"text": chunk, "metadata": meta})
    return records


def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return embeddings


def build_chroma_store(
    chunks: list[dict],
    embeddings: np.ndarray,
    persist_dir: str = "vector_store/chroma",
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """Persist chunks + embeddings in a ChromaDB collection."""
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # ChromaDB has a 41,666-document limit per add call
    batch = 5000
    for start in range(0, len(ids), batch):
        end = start + batch
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
    return collection


def load_chroma_store(
    persist_dir: str = "vector_store/chroma",
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_collection(collection_name)


def query_store(
    collection: chromadb.Collection,
    question: str,
    model: SentenceTransformer,
    k: int = 5,
    product_filter: Optional[str] = None,
) -> list[dict]:
    """Return top-k chunks most similar to the question."""
    q_embedding = model.encode([question], convert_to_numpy=True).tolist()
    where = {"product_category": product_filter} if product_filter else None
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "metadata": meta, "distance": dist})
    return hits

