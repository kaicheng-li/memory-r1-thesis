"""Baseline-lora MiniLM retrieval; lexical retrieval is an explicit ablation."""
import os
from functools import lru_cache

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def retrieval_config():
    backend = os.environ.get("MEMORY_RETRIEVAL_BACKEND", "minilm")
    if backend not in {"minilm", "lexical"}:
        raise ValueError(f"Unknown retrieval backend: {backend}")
    return {"backend": backend,
            "model": os.environ.get("MEMORY_EMBEDDING_MODEL", DEFAULT_MODEL),
            "device": os.environ.get("MEMORY_EMBEDDING_DEVICE", "cpu")}


@lru_cache(maxsize=2)
def embedding_model(model_name, device):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device)


@lru_cache(maxsize=50000)
def embedding(text, model_name, device):
    return embedding_model(model_name, device).encode(
        text, normalize_embeddings=True, convert_to_numpy=True)


def semantic_retrieve(query, bank, top_k):
    if not bank or top_k <= 0:
        return []
    config = retrieval_config()
    model, device = config["model"], config["device"]
    vector = embedding(query, model, device)
    return sorted(bank, key=lambda row: float(embedding(row["text"], model, device) @ vector),
                  reverse=True)[:top_k]
