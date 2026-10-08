"""Client helpers for the local embedding API."""

import os
from collections.abc import Sequence
from config import settings
import requests


_API_URL = settings['EMBEDDING_API_URL']
_REQUEST_TIMEOUT_SECONDS = 300


def embed_text(text: str) -> list[float]:
    response = requests.post(
        f"{_API_URL}/embed",
        json={"text": text},
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []

    response = requests.post(
        f"{_API_URL}/embed/batch",
        json={"texts": list(texts)},
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    embeddings = response.json()["embeddings"]

    if len(embeddings) != len(texts):
        raise ValueError(
            f"Embedding API returned {len(embeddings)} vectors "
            f"for {len(texts)} texts"
        )
    return embeddings
