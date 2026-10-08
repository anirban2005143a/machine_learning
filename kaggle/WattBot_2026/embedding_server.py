"""Embedding API. Run from this directory with:

    python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

    uvicorn embedding_server:app --host 127.0.0.1 --port 8000
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        cache_folder=".model",
        model_kwargs={"device": "cuda", "local_files_only": False},
    )
    yield


app = FastAPI(lifespan=lifespan)


class EmbedRequest(BaseModel):
    text: str = Field(min_length=1)


class EmbedBatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1)


class EmbeddingResponse(BaseModel):
    embedding: list[float]


class EmbeddingBatchResponse(BaseModel):
    embeddings: list[list[float]]


def _embed(texts: list[str], request: Request) -> list[list[float]]:
    if any(not text.strip() for text in texts):
        raise HTTPException(status_code=422, detail="Text must not be blank")

    vectors = request.app.state.embedding.embed_documents(texts)
    if any(len(vector) != 1024 for vector in vectors):
        raise HTTPException(
            status_code=500,
            detail="BGE-large-en-v1.5 must produce 1024-dimensional vectors",
        )
    return vectors


@app.post("/embed", response_model=EmbeddingResponse)
def embed_text(payload: EmbedRequest, request: Request) -> EmbeddingResponse:
    vector = _embed([payload.text], request)[0]
    return EmbeddingResponse(embedding=vector)


@app.post("/embed/batch", response_model=EmbeddingBatchResponse)
def embed_batch(
    payload: EmbedBatchRequest,
    request: Request,
) -> EmbeddingBatchResponse:
    vectors = _embed(payload.texts, request)
    return EmbeddingBatchResponse(embeddings=vectors)
