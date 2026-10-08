from DocumentHandler import DocumentHandler
from langchain_core.documents import Document
from Elasticsearch.ElasticSearchClient import es
from embedding_client import embed_documents
from typing import Iterable
from config import settings
import pandas as pd

es.connect()

# ---------------------------------------------------------
# Store multiple LangChain Documents efficiently
# ---------------------------------------------------------


def store_documents(
    documents: Iterable[Document],
    index_name: str,
) -> None:
    """
    Generate embeddings for multiple LangChain Documents and
    store them in Elasticsearch.
    """

    documents = list(documents)

    if not documents:
        return

    for document in documents:
        if not isinstance(document, Document):
            raise TypeError("All items must be LangChain Document objects")

        if not document.page_content or not document.page_content.strip():
            raise ValueError("Document page_content is empty")

    # Batch embedding generation
    texts = [document.page_content for document in documents]
    print("Generating Embedding for document : ", documents[0].metadata['id'])

    embeddings = embed_documents(texts)

    if any(len(vector) != 1024 for vector in embeddings):
        raise ValueError("BGE-large-en-v1.5 must produce 1024-dimensional vectors")

    # Prepare Elasticsearch bulk actions
    operations = []

    for document, vector in zip(documents, embeddings):
        metadata = dict(document.metadata)

        elastic_id = str(metadata.get("id", "document"))

        operations.append(
            {
                "index": {
                    "_index": index_name,
                    "_id": elastic_id,
                }
            }
        )

        operations.append(
            {
                "page_content": document.page_content,
                "metadata": metadata,
                "content_embedding": vector,
            }
        )

    print("Storing documents of id : ", documents[0].metadata['id'])
    
    es.get_client().bulk(operations=operations, refresh=True)


df = pd.read_csv("./metadata.csv")

document_handler = DocumentHandler(
    metadata_csv="./metadata.csv"
)

for doc_id in df["id"]:
    store_documents(
        documents=document_handler.get_chunks_from_document(doc_id),
        index_name=settings["INDEX_NAME"],
    )
