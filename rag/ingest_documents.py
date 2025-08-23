from rag.loader import load_documents
from rag.splitter import split_documents
from rag.embedder import get_embedder
from rag.vector_store import build_vectorstore


def ingest():
    docs = load_documents()
    chunks = split_documents(docs)

    # Build vectorstore
    embedder = get_embedder()
    build_vectorstore(chunks, embedder)

    return chunks  # return chunks for BM25 retriever
