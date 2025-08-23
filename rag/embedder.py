from langchain_community.embeddings import HuggingFaceEmbeddings

import warnings

def get_embedder():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )