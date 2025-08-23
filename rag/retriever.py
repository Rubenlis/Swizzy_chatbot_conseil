from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import EnsembleRetriever
from rank_bm25 import BM25Okapi
from pydantic import Field, PrivateAttr


class BM25Retriever(BaseRetriever):
    documents: List[Document] = Field(...)
    bm25: BM25Okapi = Field(...)
    tokenized_docs: List[List[str]] = Field(...)

    def __init__(self, documents: List[Document]):
        # Pre-tokenize docs
        tokenized_docs = [self._tokenize(doc.page_content) for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)

        # Pass fields to BaseRetriever/Pydantic
        super().__init__(documents=documents, tokenized_docs=tokenized_docs, bm25=bm25)

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def _get_relevant_documents(self, query: str) -> List[Document]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:5]
        return [self.documents[i] for i in top_indices]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


class FilteredRetriever(BaseRetriever):
    """Un retriever qui filtre les documents par canton avant de les renvoyer."""

    canton: Optional[str] = Field(default=None)
    _base_retriever: BaseRetriever = PrivateAttr()  # Use PrivateAttr for non-field attributes

    def __init__(self, base_retriever: BaseRetriever, canton: Optional[str] = None):
        super().__init__(canton=canton)
        self._base_retriever = base_retriever  # Use PrivateAttr instead of regular field

    def set_canton(self, canton: str):
        """Permet de mettre à jour dynamiquement le canton en cours."""
        self.canton = canton

    def _filter_docs(self, docs: List[Document]) -> List[Document]:
        if not self.canton:
            return docs  # Pas de filtre si le canton n'est pas encore connu
        return [d for d in docs if d.metadata.get("canton") == self.canton]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self._base_retriever.get_relevant_documents(query)
        return self._filter_docs(docs)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        docs = await self._base_retriever.aget_relevant_documents(query)
        return self._filter_docs(docs)

def get_retriever(vectorstore, documents, canton: Optional[str] = None):
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    bm25_retriever = BM25Retriever(documents=documents)

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )

    return FilteredRetriever(ensemble, canton=canton)
