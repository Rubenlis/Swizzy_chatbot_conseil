from langchain_community.vectorstores import FAISS

def build_vectorstore(docs, embedder):
    vectorstore = FAISS.from_documents(docs, embedder)
    vectorstore.save_local("faiss_index")  # Save to disk
    return vectorstore
