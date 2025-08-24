import os
import sys
import logging
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("document_loader.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 📌 Exclusion
EXCLUDED_PDFS = {"Helsana_Primes_LCA.pdf"}


class TableAwareTextSplitter(RecursiveCharacterTextSplitter):
    """Évite de découper les tableaux (metadata type=insurance_table)."""
    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for doc in documents:
            if doc.metadata.get("type") == "insurance_table":
                logger.debug(f"Préservation tableau complet : {doc.metadata}")
                chunks.append(doc)
            else:
                chunks.extend(super().split_documents([doc]))
        return chunks


def generate_md_chunks_from_pdf(pdf_path: str, output_dir: str = "data/chunks"):
    """Convertit automatiquement un PDF primes → markdown chunks."""
    try:
        if "rag.pdf_primes_to_markdown" in sys.modules:
            del sys.modules["rag.pdf_primes_to_markdown"]

        from preprocess.xlsx_primes_to_markdown import save_chunks_as_files
        logger.info(f"🔄 Génération des fichiers .md depuis {pdf_path}")
        save_chunks_as_files(pdf_path, output_dir=output_dir)

    except Exception as e:
        logger.error(f"Erreur génération .md depuis {pdf_path}: {e}", exc_info=True)


def load_documents(data_dir: str = "data") -> List[Document]:
    """
    Charge tous les documents (PDF, MD, TXT) depuis `data/` et `data/chunks`,
    applique le split intelligent, retourne une liste de chunks.
    """
    logger.info("📂 Début du chargement des documents")
    all_docs = []

    # 🔹 Génération automatique des chunks.md depuis Helsana_Primes_LCA.pdf
    pdf_source = os.path.join(data_dir, "Helsana_Primes_LCA.pdf")
    if os.path.exists(pdf_source):
        generate_md_chunks_from_pdf(pdf_source, output_dir=os.path.join(data_dir, "chunks"))

    # 🔹 Inclure `data/` + `data/chunks`
    folders_to_scan = [data_dir, os.path.join(data_dir, "chunks")]

    for folder in folders_to_scan:
        if not os.path.exists(folder):
            continue

        for root, _, files in os.walk(folder):
            for file in files:
                if file in EXCLUDED_PDFS:
                    logger.info(f"⏭ Fichier exclu : {file}")
                    continue

                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()

                try:
                    docs = []

                    if ext == ".pdf":
                        docs = PyPDFLoader(file_path).load()

                    elif ext == ".md":
                        docs = UnstructuredMarkdownLoader(file_path).load()
                        # ✅ flatten.md = haute priorité
                        if "flatten" in file.lower():
                            for d in docs:
                                d.metadata["priority"] = "high"
                        # ✅ marquer les chunks générés
                        if "chunk_" in file.lower() or "_md" in file.lower():
                            for d in docs:
                                d.metadata["type"] = "insurance_table"

                    elif ext == ".txt":
                        docs = TextLoader(file_path, encoding="utf-8").load()

                    else:
                        logger.warning(f"⏭ Ignoré (format non supporté) : {file}")
                        continue

                    for d in docs:
                        d.metadata["source"] = file

                    all_docs.extend(docs)
                    logger.info(f"{len(docs)} docs chargés depuis {file}")

                except Exception as e:
                    logger.error(f"Erreur chargement {file}: {e}", exc_info=True)

    logger.info(f"📄 Total documents avant split : {len(all_docs)}")

    # ✅ Split intelligent
    final_chunks = []
    splitter_default = TableAwareTextSplitter(chunk_size=1000, chunk_overlap=200)

    for doc in all_docs:
        if doc.metadata.get("type") == "insurance_table" or "flatten" in doc.metadata.get("source", "").lower():
            final_chunks.append(doc)  # pas de split
        else:
            final_chunks.extend(splitter_default.split_documents([doc]))

    logger.info(f"✂️ Total chunks après split : {len(final_chunks)}")
    return final_chunks


if __name__ == "__main__":
    load_documents()
