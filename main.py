import os
from dotenv import load_dotenv

from rag.loader import load_documents, logger
from rag.embedder import get_embedder
from rag.vector_store import build_vectorstore
from rag.retriever import get_retriever
from preprocess.xlsx_primes_to_markdown import save_chunks_as_files
from preprocess.extract_prestations import extract_prestations
from chatbot.llm_interface import build_qa_chain
from chatbot.chat_flow import ChatFlow
from langchain_community.vectorstores import FAISS


def run_preprocess():
    """
    Étape 1 - Lancer tous les preprocess avant le RAG
    """
    print("🔁 Préprocessing des données...")

    # 1. Transformer le fichier Excel des primes en chunks markdown
    excel_path = "data/Helsana/Helsana_Primes_LCA.xlsx"
    save_chunks_as_files(excel_path, "data/chunks")

    # 2. Extraire les prestations depuis les documents (CGA, PDF, etc.)
    prestations_dir = "data/Helsana/produits"
    os.makedirs(prestations_dir, exist_ok=True)
    extract_prestations(
        input_path="data/Helsana/Table_conditions_assura.pdf",
        output_dir=prestations_dir
    )

    print("✅ Préprocess terminé.")


def initialize_rag():
    """
    Étape 2 - Charger le RAG avec les documents + prestations
    """
    print("🔁 Initialisation du système RAG...")
    load_dotenv()

    # 📥 Charger tous les documents (Excel transformé + PDF + prestations.json)
    docs = load_documents()
    print(f"{len(docs)} chunks prêts à être indexés.")
    logger.info(f"Exemple metadata chunk 1 : {docs[0].metadata}")

    # 🧠 Créer embeddings
    embedder = get_embedder()
    print("🧠 Modèle d'embedding chargé.")

    # 📦 Charger ou créer l’index FAISS
    faiss_index_dir = "faiss_index"
    faiss_index_file = os.path.join(faiss_index_dir, "index.faiss")
    if os.path.exists(faiss_index_file):
        print("📦 Chargement index FAISS existant...")
        vs = FAISS.load_local(faiss_index_dir, embedder, allow_dangerous_deserialization=True)
    else:
        print("⚙️ Aucun index trouvé. Création d’un nouvel index FAISS...")
        vs = build_vectorstore(docs, embedder)
        vs.save_local(faiss_index_dir)
        print("💾 Nouvel index FAISS sauvegardé.")

    print("✅ Vector store prêt.")

    # 🔎 Construire retriever hybride (BM25 + FAISS)
    retriever = get_retriever(vs, docs)
    print("🔗 Retriever hybride prêt.")

    return build_qa_chain(retriever)


def terminal_chat(qa_chain):
    """
    Étape 3 - Lancer une session terminal avec mémoire
    """
    print("\n" + "=" * 50)
    print("💬 Chatbot Conseil Assurance - Mode Terminal")
    print("Tapez 'quit' pour quitter")
    print("=" * 50 + "\n")

    chatflow = ChatFlow(qa_chain)

    while True:
        try:
            query = input("👤 Vous: ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                print("👋 Merci, à bientôt !")
                break

            bot_response = chatflow.handle_message(query)
            print(f"\n🤖 Bot: {bot_response}\n")

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    run_preprocess()
    qa_chain = initialize_rag()
    terminal_chat(qa_chain)
