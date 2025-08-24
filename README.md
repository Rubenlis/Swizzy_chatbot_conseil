Project Architecture : 

Chatbot_consel/
│
├── Fichiers principaux du chatbot/   <-- logique métier & conversation
│   ├── chat_flow.py          # Orchestration du dialogue (profil → scoring → réponse)
│   ├── llm_interface.py      # Centralise les modèles LLM (flexibilité, maintenance)
│   ├── question_engine.py    # Gère les slots manquants → quelles questions poser ?
│   ├── response_generator.py # Génère la réponse finale "humaine" du chatbot
│   ├── rules_engine.py       # (optionnel / garde-fou) exclusions strictes
│   ├── scoring_engine.py     # Scoring explicable (0–20) sur les critères utilisateurs
│   └── slot_filling.py       # Extraction infos utilisateur (profil, besoins, contraintes)
│
├── data/
│   └── Helsana_documents/    # Base documentaire assurance
│       ├── assurances.json   # Données structurées des produits
│       ├── *.pdf, *.xlsx     # CGA, primes, conditions
│
├── rag/                      # Tout ce qui est RAG (VectorDB + ingestion)
│   ├── embedder.py
│   ├── ingest_documents.py
│   ├── loader.py
│   ├── retriever.py
│   ├── splitter.py
│   ├── vector_store.py
│   └── xlsx_primes_to_markdown.py
│
├── main.py                   # Point d’entrée → démarre le chatbot
│
├── External Libraries/       # (optionnel) dépendances externes, wrappers
└── Scratches and Consoles/   # (optionnel) tests rapides, explorations
