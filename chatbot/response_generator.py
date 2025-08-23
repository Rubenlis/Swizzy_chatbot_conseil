from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

def generate_final_answer(profile: Dict[str, Any],
                          scored: List[Dict[str, Any]]) -> str:
    """
    Demande au LLM de générer une réponse structurée (analyse + tableau + explications + prochaines questions).
    On reste côté génération de texte (le scoring est déjà fait).
    """
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    sys = SystemMessage(content="""
Tu es un conseiller en assurance santé suisse. Tu expliques de manière claire et transparente.
Formate la réponse avec :
- **Analyse du besoin**
- **Tableau comparatif** (Produit | Score total /20 | Points forts | Points faibles)
- **Détails explicatifs** avec justifications par critère
- **Prochaine question** pour affiner si nécessaire
""")
    user = HumanMessage(content=f"Profil: {profile}\n\nScores: {scored}")
    return llm.invoke([sys, user]).content
