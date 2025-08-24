# -*- coding: utf-8 -*-
"""
ChatFlow entièrement refondu :
- 1) Extraction slots via LLM (profil utilisateur)
- 2) Questions ciblées si info critique manquante
- 3) RAG: récupère extraits CGA/primes (canton/âge/sexe)
- 4) Parsing LLM des PRODUITS + PRESTATIONS à partir des extraits
- 5) Exclusions dures (garde-fou)
- 6) Scoring LLM explicable /20 sur critères détaillés
- 7) Sélection TOP-3 + réponse flexible (tableau + justifs + prochaines questions)

Dépendances internes (déjà données dans les messages précédents) :
- slot_filling.py, question_engine.py, rules_engine.py, scoring_engine.py,
- response_generator.py, rag_utils.py
"""

from typing import List, Dict, Any
from langchain.memory import ConversationBufferMemory
from chatbot.intent_classifier import classify_intent
from chatbot.slot_filling import SLOT_SCHEMA, extract_slots_with_llm, merge_profiles
from chatbot.question_engine import next_question
from chatbot.rules_engine import hard_exclusions
from chatbot.scoring_engine import score_products_with_llm
from chatbot.response_generator import generate_final_answer
from chatbot.rag_utils import retrieve_snippets, product_candidates_from_docs
from chatbot.llm_interface import get_llm_extraction


REQUIRED_SLOTS_MIN = ["age", "sexe", "canton"]  # minimum pour tarifer/filtrer


class ChatFlow:
    """
    Orchestrateur de conversation + RAG + scoring.
    Utilisation : cf. app.py qui instancie ChatFlow(qa_chain)
    """

    def __init__(self, qa_chain):
        # qa_chain est un objet simple avec .retriever (voir llm_interface.build_qa_chain)
        self.qa_chain = qa_chain
        self.profile: Dict[str, Any] = {k: v for k, v in SLOT_SCHEMA.items()}
        self.memory = ConversationBufferMemory(return_messages=True)

    # ---------- Étapes principales ----------

    def _need_core_info(self) -> bool:
        for k in REQUIRED_SLOTS_MIN:
            v = self.profile.get(k)
            if v in (None, "", []):
                return True
        return False

    def _collect_rag_snippets(self, user_input: str) -> List[str]:
        """Construit plusieurs requêtes focalisées (prix/conditions/prestations) et agrège les extraits RAG."""
        sexe = self.profile.get("sexe") or ""
        canton = self.profile.get("canton") or ""
        age = self.profile.get("age")
        age_str = f"{age} ans" if age else ""

        # Intent légère pour orienter les requêtes
        intent = classify_intent(user_input)

        queries = [
            f"Tarifs par âge {age_str} et sexe {sexe} dans le canton {canton}",
            f"Conditions générales (CGA) helsana prestations, limitations, fréquences dans {canton}",
            f"Prestations lunettes dentaires médecines alternatives {canton} {sexe} {age_str}",
            f"Hospitalisation privée semi-privée éligibilité refus {canton} {sexe} {age_str}",
        ]

        # Renfort selon intent
        if intent in ("tableau", "comparaison", "prix"):
            queries.append(f"Tableaux primes produits par tranche d'âge {age_str} {sexe} {canton}")
        if intent in ("details", "conditions"):
            queries.append(f"Détails CGA: exclusions, plafonds, franchises complémentaires {canton}")

        snippets: List[str] = []
        for q in queries:
            snippets.extend(retrieve_snippets(self.qa_chain.retriever, q, k=4))

        # Fallback générique si rien
        if not snippets:
            snippets.extend(retrieve_snippets(self.qa_chain.retriever, "Helsana produits conditions primes", k=8))

        # Déduplication légère
        seen = set()
        uniq = []
        for s in snippets:
            if s not in seen:
                uniq.append(s)
                seen.add(s)
        return uniq[:20]

    def _parse_products_and_prestations(self, snippets: List[str]) -> List[Dict[str, Any]]:
        """
        Utilise un LLM d'extraction pour transformer des extraits RAG (CGA/primes)
        en une liste de produits structurés avec prestations détectées.
        """
        if not snippets:
            return []

        llm = get_llm_extraction()
        # On fournit une taxonomie simple de prestations pour aider le modèle
        taxonomy = [
            "lunettes/lentilles", "soins dentaires", "médecines alternatives",
            "hospitalisation privée", "hospitalisation semi-privée", "hospitalisation générale",
            "kiné/ostéopathie", "psychologie/psychothérapie", "soins à domicile",
            "voyage/rapatriement", "appareillages/prothèses", "perte de gain", "capital décès"
        ]

        system = (
            "Tu es un extracteur structuré. À partir d'extraits CGA/primes suisses, "
            "identifie les produits d'assurance et liste leurs prestations clés. "
            "Retourne uniquement un JSON valide (liste)."
        )
        user = f"""
Profil utilisateur (contexte) :
{self.profile}

Taxonomie de prestations (à utiliser si possible) :
{taxonomy}

Extraits RAG (texte brut) :
{snippets[:12]}

Format JSON attendu (liste) :
[
  {{
    "nom": "Nom produit (avec variation si présente)",
    "type": "catégorie simplifiée (ex: hospitalisation, dentaires, lunettes, voyage, med_naturelles, etc.)",
    "prestations": [
      {{"label": "lunettes/lentilles", "details": "plafond, fréquence, conditions"}},
      {{"label": "soins dentaires", "details": "plafonds, exclusions"}}
    ],
    "conditions_importantes": ["carence 6 mois", "plafond 300 CHF/an", "limité aux opticiens partenaires"],
    "eligibilite": ["sexe F/M", "âges 26-30 ok", "refus après 55 ans le cas échéant"]
  }}
]
IMPORTANT :
- Concentre-toi sur ce qui est explicitement lisible ou fortement implicite dans les extraits.
- N'invente pas de produits ; si incertain, omets.
- N'ajoute pas de texte hors JSON.
"""
        resp = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}]).content
        try:
            products = __import__("json").loads(resp)
        except Exception:
            # Fallback simple si parsing échoue : utiliser les entêtes "### Produit (Variation)" détectés
            products = product_candidates_from_docs(snippets)
            # homogénéiser
            for p in products:
                p.setdefault("prestations", [])
                p.setdefault("conditions_importantes", [])
                p.setdefault("eligibilite", [])
        return products

    # ---------- API principale ----------

    def handle_message(self, user_input: str) -> str:
        # 1) Historique utilisateur
        self.memory.chat_memory.add_user_message(user_input)

        # 2) Extraction slots (profil) via LLM
        patch = extract_slots_with_llm(user_input, self.profile)
        self.profile = merge_profiles(self.profile, patch)

        # 3) Si infos minimales manquent → question ciblée
        q = next_question(self.profile)
        if q:
            self.memory.chat_memory.add_ai_message(q)
            return q

        # 4) Réglage dynamique du retriever par canton si supporté
        if "canton" in self.profile and hasattr(self.qa_chain.retriever, "set_canton"):
            self.qa_chain.retriever.set_canton(self.profile["canton"])

        # 5) RAG: récupérer extraits
        snippets = self._collect_rag_snippets(user_input)

        # 6) Parsing LLM : produits + prestations
        products = self._parse_products_and_prestations(snippets)

        # 7) Exclusions dures (garde-fou) : on marque les exclusions au lieu de supprimer
        products = hard_exclusions(self.profile, products)

        # 8) Scoring explicable /20 (profil + extraits)
        scored = score_products_with_llm(self.profile, products, snippets)

        # 9) Top-3 (ou Top-2 si peu de candidats)
        scored_sorted = sorted(scored, key=lambda x: x.get("score_total", 0), reverse=True)
        top_n = scored_sorted[:3] if len(scored_sorted) >= 3 else scored_sorted[:2]
        # On garde les prestations extraites pour l'affichage final (si available)
        # On merge par nom de produit
        products_by_name = {p.get("nom") or p.get("produit"): p for p in products}
        for item in top_n:
            pkey = item.get("produit")
            if pkey in products_by_name:
                item["prestations"] = products_by_name[pkey].get("prestations", [])
                item["conditions_importantes"] = products_by_name[pkey].get("conditions_importantes", [])
                item["eligibilite"] = products_by_name[pkey].get("eligibilite", [])

        # 10) Génération de la réponse finale (flexible : analyse, tableau, justifs)
        final = generate_final_answer(self.profile, top_n)
        final += "\n\n🔎 Sources : extraits indexés (CGA, tableaux de primes) via RAG Helsana."

        # 11) Mémoire
        self.memory.chat_memory.add_ai_message(final)
        return final
