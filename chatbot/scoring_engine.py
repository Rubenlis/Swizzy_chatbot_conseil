import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

CRITERES_DETAILLES = {
    "Adéquation à la situation familiale":
        "Mesure si le produit est utile compte tenu de la situation familiale. "
        "0 = totalement inutile (ex: capital décès élevé pour un célibataire), "
        "20 = parfaitement adapté (ex: hospitalisation renforcée pour un couple avec enfants).",

    "Compatibilité avec l’âge":
        "Mesure si le produit est accessible/pertinent selon l’âge. "
        "0 = produit inadapté ou refusé (ex: hospitalisation privée après 55 ans), "
        "20 = produit parfaitement adapté à l’âge (ex: soins dentaires pour jeunes adultes).",

    "Pertinence selon l’activité professionnelle":
        "Mesure la valeur ajoutée selon la profession. "
        "0 = aucun intérêt (ex: perte de gain pour un retraité), "
        "20 = indispensable (ex: protection revenu pour un indépendant).",

    "Adaptation à l’état de santé":
        "Mesure si le produit couvre bien l’état de santé de l’utilisateur. "
        "0 = inadapté ou refus probable (ex: médecines naturelles pour santé fragile), "
        "20 = couverture optimale (ex: soins à domicile pour santé fragile).",

    "Cohérence avec le budget":
        "Mesure la compatibilité avec le budget déclaré. "
        "0 = produit trop cher, irréaliste (hors budget), "
        "20 = produit parfaitement en ligne avec les moyens financiers de l’utilisateur.",

    "Adéquation au style de vie / habitudes":
        "Mesure si le produit correspond au mode de vie. "
        "0 = pas du tout adapté (ex: assurance voyage pour casanier), "
        "20 = totalement pertinent (ex: assurance voyage pour grand voyageur).",

    "Alignement avec les objectifs personnels":
        "Mesure si le produit aide à atteindre les objectifs de confort, sécurité ou économie. "
        "0 = aucun alignement (ex: confort recherché mais produit très basique), "
        "20 = correspond exactement aux objectifs (ex: chambre privée pour confort).",

    "Tolérance au risque financier":
        "Mesure l’adéquation entre le produit et la tolérance au risque. "
        "0 = totalement contradictoire (ex: franchise très haute pour profil refusant les surprises), "
        "20 = parfaitement aligné (ex: franchise haute pour profil acceptant le risque)."
}


def score_products_with_llm(profile: Dict[str, Any],
                            products: List[Dict[str, Any]],
                            retrieved_snippets: List[str]) -> List[Dict[str, Any]]:
    """
    Scoring explicable par LLM. On passe:
    - le profil,
    - la liste de produits (nom, type, tags…),
    - les extraits RAG (conditions, CGA, prix/canton/âge/sexe).
    Le LLM attribue une note /20 par critère + justification, et une moyenne finale.
    """
    if not products:
        return []

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    sys = SystemMessage(content=f"""
Tu es un expert-conseiller en assurance santé en Suisse.
Tu dois évaluer chaque produit à l'aide d'extraits fiables (CGA, prix) fournis.
Applique STRICTEMENT l’échelle 0–20 décrite par critère ci-dessous.
Calcule "score_total" = moyenne arithmétique des notes (arrondie à l'entier).
Retourne uniquement un JSON valide (liste), pas de texte libre.

Critères et échelles (0=très inadapté / 20=parfaitement adapté) :
{json.dumps(CRITERES_DETAILLES, ensure_ascii=False, indent=2)}
""")

    payload = {
        "profil": profile,
        "produits": products,
        "extraits_RAG": retrieved_snippets[:12]  # limite pour rester concis
    }
    user = HumanMessage(content=f"""
Voici le contexte JSON à évaluer :
{json.dumps(payload, ensure_ascii=False, indent=2)}

Structure attendue :
[
  {{
    "produit": "Nom",
    "scores": [
      {{"critere":"Adéquation à la situation familiale","note":0,"justification":"..."}},
      ...
    ],
    "score_total": 0
  }}
]
""")
    resp = llm.invoke([sys, user]).content
    try:
        data = json.loads(resp)
        # Sécurité: borne 0..20
        for item in data:
            for s in item.get("scores", []):
                try:
                    s["note"] = max(0, min(20, int(round(float(s.get("note", 0))))))
                except Exception:
                    s["note"] = 0
            try:
                item["score_total"] = max(0, min(20, int(round(float(item.get("score_total", 0))))))
            except Exception:
                item["score_total"] = 0
        return data
    except Exception:
        # En cas d'erreur parsing, on renvoie une structure "vide"
        return [{"produit": p.get("nom","?"), "scores": [], "score_total": 0, "parse_error": True} for p in products]