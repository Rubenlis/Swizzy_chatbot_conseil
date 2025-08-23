import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Schéma de slots centralisé
SLOT_SCHEMA: Dict[str, Any] = {
    "age": None,                       # int
    "sexe": None,                      # "homme" | "femme"
    "canton": None,                    # Nom officiel (Vaud, Genève...)
    "situation_familiale": None,       # "célibataire" | "couple" | "avec enfants"
    "activite_pro": None,              # "salarié" | "indépendant" | "retraité"
    "franchise": None,                 # "basse" | "haute"
    "etat_sante": None,                # "bonne santé" | "fragile"
    "habitudes": [],                   # ["sportif","voyageur","casanier"]
    "budget": None,                    # "faible" | "moyen" | "élevé"
    "tol_risque": None,                # "accepte surprises" | "refuse surprises"
    "objectifs": []                    # ["économie","confort","sécurité","lunettes","dentaire","médecines naturelles"]
}

NORMALISATIONS = {
    "sexe": {"h": "homme", "f": "femme"},
    "franchise": {"basse": "basse", "haute": "haute"},
    "budget": {"faible": "faible", "moyen": "moyen", "élevé": "élevé", "eleve": "élevé"},
}

def _normalize(profil_patch: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in profil_patch.items():
        if v is None:
            continue
        if k in ("habitudes", "objectifs") and isinstance(v, list):
            out[k] = list({s.lower().strip() for s in v if isinstance(s, str) and s.strip()})
            continue
        if isinstance(v, str):
            v_norm = v.lower().strip()
            if k in NORMALISATIONS and v_norm in NORMALISATIONS[k]:
                out[k] = NORMALISATIONS[k][v_norm]
            else:
                out[k] = v_norm
        else:
            out[k] = v
    return out

def merge_profiles(current: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = {**current}
    for k, v in patch.items():
        if v in (None, "", []):
            continue
        merged[k] = v
    return merged

def extract_slots_with_llm(utterance: str, current_profile: Dict[str, Any]) -> Dict[str, Any]:
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    sys = SystemMessage(content=f"""
Tu es un extracteur de profil d’assurance en Suisse.
Rends UNIQUEMENT un JSON valide avec les clés du schéma ci-dessous. Si une info n’est pas présente, mets null.
Normalise les valeurs dans les vocabulaires suivants :
- sexe: "homme" | "femme"
- franchise: "basse" | "haute"
- budget: "faible" | "moyen" | "élevé"
- habitudes: sous-ensemble de ["sportif","voyageur","casanier"]
- tol_risque: "accepte surprises" | "refuse surprises"

Schéma:
{json.dumps(SLOT_SCHEMA, ensure_ascii=False, indent=2)}
""")
    user = HumanMessage(content=f"Message: {utterance}\nProfil actuel (contexte): {json.dumps(current_profile, ensure_ascii=False)}")
    resp = llm.invoke([sys, user]).content
    try:
        data = json.loads(resp)
    except Exception:
        # fallback minimal
        return {}
    return _normalize(data)
