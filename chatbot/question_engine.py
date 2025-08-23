from typing import Optional, Dict, Any

QUESTIONS = {
    "age": "Quel âge avez-vous ?",
    "sexe": "Êtes-vous un homme ou une femme ?",
    "canton": "Dans quel canton résidez-vous ?",
    "situation_familiale": "Pouvez-vous préciser votre situation familiale ? (célibataire, couple, avec enfants)",
    "activite_pro": "Quelle est votre activité professionnelle ? (salarié, indépendant, retraité)",
    "franchise": "Préférez-vous une franchise basse (sécurité) ou haute (primes plus basses) ?",
    "etat_sante": "Comment décririez-vous votre état de santé ? (bonne santé, fragile)",
    "budget": "Quel est votre budget mensuel pour l’assurance ? (faible, moyen, élevé)",
    "tol_risque": "Préférez-vous payer une prime plus élevée pour éviter les surprises, ou accepter plus de risque pour payer moins ?",
    "objectifs": "Quelles sont vos priorités ? (économie, confort, sécurité, lunettes, dentaire, médecines naturelles)"
}

CORE_MINIMUM = ["age", "sexe", "canton", "budget", "franchise"]

def next_question(profile: Dict[str, Any]) -> Optional[str]:
    # On priorise le “core minimum” pour pouvoir tarifer/filtrer rapidement
    for k in CORE_MINIMUM:
        v = profile.get(k)
        if v in (None, "", []):
            return QUESTIONS[k]
    # Puis le reste
    for k, q in QUESTIONS.items():
        v = profile.get(k)
        if v in (None, "", []):
            return q
    return None
