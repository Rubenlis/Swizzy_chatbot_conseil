from typing import List, Dict, Any

def hard_exclusions(profile: Dict[str, Any], products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filtre dur avant scoring LLM (règles non négociables).
    On suppose que chaque produit dict possède au moins {"nom": str, "type": str}
    """
    age = profile.get("age")
    budget = profile.get("budget")
    etat = profile.get("etat_sante")

    filtered = []
    for p in products:
        t = (p.get("type") or "").lower()
        motifs = []

        if isinstance(age, int) and age >= 55 and t in ("semi_privée", "semi-privee", "privée", "privee"):
            motifs.append("Souvent refusé après 55 ans")

        if budget == "faible" and t in ("privée", "privee"):
            motifs.append("Chambre privée hors budget faible")

        if etat == "fragile" and t in ("médecines naturelles", "medecines naturelles"):
            motifs.append("Souvent inadapté/Refus probable profil santé fragile")

        if motifs:
            # On garde mais on marque “exclu” pour la transparence vis-à-vis du LLM/rendu utilisateur
            p = {**p, "_exclu_dur": True, "_motifs": motifs}
        filtered.append(p)
    return filtered

### GARDE FOU