import os
import json
from pathlib import Path
from PyPDF2 import PdfReader
from llm_interface import call_llm  # Ton wrapper LLM existant

# 📑 Schéma minimal : mais on autorise le LLM à ajouter des champs
BASE_SCHEMA = {
    "produit": None,
    "assureur": None,
    "description": None,
    "prestations": {},     # non-exhaustif : le LLM peut en rajouter
    "eligibilité": {},
    "exclusions": []
}

def extract_text_from_pdf(pdf_path):
    """Lit un PDF et retourne tout le texte brut."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, max_chars=4000):
    """Découpe un texte long en morceaux (chunks)."""
    chunks = []
    for i in range(0, len(text), max_chars):
        chunks.append(text[i:i+max_chars])
    return chunks

def query_llm_for_prestations(chunk, produit, assureur):
    """Demande au LLM d'extraire les prestations d'un chunk de CGA."""
    prompt = f"""
    Tu es un expert en assurance maladie suisse.
    Extrait les prestations clés de ce produit d’assurance, au format JSON.

    ⚠️ La liste des prestations n'est PAS exhaustive : 
    - Inclue toutes les prestations que tu trouves dans le texte.
    - Tu peux ajouter des catégories si nécessaires (ex: maternité, prévention, psychothérapie).
    - Si une info n’est pas trouvée, laisse-la vide ou ne la mets pas.

    Format attendu (exemple) :
    {{
      "produit": "{produit}",
      "assureur": "{assureur}",
      "description": "...",
      "prestations": {{
        "hospitalisation": "semi-privée",
        "médecine_alternative": "jusqu'à 1000 CHF/an",
        "lunettes": "200 CHF / 3 ans",
        "dentiste": "75% enfants, max 1000 CHF/an",
        "voyage": "couverture mondiale",
        "maternité": "...",
        "psychothérapie": "...",
        "...": "..."
      }},
      "eligibilité": {{
        "âge_max": 65,
        "restrictions": ["refus si maladie chronique lourde"]
      }},
      "exclusions": [
        "traitements expérimentaux",
        "soins à l’étranger hors urgences"
      ]
    }}

    Texte d'entrée :
    {chunk}
    """
    response = call_llm(prompt, temperature=0.2)
    try:
        return json.loads(response)
    except Exception:
        return {}

def merge_json(base, new_data):
    """Fusionne récursivement deux dictionnaires JSON."""
    for key, value in new_data.items():
        if isinstance(value, dict):
            node = base.setdefault(key, {})
            merge_json(node, value)
        elif isinstance(value, list):
            base.setdefault(key, [])
            base[key] = list(set(base[key] + value))  # fusion unique
        else:
            if key not in base or not base[key]:
                base[key] = value
    return base

def extract_prestations(pdf_path, output_path, produit, assureur):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    result = BASE_SCHEMA.copy()
    result["produit"] = produit
    result["assureur"] = assureur

    for chunk in chunks:
        json_chunk = query_llm_for_prestations(chunk, produit, assureur)
        result = merge_json(result, json_chunk)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Prestations extraites et sauvegardées dans {output_path}")

def process_all_assureurs(base_dir="../data"):
    for assureur in os.listdir(base_dir):
        assureur_path = os.path.join(base_dir, assureur, "produits")
        if not os.path.exists(assureur_path):
            continue

        for produit in os.listdir(assureur_path):
            produit_path = os.path.join(assureur_path, produit)
            pdf_path = os.path.join(produit_path, "CGA.pdf")
            output_path = os.path.join(produit_path, "prestations.json")

            if not os.path.exists(pdf_path):
                print(f"⚠️ Pas de CGA trouvé pour {produit}")
                continue

            print(f"📄 Extraction pour {assureur}/{produit}...")
            extract_prestations(pdf_path, output_path, produit, assureur)



if __name__ == "__main__":
    process_all_assureurs("../data")
