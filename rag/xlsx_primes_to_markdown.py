from langchain.schema import Document
import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AGE_COLUMNS = [
    "00-05", "06-10", "11-15", "16-20", "21-25", "26-30", "31-35",
    "36-40", "41-45", "46-50", "51-55", "56-60", "61-65"
]

def expand_age_range(age_range: str):
    """Convertit une tranche d'âge '06-10' en liste d'entiers [6,7,8,9,10]."""
    if "-" in age_range:
        start, end = age_range.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(age_range)]


def xlsx_to_documents(xlsx_path: str):
    """Lit un Excel et le convertit en liste de Documents LangChain enrichis en métadonnées."""
    df = pd.read_excel(xlsx_path)
    rows = []

    for _, row in df.iterrows():
        try:
            canton_code = row.get("Canton")
            canton_nom = row.get("Canton Nom")
            canton = str(canton_nom).strip() if pd.notna(canton_nom) else str(canton_code).strip()

            produit = str(row.get("Produit", "-")).strip()
            variation = str(row.get("Variation", "Standard")).strip()
            sexe = str(row.get("Sexe", "F/M")).strip()
            categorie = str(row.get("Catégorie", "Assurance maladie")).strip()

            for age_col in AGE_COLUMNS:
                prix = row.get(age_col)
                if pd.isna(prix) or prix == 0:
                    continue

                prix_str = str(prix).replace(",", ".")
                ages_list = expand_age_range(age_col)
                description = (
                    f"Dans le canton {canton}, pour le produit '{produit}' ({categorie})"
                    f"{' - ' + variation if variation else ''}, sexe {sexe}, "
                    f"la prime mensuelle est de {prix_str} CHF pour les âges {age_col} ans "
                    f"(couvrant {', '.join(str(a) for a in ages_list)} ans)."
                )

                rows.append({
                    "canton": canton,
                    "categorie": categorie,
                    "produit": produit,
                    "variation": variation,
                    "sexe": sexe,
                    "age": age_col,
                    "prix": prix_str,
                    "description": description
                })
        except Exception as e:
            logger.warning(f"Erreur traitement ligne Excel: {e}")
            continue

    # Regroupement par produit/variation/sexe/canton
    grouped_chunks = {}
    resume_data = []

    for r in rows:
        key = (r["canton"], r["produit"], r["variation"], r["sexe"])
        if key not in grouped_chunks:
            grouped_chunks[key] = []
        grouped_chunks[key].append(r)
        resume_data.append((r["produit"], r["canton"], r["variation"]))

    docs = []
    for (canton, produit, variation, sexe), group in grouped_chunks.items():
        chunk_text = f"### {produit} ({variation})\n"
        chunk_text += f"Canton: {canton} | Sexe: {sexe} | Catégorie: {group[0]['categorie']}\n\n"
        for r in group:
            chunk_text += f"- {r['description']}\n"
        chunk_text += "\n_Source: tableau officiel Helsana_"

        # 🔥 Ici on enrichit les métadonnées avec canton, produit, variation, sexe
        docs.append(
            Document(
                page_content=chunk_text,
                metadata={
                    "source": os.path.basename(xlsx_path),
                    "canton": canton,
                    "produit": produit,
                    "variation": variation,
                    "sexe": sexe
                }
            )
        )

    # Résumé global
    if resume_data:
        resume_text = "# Résumé des produits et disponibilités par canton\n\n"
        for produit, canton, variation in sorted(set(resume_data)):
            resume_text += f"- **{produit}** ({variation}) — {canton}\n"
        docs.append(
            Document(
                page_content=resume_text,
                metadata={"source": "resume_produits.md"}
            )
        )

    return docs


def save_chunks_as_files(xlsx_path: str, output_dir: str = "data/chunks"):
    """Sauvegarde les Documents en fichiers Markdown dans output_dir."""
    docs = xlsx_to_documents(xlsx_path)
    if not docs:
        logger.warning("Aucune donnée trouvée dans l'Excel.")
        return
    os.makedirs(output_dir, exist_ok=True)
    for i, doc in enumerate(docs, start=1):
        file_name = "resume_produits.md" if doc.page_content.startswith("# Résumé") else f"chunk_{i:05d}.md"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc.page_content)
    logger.info(f"{len(docs)} fichiers sauvegardés dans {output_dir}")
    return docs  # 👈 retourne les documents pour ingestion directe
