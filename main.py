import json
from chatbot.slot_filling import extract_slots, merge_profiles, SLOT_SCHEMA
from chatbot.rules_engine import apply_rules
from chatbot.response_generator import generate_response

# Chargement des produits disponibles
with open("data/assurances.json", "r", encoding="utf-8") as f:
    assurances = json.load(f)

profil = SLOT_SCHEMA.copy()

# Exemple de conversation
message1 = "J’ai 32 ans, je voyage souvent et je suis indépendant."
slots1 = extract_slots(message1, profil)
profil = merge_profiles(profil, slots1)

message2 = "Je préfère payer un peu plus cher pour être tranquille."
slots2 = extract_slots(message2, profil)
profil = merge_profiles(profil, slots2)

# Application des règles métier
recommandations = apply_rules(profil, assurances)

# Génération de la réponse utilisateur
reponse = generate_response(profil, recommandations)

print("=== Réponse chatbot ===")
print(reponse)
