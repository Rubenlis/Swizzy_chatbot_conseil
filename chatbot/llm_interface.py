# -*- coding: utf-8 -*-
"""
Interface LLM minimale et rétrocompatible.
- expose des constructeurs de modèles (gpt-4o / gpt-4o-mini)
- fournit build_qa_chain(retriever) qui retourne un objet avec attribut .retriever
  pour rester compatible avec app.py et le reste du projet.
"""

from types import SimpleNamespace
from langchain_openai import ChatOpenAI


def get_llm_generation(temperature: float = 0.0):
    """LLM pour la génération (explications, réponses longues)."""
    return ChatOpenAI(model_name="gpt-4o", temperature=temperature)


def get_llm_extraction():
    """LLM pour extraction/structuration (slots, parsing prestations)."""
    return ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


def build_qa_chain(retriever):
    """
    Rétrocompatibilité avec l'ancien code : renvoie un objet très simple
    qui expose .retriever (utilisé par ChatFlow). Pas de chaîne combinatoire.
    """
    return SimpleNamespace(retriever=retriever)
