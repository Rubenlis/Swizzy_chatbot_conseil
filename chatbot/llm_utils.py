import json
from openai import OpenAI

client = OpenAI()

def call_llm(messages, model="gpt-4o-mini", temperature=0):
    """Appel générique au LLM"""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content
