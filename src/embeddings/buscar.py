"""
Paso 3: Búsqueda semántica sobre las 98 ofertas.
Le pasás un texto (tu perfil, o una oferta nueva) y te devuelve
las N ofertas más similares por significado, no por keyword matching.
"""

import json
import requests
import numpy as np
import faiss

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODELO = "nomic-embed-text"

RUTA_INDICE = "data/processed/ofertas.index"
RUTA_METADATA = "data/processed/ofertas_metadata.json"


def obtener_embedding(texto: str) -> list[float]:
    resp = requests.post(OLLAMA_URL, json={"model": MODELO, "prompt": texto})
    resp.raise_for_status()
    return resp.json()["embedding"]


def buscar(query: str, top_k: int = 5):
    indice = faiss.read_index(RUTA_INDICE)
    with open(RUTA_METADATA, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    vec = np.array([obtener_embedding(query)], dtype="float32")
    faiss.normalize_L2(vec)

    scores, indices = indice.search(vec, top_k)

    print(f"\nTop {top_k} ofertas más similares a tu búsqueda:\n")
    for score, idx in zip(scores[0], indices[0]):
        m = metadata[idx]
        print(f"  score={score:.3f}  |  oferta #{m['id']} ({m['origen']})")
        print(f"    {m['titulo'][:100]}")
        print()


if __name__ == "__main__":
    # Ejemplo: buscar con tu propio perfil de skills como query
    MI_PERFIL_TEXTO = """
    Ingeniero con experiencia en desarrollo backend, construcción de agentes
    de IA con tool calling, automatización de procesos con n8n, integración
    de APIs REST, y despliegue de aplicaciones en Supabase y Vercel.
    """

    buscar(MI_PERFIL_TEXTO, top_k=5)
