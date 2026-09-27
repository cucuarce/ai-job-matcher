"""
Paso 2: Generar embeddings de las 98 ofertas usando Ollama local,
y armar un índice FAISS para búsqueda semántica.

Requisitos antes de correr esto:
    ollama pull nomic-embed-text
    pip install faiss-cpu requests numpy

Ollama debe estar corriendo (ollama serve, o la app de escritorio activa).
"""

import json
import requests
import numpy as np
import faiss

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODELO = "nomic-embed-text"

RUTA_OFERTAS = "data/raw/ofertas.json"
RUTA_INDICE = "data/processed/ofertas.index"
RUTA_METADATA = "data/processed/ofertas_metadata.json"


def obtener_embedding(texto: str) -> list[float]:
    resp = requests.post(OLLAMA_URL, json={"model": MODELO, "prompt": texto})
    resp.raise_for_status()
    return resp.json()["embedding"]


def main():
    with open(RUTA_OFERTAS, "r", encoding="utf-8") as f:
        ofertas = json.load(f)

    print(f"Generando embeddings para {len(ofertas)} ofertas con {MODELO}...")

    embeddings = []
    metadata = []

    for i, oferta in enumerate(ofertas, 1):
        # Truncamos a un largo razonable: el modelo de embeddings tiene límite
        # de contexto, y para similitud semántica no necesitamos el texto entero.
        texto = oferta["texto_completo"][:2000]

        try:
            vec = obtener_embedding(texto)
            embeddings.append(vec)
            metadata.append({
                "id": oferta["id"],
                "origen": oferta["origen"],
                "titulo": oferta["titulo"]
            })
            print(f"  [{i}/{len(ofertas)}] oferta #{oferta['id']} OK (dim={len(vec)})")
        except Exception as e:
            print(f"  [{i}/{len(ofertas)}] oferta #{oferta['id']} FALLÓ: {e}")

    embeddings_np = np.array(embeddings, dtype="float32")

    # Índice FAISS simple: distancia L2 sobre vectores normalizados
    # (equivalente a similitud de coseno cuando los vectores están normalizados)
    faiss.normalize_L2(embeddings_np)
    dimension = embeddings_np.shape[1]
    indice = faiss.IndexFlatIP(dimension)  # Inner Product = coseno si está normalizado
    indice.add(embeddings_np)

    faiss.write_index(indice, RUTA_INDICE)
    with open(RUTA_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nÍndice FAISS guardado en {RUTA_INDICE}")
    print(f"Metadata guardada en {RUTA_METADATA}")
    print(f"Dimensión de los embeddings: {dimension}")


if __name__ == "__main__":
    main()
