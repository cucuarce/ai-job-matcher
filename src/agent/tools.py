"""
Herramientas del agente: buscar_ofertas y leer_oferta.

El texto de las ofertas es contenido externo NO confiable. Toda salida de
herramienta que incluya ese texto se envuelve en <oferta_no_confiable> para
que el modelo lo trate como datos, nunca como instrucciones.
"""

import json
from pathlib import Path

import faiss
import numpy as np

from src.agent.sanitizer import sanitizar
from src.embeddings.buscar import obtener_embedding

RAIZ = Path(__file__).resolve().parents[2]
RUTA_INDICE = RAIZ / "data" / "processed" / "ofertas.index"
RUTA_METADATA = RAIZ / "data" / "processed" / "ofertas_metadata.json"
RUTA_OFERTAS = RAIZ / "data" / "raw" / "ofertas.json"

MAX_CHARS_OFERTA = 3000
MAX_TOP_K = 10
SNIPPET_CHARS = 400

TOOLS = [
    {"type": "function", "function": {
        "name": "buscar_ofertas",
        "description": "Búsqueda semántica: devuelve las ofertas de trabajo más similares en significado "
                       "a un texto (por ejemplo un perfil de habilidades). Devuelve id, score y un fragmento inicial de cada oferta.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Texto a buscar, en lenguaje natural."},
            "top_k": {"type": "integer", "description": "Cantidad de resultados (1-10). Por defecto 5."},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "leer_oferta",
        "description": "Devuelve el texto completo de una oferta por su id, para analizarla en detalle.",
        "parameters": {"type": "object", "properties": {
            "id": {"type": "integer", "description": "Id de la oferta (el campo 'id' de buscar_ofertas)."},
        }, "required": ["id"]},
    }},
]


def cargar_ofertas() -> list[dict]:
    return json.loads(RUTA_OFERTAS.read_text(encoding="utf-8"))


def envolver_no_confiable(texto: str, oferta_id: int) -> str:
    """Sanitiza (capa 1) y envuelve (capa 2) el texto de una oferta antes de dárselo al modelo."""
    # Neutralizamos cierres falsos de la etiqueta que vengan dentro del texto de la oferta.
    limpio = texto.replace("</oferta_no_confiable>", "[etiqueta eliminada]")
    limpio, _ = sanitizar(limpio)
    return f'<oferta_no_confiable id="{oferta_id}">\n{limpio}\n</oferta_no_confiable>'


def buscar_ofertas(query: str, top_k: int = 5) -> str:
    top_k = max(1, min(int(top_k), MAX_TOP_K))
    indice = faiss.read_index(str(RUTA_INDICE))
    metadata = json.loads(RUTA_METADATA.read_text(encoding="utf-8"))

    vec = np.array([obtener_embedding(query)], dtype="float32")
    faiss.normalize_L2(vec)
    scores, indices = indice.search(vec, top_k)

    textos = {o["id"]: o["texto_completo"] for o in cargar_ofertas()}
    resultados = [{"id": metadata[i]["id"], "score": round(float(s), 3),
                   "fragmento": envolver_no_confiable(textos[metadata[i]["id"]][:SNIPPET_CHARS],
                                                      metadata[i]["id"])}
                  for s, i in zip(scores[0], indices[0]) if i != -1]
    return json.dumps(resultados, ensure_ascii=False)


def leer_oferta(id: int) -> str:
    for o in cargar_ofertas():
        if o["id"] == int(id):
            return envolver_no_confiable(o["texto_completo"][:MAX_CHARS_OFERTA], o["id"])
    return json.dumps({"error": f"No existe la oferta con id {id}"}, ensure_ascii=False)


FUNCIONES = {"buscar_ofertas": buscar_ofertas, "leer_oferta": leer_oferta}


def ejecutar_tool(nombre: str, argumentos: dict) -> str:
    """Ejecuta una tool; los errores vuelven como texto para que el modelo pueda corregirse."""
    if nombre not in FUNCIONES:
        return json.dumps({"error": f"Tool desconocida: {nombre}. Disponibles: {sorted(FUNCIONES)}"}, ensure_ascii=False)
    if "__invalid_json__" in argumentos:
        return json.dumps({"error": "Los argumentos no eran JSON válido. Reintentá con JSON válido."}, ensure_ascii=False)
    try:
        return FUNCIONES[nombre](**argumentos)
    except TypeError as e:
        return json.dumps({"error": f"Argumentos inválidos: {e}"}, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001 - devolvemos el error al modelo en vez de romper el loop
        return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
