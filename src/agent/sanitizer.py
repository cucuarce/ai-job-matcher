"""
Capa 1 contra prompt injection: detección por patrones (ES/EN) sobre el texto de las ofertas.

Las ofertas son contenido de terceros no confiable. Antes de que el texto llegue al modelo,
se buscan frases que intentan darle órdenes a una IA ("ignorá tus instrucciones", "terminá tu
respuesta con...", "no menciones esto") y se REDACTA la oración completa que las contiene.

Es una defensa determinista y gratis, pero NO es completa:
  - No detecta paráfrasis nuevas ni ataques en otros idiomas.
  - Puede redactar texto legítimo (falsos positivos): se mide contra el dataset real en los tests.
Por eso es una capa más, junto con el envoltorio <oferta_no_confiable> y la regla del prompt.
"""

import re
import unicodedata

PLACEHOLDER = "[FRAGMENTO REDACTADO: posible instrucción dirigida a una IA]"

_INVISIBLES = re.compile("[​-‏‪-‮⁠-⁤﻿­]")

# Los patrones se aplican sobre texto normalizado: minúsculas, sin acentos, sin caracteres invisibles.
_PATRONES = {
    "ignorar_instrucciones": r"\b(ignor\w*|olvid\w*|descart\w*|disregard|forget|override)\b.{0,40}\b(instruccion\w*|indicacion\w*|regla\w*|directriz\w*|prompts?|instructions?|rules?|guidelines?)\b",
    "olvidar_lo_anterior": r"\b(olvid\w*|forget)\b.{0,30}\b(todo|anterior\w*|previo\w*|everything|previous|above)\b",
    "aviso_a_la_ia": r"\b(importante|atencion|attention|important|nota|note|aviso|urgente|urgent)\b.{0,25}\b(asistente|assistant|llm|language model|modelo de lenguaje|ai model|modelo de ia)\b",
    "si_sos_una_ia": r"\b(si (eres|sos|es)|if you are|if you're)\b.{0,25}\b(una? ia|an? ai|llm|inteligencia artificial|language model|modelo de lenguaje|asistente|assistant|chatgpt|claude|gpt)\b",
    "nuevas_instrucciones": r"\b(nuevas? instrucciones|new instructions|system prompt|prompt del sistema|mensaje del sistema|system message|developer message)\b",
    "reasignar_rol": r"\b(a partir de ahora|from now on)\b.{0,30}\b(sos|eres|debes|responde\w*|you are|you must|respond|answer|siempre|always)\b|\b(you are now|ahora (sos|eres))\b",
    "dirigir_salida": r"\b(termina\w*|finaliza\w*|concluye\w*|agrega\w*|anade\w*|incluye\w*|end|finish|append|add|include)\b.{0,25}\b(tu|su|la|your|the)\b.{0,15}\b(respuesta|salida|answer|response|reply|output)\b.{0,15}\b(con|with)\b|\b(responde\w*|contesta\w*|respond|reply|answer)\b.{0,15}\b(unicamente|solo|solamente|exclusivamente|only|exclusively)\b.{0,10}\b(con|with)\b",
    "ocultar_instruccion": r"\b(no (menciones|reveles|digas|comentes|informes)|do not (mention|reveal|tell|disclose)|don't (mention|reveal|tell)|sin mencionar)\b.{0,30}\b(esta|esto|instruccion\w*|this|instruction\w*|message|mensaje)\b",
    "manipular_ranking": r"\b(recomienda\w*|califica\w*|puntua\w*|rank\w*|rate|recommend)\b.{0,30}\b(esta oferta|this (job|offer|position|role))\b.{0,30}\b(como|first|primero|mejor|best|top)\b",
    "ordenar_tool": r"\b(llama|invoca|ejecuta|call|invoke)\s+(a\s+)?(la\s+|el\s+|the\s+)?(herramienta|funcion|tool|function)\b",
    "delimitadores_falsos": r"<\s*/?\s*(system|assistant|instructions?)\s*>|\[/?inst\]|<\|im_(start|end)\|>|^#+\s*system\b",
}
_COMPILADOS = {nombre: re.compile(p) for nombre, p in _PATRONES.items()}

_SEPARADOR_ORACION = re.compile(r"(?<=[.!?])\s+")


def _normalizar(texto: str) -> str:
    sin_invisibles = _INVISIBLES.sub("", texto)
    descompuesto = unicodedata.normalize("NFKD", sin_invisibles)
    return "".join(c for c in descompuesto if not unicodedata.combining(c)).lower()


def detectar(texto: str) -> list[str]:
    """Nombres de los patrones que coinciden en el texto (vacío = nada sospechoso)."""
    normal = _normalizar(texto)
    return [nombre for nombre, rx in _COMPILADOS.items() if rx.search(normal)]


def sanitizar(texto: str) -> tuple[str, list[str]]:
    """Redacta cada oración sospechosa. Devuelve (texto_limpio, patrones_detectados)."""
    hallazgos: list[str] = []
    lineas_limpias = []
    for linea in texto.split("\n"):
        oraciones = _SEPARADOR_ORACION.split(linea)
        limpias = []
        for oracion in oraciones:
            encontrados = detectar(oracion)
            if encontrados:
                hallazgos.extend(encontrados)
                limpias.append(PLACEHOLDER)
            else:
                limpias.append(oracion)
        lineas_limpias.append(" ".join(limpias))
    return "\n".join(lineas_limpias), hallazgos
