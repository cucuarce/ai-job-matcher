"""
Observabilidad: cada corrida del agente deja una traza estructurada (JSONL).

Se registra QUÉ decidió el agente (qué tools llamó, con qué argumentos, cuánto
tardó, si la guarda intervino), no el contenido de las ofertas: es texto de
terceros y no se persiste en los logs.
"""

import json
import time
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
RUTA_LOGS = RAIZ / "logs" / "runs.jsonl"


class Trace:
    def __init__(self, provider: str, model: str, pregunta: str):
        self.run_id = uuid.uuid4().hex[:8]
        self.provider = provider
        self.model = model
        self.pregunta = pregunta
        self.eventos: list[dict] = []
        self.respuesta_final: str | None = None
        self.terminado = False  # True solo si el modelo dio una respuesta final
        self.respuesta_forzada = False  # True si se le quitaron las tools para obligarlo a responder
        self.redacciones = 0  # fragmentos de OFERTAS redactados por el sanitizador (capa de entrada)
        self.redacciones_salida = 0  # patrones sospechosos redactados de la RESPUESTA del modelo (capa de salida)
        self.ids_buscadas: set[int] = set()
        self.ids_leidas: set[int] = set()
        self.citados_sin_leer: list[int] = []  # ids citados en la respuesta final sin haberlos leído
        self.sin_veredicto = False  # True si la respuesta final quedó sin veredicto explícito
        self._t0 = time.time()

    def add(self, tipo: str, **datos) -> None:
        self.eventos.append({"t": round(time.time() - self._t0, 2), "tipo": tipo, **datos})

    def _contar(self, tipo: str) -> int:
        return sum(1 for e in self.eventos if e["tipo"] == tipo)

    def resumen(self) -> dict:
        tools = [e for e in self.eventos if e["tipo"] == "tool_call"]
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "pregunta": self.pregunta,
            "terminado": self.terminado,
            "respuesta_forzada": self.respuesta_forzada,
            "redacciones": self.redacciones,
            "redacciones_salida": self.redacciones_salida,
            "latencia_s": round(time.time() - self._t0, 1),
            "llamadas_llm": self._contar("llm_call"),
            "tool_calls": len(tools),
            "tool_calls_invalidas": sum(1 for e in tools if not e["valida"]),
            "intervenciones_guarda": self._contar("guardrail"),
            "ids_buscadas": sorted(self.ids_buscadas),
            "ids_leidas": sorted(self.ids_leidas),
            "citados_sin_leer": self.citados_sin_leer,
            "sin_veredicto": self.sin_veredicto,
        }

    def guardar(self, ruta: Path | None = None) -> None:
        ruta = ruta or RUTA_LOGS
        ruta.parent.mkdir(parents=True, exist_ok=True)
        registro = {**self.resumen(), "eventos": self.eventos}
        with open(ruta, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
