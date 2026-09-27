"""
Proveedor para cualquier API compatible con OpenAI /chat/completions.
Sirve para Ollama (local, gratis) y Groq (cloud, free tier) cambiando solo
base_url, api_key y modelo.
"""

import json
import time

import requests

from .base import LLMResponse, ToolCall


class OpenAICompatProvider:
    def __init__(self, name: str, base_url: str, model: str, api_key: str | None = None,
                 timeout: int = 600, max_tokens: int = 700,
                 max_reintentos: int = 4):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_reintentos = max_reintentos
        self.reintentos = 0  # contador acumulado, útil para observabilidad

    def _post_con_reintentos(self, payload: dict, headers: dict) -> requests.Response:
        """Los tiers gratis limitan por minuto (429): esperamos lo que pide Retry-After y reintentamos."""
        url = f"{self.base_url}/chat/completions"
        for intento in range(self.max_reintentos + 1):
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            if resp.status_code not in (429, 500, 502, 503) or intento == self.max_reintentos:
                resp.raise_for_status()
                return resp
            try:
                espera = float(resp.headers.get("retry-after", ""))
            except ValueError:
                espera = 2 ** (intento + 1)
            self.reintentos += 1
            time.sleep(min(espera, 60))
        raise AssertionError("inalcanzable")

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        payload = {"model": self.model, "messages": messages, "temperature": 0,
                   "max_tokens": self.max_tokens}
        if tools:
            payload["tools"] = tools

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = self._post_con_reintentos(payload, headers)
        data = resp.json()

        msg = data["choices"][0]["message"]
        tool_calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            args = tc["function"].get("arguments", {})
            # OpenAI/Groq mandan los argumentos como string JSON; Ollama a veces como dict.
            # Modelos chicos pueden devolver JSON inválido: no lo ocultamos, lo dejamos
            # visible para que el agente y los evals lo registren.
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {"__invalid_json__": args}
            tool_calls.append(ToolCall(id=tc.get("id") or f"call_{i}",
                                       name=tc["function"]["name"], arguments=args))

        return LLMResponse(text=msg.get("content") or "", tool_calls=tool_calls, raw=data)
