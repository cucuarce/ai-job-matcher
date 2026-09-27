"""
Interfaz común para los proveedores de LLM.

El agente habla siempre en formato "OpenAI-style" (messages con role/content,
tools como JSON Schema). Cada proveedor traduce a su API. Así cambiar de
Ollama a Groq o Claude no toca el agente.
"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict | None = None


class LLMProvider(Protocol):
    name: str
    model: str

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        """messages: [{'role','content', ...}]. tools: [{'type':'function','function':{...}}]."""
        ...
