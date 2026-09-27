"""
Fábrica de proveedores. Se elige con LLM_PROVIDER en el entorno / .env:

    LLM_PROVIDER=ollama      (default, local y gratis)
    LLM_PROVIDER=groq        (requiere GROQ_API_KEY)
    LLM_PROVIDER=anthropic   (requiere ANTHROPIC_API_KEY, de pago)

LLM_MODEL sobrescribe el modelo por defecto de cada proveedor.
"""

import os

from .base import LLMProvider, LLMResponse, ToolCall
from .openai_compat import OpenAICompatProvider

DEFAULT_MODELS = {
    "ollama": "qwen2.5:7b",
    "groq": "openai/gpt-oss-120b",
    "anthropic": "claude-sonnet-5",
}


def get_provider(provider: str | None = None, model: str | None = None) -> LLMProvider:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"LLM_PROVIDER desconocido: {provider!r}. Opciones: {sorted(DEFAULT_MODELS)}")
    model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider]

    if provider == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OpenAICompatProvider("ollama", f"{base.rstrip('/')}/v1", model)

    if provider == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("Falta GROQ_API_KEY en el entorno / .env")
        return OpenAICompatProvider("groq", "https://api.groq.com/openai/v1", model, api_key=key)

    from .anthropic_provider import AnthropicProvider
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Falta ANTHROPIC_API_KEY en el entorno / .env")
    return AnthropicProvider(model, api_key=key)


__all__ = ["get_provider", "LLMProvider", "LLMResponse", "ToolCall", "OpenAICompatProvider"]
