"""
Proveedor para Claude (Anthropic API). Es de pago: solo se usa si
LLM_PROVIDER=anthropic. Traduce del formato OpenAI-style al de Anthropic.
"""

import json

from .base import LLMResponse, ToolCall


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 2048):
        import anthropic  # import perezoso: no hace falta para usar Ollama/Groq

        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key)

    @staticmethod
    def _convert_tools(tools: list[dict] | None) -> list[dict]:
        return [{"name": t["function"]["name"],
                 "description": t["function"].get("description", ""),
                 "input_schema": t["function"]["parameters"]} for t in tools or []]

    @staticmethod
    def _convert_messages(messages: list[dict]) -> tuple[str, list[dict]]:
        system_parts, out = [], []
        for m in messages:
            role = m["role"]
            if role == "system":
                system_parts.append(m["content"])
            elif role == "assistant" and m.get("tool_calls"):
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                for tc in m["tool_calls"]:
                    args = tc["function"]["arguments"]
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"],
                                   "input": json.loads(args) if isinstance(args, str) else args})
                out.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                # Anthropic exige que los tool_result consecutivos vayan en un solo mensaje user.
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                        and out[-1]["content"][0].get("type") == "tool_result":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            else:
                out.append({"role": role, "content": m["content"]})
        return "\n\n".join(system_parts), out

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        system, msgs = self._convert_messages(messages)
        kwargs = {"model": self.model, "max_tokens": self.max_tokens, "messages": msgs}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        resp = self._client.messages.create(**kwargs)

        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(id=b.id, name=b.name, arguments=b.input)
                 for b in resp.content if b.type == "tool_use"]
        return LLMResponse(text=text, tool_calls=calls, raw=resp.model_dump())
