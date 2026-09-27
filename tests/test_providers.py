from unittest.mock import MagicMock, patch

import pytest

from src.agent.providers import get_provider
from src.agent.providers.anthropic_provider import AnthropicProvider
from src.agent.providers.openai_compat import OpenAICompatProvider


def _fake_response(message: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"choices": [{"message": message}]}
    return resp


def _provider() -> OpenAICompatProvider:
    return OpenAICompatProvider("ollama", "http://localhost:11434/v1", "qwen2.5:7b")


def test_text_response():
    with patch("src.agent.providers.openai_compat.requests.post",
               return_value=_fake_response({"content": "hola"})):
        r = _provider().chat([{"role": "user", "content": "hi"}])
    assert r.text == "hola"
    assert r.tool_calls == []


def test_tool_call_arguments_as_json_string():
    msg = {"content": None, "tool_calls": [
        {"id": "c1", "function": {"name": "buscar_ofertas", "arguments": '{"query": "n8n", "top_k": 3}'}}]}
    with patch("src.agent.providers.openai_compat.requests.post", return_value=_fake_response(msg)):
        r = _provider().chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    assert r.tool_calls[0].name == "buscar_ofertas"
    assert r.tool_calls[0].arguments == {"query": "n8n", "top_k": 3}


def test_tool_call_arguments_as_dict_and_missing_id():
    msg = {"content": "", "tool_calls": [{"function": {"name": "leer_oferta", "arguments": {"id": 5}}}]}
    with patch("src.agent.providers.openai_compat.requests.post", return_value=_fake_response(msg)):
        r = _provider().chat([{"role": "user", "content": "x"}])
    assert r.tool_calls[0].id == "call_0"
    assert r.tool_calls[0].arguments == {"id": 5}


def test_invalid_json_arguments_are_surfaced_not_hidden():
    msg = {"content": "", "tool_calls": [{"id": "c", "function": {"name": "t", "arguments": "{not json"}}]}
    with patch("src.agent.providers.openai_compat.requests.post", return_value=_fake_response(msg)):
        r = _provider().chat([{"role": "user", "content": "x"}])
    assert r.tool_calls[0].arguments == {"__invalid_json__": "{not json"}


def test_request_payload_and_auth_header():
    p = OpenAICompatProvider("groq", "https://api.groq.com/openai/v1/", "m", api_key="secret")
    with patch("src.agent.providers.openai_compat.requests.post",
               return_value=_fake_response({"content": "ok"})) as post:
        p.chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    assert post.call_args.args[0] == "https://api.groq.com/openai/v1/chat/completions"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer secret"
    assert post.call_args.kwargs["json"]["tools"] == [{"type": "function"}]


def test_factory_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    p = get_provider()
    assert p.name == "ollama" and p.model == "qwen2.5:7b"


def test_factory_groq_requires_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with patch("dotenv.load_dotenv"):
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            get_provider("groq")


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError):
        get_provider("nope")


def test_anthropic_message_conversion():
    messages = [
        {"role": "system", "content": "sos un agente"},
        {"role": "user", "content": "buscá"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "function": {"name": "buscar_ofertas", "arguments": '{"query": "x"}'}},
            {"id": "b", "function": {"name": "leer_oferta", "arguments": {"id": 1}}}]},
        {"role": "tool", "tool_call_id": "a", "content": "r1"},
        {"role": "tool", "tool_call_id": "b", "content": "r2"},
    ]
    system, out = AnthropicProvider._convert_messages(messages)
    assert system == "sos un agente"
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[1]["content"][0] == {"type": "tool_use", "id": "a", "name": "buscar_ofertas",
                                    "input": {"query": "x"}}
    # los dos tool_result quedan juntos en un único mensaje user
    assert [b["tool_use_id"] for b in out[2]["content"]] == ["a", "b"]


def test_anthropic_tool_conversion():
    tools = [{"type": "function", "function": {"name": "t", "description": "d",
                                               "parameters": {"type": "object", "properties": {}}}}]
    assert AnthropicProvider._convert_tools(tools) == [
        {"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {}}}]


def _resp(status, headers=None, message=None):
    r = MagicMock()
    r.status_code = status
    r.headers = headers or {}
    r.json.return_value = {"choices": [{"message": message or {"content": "ok"}}]}
    if status >= 400:
        import requests
        r.raise_for_status.side_effect = requests.HTTPError(f"{status}")
    return r


def test_reintenta_ante_429_respetando_retry_after():
    p = _provider()
    with patch("src.agent.providers.openai_compat.requests.post",
               side_effect=[_resp(429, {"retry-after": "3"}), _resp(200)]) as post, \
            patch("src.agent.providers.openai_compat.time.sleep") as sleep:
        assert p.chat([{"role": "user", "content": "x"}]).text == "ok"
    assert post.call_count == 2 and p.reintentos == 1
    sleep.assert_called_once_with(3.0)


def test_429_persistente_termina_lanzando_el_error():
    import requests
    p = OpenAICompatProvider("groq", "http://x/v1", "m", max_reintentos=2)
    with patch("src.agent.providers.openai_compat.requests.post", return_value=_resp(429)), \
            patch("src.agent.providers.openai_compat.time.sleep"):
        with pytest.raises(requests.HTTPError):
            p.chat([{"role": "user", "content": "x"}])
    assert p.reintentos == 2


def test_errores_4xx_que_no_son_429_no_se_reintentan():
    import requests
    p = _provider()
    with patch("src.agent.providers.openai_compat.requests.post", return_value=_resp(404)) as post, \
            patch("src.agent.providers.openai_compat.time.sleep") as sleep:
        with pytest.raises(requests.HTTPError):
            p.chat([{"role": "user", "content": "x"}])
    assert post.call_count == 1 and sleep.call_count == 0
