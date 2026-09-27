import pytest

from src.agent import observability


@pytest.fixture(autouse=True)
def logs_en_tmp(tmp_path, monkeypatch):
    """Ningún test escribe en logs/ reales."""
    monkeypatch.setattr(observability, "RUTA_LOGS", tmp_path / "runs.jsonl")
