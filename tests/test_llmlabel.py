"""LLM provider detection + selection + the CONCORD_NO_LLM kill switch (no API calls)."""
from __future__ import annotations

import pytest

from concordai import llmlabel

_KEYS = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY",
         "MISTRAL_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _KEYS + ["CONCORD_NO_LLM", "CONCORD_LLM"]:
        monkeypatch.delenv(k, raising=False)
    llmlabel.set_provider("auto")
    yield
    llmlabel.set_provider("auto")


def test_no_keys_means_unavailable():
    s = llmlabel.status()
    assert s["available"] is False and s["providers"] == []


def test_detects_and_prefers_anthropic(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-y")
    s = llmlabel.status()
    assert s["available"] and s["provider"] == "anthropic"          # preference order
    assert {p["name"] for p in s["providers"]} == {"anthropic", "openai"}


def test_explicit_provider_selection(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-y")
    llmlabel.set_provider("openai")
    assert llmlabel.status()["provider"] == "openai"


def test_no_llm_kill_switch(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("CONCORD_NO_LLM", "1")
    assert llmlabel.status()["available"] is False


def test_off_selection_disables(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    llmlabel.set_provider("off")
    assert llmlabel.available() is False
