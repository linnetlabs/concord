"""Optional LLM cluster naming — clean topic names instead of tf-idf keyword bags.

Best-effort and dependency-free: if ANTHROPIC_API_KEY (or OPENAI_API_KEY) is set, name
every cluster from a few representative passages in ONE batched call via stdlib urllib.
On no key / any error, returns None and the caller keeps its deterministic tf-idf label.
This is the "grounded AI" the tool is about — the model names what retrieval found.
"""
from __future__ import annotations

import json
import os
import urllib.request


def _prompt(samples) -> str:
    blocks = []
    for i, s in enumerate(samples):
        joined = "  ·  ".join(t.replace("\n", " ")[:160] for t in s[:3])
        blocks.append(f"{i}: {joined}")
    return (
        "Each numbered item is a cluster of passages from one repository. Give each a "
        "2-4 word human-readable topic label in Title Case (no punctuation, no numbering). "
        "Reply with ONLY a JSON array of strings, in the same order.\n\n" + "\n".join(blocks)
    )


def _parse(text: str, n: int):
    """Lenient: accept any non-empty JSON string array (the model occasionally
    returns a few fewer/more than n; the caller pads/truncates to exactly n)."""
    a, b = text.find("["), text.rfind("]")
    if a >= 0 and b > a:
        try:
            arr = json.loads(text[a:b + 1])
            if isinstance(arr, list) and arr and all(isinstance(x, str) for x in arr):
                return [x.strip()[:48] for x in arr]
        except Exception:
            return None
    return None


def _post(url, headers, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    return json.loads(urllib.request.urlopen(req, timeout=40).read())


# Provider registry, in preference order. DeepSeek is FIRST so it is the default whenever
# DEEPSEEK_API_KEY is present: concord's LLM use (verify/label/resolve judging) is frequent and
# cheap-by-design, and DeepSeek runs it at a fraction of OpenAI/Anthropic cost with no quality loss
# on this kind of constrained JSON judging. Auto falls through to the next available key if no
# DeepSeek key is set. All but Anthropic/Gemini are OpenAI-compatible chat APIs. The API KEY IS THE
# USER'S — they pay for usage.
PROVIDERS = [
    {"name": "deepseek", "label": "DeepSeek", "env": ["DEEPSEEK_API_KEY"],
     "model": "deepseek-chat", "kind": "openai", "url": "https://api.deepseek.com/chat/completions"},
    {"name": "anthropic", "label": "Anthropic (Claude)", "env": ["ANTHROPIC_API_KEY"],
     "model": "claude-haiku-4-5-20251001", "kind": "anthropic", "url": "https://api.anthropic.com/v1/messages"},
    {"name": "openai", "label": "OpenAI", "env": ["OPENAI_API_KEY"],
     "model": "gpt-4o-mini", "kind": "openai", "url": "https://api.openai.com/v1/chat/completions"},
    {"name": "groq", "label": "Groq", "env": ["GROQ_API_KEY"],
     "model": "llama-3.3-70b-versatile", "kind": "openai", "url": "https://api.groq.com/openai/v1/chat/completions"},
    {"name": "mistral", "label": "Mistral", "env": ["MISTRAL_API_KEY"],
     "model": "mistral-small-latest", "kind": "openai", "url": "https://api.mistral.ai/v1/chat/completions"},
    {"name": "openrouter", "label": "OpenRouter", "env": ["OPENROUTER_API_KEY"],
     "model": "anthropic/claude-3.5-haiku", "kind": "openai", "url": "https://openrouter.ai/api/v1/chat/completions"},
    {"name": "gemini", "label": "Google Gemini", "env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
     "model": "gemini-2.0-flash", "kind": "gemini",
     "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"},
]

_selected = None  # runtime override: None=auto, "off"=disabled, else a provider name


def _key(p):
    for e in p["env"]:
        if os.environ.get(e):
            return os.environ[e]
    return None


def available_providers():
    """Providers that have a key in the environment, in preference order."""
    return [p for p in PROVIDERS if _key(p)]


def set_provider(name):
    """UI/API selector: a provider name, 'off', or 'auto'/None."""
    global _selected
    _selected = None if name in (None, "auto", "") else name


def _active():
    if os.environ.get("CONCORD_NO_LLM") or _selected == "off":
        return None
    avail = available_providers()
    if not avail:
        return None
    names = [p["name"] for p in avail]
    if _selected in names:
        pick = _selected
    else:
        force = os.environ.get("CONCORD_LLM", "").lower()
        pick = force if force in names else names[0]
    return next(p for p in avail if p["name"] == pick)


def status() -> dict:
    """Active LLM + the list of available providers. CONCORD_NO_LLM disables everything."""
    p = _active()
    return {
        "available": p is not None,
        "provider": p["name"] if p else None,
        "model": p["model"] if p else None,
        "disabled": bool(os.environ.get("CONCORD_NO_LLM")) or _selected == "off",
        "providers": [{"name": x["name"], "label": x["label"], "model": x["model"]} for x in available_providers()],
    }


def available() -> bool:
    return _active() is not None


def _llm(prompt: str, max_tokens: int = 900):
    """One LLM call -> text, billed to the USER's key. None if unavailable/failed."""
    p = _active()
    if not p:
        return None
    key = _key(p)
    try:
        if p["kind"] == "anthropic":
            r = _post(p["url"], {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                      {"model": p["model"], "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]})
            return r["content"][0]["text"]
        if p["kind"] == "gemini":
            r = _post(p["url"] + "?key=" + key, {"content-type": "application/json"},
                      {"contents": [{"parts": [{"text": prompt}]}]})
            return r["candidates"][0]["content"]["parts"][0]["text"]
        r = _post(p["url"], {"authorization": f"Bearer {key}", "content-type": "application/json"},
                  {"model": p["model"], "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]})
        return r["choices"][0]["message"]["content"]
    except Exception:
        return None


def label_clusters(samples):
    """samples: list[list[str]] — returns list[str] names (same length) or None."""
    if not samples:
        return None
    raw = _llm(_prompt(samples), 700)
    return _parse(raw, len(samples)) if raw else None
