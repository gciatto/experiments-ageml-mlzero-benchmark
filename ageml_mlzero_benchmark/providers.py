"""Live checks of the OpenAI-compatible Gemini endpoint (explicitly requested only)."""

import json
import os
import urllib.request
from typing import Any


def _request(url: str, key: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def check_llm(llm: dict[str, Any]) -> dict[str, Any]:
    """Model discovery and one tiny chat on the SAME endpoint the runs use."""
    key = os.environ.get(llm["api_key_env"])
    if not key:
        return {"ok": False, "error": f"${llm['api_key_env']} is not set"}
    base = llm["endpoint"].rstrip("/") + "/"
    listed = [
        m["id"].removeprefix("models/") for m in _request(base + "models", key)["data"]
    ]
    if llm["model"] not in listed:
        return {
            "ok": False,
            "error": f"{llm['model']} not served; similar: "
            f"{[m for m in listed if llm['model'].split('-')[0] in m][:10]}",
        }
    body = {
        "model": llm["model"],
        "messages": [{"role": "user", "content": "Reply with OK."}],
        **(llm.get("settings") or {}),
    }
    body.pop("max_tokens", None)
    reply = _request(base + "chat/completions", key, body)
    return {
        "ok": True,
        "listed": True,
        "reported_model": reply.get("model"),
        "usage": reply.get("usage"),
    }
