"""Usage, cost and resource accounting from incrementally written logs."""

from pathlib import Path
from typing import Any

import yaml


def _detail(usage: dict[str, Any], group: str, key: str) -> int | None:
    v = (usage.get(group) or {}).get(key)
    return int(v) if v is not None else None


def summarize_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate ``usage.jsonl`` rows (one per completed LLM call).

    LangChain's ``input_token_details.cache_read`` and
    ``output_token_details.reasoning`` are subsets of input/output tokens, so
    they are reported, never added. When the raw provider ``total_tokens``
    shows reasoning was billed outside ``completion_tokens``, it is added to
    the billable output instead.
    """
    calls = len(rows)
    with_usage = [r for r in rows if r.get("usage_metadata")]
    if not with_usage:
        return {
            "llm_calls": calls or None,
            "input_tokens": None,
            "output_tokens": None,
            "cache_read_tokens": None,
            "reasoning_tokens": None,
            "billable_output_tokens": None,
            "reported_models": [],
            "usage_coverage": "none" if calls else "no_calls",
        }
    inp = out = billable_out = 0
    cache: int | None = None
    reasoning: int | None = None
    for r in with_usage:
        u = r["usage_metadata"]
        i, o = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
        inp, out = inp + i, out + o
        c, rs = (
            _detail(u, "input_token_details", "cache_read"),
            _detail(u, "output_token_details", "reasoning"),
        )
        if c is not None:
            cache = (cache or 0) + c
        if rs is not None:
            reasoning = (reasoning or 0) + rs
        raw_total = (r.get("token_usage") or {}).get("total_tokens")
        extra = rs if rs and raw_total == i + o + rs else 0
        billable_out += o + extra
    return {
        "llm_calls": calls,
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_tokens": cache,
        "reasoning_tokens": reasoning,
        "billable_output_tokens": billable_out,
        "reported_models": sorted(
            {r["reported_model"] for r in rows if r.get("reported_model")}
        ),
        "usage_coverage": "complete" if len(with_usage) == calls else "partial",
    }


def load_pricing(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def estimate_cost(
    usage: dict[str, Any], model: str, pricing: dict[str, Any]
) -> dict[str, Any]:
    """Cost from a dated pricing file; None when prices or counts are missing."""
    p = (pricing.get("models") or {}).get(model)
    none = {"cost": None, "currency": None, "pricing_version": pricing.get("version")}
    if not p or usage.get("input_tokens") is None:
        return none
    if p.get("input_per_mtok") is None or p.get("output_per_mtok") is None:
        return none
    cached = usage.get("cache_read_tokens") or 0
    cached_price = p.get("cached_input_per_mtok", p["input_per_mtok"])
    cost = (
        (usage["input_tokens"] - cached) * p["input_per_mtok"]
        + cached * cached_price
        + usage["billable_output_tokens"] * p["output_per_mtok"]
    ) / 1e6
    return {
        "cost": cost,
        "currency": pricing.get("currency", "USD"),
        "pricing_version": pricing.get("version"),
    }


def summarize_resources(res: dict[str, Any] | None) -> dict[str, Any]:
    if not res:
        return {"cpu_s": None, "peak_rss_bytes": None, "resources_final": None}
    cpu = res.get("cgroup_cpu_s")
    peak = res.get("cgroup_peak_bytes")
    return {
        "cpu_s": cpu if cpu is not None else res.get("rusage_children_cpu_s"),
        "peak_rss_bytes": peak
        if peak is not None
        else res.get("rusage_children_maxrss_bytes"),
        "resources_source": "cgroup" if cpu is not None else "rusage_children",
        # False: the process was killed and these are the last periodic sample
        "resources_final": bool(res.get("final")),
    }
