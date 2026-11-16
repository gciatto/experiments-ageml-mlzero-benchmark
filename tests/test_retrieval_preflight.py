"""Retrieval failures must be detected before any timed search or Gemini call."""

import importlib.util
from pathlib import Path

import pytest


def module():
    p = Path(__file__).resolve().parents[1] / "environments/mlzero/check_retrieval.py"
    spec = importlib.util.spec_from_file_location("check_retrieval", p)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Indexer:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def _load_embedding_model(self):
        if self.fail:
            raise TypeError("BertModel unexpected keyword argument dtype")

    def build_indices(self):
        self.calls.append("build")

    def save_indices(self):
        self.calls.append("save")

    def load_indices(self):
        self.calls.append("load")

    def search(self, query, tool, **kwargs):
        self.calls.append((tool, kwargs["condensed"]))
        return [{"content": "A real tutorial", "score": 0.5}]


def test_load_failure_is_not_swallowed_by_upstream_builder():
    indexer = Indexer(fail=True)
    with pytest.raises(TypeError, match="dtype"):
        module().smoke(indexer, rebuild=True)
    assert not indexer.calls


def test_checks_both_indices_of_all_required_tools():
    indexer = Indexer()
    result = module().smoke(indexer, rebuild=True)
    assert len(result) == 6
    assert indexer.calls[:3] == ["build", "save", "load"]


@pytest.mark.parametrize(
    "results",
    [
        [],
        [{"content": "", "score": 0.5}],
        [{"content": "tutorial", "score": float("nan")}],
    ],
)
def test_empty_or_invalid_retrieval_blocks_campaign(results):
    with pytest.raises(RuntimeError, match="Retrieval smoke test"):
        module().validate_results(results, "autogluon.tabular")
