"""Functional retrieval preflight; no Gemini calls and no swallowed build errors."""

import argparse
import importlib.metadata
import json
import math

PINS = {
    "FlagEmbedding": "1.3.5",
    "transformers": "4.48.3",
    "sentence-transformers": "3.4.1",
    "peft": "0.15.2",
    "pandas": "2.2.3",
}


def validate_results(results, tool):
    if not results:
        raise RuntimeError(f"Retrieval smoke test returned no tutorials for {tool}")
    if not all(
        r.get("content", "").strip() and math.isfinite(r["score"]) for r in results
    ):
        raise RuntimeError(
            f"Retrieval smoke test returned invalid content/scores for {tool}"
        )


def smoke(indexer, rebuild=False):
    if rebuild:
        # Explicit model load propagates dtype/import errors, unlike build_indices.
        indexer._load_embedding_model()
        indexer.build_indices()
        indexer.save_indices()
    indexer.load_indices()
    results = {}
    for tool in ("autogluon.tabular", "machine learning", "autogluon.multimodal"):
        for condensed in (False, True):
            found = indexer.search(
                "classification regression training tutorial",
                tool,
                condensed=condensed,
                top_k=1,
            )
            validate_results(found, tool)
            results[f"{tool}:{condensed}"] = len(found)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    observed = {name: importlib.metadata.version(name) for name in PINS}
    if observed != PINS:
        raise RuntimeError(
            f"Retrieval dependency mismatch: {observed}; expected {PINS}"
        )
    from autogluon.assistant.tools_registry.indexing import TutorialIndexer

    results = smoke(TutorialIndexer(), args.rebuild)
    print(
        json.dumps(
            {"retrieval_smoke": "passed", "versions": observed, "queries": results}
        )
    )


if __name__ == "__main__":
    main()
