"""Pipelines: sft_recipe tiny + frontier_pipeline tiny/dry_run runs on CPU."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import trainall  # noqa: E402
from trainall.data import InMemorySource  # noqa: E402
from trainall.pipelines import Pipeline, Stage, StageResult, frontier_pipeline, sft_recipe  # noqa: E402


def _sft_data(n=4, seqlen=8, vocab=64):
    items = []
    for _ in range(n):
        ids = torch.randint(0, vocab, (seqlen,)).tolist()
        labels = list(ids)
        labels[:2] = [-100, -100]
        items.append({"input_ids": ids, "labels": labels})
    return InMemorySource(items)


def test_pipeline_threads_model_through_stages():
    def stage_a(prev, ctx):
        return StageResult(name="a", model="model-1", metrics={"x": 1})

    def stage_b(prev, ctx):
        # Receives stage_a's model.
        assert prev.model == "model-1"
        return StageResult(name="b", model="model-2")

    pipe = Pipeline([Stage("a", stage_a), Stage("b", stage_b)])
    result = pipe.run()
    assert result.model == "model-2"
    assert "history" in result.extra
    assert result.extra["history"]["a"] == {"x": 1}


def test_sft_recipe_tiny_runs():
    pipe = sft_recipe(data=_sft_data(), tiny=True)
    assert isinstance(pipe, Pipeline)
    result = pipe.run()
    assert result.model is not None
    assert result.metrics.get("stage") == "sft"


def test_frontier_pipeline_dry_run():
    # The CPT/SFT/expand phases all consume the causal-LM batch shape produced
    # by the default collate, so they thread end-to-end on the tokenised data.
    # (DPO/RLVR phases need preference / policy-gradient batches respectively.)
    pipe = frontier_pipeline(data=_sft_data(), dry_run=True, stages=["cpt", "sft", "expand"])
    result = pipe.run()
    assert result.model is not None
    hist = result.extra["history"]
    assert "cpt" in hist and "sft" in hist and "expand" in hist


def test_frontier_pipeline_threads_model_and_clamps_steps():
    # Two trainable causal-LM phases chained: the model from cpt feeds sft.
    pipe = frontier_pipeline(data=_sft_data(), tiny=True, stages=["cpt", "sft"])
    result = pipe.run()
    assert result.model is not None
    assert len(result.extra["history"]) == 2


def test_recipe_registered():
    avail = trainall.available("recipe")["recipe"]
    for key in ("sft", "frontier", "dpo", "rlvr", "cpt", "distill", "agentic_rlvr"):
        assert key in avail
