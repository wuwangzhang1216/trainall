"""Core: config round-trip + overrides, registry build/available, data types."""
from __future__ import annotations

import pytest

import trainall
from trainall.config import RunConfig, load_config
from trainall.types import Batch, VerifierResult


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_runconfig_roundtrip():
    cfg = RunConfig()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    # Nested dataclasses serialise to nested dicts.
    assert isinstance(d["objective"], dict)
    assert d["objective"]["name"] == "sft"
    rebuilt = RunConfig.from_dict(d)
    assert rebuilt.to_dict() == d


def test_load_config_from_dict_nested_and_unknown_keys():
    cfg = load_config(
        {
            "name": "demo",
            "objective": {"name": "dpo", "options": {"beta": 0.2}},
            "totally_unknown_key": 123,  # tolerated for forward-compat
        }
    )
    assert isinstance(cfg, RunConfig)
    assert cfg.name == "demo"
    assert cfg.objective.name == "dpo"
    assert cfg.objective.options["beta"] == 0.2
    # Defaults preserved for unspecified sections.
    assert cfg.algorithm.name == "full"


def test_config_merge_override():
    cfg = RunConfig()
    merged = cfg.merge(name="override")
    assert merged.name == "override"
    assert cfg.name == "trainall-run"  # original unchanged (replace is a copy)


def test_load_config_passthrough_runconfig():
    cfg = RunConfig(name="x")
    assert load_config(cfg) is cfg


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_available_has_expected_categories():
    avail = trainall.available()
    for cat in ("objective", "algorithm", "verifier", "reward", "datasource", "environment", "recipe"):
        assert cat in avail and avail[cat], f"missing category {cat}"
    assert "dpo" in avail["objective"]
    assert "math" in avail["verifier"]
    assert "lora" in avail["algorithm"]


def test_build_objective_and_aliases():
    obj = trainall.build("dpo", category="objective", beta=0.25)
    assert obj.beta == 0.25
    # Alias: clm -> CausalLMObjective.
    clm = trainall.build("clm", category="objective")
    assert clm.__class__.__name__ == "CausalLMObjective"


def test_build_unscoped_and_priority_resolution():
    # Unscoped lookup works for a unique key.
    v = trainall.build("composite", components=[trainall.build("regex", category="verifier")])
    assert v.__class__.__name__ == "CompositeVerifier"
    # Names registered in several categories resolve by priority (objective wins),
    # so the common `build("dpo")` returns the objective, not the same-named recipe.
    assert trainall.get("dpo").__name__ == "DPOObjective"
    assert trainall.get("dpo", category="recipe").__name__ == "dpo_recipe"
    # 'reward_model' exists as both an objective and a reward -> objective wins.
    assert trainall.get("reward_model").__name__ == "BradleyTerryObjective"
    assert trainall.get("reward_model", category="reward").__name__ == "RewardModelReward"


def test_build_unknown_key_raises():
    with pytest.raises(KeyError):
        trainall.build("does_not_exist", category="objective")


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #
def test_batch_accessors():
    b = Batch.of(a=1, b=2)
    assert b["a"] == 1
    assert "a" in b and "z" not in b
    assert b.get("z", 99) == 99
    assert set(b.keys()) == {"a", "b"}


def test_batch_to_moves_tensors():
    torch = pytest.importorskip("torch")
    b = Batch.of(x=torch.ones(2), label="not-a-tensor")
    b.to("cpu")  # should not raise; non-tensor passes through
    assert b["x"].device.type == "cpu"
    assert b["label"] == "not-a-tensor"


def test_verifier_result_dunders_and_ctors():
    ok = VerifierResult.ok(reward=0.8, detail="good")
    assert float(ok) == pytest.approx(0.8)
    assert bool(ok) is True
    fail = VerifierResult.fail(detail="bad")
    assert float(fail) == 0.0
    assert bool(fail) is False
