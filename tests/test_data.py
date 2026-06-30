"""Data: templates + masking, pack_sequences, synthetic/RS/self-play flywheels."""
from __future__ import annotations

import pytest

from trainall.data import (
    ChatTemplate,
    Curriculum,
    InMemorySource,
    Packer,
    RejectionSampler,
    SelfPlayLoop,
    SyntheticDataEngine,
    TaskProposer,
    apply_template,
    mask_prompt,
    pack_sequences,
)
from trainall.types import Message, Sample


# --------------------------------------------------------------------------- #
# Templates + masking
# --------------------------------------------------------------------------- #
def test_chatml_render_and_assistant_span():
    tmpl = ChatTemplate(style="chatml")
    msgs = [Message("user", "hi"), Message("assistant", "hello there")]
    text, spans = tmpl.render_with_mask(msgs)
    assert "<|im_start|>user" in text
    assert "<|im_start|>assistant" in text
    # The reported span covers exactly the assistant content.
    assert len(spans) == 1
    start, end = spans[0]
    assert text[start:end] == "hello there"


def test_apply_template_styles():
    msgs = [{"role": "user", "content": "q"}]
    assert "<|im_start|>" in apply_template(msgs, "chatml")
    assert "<|start_header_id|>" in apply_template(msgs, "llama3")
    assert "User:" in apply_template(msgs, "plain")


def test_mask_prompt():
    input_ids, labels = mask_prompt([1, 2, 3], [4, 5])
    assert input_ids == [1, 2, 3, 4, 5]
    assert labels == [-100, -100, -100, 4, 5]


# --------------------------------------------------------------------------- #
# Packing
# --------------------------------------------------------------------------- #
def test_pack_sequences_fills_and_pads():
    packed = pack_sequences([[1, 2, 3], [4, 5], [6]], max_len=4, pad_id=0)
    # 6 tokens -> first bin full [1,2,3,4], second [5,6,0,0] padded.
    assert packed[0] == [1, 2, 3, 4]
    assert packed[1] == [5, 6, 0, 0]
    assert all(len(b) == 4 for b in packed)


def test_pack_sequences_chunks_long():
    packed = pack_sequences([[1, 2, 3, 4, 5]], max_len=2, drop_last=False)
    assert packed[0] == [1, 2] and packed[1] == [3, 4]
    assert packed[2] == [5, 0]  # padded


def test_packer_streaming():
    p = Packer(max_len=3, pad_id=0)
    out = p.add([1, 2])
    assert out == []  # not full yet
    out = p.add([3, 4])
    assert out == [[1, 2, 3]]
    rest = p.flush()
    assert rest == [[4, 0, 0]]


# --------------------------------------------------------------------------- #
# In-memory source
# --------------------------------------------------------------------------- #
def test_in_memory_source():
    src = InMemorySource([{"prompt": "p", "response": "r"}, Sample(prompt="x", response="y")])
    items = list(src)
    assert len(items) == 2
    assert isinstance(items[0], Sample)
    assert items[0].prompt == "p"


# --------------------------------------------------------------------------- #
# Synthetic data engine (callable proposer/solver/verifier)
# --------------------------------------------------------------------------- #
def test_synthetic_data_engine():
    counter = {"i": 0}

    def proposer():
        counter["i"] += 1
        n = counter["i"]
        return {"prompt": f"add {n}+{n}", "reference": str(2 * n)}

    def solver(prompt):
        # Parse "add a+b" and return the (correct) sum as a boxed answer.
        a, b = prompt.replace("add ", "").split("+")
        return [rf"\boxed{{{int(a) + int(b)}}}"]

    from trainall.verifiers import MathVerifier

    engine = SyntheticDataEngine(proposer, solver, MathVerifier(), k=2)
    samples = engine.generate(3)
    assert len(samples) == 3
    assert all(isinstance(s, Sample) for s in samples)
    assert all(s.meta["synthetic"] for s in samples)
    assert all(s.meta["difficulty"] in {"easy", "medium", "hard", "unsolved"} for s in samples)


# --------------------------------------------------------------------------- #
# Rejection sampler
# --------------------------------------------------------------------------- #
def test_rejection_sampler_keeps_passing():
    # Solver returns a mix; verifier passes only the correct ones.
    def solver(prompt):
        return [r"\boxed{42}", r"\boxed{0}", r"\boxed{42}"]

    from trainall.verifiers import MathVerifier

    rs = RejectionSampler(solver, MathVerifier(), n=3, keep="all")
    out = rs.run([{"prompt": "q", "reference": "42"}])
    assert len(out) == 2  # two correct candidates kept
    assert all(s.response == r"\boxed{42}" for s in out)


def test_rejection_sampler_none_pass():
    def solver(prompt):
        return r"\boxed{1}"

    from trainall.verifiers import MathVerifier

    rs = RejectionSampler(solver, MathVerifier(), n=2)
    out = rs.run([("q", "42")])
    assert out == []


# --------------------------------------------------------------------------- #
# Self-play loop
# --------------------------------------------------------------------------- #
def test_selfplay_loop_with_curriculum():
    def proposer(difficulty=0.5):
        return {"prompt": "add 1+1", "reference": "2"}

    def solver(prompt):
        return r"\boxed{2}"

    from trainall.verifiers import MathVerifier

    loop = SelfPlayLoop(
        TaskProposer(proposer),
        solver,
        MathVerifier(),
        curriculum=Curriculum(difficulty=0.2, step=0.1),
        rounds=2,
        tasks_per_round=2,
        k=2,
    )
    samples = loop.run()
    assert len(samples) >= 1
    # Curriculum recorded two rounds; high pass-rate should push difficulty up.
    assert len(loop.curriculum.history) == 2
    assert loop.curriculum.history[0]["decision"] in {"harder", "hold", "easier"}


def test_curriculum_adapts_difficulty():
    c = Curriculum(difficulty=0.5, step=0.1, target_high=0.8, target_low=0.4)
    # High pass-rate -> harder.
    c.update(0.95, ["a", "b"])
    assert c.difficulty == pytest.approx(0.6)
    # Low pass-rate -> easier.
    c.update(0.1, ["a", "b"])
    assert c.difficulty == pytest.approx(0.5)
