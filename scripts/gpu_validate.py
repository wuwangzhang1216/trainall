#!/usr/bin/env python3
"""End-to-end execution + short-convergence validation for the whole trainall stack.

This is a *field test* harness: it proves every advertised training path actually
runs on real hardware with a real model, and that two of them (SFT, GRPO) show a
real learning signal over a short run.  It is deliberately split from the unit
tests, which only exercise toy models on CPU.

Stages
------
0  preflight  — torch / CUDA / compute-capability (sm_120 for RTX 5090) / bf16 /
                bitsandbytes report.  This is the go/no-go for the GPU stages.
1  suite      — run ``pytest`` and the 7 example scripts as subprocesses (confirms
                the install is intact on this box).  Skip with ``--skip-suite``.
2  smoke      — every training path runs a handful of steps on the chosen backend
                and device: SFT(full/lora/qlora), DPO, GRPO+verifier, agentic,
                pretrain-from-scratch, distillation.  PASS = no crash + finite loss.
3  converge   — a short SFT run (loss must trend down) and a short GRPO-on-arithmetic
                run (mean reward must trend up).  Curves are saved as CSV (+ PNG if
                matplotlib is present).

Backends
--------
``--backend toy``  a few-thousand-param ``DecoderLM`` + byte tokenizer.  No network,
                   no GPU needed — used to debug this harness's orchestration on CPU.
``--backend hf``   a real HF causal-LM (default ``Qwen/Qwen2.5-0.5B-Instruct``) +
                   its tokenizer.  This is what you run on the 5090.

Examples
--------
    # CPU dry-run of the orchestration (no downloads):
    python scripts/gpu_validate.py --backend toy --device cpu --quick

    # the real thing on the RTX 5090:
    python scripts/gpu_validate.py --backend hf --device cuda \
        --model Qwen/Qwen2.5-0.5B-Instruct --out gpu_validation_out
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

# --- make `trainall` importable whether or not PYTHONPATH=src is set ---------- #
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ============================================================================ #
# Result bookkeeping
# ============================================================================ #
@dataclass
class Result:
    stage: str
    name: str
    status: str  # PASS | FAIL | SKIP | WARN
    detail: str = ""
    seconds: float = 0.0
    peak_mb: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


RESULTS: List[Result] = []

_ICON = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️ ", "WARN": "⚠️ "}


def _emit(r: Result) -> None:
    RESULTS.append(r)
    mem = f"  {r.peak_mb:.0f}MB" if r.peak_mb else ""
    print(f"{_ICON.get(r.status, '  ')} [{r.stage}] {r.name:<22} {r.status:<4} "
          f"{r.seconds:6.2f}s{mem}  {r.detail}", flush=True)


def _torch():
    import torch  # lazy: keeps `import gpu_validate` cheap and torch-free
    return torch


def _reset_peak(device: str) -> None:
    if device == "cuda":
        t = _torch()
        if t.cuda.is_available():
            t.cuda.reset_peak_memory_stats()


def _peak_mb(device: str) -> Optional[float]:
    if device == "cuda":
        t = _torch()
        if t.cuda.is_available():
            return t.cuda.max_memory_allocated() / 1e6
    return None


def run_step(stage: str, name: str, fn: Callable[[], str], device: str) -> Result:
    """Run one validation unit, capturing time / peak-mem / pass-fail."""
    _reset_peak(device)
    t0 = time.time()
    try:
        detail = fn() or ""
        r = Result(stage, name, "PASS", detail, time.time() - t0, _peak_mb(device))
    except Exception as exc:  # noqa: BLE001 - this harness must survive any failure
        tb = traceback.format_exc(limit=3).strip().splitlines()
        r = Result(stage, name, "FAIL", f"{type(exc).__name__}: {exc}",
                   time.time() - t0, _peak_mb(device), {"traceback": tb})
    _emit(r)
    return r


# ============================================================================ #
# Backend: toy DecoderLM+bytes, or a real HF model+tokenizer
# ============================================================================ #
MAX_LEN = 64


class ByteTok:
    """Minimal stand-in for an HF tokenizer (byte-level, vocab 256)."""

    vocab_size = 256

    def ids(self, text: str, max_len: int = MAX_LEN) -> List[int]:
        return list(text.encode("utf-8"))[:max_len]

    def text(self, ids) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


class Backend:
    """Holds the model factory + a uniform tokenize/detokenize surface."""

    def __init__(self, kind: str, device: str, model_name: str):
        self.kind = kind
        self.device = device
        self.model_name = model_name
        self.is_hf = kind == "hf"
        self._hf_tok = None
        if self.is_hf:
            from trainall._optional import require
            tf = require("transformers", feature="hf backend")
            self._hf_tok = tf.AutoTokenizer.from_pretrained(model_name)
            if self._hf_tok.pad_token is None:
                self._hf_tok.pad_token = self._hf_tok.eos_token
        else:
            self._byte = ByteTok()

    # -- model construction -------------------------------------------------- #
    def new_model(self, **arch_overrides):
        if self.is_hf:
            from trainall._optional import require
            tf = require("transformers", feature="hf backend")
            model = tf.AutoModelForCausalLM.from_pretrained(self.model_name)
            return model
        from trainall.models import ArchConfig, DecoderLM
        cfg = ArchConfig(vocab_size=256, dim=64, n_layers=2, n_heads=4,
                         n_kv_heads=2, ffn_dim=128, max_seq_len=MAX_LEN,
                         tie_embeddings=True)
        for k, v in arch_overrides.items():
            setattr(cfg, k, v)
        return DecoderLM.from_config(cfg)

    def tiny_decoder(self, n_layers: int = 1):
        """A from-scratch DecoderLM (used for pretrain/distill regardless of backend)."""
        from trainall.models import ArchConfig, DecoderLM
        cfg = ArchConfig(vocab_size=256, dim=64, n_layers=n_layers, n_heads=4,
                         n_kv_heads=2, ffn_dim=128, max_seq_len=MAX_LEN, tie_embeddings=True)
        return DecoderLM.from_config(cfg)

    # -- tokenization -------------------------------------------------------- #
    def ids(self, text: str) -> List[int]:
        if self.is_hf:
            return self._hf_tok(text, add_special_tokens=False)["input_ids"][:MAX_LEN]
        return self._byte.ids(text)

    @staticmethod
    def byte_ids(text: str) -> List[int]:
        # Always byte-level (vocab 256). The from-scratch DecoderLM paths use a
        # vocab-256 model regardless of backend, so their data MUST be byte ids —
        # feeding a 151k-vocab HF tokenizer's ids into it indexes out of range.
        return list(text.encode("utf-8"))[:MAX_LEN]

    @property
    def hf_tokenizer(self):
        return self._hf_tok

    @property
    def lora_targets(self) -> List[str]:
        # Qwen2 / Llama-style projection names; DecoderLM uses the same q/k/v/o names.
        return ["q_proj", "v_proj"]


# ============================================================================ #
# Shared data builders
# ============================================================================ #
def sft_record(be: Backend, prompt: str, response: str) -> Dict[str, List[int]]:
    p, r = be.ids(prompt), be.ids(response)
    ids = (p + r)[:MAX_LEN]
    labels = ([-100] * len(p) + r)[:MAX_LEN]
    return {"input_ids": ids, "labels": labels}


def byte_sft_record(prompt: str, response: str) -> Dict[str, List[int]]:
    """Byte-level SFT record for the vocab-256 from-scratch DecoderLM paths."""
    p = list(prompt.encode("utf-8"))[:MAX_LEN]
    r = list(response.encode("utf-8"))[:MAX_LEN]
    ids = (p + r)[:MAX_LEN]
    labels = ([-100] * len(p) + r)[:MAX_LEN]
    return {"input_ids": ids, "labels": labels}


SFT_PAIRS = [
    ("Q: capital of Japan?\nA:", " Tokyo"),
    ("Q: 2+2?\nA:", " 4"),
    ("Q: opposite of hot?\nA:", " cold"),
    ("Q: color of the sky?\nA:", " blue"),
]

ARITH = [("2+2", "4"), ("10-3", "7"), ("5*2", "10"), ("9+6", "15"), ("8-5", "3")]


# ============================================================================ #
# Stage 0 — preflight
# ============================================================================ #
def stage_preflight(args) -> Dict[str, Any]:
    print("\n=== Stage 0: preflight ===", flush=True)
    info: Dict[str, Any] = {}
    try:
        t = _torch()
        info["torch"] = t.__version__
        info["torch_cuda_build"] = getattr(t.version, "cuda", None)
        info["cuda_available"] = bool(t.cuda.is_available())
        if t.cuda.is_available():
            info["device_name"] = t.cuda.get_device_name(0)
            cap = t.cuda.get_device_capability(0)
            info["capability"] = f"sm_{cap[0]}{cap[1]}"
            info["bf16"] = bool(t.cuda.is_bf16_supported())
            try:
                info["vram_gb"] = round(t.cuda.get_device_properties(0).total_memory / 1e9, 1)
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = str(exc)

    try:
        import bitsandbytes as bnb
        info["bitsandbytes"] = getattr(bnb, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        info["bitsandbytes"] = f"absent ({type(exc).__name__})"

    for k, v in info.items():
        print(f"    {k:20} {v}", flush=True)

    # Blackwell sanity check + go/no-go for the cuda stages.
    if args.device == "cuda":
        if not info.get("cuda_available"):
            _emit(Result("0", "cuda_available", "FAIL",
                         "device=cuda requested but torch.cuda.is_available() is False "
                         "(wrong torch build? need a cu128 wheel for sm_120)"))
        else:
            cap = info.get("capability", "?")
            note = "Blackwell" if cap == "sm_120" else ""
            _emit(Result("0", "cuda_available", "PASS",
                         f"{info.get('device_name','?')} {cap} {note}".strip()))
    else:
        _emit(Result("0", "device", "PASS", f"running on {args.device}"))
    return info


# ============================================================================ #
# Stage 1 — repo suite (pytest + examples) as subprocesses
# ============================================================================ #
def stage_suite(args) -> None:
    print("\n=== Stage 1: repo suite (pytest + examples) ===", flush=True)
    env = dict(os.environ, PYTHONPATH=_SRC + os.pathsep + os.environ.get("PYTHONPATH", ""))

    def _run(name, cmd, cwd):
        t0 = time.time()
        try:
            p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                               text=True, timeout=args.suite_timeout)
            tail = (p.stdout + p.stderr).strip().splitlines()[-1:] or [""]
            status = "PASS" if p.returncode == 0 else "FAIL"
            _emit(Result("1", name, status, tail[-1][:80], time.time() - t0))
        except Exception as exc:  # noqa: BLE001
            _emit(Result("1", name, "FAIL", f"{type(exc).__name__}: {exc}", time.time() - t0))

    _run("pytest", [sys.executable, "-m", "pytest", "-q"], _REPO)
    ex_dir = os.path.join(_REPO, "examples")
    for f in sorted(x for x in os.listdir(ex_dir) if x[0].isdigit() and x.endswith(".py")):
        _run(f"example:{f[:2]}", [sys.executable, f], ex_dir)


# ============================================================================ #
# Stage 2 — smoke matrix (a few steps each, on the chosen backend/device)
# ============================================================================ #
def _trainer_cfg(args, **over):
    from trainall.training import TrainerConfig
    base = dict(lr=1e-3 if args.backend == "hf" else 1e-2, batch_size=2,
                max_steps=args.smoke_steps, device=args.device, log_every=999,
                bf16=(args.device == "cuda"))
    base.update(over)
    return TrainerConfig(**base)


def _finite(x) -> bool:
    import math
    try:
        return math.isfinite(float(x))
    except Exception:  # noqa: BLE001
        return False


def _make_tap():
    """A Callback that records per-step loss (the Trainer keeps no history itself)."""
    from trainall.training import Callback

    class _Tap(Callback):
        def __init__(self):
            self.losses: List[float] = []

        def on_step_end(self, step, metrics, trainer=None, **kw):  # trainer's real hook signature
            if metrics and "loss" in metrics:
                self.losses.append(float(metrics["loss"]))

    return _Tap()


def _train(args, *, model, objective, data, collate=None, algorithm=None, **cfg_over):
    """Build a Trainer (capturing per-step loss), run it, return (trained_model, losses)."""
    from trainall.training import Trainer
    tap = _make_tap()
    kw: Dict[str, Any] = dict(model=model, objective=objective, data=data,
                              config=_trainer_cfg(args, **cfg_over), callbacks=[tap])
    if collate is not None:
        kw["collate"] = collate
    if algorithm is not None:
        kw["algorithm"] = algorithm
    trained = Trainer(**kw).train()
    return trained, tap.losses


def smoke_sft(be: Backend, args, algo_name: str) -> str:
    import trainall
    from trainall.data import InMemorySource
    from trainall.training import Trainer

    trainall.seed_everything(0)
    model = be.new_model()
    objective = trainall.build("sft", category="objective")
    algo_kwargs = {}
    if algo_name in ("lora", "qlora"):
        algo_kwargs = dict(r=8, alpha=16, target_modules=be.lora_targets)
    algorithm = trainall.build(algo_name, category="algorithm", **algo_kwargs)
    data = InMemorySource([sft_record(be, p, r) for p, r in SFT_PAIRS] * 2)
    trained, losses = _train(args, model=model, objective=objective, algorithm=algorithm, data=data)
    trainable = sum(p.numel() for p in trained.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trained.parameters())
    last = losses[-1] if losses else float("nan")
    note = ""
    if algo_name == "qlora":
        quantized = any("4bit" in type(m).__name__.lower() or "Params4bit" in type(getattr(m, "weight", None)).__name__
                        for m in trained.modules())
        note = "real 4-bit" if quantized else "fp fallback (no bnb/cuda)"
    if not _finite(last):
        raise RuntimeError(f"non-finite loss {last}")
    return f"{algo_name}: {trainable:,}/{total:,} trainable, loss~{last:.3f} {note}".strip()


def smoke_dpo(be: Backend, args) -> str:
    import copy
    import torch
    import trainall
    from trainall.data import InMemorySource
    from trainall.training import Trainer
    from trainall.types import Batch, PreferenceSample

    trainall.seed_everything(0)
    model = be.new_model()
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.to(args.device)  # the Trainer moves the policy to device but not batch.extra["ref_model"]

    def _pad(seqs, fill):
        m = max(len(s) for s in seqs)
        return torch.tensor([s + [fill] * (m - len(s)) for s in seqs], dtype=torch.long)

    def collate(items):
        c_ids, c_lab, r_ids, r_lab = [], [], [], []
        for s in items:
            p, cho, rej = be.ids(s.prompt), be.ids(s.chosen), be.ids(s.rejected)
            c_ids.append(p + cho); c_lab.append([-100] * len(p) + cho)
            r_ids.append(p + rej); r_lab.append([-100] * len(p) + rej)
        b = Batch.of(
            chosen_input_ids=_pad(c_ids, 0), chosen_attention_mask=_pad([[1] * len(x) for x in c_ids], 0),
            chosen_labels=_pad(c_lab, -100),
            rejected_input_ids=_pad(r_ids, 0), rejected_attention_mask=_pad([[1] * len(x) for x in r_ids], 0),
            rejected_labels=_pad(r_lab, -100),
        )
        b.extra["ref_model"] = ref
        return b

    prefs = [
        PreferenceSample(prompt="Be helpful: ", chosen="here is a clear answer", rejected="no."),
        PreferenceSample(prompt="Greet: ", chosen="hello, how can I help?", rejected="go away"),
        PreferenceSample(prompt="Explain: ", chosen="step by step reasoning", rejected="idk"),
    ] * 2
    objective = trainall.build("dpo", category="objective", beta=0.1)
    _, losses = _train(args, model=model, objective=objective, data=InMemorySource(prefs),
                       collate=collate, batch_size=3)
    last = losses[-1] if losses else float("nan")
    if not _finite(last):
        raise RuntimeError(f"non-finite loss {last}")
    return f"ref-model DPO ok, loss~{last:.3f}"


def _grpo_collate(be: Backend, trajs):
    import torch
    from trainall.types import Batch
    rows, masks, advs = [], [], []
    for t in trajs:
        p, r = be.ids(t.prompt), be.ids(t.response or " ")
        rows.append(p + r)
        masks.append([0] * len(p) + [1] * len(r))
        advs.append(float(t.advantage or 0.0))
    m = max(len(x) for x in rows)
    pad = lambda seqs, f: torch.tensor([s + [f] * (m - len(s)) for s in seqs], dtype=torch.long)
    return Batch.of(input_ids=pad(rows, 0), attention_mask=pad([[1] * len(x) for x in rows], 0),
                    response_mask=pad(masks, 0), advantages=torch.tensor(advs, dtype=torch.float32))


def _score_and_advantage(trajs, answers):
    import trainall
    from trainall.rl import compute_group_advantages
    verifier = trainall.build("math", category="verifier")
    reward = trainall.build("verifier", category="reward", verifier=verifier)
    for t in trajs:
        t.meta["reference"] = answers.get(t.prompt.strip(), "0")
    scores = reward.score(list(trajs))
    for t, s in zip(trajs, scores):
        t.reward = s
    compute_group_advantages(trajs)
    return sum(scores) / max(len(scores), 1)


def _grpo_tasks(be: Backend, n: Optional[int] = None):
    """(prompts, answers) for GRPO. Real models get an instruction-formatted prompt
    so they actually emit a parseable \\boxed answer the verifier can reward."""
    pairs = ARITH if n is None else ARITH[:n]
    if be.is_hf:
        items = [(f"Compute {expr}. Give only the final answer in \\boxed{{}}.", a) for expr, a in pairs]
    else:
        items = [(expr, a) for expr, a in pairs]
    return [p for p, _ in items], {p: a for p, a in items}


def smoke_grpo(be: Backend, args) -> str:
    import random
    import trainall
    from trainall.rl import Rollout, RolloutConfig
    from trainall.training import Trainer

    trainall.seed_everything(0)
    random.seed(0)
    prompts, answers = _grpo_tasks(be, n=3)

    if be.is_hf:
        model = be.new_model().to(args.device)
        rollout = Rollout(policy=model, tokenizer=be.hf_tokenizer,
                          config=RolloutConfig(group_size=args.group_size, max_new_tokens=48,
                                               temperature=1.0))
        trajs = rollout.group_sample(prompts)
    else:
        def policy(prompt: str) -> str:
            correct = answers.get(prompt.strip(), "0")
            return f"\\boxed{{{correct}}}" if random.random() < 0.5 else f"\\boxed{{{random.randint(0,20)}}}"
        rollout = Rollout(policy=policy, config=RolloutConfig(group_size=args.group_size))
        trajs = rollout.group_sample(prompts)
        model = be.new_model()

    mean_r = _score_and_advantage(trajs, answers)
    objective = trainall.build("grpo", category="objective", clip_range=0.2)
    _, losses = _train(args, model=model, objective=objective, data=[_grpo_collate(be, trajs)],
                       batch_size=len(trajs))
    last = losses[-1] if losses else float("nan")
    if not _finite(last):
        raise RuntimeError(f"non-finite loss {last}")
    return f"{len(trajs)} rollouts, mean_reward={mean_r:.2f}, grpo loss~{last:.3f}"


def smoke_agentic(be: Backend, args) -> str:
    import re
    import trainall
    from trainall.rl import AgenticRunner, compute_group_advantages
    from trainall.training import Trainer
    from trainall.types import Sample

    trainall.seed_everything(0)

    def scripted(observation):
        text = str(observation)
        m = re.search(r"Compute ([\d\s\+\-\*/\.\(\)]+)\.", text)
        if m:
            return f"calculator: {m.group(1).strip()}"
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return f"answer: {nums[-1]}" if nums else "answer: 0"

    env = trainall.build("expression_env", category="environment")
    runner = AgenticRunner(env=env, policy=scripted, max_steps=4)
    tasks = [Sample(prompt="Compute 6 * 7.", reference=42.0),
             Sample(prompt="Compute 12 + 5.", reference=17.0)]
    samples = [s for s in tasks for _ in range(args.group_size)]
    trajs = runner.collect(samples)
    for i, t in enumerate(trajs):
        t.group_id = i % len(tasks)
    compute_group_advantages(trajs)
    succ = sum(t.meta.get("success", False) for t in trajs) / len(trajs)

    model = be.new_model()
    objective = trainall.build("grpo", category="objective")
    _train(args, model=model, objective=objective, data=[_grpo_collate(be, trajs)], batch_size=len(trajs))
    return f"{len(trajs)} episodes, success={succ:.2f}, grpo step ok"


def smoke_pretrain(be: Backend, args) -> str:
    import trainall
    from trainall.data import InMemorySource, pack_sequences
    from trainall.training import Trainer

    trainall.seed_everything(0)
    corpus = ["the quick brown fox jumps over the lazy dog. ",
              "language models predict the next token. ",
              "attention is all you need for sequences. "]
    token_lists = [be.byte_ids(doc * 4) for doc in corpus]  # vocab-256 model → byte ids
    packed = pack_sequences(token_lists, max_len=MAX_LEN, pad_id=0)
    data = InMemorySource([{"input_ids": ids, "labels": list(ids)} for ids in packed])
    model = be.tiny_decoder(n_layers=2)  # the from-scratch path is always a DecoderLM
    objective = trainall.build("pretrain", category="objective")
    _, losses = _train(args, model=model, objective=objective, data=data)
    last = losses[-1] if losses else float("nan")
    if not _finite(last):
        raise RuntimeError(f"non-finite loss {last}")
    return f"packed {len(packed)} windows, scratch DecoderLM loss~{last:.3f}"


def smoke_distill(be: Backend, args) -> str:
    import torch
    import trainall
    from trainall.data import InMemorySource
    from trainall.training import Trainer
    from trainall.types import Batch

    trainall.seed_everything(0)
    teacher = be.tiny_decoder(n_layers=2).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = be.tiny_decoder(n_layers=1)
    dev = args.device
    teacher.to(dev)

    def collate(items):
        ids = [it["input_ids"] for it in items]
        m = max(len(x) for x in ids)
        input_ids = torch.tensor([x + [0] * (m - len(x)) for x in ids], dtype=torch.long)
        labels = torch.tensor([it["labels"] + [-100] * (m - len(it["labels"])) for it in items], dtype=torch.long)
        attn = torch.tensor([[1] * len(x) + [0] * (m - len(x)) for x in ids], dtype=torch.long)
        with torch.no_grad():
            tl = teacher(input_ids=input_ids.to(dev), attention_mask=attn.to(dev)).logits
        b = Batch.of(input_ids=input_ids, attention_mask=attn, labels=labels)
        b.extra["teacher_logits"] = tl
        return b

    data = InMemorySource([byte_sft_record(p, r) for p, r in SFT_PAIRS] * 2)  # vocab-256 models
    objective = trainall.build("distill", category="objective", temperature=2.0, alpha=0.5)
    _, losses = _train(args, model=student, objective=objective, data=data, collate=collate, batch_size=3)
    last = losses[-1] if losses else float("nan")
    if not _finite(last):
        raise RuntimeError(f"non-finite loss {last}")
    return f"KD student<-teacher loss~{last:.3f}"


def stage_smoke(args) -> None:
    # Each path runs in its own subprocess so one CUDA fault can't poison the
    # shared context and cascade into false failures for every later path.
    print("\n=== Stage 2: smoke matrix (isolated subprocesses) ===", flush=True)
    _run_units("2", SMOKE_ORDER, args)


# ============================================================================ #
# Stage 3 — short convergence (SFT loss down, GRPO reward up)
# ============================================================================ #
def _trend(series: List[float]) -> Dict[str, float]:
    n = len(series)
    if n < 4:
        return {"first": series[0] if series else 0.0, "last": series[-1] if series else 0.0, "delta": 0.0}
    k = max(1, n // 4)
    first = sum(series[:k]) / k
    last = sum(series[-k:]) / k
    return {"first": round(first, 4), "last": round(last, 4), "delta": round(last - first, 4)}


def _save_curve(out: str, name: str, xs, ys, ylabel: str) -> None:
    os.makedirs(out, exist_ok=True)
    csv = os.path.join(out, f"{name}.csv")
    with open(csv, "w") as fh:
        fh.write(f"step,{ylabel}\n")
        for x, y in zip(xs, ys):
            fh.write(f"{x},{y}\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 3))
        plt.plot(xs, ys, marker="o", ms=3)
        plt.xlabel("step"); plt.ylabel(ylabel); plt.title(name); plt.tight_layout()
        plt.savefig(os.path.join(out, f"{name}.png"), dpi=110)
        plt.close()
    except Exception:  # noqa: BLE001 - matplotlib optional
        pass


def converge_sft(be: Backend, args) -> str:
    import trainall
    from trainall.data import InMemorySource

    trainall.seed_everything(0)
    model = be.new_model()
    objective = trainall.build("sft", category="objective")
    algorithm = trainall.build("lora", category="algorithm", r=8, alpha=16, target_modules=be.lora_targets)
    data = InMemorySource([sft_record(be, p, r) for p, r in SFT_PAIRS] * 8)
    _, losses = _train(args, model=model, objective=objective, algorithm=algorithm, data=data,
                       lr=5e-4 if be.is_hf else 5e-3, batch_size=4, max_steps=args.conv_steps, log_every=1)
    if not losses:
        raise RuntimeError("no loss history captured")
    tr = _trend(losses)
    _save_curve(args.out, "sft_loss", list(range(1, len(losses) + 1)), losses, "loss")
    status = "down" if tr["delta"] < 0 else "flat/up"
    return f"loss {tr['first']:.3f}->{tr['last']:.3f} (Δ{tr['delta']:+.3f}, {status}) over {len(losses)} steps"


def converge_grpo(be: Backend, args) -> str:
    import random
    import trainall
    from trainall.rl import Rollout, RolloutConfig
    from trainall.training import Trainer

    trainall.seed_everything(0)
    random.seed(0)
    prompts, answers = _grpo_tasks(be)
    rewards_per_iter: List[float] = []

    if be.is_hf:
        model = be.new_model().to(args.device)
    else:
        model = be.new_model()

    objective = trainall.build("grpo", category="objective", clip_range=0.2)
    for it in range(args.conv_iters):
        if be.is_hf:
            rollout = Rollout(policy=model, tokenizer=be.hf_tokenizer,
                              config=RolloutConfig(group_size=args.group_size, max_new_tokens=48, temperature=1.0))
            trajs = rollout.group_sample(prompts)
        else:
            # toy policy that improves: probability of correctness rises each iter
            p_correct = min(0.2 + 0.1 * it, 0.9)

            def policy(prompt: str, _p=p_correct) -> str:
                c = answers.get(prompt.strip(), "0")
                return f"\\boxed{{{c}}}" if random.random() < _p else f"\\boxed{{{random.randint(0,20)}}}"
            rollout = Rollout(policy=policy, config=RolloutConfig(group_size=args.group_size))
            trajs = rollout.group_sample(prompts)

        mean_r = _score_and_advantage(trajs, answers)
        rewards_per_iter.append(mean_r)
        _train(args, model=model, objective=objective, data=[_grpo_collate(be, trajs)],
               batch_size=len(trajs))

    tr = _trend(rewards_per_iter)
    _save_curve(args.out, "grpo_reward", list(range(1, len(rewards_per_iter) + 1)), rewards_per_iter, "mean_reward")
    status = "up" if tr["delta"] > 0 else "flat/down"
    return f"reward {tr['first']:.2f}->{tr['last']:.2f} (Δ{tr['delta']:+.2f}, {status}) over {len(rewards_per_iter)} iters"


def stage_converge(args) -> None:
    print("\n=== Stage 3: short convergence (isolated subprocesses) ===", flush=True)
    _run_units("3", CONV_ORDER, args)


# --- unit registry + subprocess isolation ---------------------------------- #
SMOKE_ORDER = ["sft-full", "lora", "qlora", "dpo", "grpo", "agentic", "pretrain", "distill"]
CONV_ORDER = ["sft-converge", "grpo-converge"]
ALL_UNITS = {
    "sft-full": lambda be, a: smoke_sft(be, a, "full"),
    "lora": lambda be, a: smoke_sft(be, a, "lora"),
    "qlora": lambda be, a: smoke_sft(be, a, "qlora"),
    "dpo": smoke_dpo,
    "grpo": smoke_grpo,
    "agentic": smoke_agentic,
    "pretrain": smoke_pretrain,
    "distill": smoke_distill,
    "sft-converge": converge_sft,
    "grpo-converge": converge_grpo,
}


def _spawn_unit(stage: str, name: str, args) -> Result:
    """Run one validation unit in a fresh process; parse its RESULTJSON line."""
    cmd = [sys.executable, os.path.abspath(__file__), "--unit", name,
           "--backend", args.backend, "--device", args.device, "--model", args.model,
           "--smoke-steps", str(args.smoke_steps), "--conv-steps", str(args.conv_steps),
           "--conv-iters", str(args.conv_iters), "--group-size", str(args.group_size),
           "--out", args.out]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.unit_timeout,
                           env=dict(os.environ, PYTHONPATH=_SRC))
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULTJSON:")), None)
        if line:
            d = json.loads(line[len("RESULTJSON:"):])
            r = Result(stage, name, d["status"], d["detail"],
                       d.get("seconds", time.time() - t0), d.get("peak_mb"))
        else:  # no result line → the unit hard-crashed (e.g. a CUDA abort killed the process)
            err = [l for l in (p.stderr or p.stdout).splitlines() if l.strip()]
            r = Result(stage, name, "FAIL", f"subprocess rc={p.returncode}: {(err[-1] if err else '')[:90]}",
                       time.time() - t0)
    except subprocess.TimeoutExpired:
        r = Result(stage, name, "FAIL", f"timeout >{args.unit_timeout}s", time.time() - t0)
    if r.status == "PASS" and name == "sft-converge" and "down" not in r.detail:
        r.status = "WARN"
    if r.status == "PASS" and name == "grpo-converge" and "up" not in r.detail:
        r.status = "WARN"
    _emit(r)
    return r


def _run_units(stage: str, order: List[str], args) -> None:
    for name in order:
        _spawn_unit(stage, name, args)


# ============================================================================ #
# main
# ============================================================================ #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["toy", "hf"], default="toy")
    ap.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model id (--backend hf)")
    ap.add_argument("--stages", default="0,1,2,3", help="comma list of stages to run")
    ap.add_argument("--skip-suite", action="store_true", help="skip Stage 1 (pytest+examples)")
    ap.add_argument("--quick", action="store_true", help="fewer steps everywhere")
    ap.add_argument("--smoke-steps", type=int, default=5)
    ap.add_argument("--conv-steps", type=int, default=60, help="SFT convergence steps")
    ap.add_argument("--conv-iters", type=int, default=8, help="GRPO convergence iterations")
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--suite-timeout", type=int, default=900)
    ap.add_argument("--unit-timeout", type=int, default=600, help="per-path subprocess timeout")
    ap.add_argument("--unit", default=None, help="internal: run one isolated unit and print RESULTJSON")
    ap.add_argument("--out", default="gpu_validation_out")
    args = ap.parse_args(argv)

    if args.device == "auto":
        try:
            args.device = "cuda" if _torch().cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            args.device = "cpu"
    if args.quick:
        args.smoke_steps = min(args.smoke_steps, 3)
        args.conv_steps = min(args.conv_steps, 12)
        args.conv_iters = min(args.conv_iters, 4)

    # Worker mode: run exactly one unit in this (fresh) process and report it.
    if args.unit:
        be = Backend(args.backend, args.device, args.model)
        stage = "3" if args.unit in CONV_ORDER else "2"
        r = run_step(stage, args.unit, lambda: ALL_UNITS[args.unit](be, args), args.device)
        print("RESULTJSON:" + json.dumps(asdict(r)), flush=True)
        return 1 if r.status == "FAIL" else 0

    stages = set(s.strip() for s in args.stages.split(","))
    print(f"trainall gpu_validate — backend={args.backend} device={args.device} "
          f"model={args.model if args.backend == 'hf' else '(toy DecoderLM)'}", flush=True)

    env_info = stage_preflight(args) if "0" in stages else {}
    # Abort the model stages early if cuda was requested but is unusable.
    cuda_dead = args.device == "cuda" and not env_info.get("cuda_available", True)

    if "1" in stages and not args.skip_suite:
        stage_suite(args)

    if not cuda_dead:
        if "2" in stages:
            stage_smoke(args)
        if "3" in stages:
            stage_converge(args)

    # ---- summary matrix ---- #
    print("\n=== summary ===", flush=True)
    counts: Dict[str, int] = {}
    for r in RESULTS:
        counts[r.status] = counts.get(r.status, 0) + 1
    order = {"FAIL": 0, "WARN": 1, "SKIP": 2, "PASS": 3}
    for r in sorted(RESULTS, key=lambda r: (r.stage, order.get(r.status, 9), r.name)):
        if r.status in ("FAIL", "WARN"):
            print(f"  {_ICON.get(r.status,'')} [{r.stage}] {r.name}: {r.detail}", flush=True)
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump({"args": vars(args), "env": env_info,
                   "results": [asdict(r) for r in RESULTS]}, fh, indent=2)
    print(f"  report -> {os.path.join(args.out, 'report.json')}", flush=True)

    return 1 if counts.get("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
