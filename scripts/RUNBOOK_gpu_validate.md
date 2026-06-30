# Blackwell (RTX 50-series) validation runbook

End-to-end "does every training path actually run on real hardware?" check for
`trainall`, plus a short SFT/GRPO convergence signal. Driver:
[`scripts/gpu_validate.py`](gpu_validate.py).

What it proves (and what it does **not**): it proves every path *executes* on a
real model + real GPU and that two paths show a learning *signal* over a short
run. It does **not** train any method to convergence — that is days of compute,
not the point of this harness.

## Confirmed run — RTX 5080, Blackwell `sm_120`, 2026-06-30

torch `2.11.0+cu128` · bitsandbytes `0.49.2` · bf16. **8/8 smoke paths pass** with
`Qwen/Qwen2.5-0.5B-Instruct`, each in its own subprocess:

| path | peak VRAM | note |
|---|---|---|
| `sft-full` | 5.0 GB | full fine-tune, 494 M params |
| `lora` / `qlora` | ~1.0 GB | **qlora = real 4-bit on Blackwell** ✓ |
| `dpo` | 6.0 GB | policy + frozen reference |
| `grpo` / `agentic` | 5.0 GB | real `.generate` rollouts → GRPO |
| `pretrain` / `distill` | ~20 MB | from-scratch DecoderLM (vocab 256, byte ids) |

SFT convergence: loss **0.57 → 2e-5** over 60 steps. Caveats found: full-parameter
GRPO on a **1.5 B** model OOMs at 16 GB (use LoRA/QLoRA); GRPO's short-run reward
curve is noisy/unstable (see §3).

## 0. The one Blackwell gotcha

Blackwell (RTX 50-series: 5080, 5090, …) is **compute capability `sm_120`**. It needs
**CUDA 12.8+** and a **PyTorch built for sm_120** (the `cu128` wheels, torch ≥ 2.7).
A default / `cu121` / CPU wheel fails on the GPU with:

```
CUDA error: no kernel image is available for execution on the device
```

So install torch from the cu128 channel *first*, then the package extras (pip
won't clobber an already-satisfied torch):

```bash
# on the pod, in the repo root
python -m pip install --upgrade pip

# 1) Blackwell-capable torch FIRST
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2) the stack (train + verifiers + 4-bit quant + dev) — torch already satisfied
pip install -e '.[train,verify,dev]'
pip install bitsandbytes            # for real 4-bit QLoRA on Blackwell (>=0.45)
pip install matplotlib              # optional: saves loss/reward PNGs (CSV always written)
```

Stage 0 of the harness prints the verdict — confirm it shows
`cuda_available True`, `capability sm_120`, and a `bitsandbytes` version (not
`absent`) before trusting the rest.

## 1. Run it

```bash
# full run on the GPU with a small real model (downloads ~1GB on first use)
python scripts/gpu_validate.py \
    --backend hf --device cuda \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --out gpu_validation_out
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--backend toy` | no model/GPU needed — the CPU dry-run used to debug the harness |
| `--device {cuda,cpu,auto}` | where to run the model stages |
| `--model <hf id>` | swap the real model (e.g. `Qwen/Qwen2.5-1.5B-Instruct`) |
| `--stages 0,1,2,3` | pick stages: 0 preflight · 1 pytest+examples · 2 smoke · 3 convergence |
| `--skip-suite` | skip stage 1 (the pytest+examples subprocess pass) |
| `--quick` | fewer steps everywhere (fast sanity pass) |
| `--smoke-steps / --conv-steps / --conv-iters` | tune run lengths |
| `--group-size` | GRPO/agentic group size |

## 2. What you get

A pass/fail matrix on stdout, plus `gpu_validation_out/`:

- `report.json` — env (torch/cuda/sm/bf16/bnb) + every result with timing & peak VRAM
- `sft_loss.csv` / `.png` — SFT convergence curve (loss should trend **down**)
- `grpo_reward.csv` / `.png` — GRPO mean-reward curve (should trend **up**)

Stage 2 covers: `sft-full`, `lora`, `qlora`, `dpo`, `grpo+verifier`, `agentic-rl`,
`pretrain`, `distill`. Each reports PASS (ran, finite loss) / FAIL (+ traceback in
`report.json`). The `qlora` row reports **`real 4-bit`** when bitsandbytes
quantises on the GPU, or **`fp fallback`** otherwise — that line is the
bitsandbytes-on-Blackwell verdict.

## 3. Expectations / honest caveats

- **`qlora`** is the highest-risk row — Blackwell support in bitsandbytes is
  recent. *Verified working* with bnb 0.49.2 on `sm_120` (reports `real 4-bit`); a
  FAIL here would be a real, useful finding, not a harness bug.
- **Full-parameter RL needs headroom.** Full-FT GRPO on a 1.5 B model OOMs on a
  16 GB card — the optimizer state alone exceeds it. Use `lora`/`qlora` for RL on
  limited VRAM (the QLoRA 4-bit path is the verified way to do this on Blackwell).
- **A clean upward GRPO curve is its own project.** This harness's GRPO loop is
  intentionally naive (fresh optimizer per iter, no KL leash). With the formatted
  prompt the 0.5 B base scores a *nonzero* reward, but a naive full-FT update with
  no KL penalty destabilises it back down — exactly the hard part trainall's docs
  call out. Reported as `WARN`, not `FAIL`. A real curve needs LoRA + a KL leash +
  more iters.
- First HF run downloads the model + tokenizer (needs network on the pod).

## 4. CPU dry-run (already green on the dev box)

```bash
python scripts/gpu_validate.py --backend toy --device cpu --quick --stages 0,2,3
# → PASS=11 ; sft loss down, grpo reward up
```
