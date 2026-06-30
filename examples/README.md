# trainall examples

Runnable, self-contained examples for the `trainall` library. Every script runs
**end-to-end on CPU in a few seconds** using a tiny `DecoderLM` (byte-level
vocab) and in-memory toy data — no downloads, no GPU, no real datasets.

Each script has two paths:

- **tiny** (default): the fast, dependency-light demo described above.
- **`--real`**: an opt-in pointer/sketch for scaling up to real HF models or
  datasets. These are gated behind the `--real` flag and are *not* run by the
  tiny-mode validation.

## Running

From the repo root, with the package on the path:

```bash
PYTHONPATH=src python3 examples/01_sft.py
PYTHONPATH=src python3 examples/02_dpo.py
# ... etc.
```

(The scripts import a shared `examples/_toy.py` helper — Python adds the script's
own directory to `sys.path`, so running them from anywhere works as long as
`trainall` itself is importable via `PYTHONPATH=src`.)

## Index

| Script | Demonstrates | Key public API |
| --- | --- | --- |
| [`01_sft.py`](01_sft.py) | Supervised fine-tuning (response-only NLL) | `Trainer`, `build("sft")`, `InMemorySource` |
| [`02_dpo.py`](02_dpo.py) | Direct Preference Optimisation on chosen/rejected pairs | `build("dpo")`, custom preference `collate`, frozen ref model in `batch.extra` |
| [`03_rlvr_grpo.py`](03_rlvr_grpo.py) | RL with verifiable rewards via GRPO | `rl.Rollout`, `build("math")` verifier, `verifier` reward, `compute_group_advantages`, `build("grpo")` |
| [`04_agentic_rl.py`](04_agentic_rl.py) | Multi-step tool-use RL (calculator env) | `rl.AgenticRunner`, `expression_env`, `CalculatorTool`, GRPO |
| [`05_distill_and_selfplay.py`](05_distill_and_selfplay.py) | Distillation + the verifier-gated data flywheel | `build("distill")`, `RejectionSampler`, `SyntheticDataEngine`, `SelfPlayLoop`, `Curriculum` |
| [`06_pretrain_from_scratch.py`](06_pretrain_from_scratch.py) | Pre-training a fresh `DecoderLM` on packed text | `build("pretrain")`, `pack_sequences`, `models.ArchConfig` / `DecoderLM` |
| [`07_lora_qlora.py`](07_lora_qlora.py) | Parameter-efficient fine-tuning | `build("lora")` / `build("qlora")`, `LoRAConfig` |

## Declarative configs

[`configs/`](configs/) holds illustrative `RunConfig` YAML files loadable with
`trainall.load_config(...)` and runnable via `Trainer.from_config(cfg)`:

- [`configs/sft.yaml`](configs/sft.yaml) — SFT, full fine-tuning.
- [`configs/dpo.yaml`](configs/dpo.yaml) — DPO with a LoRA adapter.
- [`configs/grpo.yaml`](configs/grpo.yaml) — GRPO/RLVR with an `rl` block.

```python
import trainall
from trainall.training import Trainer

cfg = trainall.load_config("examples/configs/sft.yaml")
Trainer.from_config(cfg).train()   # builds model + objective + data from the registry
```

## Pipelines / recipes

The single-phase scripts above map onto the named recipes in
`trainall.pipelines` (`sft_recipe`, `dpo_recipe`, `rlvr_recipe`,
`agentic_rlvr_recipe`, `distill_recipe`, `cpt_recipe`) and the end-to-end
`frontier_pipeline(...)` that chains CPT → SFT → expand → DPO → RLVR → Agentic →
Distill. Pass `tiny=True` to any of them for the same fast CPU path:

```python
from trainall.pipelines import sft_recipe
from trainall.models import ArchConfig, DecoderLM

# Pass a model whose vocab matches your tokenisation (the byte-level toy data
# in these examples uses vocab 256; the recipe's built-in default is vocab 64).
model = DecoderLM.from_config(ArchConfig(vocab_size=256, dim=32, n_layers=2, max_seq_len=64))
result = sft_recipe(model=model, data=my_tokenised_records, tiny=True).run()
```
