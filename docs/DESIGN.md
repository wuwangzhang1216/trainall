# Design

How `trainall` is put together, and why.

## The spine and the leaves

The library is split into a small **spine** that defines fixed contracts and a set of **leaf** modules that fill them in.

The spine is four pure-python files, importable with no ML stack:

| File | Role |
| --- | --- |
| `types.py` | Data contracts. Plain dataclasses: `Sample`, `PreferenceSample`, `Batch`, `Trajectory`, `Episode`, `Transition`, `Message`, `VerifierResult`. These are the only currency that flows between components. |
| `base.py` | The abstract base classes every component implements: `Objective`, `Algorithm`, `Verifier`, `Reward`, `DataSource`, `Environment`. |
| `registry.py` | A tiny category-aware registry plus the `build` factory. `@register` records a class under a string key; `build(key, category=...)` instantiates it. |
| `config.py` | `RunConfig` (the universal intermediate representation) and its per-axis dataclasses, with dict/YAML round-tripping via `load_config`. |

Everything else — `models`, `objectives`, `algorithms`, `verifiers`, `rewards`, `rl`, `data`, `training`, `pipelines` — is a leaf subpackage. Leaves depend on the spine; the spine never depends on a leaf.

## The three orthogonal axes

The central design claim is that any LLM training method factors into three independent choices:

```
        data            ×          objective          ×        algorithm
  (where samples come          (what the model is           (how parameters
   from + tokenization)         rewarded to become)             update)
```

- **data** (`trainall.data`, category `datasource`) — a `DataSource` yields `Sample` / `PreferenceSample`. Includes tokenization helpers (`ChatTemplate`, `mask_prompt`, `Packer`) and the data *generation* paths (`SyntheticDataEngine`, `RejectionSampler`, `SelfPlayLoop`).
- **objective** (`trainall.objectives`, category `objective`) — an `Objective` exposes `compute_loss(model, batch) -> (loss, metrics)`. This is the only place the *loss* lives. Switching from SFT to DPO to GRPO is just swapping this one component.
- **algorithm** (`trainall.algorithms`, category `algorithm`) — an `Algorithm` exposes `prepare_model(model) -> model`. `FullFinetune` is a pass-through; `LoRA`/`QLoRA` swap target `nn.Linear`s for adapter modules and freeze the base.

Because the axes are orthogonal, the same objective runs under any algorithm and over any data source. QLoRA is therefore not a training *method* — it is an efficiency layer composable under *every* objective. The `Trainer` is the small piece of glue that holds one choice from each axis:

```python
Trainer(model, objective, algorithm=None, data=None, config=TrainerConfig(...))
```

`RunConfig` mirrors the axes one-to-one (`model`, `data`, `objective`, `algorithm`, `optim`, `train`, `rl`), so a YAML file *is* a point in the cross-product, and `Trainer.from_config(cfg)` reconstructs it by calling `build` on each axis's `name`.

## The registry and `build`

The registry is keyed by **category** so the same short name can live in two categories without colliding:

- `build(key, category)` looks up exactly one bucket.
- `build(key)` does a global search and instantiates only if the key is unambiguous; otherwise it raises with the list of categories.

This is why `dpo` (registered as both an `objective` and a `recipe`) must be built as `build("dpo", category="objective")`, while `qlora` (only an algorithm) works bare.

`@register` also accepts `aliases=[...]`, which is how `pretrain`/`clm`, `reward_model`/`rm`/`bt`, and `kd`/`distill` map several keys onto one class. When the decorated object is a class, its `.name` attribute is set to the primary key so an instance can report its own registry identity.

`available(category=None)` returns `{category: [sorted keys]}` and powers both `trainall.available()` and `trainall list`.

## The lazy-import philosophy

`import trainall` must be cheap and must succeed with **nothing but `pyyaml`** installed. Two mechanisms enforce this:

1. **Lazy submodule access (PEP 562).** The top-level `__init__` only eagerly imports the pure-python spine. The heavy subpackages are listed in `_LAZY_SUBMODULES` and resolved on first attribute access via `__getattr__`. So `trainall.objectives` triggers the import only when you touch it.

2. **Lazy heavy dependencies inside functions.** Components never `import torch` (or transformers / trl / peft / datasets / bitsandbytes / sympy / jsonschema) at module top level. They import inside the method that needs it, through `trainall._optional`:

   ```python
   from .._optional import require
   def compute_loss(self, model, batch):
       torch = require("torch", feature="DPO loss")
       ...
   ```

   `require(pkg, feature=...)` raises a clear "install `trainall[train]`" message if the package is missing; `has(pkg)` lets a component degrade gracefully (e.g. `MathVerifier` falling back to numeric matching when `sympy` is absent, `QLoRA` falling back to fp weights with a warning when `bitsandbytes` is unavailable).

   **The two deliberate exceptions:** the `models/` subpackage and `algorithms/lora.py` / `qlora.py` import torch at module top level, because they subclass `nn.Module` at *class-definition* time. They are never imported by the cheap path.

The registry's `_bootstrap` ties these together: it imports the registrable subpackages inside a `try/except` so that the torch-required ones simply don't register when torch is missing — `import trainall` and `trainall.available()` still work, they just show fewer keys.

## How to add a new component

Every component is a class decorated with `@register` and matching the relevant ABC. The pattern is identical across axes.

### A new objective

```python
# trainall/objectives/my_loss.py
from __future__ import annotations
from ..base import Objective
from ..registry import register
from .._optional import require

@register("myloss", category="objective")
class MyLossObjective(Objective):
    """One-line idea + paper citation."""
    requires_reference_model = False    # set where relevant
    is_on_policy = False

    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def compute_loss(self, model, batch):
        torch = require("torch", feature="MyLoss")
        # read tensors from `batch` per the Batch contract, return (loss, metrics)
        ...
        return loss, {"loss": float(loss)}
```

Then export it from `trainall/objectives/__init__.py`. `build("myloss", category="objective")` now works, and so does a YAML `objective: { name: myloss }`.

### A new verifier

```python
@register("myverifier", category="verifier")
class MyVerifier(Verifier):
    def verify(self, response, reference=None, *, prompt=None, **kw):
        ...
        return VerifierResult(reward=1.0, passed=True, detail="ok")
```

`verify` must be deterministic, return a reward in `[0, 1]`, and be robust to messy input. Use `_optional.require`/`has` for optional deps (`sympy`, `jsonschema`) and degrade gracefully when they are absent. Anything that runs untrusted code (a `CodeVerifier`) does so in a subprocess with a timeout, never in-process `exec`.

### A new algorithm

```python
@register("myalgo", category="algorithm")
class MyAlgo(Algorithm):
    def prepare_model(self, model):
        # mutate / wrap `model` (freeze params, swap layers, ...) and return it
        return model
```

`prepare_model` is called once by the `Trainer` before the loop. Module-subclassing algorithms may import torch at top level (see the exceptions above).

### Wiring it into config and pipelines

Once registered and exported, a component is reachable three ways with no further changes:

1. directly — `trainall.build("myloss", category="objective", beta=0.2)`
2. by config — `RunConfig.objective.name = "myloss"`, then `Trainer.from_config`
3. inside a recipe — recipes build their objective/algorithm/data through the same `build`, so a `Stage` can name your key.

## Pipelines

`pipelines` add a fourth, *temporal* layer on top of the three axes. A `Stage(name, builder)` produces a `StageResult(model, metrics, name)`; a `Pipeline(stages)` threads each stage's output model into the next via a shared run-context. Recipe factories (`sft_recipe`, `dpo_recipe`, … and `frontier_pipeline`) just assemble `Stage`s, each of which builds a single-axis training run through the registry and a `Trainer`. A `tiny` / `dry_run` mode swaps in a tiny `DecoderLM` and clamps the step budget so the whole frontier chain runs on CPU in tests.
