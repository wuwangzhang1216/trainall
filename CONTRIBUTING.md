# Contributing to trainall

Thanks for your interest in improving **trainall** — one library for the frontier LLM training stack.

## Development setup

```bash
git clone https://github.com/wuwangzhang1216/LLM-Train-All.git
cd LLM-Train-All
pip install -e '.[all]'      # core + train + verify + quant
# or the lighter dev set:
pip install -e '.[dev]'      # core + verifiers + pytest
```

Run the test suite (CPU, seconds):

```bash
PYTHONPATH=src python3 -m pytest -q
```

Everything is built to run on CPU with tiny toy models, so you do **not** need a GPU to develop or test.

## The one hard rule: keep `import trainall` dependency-free

`import trainall` must work with **no ML stack installed**. Never import `torch` /
`transformers` / `peft` / `trl` / `datasets` at module top level — pull them in
lazily inside the function that needs them:

```python
from .._optional import require

def compute_loss(self, model, batch):
    torch = require("torch", feature="my loss")
    ...
```

The only sanctioned exceptions are the `models/` subpackage and
`algorithms/lora.py` / `qlora.py`, whose classes subclass `torch.nn.Module` at
definition time and are therefore torch-required by design.

## Adding a component

Every objective, algorithm, verifier, reward, data source and recipe registers
itself with a string key and is then reachable through `trainall.build(...)`.
See [`docs/DESIGN.md`](docs/DESIGN.md) ("Extending") for the contracts. In short:

```python
from trainall.base import Objective
from trainall.registry import register

@register("my_loss", category="objective")
class MyObjective(Objective):
    def __init__(self, beta: float = 0.1):
        self.beta = beta
    def compute_loss(self, model, batch):
        ...  # return (scalar_loss, metrics_dict)
```

Add a test under `tests/` that exercises it on a tiny CPU example, and (for a
new training method) a doc under `docs/methods/`.

## Style

- `from __future__ import annotations` at the top of every module.
- Concise docstrings that explain the *idea* and cite the source paper.
- Provide `__all__`.
- Match the surrounding code's conventions.

## Pull requests

1. Branch off `main`.
2. Keep the change focused; add/adjust tests.
3. Make sure `pytest -q` is green and `python -c "import trainall"` stays torch-free.
4. Open the PR with a clear description of *what* and *why*.

By contributing you agree that your contributions are licensed under the
[MIT License](LICENSE).
