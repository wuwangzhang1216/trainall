"""The training loop that ties objective + algorithm + data together.

:class:`Trainer` is a deliberately small, dependency-light torch loop.  It is
the one place where the three orthogonal axes of the library meet:

* a :class:`~trainall.base.DataSource` supplies samples,
* a :class:`~trainall.base.Objective` turns a collated :class:`~trainall.types.Batch`
  into a scalar loss,
* an :class:`~trainall.base.Algorithm` decides which parameters move.

Everything heavy (torch, transformers) is imported lazily inside the methods
that need it so ``import trainall`` stays cheap.  The loop supports gradient
accumulation, gradient clipping, a cosine-with-warmup schedule, ``max_steps``
early stop, device auto-resolution (cpu / cuda / mps) and callbacks.  A
best-effort :meth:`to_hf_trainer` adapter is provided for users who want the
full HuggingFace ecosystem.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ..base import Algorithm, DataSource, Objective
from ..types import Batch
from ..utils import get_logger, seed_everything
from .callbacks import Callback, LoggingCallback

__all__ = ["Trainer", "TrainerConfig"]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class TrainerConfig:
    """Knobs for the :class:`Trainer` loop (mirrors ``config.TrainConfig``)."""

    lr: float = 1e-5
    epochs: float = 1.0
    batch_size: int = 8
    grad_accum: int = 1
    max_steps: Optional[int] = None
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    bf16: bool = True
    device: str = "auto"
    log_every: int = 10
    output_dir: str = "./out"
    seed: int = 0


# --------------------------------------------------------------------------- #
# Default collate
# --------------------------------------------------------------------------- #
def default_collate(items: Sequence[Any], pad_token_id: int = 0, ignore_index: int = -100) -> Batch:
    """Collate a list of items into a causal-LM / SFT :class:`Batch`.

    Robust to three input shapes:

    * an already-built :class:`Batch` (passed straight through),
    * a single :class:`Batch` wrapped in a list,
    * a list of dicts (or objects with ``input_ids`` / ``labels`` attributes)
      carrying python-list ``input_ids`` and optional ``labels``.  These are
      right-padded to the longest sequence; ``labels`` pad with ``ignore_index``
      (``-100``) and ``input_ids`` with ``pad_token_id``; an ``attention_mask``
      is synthesised.  Already-tensor fields are stacked when shapes match.
    """
    from .._optional import require

    if isinstance(items, Batch):
        return items
    if len(items) == 1 and isinstance(items[0], Batch):
        return items[0]

    torch = require("torch", feature="batch collation")

    def _field(item: Any, key: str) -> Any:
        if isinstance(item, dict):
            return item.get(key)
        return getattr(item, key, None)

    rows: List[Dict[str, Any]] = []
    for item in items:
        ids = _field(item, "input_ids")
        if ids is None:
            raise ValueError(
                "default_collate expects items with 'input_ids'; got "
                f"{type(item).__name__}. Provide tokenised samples or a custom collate."
            )
        labels = _field(item, "labels")
        if labels is None:
            labels = ids
        rows.append({"input_ids": list(ids), "labels": list(labels)})

    max_len = max(len(r["input_ids"]) for r in rows)
    input_ids, labels, attn = [], [], []
    for r in rows:
        ids, lab = r["input_ids"], r["labels"]
        pad = max_len - len(ids)
        attn.append([1] * len(ids) + [0] * pad)
        input_ids.append(ids + [pad_token_id] * pad)
        labels.append(lab + [ignore_index] * pad)

    return Batch.of(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.tensor(attn, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
    )


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class Trainer:
    """Minimal torch training loop over an objective + algorithm + data source.

    Parameters
    ----------
    model:
        A torch ``nn.Module`` (e.g. :class:`trainall.models.DecoderLM` or a HF
        ``AutoModelForCausalLM``).
    objective:
        An :class:`~trainall.base.Objective` mapping ``(model, batch) -> (loss, metrics)``.
    algorithm:
        An :class:`~trainall.base.Algorithm`; defaults to full fine-tuning.
    data:
        A :class:`~trainall.base.DataSource` or any iterable of samples.
    collate:
        ``items -> Batch``; defaults to :func:`default_collate` (SFT padding).
    config:
        A :class:`TrainerConfig`.
    callbacks:
        A list of :class:`~trainall.training.Callback`.  A
        :class:`LoggingCallback` is added when none is supplied.
    """

    def __init__(
        self,
        model: Any,
        objective: Objective,
        algorithm: Optional[Algorithm] = None,
        data: Optional[Any] = None,
        collate: Optional[Callable[[Sequence[Any]], Batch]] = None,
        config: Optional[TrainerConfig] = None,
        callbacks: Optional[Sequence[Callback]] = None,
    ) -> None:
        self.model = model
        self.objective = objective
        self.algorithm = algorithm if algorithm is not None else _default_algorithm()
        self.data = data
        self.collate = collate or default_collate
        self.config = config or TrainerConfig()
        self.callbacks: List[Callback] = (
            list(callbacks)
            if callbacks is not None
            else [LoggingCallback(log_every=self.config.log_every)]
        )
        self.log = get_logger("trainer")
        self._prepared = False
        self.global_step = 0

    # ------------------------------------------------------------------ #
    # Construction from a RunConfig
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: Any) -> "Trainer":
        """Build a fully-wired :class:`Trainer` from a :class:`trainall.config.RunConfig`.

        Builds the model (HF ``AutoModelForCausalLM.from_pretrained`` when
        ``cfg.model.pretrained`` is set, otherwise a from-scratch
        :class:`trainall.models.DecoderLM`), then resolves the objective,
        algorithm and data source through the registry.  All heavy imports are
        lazy.
        """
        import trainall

        model = cls._build_model(cfg)

        objective = trainall.build(
            cfg.objective.name, category="objective", **dict(cfg.objective.options or {})
        )
        algorithm = trainall.build(
            cfg.algorithm.name, category="algorithm", **dict(cfg.algorithm.options or {})
        )

        data = cls._build_data(cfg)

        tconf = TrainerConfig(
            lr=cfg.optim.lr,
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            grad_accum=cfg.train.grad_accum,
            max_steps=cfg.train.max_steps,
            grad_clip=cfg.optim.grad_clip,
            weight_decay=cfg.optim.weight_decay,
            warmup_ratio=cfg.optim.warmup_ratio,
            bf16=cfg.train.bf16,
            log_every=cfg.train.log_every,
            output_dir=cfg.train.output_dir,
            seed=cfg.train.seed,
        )
        return cls(model=model, objective=objective, algorithm=algorithm, data=data, config=tconf)

    @staticmethod
    def _build_model(cfg: Any) -> Any:
        if getattr(cfg.model, "pretrained", None):
            from .._optional import require

            transformers = require("transformers", feature="pretrained model loading")
            return transformers.AutoModelForCausalLM.from_pretrained(
                cfg.model.pretrained,
                trust_remote_code=getattr(cfg.model, "trust_remote_code", False),
            )
        from ..models import ArchConfig, DecoderLM

        arch = ArchConfig(**dict(getattr(cfg.model, "options", {}) or {}))
        return DecoderLM.from_config(arch)

    @staticmethod
    def _build_data(cfg: Any) -> Any:
        import trainall

        d = cfg.data
        opts = dict(getattr(d, "options", {}) or {})
        # Pass the common DataConfig fields the source may want; sources tolerate
        # unknown kwargs via their own signatures (we only forward known ones).
        for key in ("path", "split"):
            val = getattr(d, key, None)
            if val is not None and key not in opts:
                opts[key] = val
        try:
            return trainall.build(d.source, category="datasource", **opts)
        except TypeError:
            # Source has a stricter signature; retry with just the path.
            return trainall.build(d.source, category="datasource")

    # ------------------------------------------------------------------ #
    # Device
    # ------------------------------------------------------------------ #
    def _resolve_device(self, torch: Any) -> Any:
        want = self.config.device
        if want and want != "auto":
            return torch.device(want)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _autocast_dtype(self, torch: Any, device: Any) -> Any:
        if not self.config.bf16:
            return None
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return None  # keep CPU/mps in fp32 for a robust, finite toy loop

    # ------------------------------------------------------------------ #
    # Data iteration
    # ------------------------------------------------------------------ #
    def _iter_batches(self) -> Iterable[Batch]:
        """Yield collated batches from ``self.data`` grouping ``batch_size`` items."""
        if self.data is None:
            raise ValueError("Trainer has no data; pass data= or use from_config.")
        bs = max(1, self.config.batch_size)
        buf: List[Any] = []
        for item in self.data:
            if isinstance(item, Batch):
                # Pre-collated batch: flush any pending buffer then yield it.
                if buf:
                    yield self.collate(buf)
                    buf = []
                yield item
                continue
            buf.append(item)
            if len(buf) >= bs:
                yield self.collate(buf)
                buf = []
        if buf:
            yield self.collate(buf)

    # ------------------------------------------------------------------ #
    # Scheduler
    # ------------------------------------------------------------------ #
    @staticmethod
    def _lr_lambda(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
        total = max(1, total_steps)
        warmup = max(0, min(warmup_steps, total - 1))

        def fn(step: int) -> float:
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(warmup)
            progress = float(step - warmup) / float(max(1, total - warmup))
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return fn

    def _estimate_total_steps(self) -> int:
        if self.config.max_steps:
            return int(self.config.max_steps)
        # Best-effort: count optimiser updates across epochs.
        n_items = None
        try:
            n_items = len(self.data)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            n_items = None
        if not n_items:
            return 1
        batches = math.ceil(n_items / max(1, self.config.batch_size))
        updates = math.ceil(batches / max(1, self.config.grad_accum))
        return max(1, int(updates * max(1.0, self.config.epochs)))

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    def train(self) -> Any:
        """Run the training loop and return the (possibly adapter-wrapped) model."""
        from .._optional import require

        torch = require("torch", feature="training loop")
        seed_everything(self.config.seed)

        device = self._resolve_device(torch)
        autocast_dtype = self._autocast_dtype(torch, device)

        if not self._prepared:
            self.model = self.algorithm.prepare_model(self.model)
            self._prepared = True
        self.model.to(device)
        self.model.train()

        params = [p for p in self.algorithm.trainable_parameters(self.model) if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=self.config.lr, weight_decay=self.config.weight_decay
        )

        total_steps = self._estimate_total_steps()
        warmup_steps = int(round(self.config.warmup_ratio * total_steps))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, self._lr_lambda(total_steps, warmup_steps)
        )

        for cb in self.callbacks:
            cb.on_train_begin(trainer=self)

        grad_accum = max(1, self.config.grad_accum)
        # When max_steps is set it caps the run; allow extra epochs so the loop
        # can reach it even if one pass over the data yields fewer updates.
        if self.config.max_steps:
            epochs = 1 << 30
        else:
            epochs = max(1, int(math.ceil(self.config.epochs)))
        micro = 0
        last_metrics: Dict[str, float] = {}
        stop = False

        optimizer.zero_grad(set_to_none=True)
        for epoch in range(epochs):
            if stop:
                break
            for batch in self._iter_batches():
                if not isinstance(batch, Batch):
                    batch = self.collate(batch)
                batch.to(device)
                batch = self.objective.prepare_batch(batch)

                if autocast_dtype is not None:
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                        loss, metrics = self.objective.compute_loss(self.model, batch)
                else:
                    loss, metrics = self.objective.compute_loss(self.model, batch)

                (loss / grad_accum).backward()
                micro += 1
                last_metrics = {k: float(v) for k, v in (metrics or {}).items()}
                last_metrics.setdefault("loss", float(loss.detach()))

                if micro % grad_accum != 0:
                    continue

                if self.config.grad_clip and self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, self.config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                last_metrics["lr"] = float(scheduler.get_last_lr()[0])
                step_metrics = dict(last_metrics)
                for cb in self.callbacks:
                    cb.on_step_end(self.global_step, step_metrics, trainer=self)
                if self.global_step % max(1, self.config.log_every) == 0:
                    self.log.debug("step %d loss=%.4f", self.global_step, step_metrics.get("loss", float("nan")))

                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    stop = True
                    break

        for cb in self.callbacks:
            cb.on_train_end(trainer=self)
        return self.model

    # ------------------------------------------------------------------ #
    # Evaluate / save
    # ------------------------------------------------------------------ #
    def evaluate(self, data: Optional[Any] = None) -> Dict[str, float]:
        """Compute the mean objective loss over ``data`` (defaults to ``self.data``).

        A light stub: no metric beyond the objective's own ``loss``/metrics,
        averaged over batches, run under ``torch.no_grad()`` in eval mode.
        """
        from .._optional import require

        torch = require("torch", feature="evaluation")
        source = data if data is not None else self.data
        if source is None:
            return {}

        device = self._resolve_device(torch)
        self.model.to(device)
        self.model.eval()

        bs = max(1, self.config.batch_size)
        buf: List[Any] = []
        agg: Dict[str, float] = {}
        n = 0

        def _run(batch: Batch) -> None:
            nonlocal n
            batch.to(device)
            batch = self.objective.prepare_batch(batch)
            with torch.no_grad():
                loss, metrics = self.objective.compute_loss(self.model, batch)
            row = {k: float(v) for k, v in (metrics or {}).items()}
            row.setdefault("loss", float(loss.detach()))
            for k, v in row.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1

        for item in source:
            if isinstance(item, Batch):
                if buf:
                    _run(self.collate(buf))
                    buf = []
                _run(item)
                continue
            buf.append(item)
            if len(buf) >= bs:
                _run(self.collate(buf))
                buf = []
        if buf:
            _run(self.collate(buf))

        self.model.train()
        return {f"eval_{k}": (v / max(1, n)) for k, v in agg.items()}

    def save(self, path: Optional[str] = None) -> str:
        """Persist the trained model via the algorithm's ``save`` policy.

        HF models (with ``save_pretrained``) get a directory; bare torch models
        are written to a ``model.pt`` state-dict file inside ``output_dir``.
        Returns the path that was written.
        """
        out = path or self.config.output_dir
        os.makedirs(out, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            target = out
        else:
            target = os.path.join(out, "model.pt")
        self.algorithm.save(self.model, target)
        self.log.info("saved model artifact to %s", target)
        return target

    # ------------------------------------------------------------------ #
    # HuggingFace adapter (best-effort)
    # ------------------------------------------------------------------ #
    def to_hf_trainer(self, **kwargs: Any) -> Any:
        """Return a ``transformers.Trainer`` wired to this trainer's pieces.

        Best-effort interop for users who want the HF ecosystem (sharding,
        deepspeed, etc.).  Requires ``transformers``; the objective's loss is
        delegated through a thin ``Trainer.compute_loss`` override so any
        :class:`~trainall.base.Objective` keeps working.
        """
        from .._optional import require

        transformers = require("transformers", feature="to_hf_trainer")
        objective = self.objective
        collate = self.collate

        targs = transformers.TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.grad_accum,
            learning_rate=self.config.lr,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.grad_clip,
            num_train_epochs=float(self.config.epochs),
            max_steps=int(self.config.max_steps) if self.config.max_steps else -1,
            logging_steps=self.config.log_every,
            seed=self.config.seed,
            report_to=[],
            **kwargs.pop("training_args", {}),
        )

        def _collate_fn(features: Sequence[Any]) -> Dict[str, Any]:
            batch = collate(features)
            return dict(batch.tensors)

        class _ObjectiveTrainer(transformers.Trainer):  # type: ignore[misc]
            def compute_loss(self, model, inputs, return_outputs=False, **kw):  # noqa: ANN001
                loss, _metrics = objective.compute_loss(model, Batch.of(**inputs))
                return (loss, None) if return_outputs else loss

        self.model = self.algorithm.prepare_model(self.model)
        self._prepared = True
        return _ObjectiveTrainer(
            model=self.model,
            args=targs,
            train_dataset=kwargs.pop("train_dataset", self.data),
            data_collator=_collate_fn,
            **kwargs,
        )


def _default_algorithm() -> Algorithm:
    """Lazily build the registry's ``full`` algorithm as the default."""
    import trainall

    return trainall.build("full", category="algorithm")
