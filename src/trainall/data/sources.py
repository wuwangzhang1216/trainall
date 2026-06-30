"""Static data sources — JSONL files, HuggingFace datasets, in-memory lists.

Each source is a :class:`trainall.base.DataSource` that yields
:class:`~trainall.types.Sample` (SFT / pretrain) or
:class:`~trainall.types.PreferenceSample` (DPO & friends) objects, so any
objective can consume any source.  ``datasets`` is imported lazily so the
core library stays dependency-free.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional

from ..base import DataSource
from ..registry import register
from ..types import Message, PreferenceSample, Sample

__all__ = ["JsonlSource", "HFDatasetSource", "InMemorySource"]


# --------------------------------------------------------------------------- #
# Record -> Sample / PreferenceSample mapping
# --------------------------------------------------------------------------- #
def _messages_from(raw: Any) -> Optional[List[Message]]:
    if not isinstance(raw, list):
        return None
    out: List[Message] = []
    for m in raw:
        if isinstance(m, Message):
            out.append(m)
        elif isinstance(m, dict):
            out.append(
                Message(
                    role=str(m.get("role", "user")),
                    content=str(m.get("content", "")),
                    name=m.get("name"),
                    meta=dict(m.get("meta", {})),
                )
            )
    return out or None


def record_to_sample(rec: Dict[str, Any], kind: str = "auto") -> Any:
    """Map a raw dict record into a :class:`Sample` / :class:`PreferenceSample`.

    Recognised fields: ``text``, ``prompt``, ``response``, ``messages``,
    ``chosen``, ``rejected``, ``reference``, ``label``, ``margin``, ``meta``.
    ``kind`` is ``"auto" | "sample" | "preference"``.
    """
    has_pref = "chosen" in rec or "rejected" in rec
    if kind == "preference" or (kind == "auto" and has_pref):
        return PreferenceSample(
            prompt=str(rec.get("prompt", "")),
            chosen=rec.get("chosen"),
            rejected=rec.get("rejected"),
            label=rec.get("label"),
            margin=rec.get("margin"),
            meta=dict(rec.get("meta", {})),
        )
    return Sample(
        text=rec.get("text"),
        prompt=rec.get("prompt"),
        response=rec.get("response"),
        messages=_messages_from(rec.get("messages")),
        reference=rec.get("reference"),
        meta=dict(rec.get("meta", {})),
    )


@register("jsonl", category="datasource")
class JsonlSource(DataSource):
    """Stream samples from a newline-delimited JSON file.

    Each line is a JSON object whose fields are mapped onto a
    :class:`Sample` / :class:`PreferenceSample` by :func:`record_to_sample`.
    """

    def __init__(self, path: str, kind: str = "auto", encoding: str = "utf-8") -> None:
        self.path = path
        self.kind = kind
        self.encoding = encoding

    def __iter__(self) -> Iterator[Any]:
        with open(self.path, "r", encoding=self.encoding) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield record_to_sample(json.loads(line), self.kind)


@register("hf", category="datasource")
class HFDatasetSource(DataSource):
    """Wrap a HuggingFace ``datasets`` dataset as a :class:`DataSource`.

    ``datasets`` is imported lazily.  ``mapping`` optionally renames dataset
    columns onto the canonical fields (e.g. ``{"question": "prompt"}``).
    """

    def __init__(
        self,
        path: str,
        split: Optional[str] = None,
        mapping: Optional[Dict[str, str]] = None,
        kind: str = "auto",
        **load_kwargs: Any,
    ) -> None:
        self.path = path
        self.split = split
        self.mapping = mapping or {}
        self.kind = kind
        self.load_kwargs = load_kwargs

    def _load(self) -> Any:
        from .._optional import require

        datasets = require("datasets", feature="HFDatasetSource")
        return datasets.load_dataset(self.path, split=self.split, **self.load_kwargs)

    def _remap(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        if not self.mapping:
            return rec
        out = dict(rec)
        for src, dst in self.mapping.items():
            if src in rec:
                out[dst] = rec[src]
        return out

    def __iter__(self) -> Iterator[Any]:
        ds = self._load()
        for rec in ds:
            yield record_to_sample(self._remap(dict(rec)), self.kind)


@register("memory", category="datasource")
class InMemorySource(DataSource):
    """An in-memory list of items — the simplest source, ideal for tests.

    Items may already be :class:`Sample` / :class:`PreferenceSample` objects
    or plain dicts (mapped via :func:`record_to_sample`).
    """

    def __init__(self, items: List[Any], kind: str = "auto") -> None:
        self.items = list(items)
        self.kind = kind

    def __iter__(self) -> Iterator[Any]:
        for it in self.items:
            if isinstance(it, (Sample, PreferenceSample)):
                yield it
            elif isinstance(it, dict):
                # Pre-tokenised records (already carrying ``input_ids``) are a
                # collate-ready payload, not a raw text record — pass them
                # through untouched so they reach the Trainer's collate.
                if self.kind == "auto" and "input_ids" in it:
                    yield it
                else:
                    yield record_to_sample(it, self.kind)
            else:
                yield it

    def __len__(self) -> int:
        return len(self.items)
