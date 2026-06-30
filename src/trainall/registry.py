"""A tiny, category-aware registry + ``build`` factory.

Every objective, algorithm, verifier, reward, data source, environment and
recipe registers itself with a string key so the whole library is reachable
through one config-driven entry point::

    obj   = trainall.build("dpo", beta=0.1)          # an Objective
    algo  = trainall.build("qlora", r=16)            # an Algorithm
    check = trainall.build("math", category="verifier")

Keys are namespaced by *category* so the same short name can't collide across
kinds, but a bare ``build("dpo")`` still works because lookup falls back to a
global search when the category is unambiguous.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

# category -> { name -> factory/class }
_REGISTRY: Dict[str, Dict[str, Any]] = {}

# Subpackages that own registrable objects.  ``build`` imports these on demand
# (guarded) so that a bare ``import trainall`` never drags in torch, yet
# ``build("dpo")`` still finds its target.  Order is irrelevant.
_REGISTRABLE_MODULES: Tuple[str, ...] = (
    "trainall.objectives",
    "trainall.verifiers",
    "trainall.rewards",
    "trainall.data",
    "trainall.rl",
    "trainall.pipelines",
    "trainall.algorithms",
    "trainall.models",
)

_bootstrapped = False

# When a bare name is registered in several categories (e.g. an objective *and*
# a same-named recipe), an unscoped ``build("dpo")`` resolves to the highest
# priority category below.  This makes the common case — building an objective /
# algorithm / verifier by its short name — unambiguous, while recipes remain
# reachable via ``build(name, category="recipe")`` or ``trainall.pipelines``.
_CATEGORY_PRIORITY = (
    "objective",
    "algorithm",
    "verifier",
    "reward",
    "datasource",
    "environment",
    "recipe",
)


def register(
    name: str,
    category: str = "objective",
    *,
    aliases: Optional[List[str]] = None,
) -> Callable[[Any], Any]:
    """Class/function decorator that records the target under ``name``.

    The decorated object also gets its ``.name`` attribute set (when it is a
    class) so instances can report their own registry key.
    """

    def deco(obj: Any) -> Any:
        bucket = _REGISTRY.setdefault(category, {})
        for key in [name, *(aliases or [])]:
            if key in bucket and bucket[key] is not obj:
                raise ValueError(
                    f"registry collision: '{key}' already registered in "
                    f"category '{category}' as {bucket[key]!r}"
                )
            bucket[key] = obj
        if isinstance(obj, type):
            try:
                obj.name = name  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - read-only attr edge case
                pass
        return obj

    return deco


def _bootstrap() -> None:
    global _bootstrapped
    if _bootstrapped:
        return
    _bootstrapped = True
    for mod in _REGISTRABLE_MODULES:
        try:
            importlib.import_module(mod)
        except Exception:  # pragma: no cover - missing optional deps are fine
            # e.g. torch-required subpackages when torch isn't installed.
            pass


def get(name: str, category: Optional[str] = None) -> Any:
    """Return the registered class/factory for ``name`` (no instantiation)."""
    _bootstrap()
    if category is not None:
        bucket = _REGISTRY.get(category, {})
        if name not in bucket:
            raise KeyError(_not_found_msg(name, category))
        return bucket[name]
    # Unscoped search across all categories.
    hits = [(cat, b[name]) for cat, b in _REGISTRY.items() if name in b]
    if not hits:
        raise KeyError(_not_found_msg(name, None))
    if len(hits) == 1:
        return hits[0][1]
    # Multiple categories hold this name -> resolve by priority (objective wins).
    def _rank(cat: str) -> int:
        return _CATEGORY_PRIORITY.index(cat) if cat in _CATEGORY_PRIORITY else len(_CATEGORY_PRIORITY)

    hits.sort(key=lambda ch: _rank(ch[0]))
    return hits[0][1]


def build(name: str, category: Optional[str] = None, **kwargs: Any) -> Any:
    """Look up ``name`` and instantiate it with ``**kwargs``."""
    factory = get(name, category)
    return factory(**kwargs)


def available(category: Optional[str] = None) -> Dict[str, List[str]]:
    """Return ``{category: [names...]}`` of everything currently registered."""
    _bootstrap()
    if category is not None:
        return {category: sorted(_REGISTRY.get(category, {}).keys())}
    return {cat: sorted(b.keys()) for cat, b in sorted(_REGISTRY.items())}


def _not_found_msg(name: str, category: Optional[str]) -> str:
    avail = available(category)
    lines = [f"unknown key '{name}'" + (f" in category '{category}'" if category else "")]
    for cat, names in avail.items():
        if names:
            lines.append(f"  {cat}: {', '.join(names)}")
    return "\n".join(lines)


# Convenience alias used widely in user code / docs.
REGISTRY = _REGISTRY

__all__ = ["register", "build", "get", "available", "REGISTRY"]
