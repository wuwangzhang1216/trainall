"""``trainall`` command-line entry point.

    trainall list                       # show every registered component
    trainall list --category verifier
    trainall run config.yaml            # run a training job from a config
    trainall run config.yaml --train.epochs 2 --objective.name dpo
    trainall version
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, List, Optional


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _apply_overrides(cfg_dict: dict, pairs: List[str]) -> dict:
    """Apply ``--a.b.c value`` dotted overrides onto a nested dict."""
    for raw in pairs:
        if "=" in raw:
            key, val = raw.split("=", 1)
        else:  # support "--a.b val" form already split by argparse REMAINDER
            continue
        key = key.lstrip("-")
        node = cfg_dict
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val)
    return cfg_dict


def _cmd_list(args: argparse.Namespace) -> int:
    from .registry import available

    table = available(args.category)
    for category, names in table.items():
        print(f"[{category}]")
        for n in names:
            print(f"  - {n}")
        if not names:
            print("  (none)")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .config import load_config

    cfg = load_config(args.config)
    data = cfg.to_dict()
    if args.set:
        data = _apply_overrides(data, args.set)
    cfg = load_config(data)
    import trainall

    print(f"running '{cfg.name}': objective={cfg.objective.name} "
          f"algorithm={cfg.algorithm.name} model={cfg.model.pretrained or cfg.model.arch}")
    trainall.train(cfg)
    return 0


def _cmd_version(_: argparse.Namespace) -> int:
    from . import __version__

    print(__version__)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trainall", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    lp = sub.add_parser("list", help="list registered components")
    lp.add_argument("--category", default=None)
    lp.set_defaults(func=_cmd_list)

    rp = sub.add_parser("run", help="run a training job from a config file")
    rp.add_argument("config")
    rp.add_argument("--set", nargs="*", default=[], metavar="key=value",
                    help="dotted config overrides, e.g. train.epochs=2")
    rp.set_defaults(func=_cmd_run)

    vp = sub.add_parser("version", help="print version")
    vp.set_defaults(func=_cmd_version)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
