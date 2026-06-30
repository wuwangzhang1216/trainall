"""The action space for agentic environments: tools the policy can call.

A :class:`Tool` is a named, side-effect-bounded function ``str -> str`` the
agent invokes by emitting a tool-call string.  Two reference tools ship here:

* :class:`PythonTool`     — runs a code snippet in a *subprocess* (never an
  in-process ``exec``) with a wall-clock timeout, so a runaway or malicious
  snippet cannot block or corrupt the host process.
* :class:`CalculatorTool` — evaluates arithmetic safely by walking an ``ast``
  tree, refusing names, attribute access and function calls, so there is no
  ``eval`` of arbitrary code.

:class:`ToolRegistry` holds a set of tools and dispatches a tool-call string
(``"name: arg"`` / ``"name(arg)"``) to the right one.
"""
from __future__ import annotations

import ast
import operator
import os
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from ..registry import register


class Tool(ABC):
    """A named callable tool exposed to an agent.

    Subclasses set :attr:`name` / :attr:`description` and implement
    :meth:`run`, which takes the raw argument string and returns a string
    observation (results or an error message — tools should not raise into the
    agent loop).
    """

    name: str = "tool"
    description: str = ""

    @abstractmethod
    def run(self, arg: str) -> str:
        """Execute the tool on ``arg`` and return a string observation."""

    def __call__(self, arg: str) -> str:
        return self.run(arg)


@register("python", category="environment")
class PythonTool(Tool):
    """Run a Python snippet in an isolated subprocess with a timeout.

    The snippet executes via ``sys.executable`` on a temp file; stdout (and
    stderr on failure) is returned as the observation.  Using a subprocess —
    rather than ``exec`` — bounds runtime via ``timeout`` and isolates state.
    """

    name = "python"
    description = "Execute a Python code snippet and return its stdout."

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout

    def run(self, arg: str) -> str:
        path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".py", delete=False, encoding="utf-8"
            ) as fh:
                fh.write(arg)
                path = fh.name
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if proc.returncode != 0:
                return f"error: {proc.stderr.strip() or 'non-zero exit'}"
            return proc.stdout.strip()
        except subprocess.TimeoutExpired:
            return f"error: timeout after {self.timeout}s"
        except Exception as exc:  # pragma: no cover - defensive
            return f"error: {exc}"
        finally:
            if path is not None:
                try:
                    os.unlink(path)
                except OSError:  # pragma: no cover - best effort cleanup
                    pass


@register("calculator", category="environment")
class CalculatorTool(Tool):
    """Evaluate an arithmetic expression safely via the ``ast`` module.

    Only literals and the binary/unary numeric operators are permitted.  Any
    name, attribute, call, subscript or comprehension raises, so there is no
    path to ``eval`` arbitrary code or read globals.
    """

    name = "calculator"
    description = "Evaluate an arithmetic expression (e.g. '2 * (3 + 4)')."

    _BIN_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _UNARY_OPS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def run(self, arg: str) -> str:
        try:
            value = self._eval(ast.parse(arg, mode="eval").body)
        except Exception as exc:
            return f"error: {exc}"
        return repr(value)

    def _eval(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("only numeric literals are allowed")
            return node.value
        if isinstance(node, ast.BinOp):
            op = self._BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"operator {type(node.op).__name__} not allowed")
            return op(self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = self._UNARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"operator {type(node.op).__name__} not allowed")
            return op(self._eval(node.operand))
        raise ValueError(f"unsupported expression node: {type(node).__name__}")


class ToolRegistry:
    """A small set of tools plus a dispatcher for tool-call strings.

    A tool call is parsed from either ``"name: arg"`` or ``"name(arg)"``; the
    matching tool is looked up by name and run.  Unknown tools return an error
    observation rather than raising, keeping the agent loop robust.
    """

    def __init__(self, tools: Optional[List[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> "ToolRegistry":
        self._tools[tool.name] = tool
        return self

    def lookup(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    @staticmethod
    def parse(call: str) -> Tuple[str, str]:
        """Split a tool-call string into ``(name, arg)``.

        Accepts ``"name(arg)"`` and ``"name: arg"`` (and bare ``"name"``).
        """
        call = call.strip()
        paren = call.find("(")
        colon = call.find(":")
        # Prefer "name: arg" when the colon precedes any parenthesis, so that
        # an arg containing '(' (e.g. "python: print(9)") is not mis-split.
        if colon != -1 and (paren == -1 or colon < paren):
            name, _, rest = call.partition(":")
            return name.strip(), rest.strip()
        if paren != -1 and call.endswith(")"):
            name, _, rest = call.partition("(")
            return name.strip(), rest[:-1].strip()
        return call, ""

    def dispatch(self, call: str) -> str:
        """Parse and run a tool-call string; return the tool's observation."""
        name, arg = self.parse(call)
        tool = self.lookup(name)
        if tool is None:
            return f"error: unknown tool '{name}' (have: {', '.join(self.names())})"
        return tool.run(arg)


__all__ = ["Tool", "PythonTool", "CalculatorTool", "ToolRegistry"]
