"""SQL verifier.

Builds a throwaway in-memory SQLite database from a schema + seed rows, runs the
candidate SQL query against it, and compares the result set to an expected one
(either given directly as rows, or computed by running a gold query).  This is
the execution-accuracy metric used by text-to-SQL benchmarks such as Spider
(Yu et al. 2018).
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence

from ..base import Verifier
from ..registry import register
from ..types import VerifierResult

__all__ = ["SQLVerifier"]

_FENCE_RE = re.compile(r"```(?:sql)?\n(.*?)```", re.DOTALL | re.IGNORECASE)


@register("sql", category="verifier")
class SQLVerifier(Verifier):
    """Compare a candidate SQL query's result set to an expected one.

    ``reference`` is a dict with keys:

    * ``schema``: SQL DDL string (or list of statements) creating the tables.
    * ``seed``: list of SQL ``INSERT`` statements (or a single string).
    * ``expected_sql``: gold query to derive expected rows, **or**
    * ``expected_rows``: an explicit iterable of rows.

    Parameters
    ----------
    order_sensitive:
        If False (default) result sets are compared as multisets, ignoring row
        order (the standard execution-accuracy convention).
    """

    name = "sql"

    def __init__(self, order_sensitive: bool = False) -> None:
        self.order_sensitive = bool(order_sensitive)

    @staticmethod
    def extract_sql(text: str) -> str:
        if text is None:
            return ""
        m = _FENCE_RE.search(text)
        if m:
            return m.group(1).strip()
        return text.strip()

    @staticmethod
    def _as_statements(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        return [str(value)]

    def _normalize_rows(self, rows: Sequence[Any]) -> Any:
        norm = [tuple(r) for r in rows]
        if self.order_sensitive:
            return norm
        # Multiset comparison independent of ordering.
        return sorted(norm, key=lambda r: tuple(repr(x) for x in r))

    def verify(
        self,
        response: str,
        reference: Any = None,
        *,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> VerifierResult:
        import sqlite3

        if not isinstance(reference, dict):
            return VerifierResult.fail(
                detail="reference must be a dict with schema/seed/expected_*"
            )
        candidate = self.extract_sql(response)
        if not candidate:
            return VerifierResult.fail(detail="no SQL found in response")

        conn = sqlite3.connect(":memory:")
        try:
            cur = conn.cursor()
            for stmt in self._as_statements(reference.get("schema")):
                cur.executescript(stmt)
            for stmt in self._as_statements(reference.get("seed")):
                cur.executescript(stmt)
            conn.commit()

            # Expected rows: explicit, or derived from a gold query.
            if "expected_rows" in reference and reference["expected_rows"] is not None:
                expected = list(reference["expected_rows"])
            elif reference.get("expected_sql"):
                cur.execute(str(reference["expected_sql"]))
                expected = cur.fetchall()
            else:
                return VerifierResult.fail(
                    detail="reference needs expected_rows or expected_sql"
                )

            try:
                cur.execute(candidate)
                actual = cur.fetchall()
            except sqlite3.Error as exc:
                return VerifierResult.fail(detail=f"candidate SQL error: {exc}")

            match = self._normalize_rows(actual) == self._normalize_rows(expected)
            detail = (
                f"candidate returned {len(actual)} row(s); "
                f"expected {len(expected)} row(s); "
                f"order_sensitive={self.order_sensitive}; match={match}"
            )
            if match:
                return VerifierResult.ok(detail=detail)
            return VerifierResult.fail(detail=detail)
        finally:
            conn.close()
