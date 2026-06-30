"""Verifiers: math, code (subprocess), sql, json, format/regex, citation, composite."""
from __future__ import annotations

import pytest

import trainall
from trainall.types import VerifierResult
from trainall.verifiers import (
    CitationVerifier,
    CodeVerifier,
    CompositeVerifier,
    FormatVerifier,
    JSONVerifier,
    MathVerifier,
    RegexVerifier,
    SQLVerifier,
)


# --------------------------------------------------------------------------- #
# Math
# --------------------------------------------------------------------------- #
def test_math_correct_boxed():
    v = MathVerifier()
    res = v.verify(r"The answer is \boxed{42}.", reference="42")
    assert res.passed and res.reward == 1.0


def test_math_correct_last_number():
    v = MathVerifier()
    res = v.verify("After computation we get 3.5", reference="3.5")
    assert res.passed


def test_math_incorrect():
    v = MathVerifier()
    res = v.verify(r"\boxed{41}", reference="42")
    assert not res.passed and res.reward == 0.0


def test_math_numeric_tolerance():
    v = MathVerifier()
    res = v.verify("answer: 0.3333333", reference="1/3")
    assert res.passed


# --------------------------------------------------------------------------- #
# Code (subprocess)
# --------------------------------------------------------------------------- #
def test_code_pass():
    v = CodeVerifier(timeout=15.0)
    response = "```python\ndef add(a, b):\n    return a + b\n```"
    res = v.verify(response, reference="assert add(2, 3) == 5")
    assert res.passed and res.reward == 1.0


def test_code_fail():
    v = CodeVerifier(timeout=15.0)
    response = "```python\ndef add(a, b):\n    return a - b\n```"
    res = v.verify(response, reference="assert add(2, 3) == 5")
    assert not res.passed
    assert "exit=" in res.detail


def test_code_no_block():
    v = CodeVerifier()
    res = v.verify("", reference="assert True")
    assert not res.passed


# --------------------------------------------------------------------------- #
# SQL
# --------------------------------------------------------------------------- #
def test_sql_correct():
    v = SQLVerifier()
    ref = {
        "schema": "CREATE TABLE t (id INTEGER, v INTEGER);",
        "seed": "INSERT INTO t VALUES (1, 10), (2, 20);",
        "expected_sql": "SELECT v FROM t ORDER BY id;",
    }
    res = v.verify("```sql\nSELECT v FROM t ORDER BY id;\n```", reference=ref)
    assert res.passed


def test_sql_wrong_result():
    v = SQLVerifier()
    ref = {
        "schema": "CREATE TABLE t (id INTEGER, v INTEGER);",
        "seed": "INSERT INTO t VALUES (1, 10), (2, 20);",
        "expected_rows": [(10,), (20,)],
    }
    res = v.verify("SELECT id FROM t;", reference=ref)
    assert not res.passed


def test_sql_invalid_candidate():
    v = SQLVerifier()
    ref = {"schema": "CREATE TABLE t (id INTEGER);", "expected_rows": []}
    res = v.verify("SELECT FROM nope", reference=ref)
    assert not res.passed


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def test_json_valid_no_schema():
    v = JSONVerifier()
    res = v.verify('{"a": 1, "b": [1,2,3]}')
    assert res.passed
    assert res.reward == pytest.approx(0.5)


def test_json_extracted_from_prose():
    v = JSONVerifier()
    res = v.verify('Sure! Here you go: {"x": 1} hope that helps')
    assert res.passed


def test_json_invalid():
    v = JSONVerifier()
    res = v.verify("not json at all {{{")
    assert not res.passed


def test_json_schema_validation_optional():
    pytest.importorskip("jsonschema")
    v = JSONVerifier()
    schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}
    ok = v.verify('{"a": 5}', reference=schema)
    assert ok.passed and ok.reward == 1.0
    bad = v.verify('{"a": "str"}', reference=schema)
    assert not bad.passed


# --------------------------------------------------------------------------- #
# Format / Regex
# --------------------------------------------------------------------------- #
def test_format_tags_and_keys():
    v = FormatVerifier(must_have_tags=["think", "answer"])
    good = v.verify("<think>reason</think><answer>42</answer>")
    assert good.passed and good.reward == 1.0
    bad = v.verify("<think>reason</think>")
    assert not bad.passed
    assert bad.reward == pytest.approx(0.5)


def test_format_no_constraints_vacuous_pass():
    v = FormatVerifier()
    assert v.verify("anything").passed


def test_regex_match_and_override():
    v = RegexVerifier(pattern=r"\d+")
    assert v.verify("abc 123").passed
    assert not v.verify("no digits").passed
    # Per-call override via string reference.
    assert v.verify("HELLO", reference=r"H.LLO").passed


# --------------------------------------------------------------------------- #
# Citation
# --------------------------------------------------------------------------- #
def test_citation_grounded():
    v = CitationVerifier()
    sources = ["The sky is blue today.", "Water boils at 100C."]
    res = v.verify('As noted, "The sky is blue today." [1]', reference=sources)
    assert res.passed and res.reward == 1.0


def test_citation_fabricated():
    v = CitationVerifier()
    sources = ["The sky is blue."]
    res = v.verify('Studies show "the moon is made of cheese" [2]', reference=sources)
    assert not res.passed
    assert res.reward < 1.0


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #
def test_composite_weighted():
    fmt = FormatVerifier(must_have_tags=["answer"])
    math = MathVerifier()
    comp = CompositeVerifier([(fmt, 1.0), (math, 1.0)], mode="weighted")
    res = comp.verify("<answer>42</answer>", reference="42")
    assert res.passed
    assert res.reward == pytest.approx(1.0)


def test_composite_any_mode():
    fmt = FormatVerifier(must_have_tags=["nope"])  # will fail
    math = MathVerifier()
    comp = CompositeVerifier([fmt, math], mode="any")
    res = comp.verify(r"\boxed{42}", reference="42")
    assert res.passed  # math passed -> any() is True


def test_verifier_registry_build():
    v = trainall.build("math", category="verifier")
    assert isinstance(v.verify("x", reference="y"), VerifierResult)
