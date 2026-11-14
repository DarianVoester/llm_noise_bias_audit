"""Tests for all 7 output schema parsers."""

import json

import pytest

from src.schemas import (
    SchemaConfig,
    SchemaType,
    build_format_instructions,
    decode_value,
    parse_response,
    strip_reasoning,
)

BOOL = SchemaConfig(type=SchemaType.BOOLEAN)
PCT = SchemaConfig(type=SchemaType.PERCENTAGE)
FLT = SchemaConfig(type=SchemaType.FLOAT)
LIK = SchemaConfig(type=SchemaType.LIKERT, likert_min=1, likert_max=7)
ENUM = SchemaConfig(type=SchemaType.ENUM, options=["Low", "Medium", "High"])
RANK = SchemaConfig(type=SchemaType.RANKING, options=["A", "B", "C"])
CONF = SchemaConfig(type=SchemaType.CONFIDENCE)


@pytest.mark.parametrize("schema", [BOOL, PCT, FLT, LIK, ENUM, RANK, CONF])
def test_format_instructions_mention_json(schema):
    text = build_format_instructions(schema)
    assert "JSON" in text and "OUTPUT FORMAT" in text


@pytest.mark.parametrize(
    "raw,expected,method",
    [
        ('{"answer": true}', 1.0, "native"),
        ('{"answer": "no"}', 0.0, "native"),
        ("Sure! ```json\n{\"answer\": false}\n```", 0.0, "json"),
        ("ANSWER: yes", 1.0, "regex"),
        ("True", 1.0, "regex"),
    ],
)
def test_boolean(raw, expected, method):
    r = parse_response(raw, BOOL, json_mode=True)
    assert r.success and r.value_num == expected and r.method == method


def test_percentage_valid_and_range():
    assert parse_response('{"answer": 42.5}', PCT).value_num == 42.5
    assert parse_response('{"answer": "73%"}', PCT).value_num == 73.0
    assert not parse_response('{"answer": 140}', PCT).success
    assert not parse_response('{"answer": -1}', PCT).success


@pytest.mark.parametrize("raw", ['{"answer": 0.425}', '{"answer": 0.8}', '{"answer": "0,42"}',
                                 "ANSWER: 0.5"])
def test_percentage_rejects_0_1_fractions(raw):
    r = parse_response(raw, PCT)
    assert not r.success and "fraction" in r.error


@pytest.mark.parametrize("raw,expected", [('{"answer": 42.5}', 42.5), ('{"answer": 0}', 0.0),
                                          ('{"answer": 1}', 1.0), ('{"answer": 100}', 100.0),
                                          ('{"answer": "42%"}', 42.0)])
def test_percentage_accepts_0_100_scale(raw, expected):
    r = parse_response(raw, PCT)
    assert r.success and r.value_num == expected


def test_percentage_instructions_forbid_fractions():
    text = build_format_instructions(PCT)
    assert "0-100" in text and "0.425" in text and "NEVER" in text


def test_confidence_rejects_fraction():
    assert not parse_response('{"decision": 60, "confidence": 0.8}', CONF).success
    assert parse_response('{"decision": 60, "confidence": 80}', CONF).success


def test_float_formats():
    assert parse_response('{"answer": 1250}', FLT).value_num == 1250.0
    assert parse_response('{"answer": "1,250,000"}', FLT).value_num == 1_250_000.0
    assert parse_response('{"answer": "12,5"}', FLT).value_num == 12.5
    assert parse_response('{"answer": "$99.90"}', FLT).value_num == 99.9
    assert not parse_response('{"answer": "NaN"}', FLT).success


def test_regex_fallback_does_not_grab_anchor_number():
    raw = "Considering the reference value of 9000 mentioned, my estimate is quite different."
    assert not parse_response(raw, FLT).success
    raw2 = "The anchor was 9000.\nANSWER: 4200"
    r = parse_response(raw2, FLT)
    assert r.success and r.value_num == 4200.0 and r.method == "regex"


def test_likert_bounds_and_integer():
    assert parse_response('{"answer": 6}', LIK).value_num == 6
    assert not parse_response('{"answer": 8}', LIK).success
    assert not parse_response('{"answer": 3.5}', LIK).success


def test_enum_case_insensitive_exact():
    r = parse_response('{"answer": "medium"}', ENUM)
    assert r.success and json.loads(r.value_json) == "Medium"
    assert not parse_response('{"answer": "Mediumish"}', ENUM).success


def test_ranking_permutation():
    r = parse_response('{"answer": ["B", "A", "C"]}', RANK)
    assert r.success and decode_value(None, r.value_json) == ["B", "A", "C"]
    assert parse_response("ANSWER: C > A > B", RANK).success
    assert not parse_response('{"answer": ["A", "A", "C"]}', RANK).success
    assert not parse_response('{"answer": ["A", "B"]}', RANK).success


def test_confidence_tuple():
    r = parse_response('{"decision": 7.5, "confidence": 80}', CONF)
    assert r.success and r.value_num == 7.5 and r.confidence == 80
    r2 = parse_response('{"decision": true, "confidence": "65%"}', CONF)
    assert r2.success and r2.value_num == 1.0 and r2.confidence == 65
    assert not parse_response('{"decision": 1, "confidence": 150}', CONF).success
    assert not parse_response('{"decision": 1}', CONF).success


def test_think_blocks_are_stripped():
    raw = '<think>The anchor says 9000, maybe {"answer": 9000}</think>{"answer": 4100}'
    assert strip_reasoning(raw) == '{"answer": 4100}'
    assert parse_response(raw, FLT).value_num == 4100.0


def test_answer_object_wins_over_earlier_json_in_the_reply():
    raw = 'Reasoning: {"base": 30}\nFinal: {"answer": 42}'
    r = parse_response(raw, PCT)
    assert r.success and r.value_num == 42.0 and r.method == "json"
    conf = 'Notes: {"anchor": 99}\n{"decision": 7, "confidence": 80}'
    r2 = parse_response(conf, CONF)
    assert r2.success and r2.value_num == 7.0 and r2.confidence == 80.0


def test_empty_and_garbage():
    assert not parse_response("", FLT).success
    r = parse_response("I cannot answer that question in this format, sorry.", BOOL)
    assert not r.success and r.method == "failed"


def test_schema_validation():
    with pytest.raises(ValueError):
        SchemaConfig(type=SchemaType.ENUM, options=["only-one"])
    with pytest.raises(ValueError):
        SchemaConfig(type=SchemaType.ENUM, options=["a", "A"])
    with pytest.raises(ValueError):
        SchemaConfig(type=SchemaType.LIKERT, likert_min=5, likert_max=5)
    assert LIK.scale_range == (1.0, 7.0)
    assert FLT.scale_range is None


@pytest.mark.parametrize("raw", [
    "{" * 100_000,
    "\n" * 100_000,
    "answer:" + " " * 100_000 + "x" + " " * 100_000,
    "[" * 100_000,
    ("{" * 5000) + ("}" * 5000),
])
def test_hostile_replies_are_parsed_quickly(raw):
    """A broken or hostile server must not be able to block the app with one reply."""
    import time

    from src.schemas import SchemaConfig, SchemaType, parse_response

    t0 = time.perf_counter()
    parse_response(raw, SchemaConfig(type=SchemaType.FLOAT))
    assert time.perf_counter() - t0 < 2.0


def test_answer_line_variants_still_parse():
    from src.schemas import SchemaConfig, SchemaType, parse_response

    schema = SchemaConfig(type=SchemaType.FLOAT)
    for raw in ("ANSWER: 42", "**Answer:** 42", "  answer = 42  ", "Reasoning...\nAnswer:\n42",
                "* answer *: 42"):
        res = parse_response(raw, schema)
        assert res.success and res.value_num == 42.0, raw
