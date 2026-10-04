"""Build the tf.Example input of Classify/Regress requests from plain JSON.

A request document lists one object per example, mapping feature names to
values; an optional "context" object holds features shared by every example:

    {"examples": [{"petal_length": 1.4, "color": "red"}, ...],
     "context":  {"site": "lab-3"}}

Each value becomes one tf.train.Feature. The list type is inferred from the
JSON literals, or given explicitly:

    5.1, [5.1, 3.0], [1, 2.5]   float_list   (any decimal point or exponent)
    3, [3, 4]                   int64_list   (integers only)
    "a", ["a", {"b64": ".."}]   bytes_list
    {"float_list": [5, 3]}      explicit; the only way to send an empty list

Like the tensor converter, this depends only on the generated protobuf
classes -- not on TensorFlow.
"""
from __future__ import annotations

import base64
import binascii
from typing import Any

from tensorflow.core.example import example_pb2, feature_pb2
from tensorflow_serving.apis import input_pb2

from json2tensor import json_bytes, json_float

LIST_TYPES = ("float_list", "int64_list", "bytes_list")
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1


class ExampleJSONError(ValueError):
    """The JSON value cannot be turned into a tf.Example."""


def _is_b64(node: Any) -> bool:
    return isinstance(node, dict) and "b64" in node


def _to_bytes(leaf: Any) -> bytes:
    if isinstance(leaf, str):
        return leaf.encode("utf-8")
    if len(leaf) != 1 or not isinstance(leaf["b64"], str):
        raise ExampleJSONError('a bytes value must be exactly {"b64": "<base64>"}')
    try:
        return base64.b64decode(leaf["b64"], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ExampleJSONError(f"invalid base64: {exc}") from None


def _leaf_kind(leaf: Any) -> str:
    if isinstance(leaf, bool):
        raise ExampleJSONError("booleans are not a feature type; use 0/1")
    if isinstance(leaf, int):
        return "int"
    if isinstance(leaf, float):
        return "float"
    if isinstance(leaf, str) or _is_b64(leaf):
        return "bytes"
    if isinstance(leaf, list):
        raise ExampleJSONError("nested arrays are not allowed; a feature is a flat list")
    if leaf is None:
        raise ExampleJSONError("null is not a feature value")
    raise ExampleJSONError(f"unsupported value {leaf!r}")


def _fill(feature: feature_pb2.Feature, list_type: str, values: list[Any]) -> None:
    if list_type == "float_list":
        for v in values:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ExampleJSONError(f"float_list needs numbers, got {v!r}")
        feature.float_list.value.extend(float(v) for v in values)
    elif list_type == "int64_list":
        for v in values:
            if isinstance(v, bool) or not isinstance(v, int):
                raise ExampleJSONError(f"int64_list needs integers, got {v!r}")
            if not _INT64_MIN <= v <= _INT64_MAX:
                raise ExampleJSONError(f"{v} is out of range for int64")
        feature.int64_list.value.extend(values)
    else:
        for v in values:
            if not (isinstance(v, str) or _is_b64(v)):
                raise ExampleJSONError(f"bytes_list needs strings, got {v!r}")
        feature.bytes_list.value.extend(_to_bytes(v) for v in values)


def feature_from_json(value: Any) -> feature_pb2.Feature:
    """Convert one JSON value (scalar, flat list, or {"<type>_list": [...]})."""
    feature = feature_pb2.Feature()
    if isinstance(value, dict) and not _is_b64(value):
        if len(value) != 1 or next(iter(value)) not in LIST_TYPES:
            raise ExampleJSONError(f"an explicit feature is one of {{{', '.join(LIST_TYPES)}: [...]}}")
        list_type, values = next(iter(value.items()))
        _fill(feature, list_type, values if isinstance(values, list) else [values])
        return feature

    values = value if isinstance(value, list) else [value]
    if not values:
        raise ExampleJSONError('an empty list has no type; write {"float_list": []} (or int64_list, bytes_list)')
    kinds = {_leaf_kind(v) for v in values}
    if kinds == {"int"}:
        list_type = "int64_list"
    elif kinds <= {"int", "float"}:
        list_type = "float_list"
    elif kinds == {"bytes"}:
        list_type = "bytes_list"
    else:
        raise ExampleJSONError("mixes strings and numbers")
    _fill(feature, list_type, values)
    return feature


def example_from_json(obj: Any) -> example_pb2.Example:
    if not isinstance(obj, dict):
        raise ExampleJSONError("must be an object of features")
    example = example_pb2.Example()
    for name, value in obj.items():
        try:
            example.features.feature[name].CopyFrom(feature_from_json(value))
        except ExampleJSONError as exc:
            raise ExampleJSONError(f'feature "{name}": {exc}') from None
    return example


def input_from_json(doc: Any) -> input_pb2.Input:
    """Convert {"examples": [...], "context": {...}} into an Input message."""
    if not isinstance(doc, dict) or not isinstance(doc.get("examples"), list) or not doc["examples"]:
        raise ExampleJSONError('request must be an object with a non-empty "examples" array')
    unknown = set(doc) - {"examples", "context"}
    if unknown:
        raise ExampleJSONError(f"unknown key {sorted(unknown)[0]!r}; expected \"examples\" and optional \"context\"")
    examples = []
    for i, obj in enumerate(doc["examples"]):
        try:
            examples.append(example_from_json(obj))
        except ExampleJSONError as exc:
            raise ExampleJSONError(f"example {i}: {exc}") from None

    result = input_pb2.Input()
    if "context" in doc:
        try:
            context = example_from_json(doc["context"])
        except ExampleJSONError as exc:
            raise ExampleJSONError(f"context: {exc}") from None
        result.example_list_with_context.examples.extend(examples)
        result.example_list_with_context.context.CopyFrom(context)
    else:
        result.example_list.examples.extend(examples)
    return result


# --------------------------------------------------------------------------
# Input -> JSON, for tests and logging
# --------------------------------------------------------------------------

def example_to_json(example: example_pb2.Example) -> dict[str, Any]:
    out = {}
    for name, feature in sorted(example.features.feature.items()):
        kind = feature.WhichOneof("kind")
        if kind == "float_list":
            out[name] = {kind: [json_float(v, True) for v in feature.float_list.value]}
        elif kind == "int64_list":
            out[name] = {kind: list(feature.int64_list.value)}
        elif kind == "bytes_list":
            out[name] = {kind: [json_bytes(v) for v in feature.bytes_list.value]}
        else:
            out[name] = {}
    return out


def input_to_json(inp: input_pb2.Input) -> dict[str, Any]:
    if inp.WhichOneof("kind") == "example_list_with_context":
        lst = inp.example_list_with_context
        return {"examples": [example_to_json(e) for e in lst.examples], "context": example_to_json(lst.context)}
    return {"examples": [example_to_json(e) for e in inp.example_list.examples]}


def describe(inp: input_pb2.Input) -> str:
    """One line for stderr: example count and each feature's list type."""
    if inp.WhichOneof("kind") == "example_list_with_context":
        examples, context = inp.example_list_with_context.examples, inp.example_list_with_context.context
    else:
        examples, context = inp.example_list.examples, None
    types = {}
    for ex in list(examples) + ([context] if context is not None else []):
        for name, feature in ex.features.feature.items():
            types.setdefault(name, feature.WhichOneof("kind") or "empty")
    feats = ", ".join(f"{n}:{t.replace('_list', '')}" for n, t in sorted(types.items()))
    ctx = " + context" if context is not None else ""
    return f"{len(examples)} examples{ctx} ({feats})"
