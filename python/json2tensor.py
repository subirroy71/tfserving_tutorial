"""Build TensorProto messages from plain JSON at runtime, and back.

Nothing here knows the model's signature in advance: the tensor's shape is
discovered by walking the nested JSON arrays and its dtype is inferred from
the JSON literals (or taken from an explicit {"dtype": ...} wrapper).

The module depends only on the generated protobuf classes -- not on
TensorFlow or NumPy.
"""
from __future__ import annotations

import base64
import binascii
import struct
from typing import Any

from tensorflow.core.framework import tensor_pb2, tensor_shape_pb2, types_pb2


class TensorJSONError(ValueError):
    """The JSON value cannot be turned into a tensor."""


# name -> DataType enum. Accepted case-insensitively, with or without "DT_".
_DTYPES = {
    "float": types_pb2.DT_FLOAT,
    "float32": types_pb2.DT_FLOAT,
    "double": types_pb2.DT_DOUBLE,
    "float64": types_pb2.DT_DOUBLE,
    "int8": types_pb2.DT_INT8,
    "int16": types_pb2.DT_INT16,
    "int32": types_pb2.DT_INT32,
    "int64": types_pb2.DT_INT64,
    "uint8": types_pb2.DT_UINT8,
    "uint32": types_pb2.DT_UINT32,
    "uint64": types_pb2.DT_UINT64,
    "bool": types_pb2.DT_BOOL,
    "string": types_pb2.DT_STRING,
    "bytes": types_pb2.DT_STRING,
}

# dtype -> (TensorProto repeated field, min, max) for integer types.
_INT_TYPES = {
    types_pb2.DT_INT8: ("int_val", -(2**7), 2**7 - 1),
    types_pb2.DT_INT16: ("int_val", -(2**15), 2**15 - 1),
    types_pb2.DT_INT32: ("int_val", -(2**31), 2**31 - 1),
    types_pb2.DT_INT64: ("int64_val", -(2**63), 2**63 - 1),
    types_pb2.DT_UINT8: ("int_val", 0, 2**8 - 1),
    types_pb2.DT_UINT32: ("uint32_val", 0, 2**32 - 1),
    types_pb2.DT_UINT64: ("uint64_val", 0, 2**64 - 1),
}

# dtype -> (TensorProto repeated field, struct code) for tensor_content.
_FIXED = {
    types_pb2.DT_FLOAT: ("float_val", "f"),
    types_pb2.DT_DOUBLE: ("double_val", "d"),
    types_pb2.DT_INT8: ("int_val", "b"),
    types_pb2.DT_INT16: ("int_val", "h"),
    types_pb2.DT_INT32: ("int_val", "i"),
    types_pb2.DT_INT64: ("int64_val", "q"),
    types_pb2.DT_UINT8: ("int_val", "B"),
    types_pb2.DT_UINT32: ("uint32_val", "I"),
    types_pb2.DT_UINT64: ("uint64_val", "Q"),
    types_pb2.DT_BOOL: ("bool_val", "?"),
}


def parse_dtype(name: Any) -> int:
    if not isinstance(name, str):
        raise TensorJSONError('"dtype" must be a string')
    key = name.strip().lower()
    if key.startswith("dt_"):
        key = key[3:]
    if key not in _DTYPES:
        raise TensorJSONError(f"unsupported dtype {name!r}")
    return _DTYPES[key]


def dtype_name(dtype: int) -> str:
    return types_pb2.DataType.Name(dtype)


# --------------------------------------------------------------------------
# JSON -> TensorProto
# --------------------------------------------------------------------------

def _is_b64(node: Any) -> bool:
    return isinstance(node, dict) and "b64" in node


def _flatten(node: Any, depth: int, shape: list[int], leaves: list[Any], rank: list[int | None]) -> None:
    """Depth-first walk that records the shape and collects leaves row-major.

    `shape` grows by one entry the first time each depth is entered; every
    later list at that depth must have the same length. `rank[0]` is fixed by
    the first leaf, after which lists may not appear at leaf depth.
    """
    if isinstance(node, list):
        if rank[0] is not None and depth >= rank[0]:
            raise TensorJSONError("ragged tensor: array found where a value was expected")
        if depth == len(shape):
            shape.append(len(node))
        elif shape[depth] != len(node):
            raise TensorJSONError(
                f"ragged tensor: dimension {depth} has length {len(node)}, expected {shape[depth]}"
            )
        for child in node:
            _flatten(child, depth + 1, shape, leaves, rank)
        return
    if depth != len(shape):
        raise TensorJSONError("ragged tensor: value found where an array was expected")
    rank[0] = depth
    leaves.append(node)


def _kind(leaf: Any) -> str:
    if isinstance(leaf, bool):  # must precede int: bool is a subclass of int
        return "bool"
    if isinstance(leaf, int):
        return "int"
    if isinstance(leaf, float):
        return "float"
    if isinstance(leaf, str):
        return "string"
    if _is_b64(leaf):
        if len(leaf) != 1 or not isinstance(leaf["b64"], str):
            raise TensorJSONError('a bytes value must be exactly {"b64": "<base64>"}')
        return "string"
    if leaf is None:
        raise TensorJSONError("null is not a valid tensor element")
    raise TensorJSONError("objects are not valid tensor elements")


def _infer_dtype(leaves: list[Any]) -> int:
    if not leaves:
        raise TensorJSONError('cannot infer the dtype of an empty tensor; use {"dtype": ..., "values": ...}')
    kinds = {_kind(leaf) for leaf in leaves}
    if kinds == {"bool"}:
        return types_pb2.DT_BOOL
    if kinds == {"string"}:
        return types_pb2.DT_STRING
    if kinds == {"int"}:
        return types_pb2.DT_INT64
    if kinds <= {"int", "float"}:
        return types_pb2.DT_FLOAT
    raise TensorJSONError(f"mixed element types: {', '.join(sorted(kinds))}")


def _to_bytes(leaf: Any) -> bytes:
    if isinstance(leaf, str):
        return leaf.encode("utf-8")
    try:
        return base64.b64decode(leaf["b64"], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise TensorJSONError(f"invalid base64: {exc}") from None


def _fill(tensor: tensor_pb2.TensorProto, dtype: int, leaves: list[Any]) -> None:
    name = dtype_name(dtype)
    kinds = [_kind(leaf) for leaf in leaves]
    if dtype in (types_pb2.DT_FLOAT, types_pb2.DT_DOUBLE):
        if any(k not in ("int", "float") for k in kinds):
            raise TensorJSONError(f"{name} needs numbers")
        field = tensor.float_val if dtype == types_pb2.DT_FLOAT else tensor.double_val
        field.extend(float(v) for v in leaves)
    elif dtype in _INT_TYPES:
        field_name, lo, hi = _INT_TYPES[dtype]
        for kind, v in zip(kinds, leaves):
            if kind != "int":
                raise TensorJSONError(f"{name} needs integers, got {v!r}")
            if not lo <= v <= hi:
                raise TensorJSONError(f"{v} is out of range for {name}")
        getattr(tensor, field_name).extend(leaves)
    elif dtype == types_pb2.DT_BOOL:
        if any(k != "bool" for k in kinds):
            raise TensorJSONError("DT_BOOL needs true/false")
        tensor.bool_val.extend(leaves)
    elif dtype == types_pb2.DT_STRING:
        if any(k != "string" for k in kinds):
            raise TensorJSONError('DT_STRING needs strings or {"b64": ...}')
        tensor.string_val.extend(_to_bytes(v) for v in leaves)
    else:  # pragma: no cover - _DTYPES and the branches above are in sync
        raise TensorJSONError(f"unsupported dtype {name}")


def tensor_from_json(value: Any) -> tensor_pb2.TensorProto:
    """Convert one decoded JSON value into a TensorProto.

    `value` is either a bare literal (scalar or nested arrays), or a spec
    object {"dtype": <name>, "shape": [..], "values": <literal>} in which
    "dtype" and "shape" are both optional.
    """
    dtype = None
    explicit_shape = None
    if isinstance(value, dict) and not _is_b64(value):
        unknown = set(value) - {"dtype", "shape", "values"}
        if unknown:
            raise TensorJSONError(f"unknown key(s) in tensor spec: {', '.join(sorted(unknown))}")
        if "values" not in value:
            raise TensorJSONError('tensor spec needs "values"')
        if "dtype" in value:
            dtype = parse_dtype(value["dtype"])
        if "shape" in value:
            explicit_shape = value["shape"]
            if not isinstance(explicit_shape, list) or any(
                isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in explicit_shape
            ):
                raise TensorJSONError('"shape" must be an array of non-negative integers')
        value = value["values"]

    shape: list[int] = []
    leaves: list[Any] = []
    _flatten(value, 0, shape, leaves, [None])

    if explicit_shape is not None:
        count = 1
        for d in explicit_shape:
            count *= d
        if count != len(leaves):
            raise TensorJSONError(f"shape {explicit_shape} needs {count} values, got {len(leaves)}")
        shape = explicit_shape
    if dtype is None:
        dtype = _infer_dtype(leaves)

    tensor = tensor_pb2.TensorProto(
        dtype=dtype,
        tensor_shape=tensor_shape_pb2.TensorShapeProto(
            dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=d) for d in shape]
        ),
    )
    _fill(tensor, dtype, leaves)
    return tensor


def inputs_from_json(doc: Any) -> dict[str, tensor_pb2.TensorProto]:
    """Convert a request document {"inputs": {name: value, ...}} to tensors."""
    if not isinstance(doc, dict) or not isinstance(doc.get("inputs"), dict):
        raise TensorJSONError('request must be an object with an "inputs" object')
    tensors = {}
    for name, value in doc["inputs"].items():
        try:
            tensors[name] = tensor_from_json(value)
        except TensorJSONError as exc:
            raise TensorJSONError(f'input "{name}": {exc}') from None
    return tensors


# --------------------------------------------------------------------------
# TensorProto -> JSON
# --------------------------------------------------------------------------

def _json_float(v: float, single: bool) -> Any:
    if v != v:
        return "NaN"
    if v in (float("inf"), float("-inf")):
        return "Infinity" if v > 0 else "-Infinity"
    if not single:
        return v
    # Shortest decimal that round-trips the float32 (0.1, not 0.10000000149).
    packed = struct.pack("<f", v)
    for digits in range(6, 10):
        short = float(f"{v:.{digits}g}")
        if struct.pack("<f", short) == packed:
            break
    return short


def _json_bytes(b: bytes) -> Any:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return {"b64": base64.b64encode(b).decode("ascii")}


def _flat_values(tensor: tensor_pb2.TensorProto, count: int) -> list[Any]:
    dtype = tensor.dtype
    if dtype == types_pb2.DT_STRING:
        values = list(tensor.string_val)
    elif dtype in _FIXED:
        field_name, code = _FIXED[dtype]
        if tensor.tensor_content:
            # Raw little-endian buffer: what TF itself writes for most tensors.
            size = struct.calcsize("<" + code)
            if len(tensor.tensor_content) != count * size:
                raise TensorJSONError(
                    f"tensor_content has {len(tensor.tensor_content)} bytes, expected {count * size}"
                )
            values = list(struct.unpack(f"<{count}{code}", tensor.tensor_content))
        else:
            values = list(getattr(tensor, field_name))
    else:
        raise TensorJSONError(f"unsupported dtype {dtype_name(dtype)}")

    # TF may send fewer values than elements: the last one is repeated
    # (and none at all means all-default).
    if len(values) < count:
        pad = values[-1] if values else (b"" if dtype == types_pb2.DT_STRING else 0)
        values.extend([pad] * (count - len(values)))
    elif len(values) > count:
        raise TensorJSONError(f"tensor has {len(values)} values for {count} elements")

    if dtype == types_pb2.DT_STRING:
        return [_json_bytes(v) for v in values]
    if dtype == types_pb2.DT_BOOL:
        return [bool(v) for v in values]
    if dtype in (types_pb2.DT_FLOAT, types_pb2.DT_DOUBLE):
        return [_json_float(v, dtype == types_pb2.DT_FLOAT) for v in values]
    return values


def tensor_shape(tensor: tensor_pb2.TensorProto) -> list[int]:
    if tensor.tensor_shape.unknown_rank:
        raise TensorJSONError("tensor has unknown rank")
    return [d.size for d in tensor.tensor_shape.dim]


def tensor_to_json(tensor: tensor_pb2.TensorProto) -> Any:
    """Convert a TensorProto into nested lists (a bare value for rank 0)."""
    shape = tensor_shape(tensor)
    count = 1
    for d in shape:
        if d < 0:
            raise TensorJSONError(f"tensor has an unknown dimension: {shape}")
        count *= d
    flat = _flat_values(tensor, count)

    def nest(dims: list[int], offset: int, stride: int) -> Any:
        if not dims:
            return flat[offset]
        stride //= dims[0] or 1
        return [nest(dims[1:], offset + i * stride, stride) for i in range(dims[0])]

    return nest(shape, 0, count)


def describe(tensor: tensor_pb2.TensorProto) -> str:
    return f"{dtype_name(tensor.dtype)} {tensor_shape(tensor)}"
