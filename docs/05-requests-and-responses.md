# 5. Requests and responses in detail

This chapter is about the payloads: how a `TensorProto` and a `tf.Example`
are built from JSON, how responses come back over gRPC and over REST, and
the conversion traps that cause most `INVALID_ARGUMENT` errors.

## Tensor requests (predict)

A predict request is `{"inputs": {<name>: <tensor>, ...}}`. The clients
write each tensor in one of two ways.

**Bare literal**: a scalar or nested arrays. The shape and dtype are
inferred:

```json
{
  "inputs": {
    "age":    [34, 51, 27],
    "income": [72000.0, 38000.5, 120000.0],
    "city":   ["Austin", "Dallas", "austin"]
  }
}
```

| JSON elements | dtype |
|---|---|
| all integers (`3`) | `DT_INT64` |
| any decimal or exponent (`3.0`, `1e3`), or integers mixed with them | `DT_FLOAT` |
| `true` / `false` | `DT_BOOL` |
| strings, or `{"b64": "..."}` for raw bytes | `DT_STRING` |

The shape is the nesting: `3.0` is rank 0, `[1, 2, 3]` is `[3]`,
`[[1, 2], [3, 4]]` is `[2, 2]`. Ragged arrays, mixed kinds (`["a", 1]`) and
`null` are errors that name the input.

**Spec object**: say it explicitly when inference would get it wrong.
`dtype` and `shape` are each optional:

```json
{
  "inputs": {
    "embedding": {"dtype": "float32", "values": [[1, 0, 0], [0, 4, 0]]},
    "x":         {"dtype": "float32", "shape": [2, 3], "values": [0, 1, 2, 3, 4, 5]},
    "ids":       {"dtype": "int32", "values": []}
  }
}
```

- `dtype`: `float32`/`float`, `float64`/`double`, `int8`, `int16`, `int32`,
  `int64`, `uint8`, `uint32`, `uint64`, `bool`, `string`/`bytes`. Names are
  case-insensitive, and `DT_FLOAT`-style names work too. Integers are
  range-checked (`300` is rejected for `uint8`).
- `shape` reshapes the flattened `values`; the element count must match.
- An empty tensor has no elements to infer from, so it needs a `dtype`.

> **The trap.** TF Serving's gRPC API does **not** convert dtypes. A float32
> input written as `[1, 0, 0]` is inferred as int64, and the server answers
> `INVALID_ARGUMENT: Expects arg[2] to be float but int64 is provided`. Write
> `1.0`, or wrap the input in `{"dtype": "float32", ...}`.
> `examples/applicants.json` does this for `embedding`. REST *does* convert,
> because it reads the signature first. That's why a request that works with
> curl can fail over gRPC.

To avoid guessing, read the signature with
[`metadata`](04-clients.md#metadata-what-does-the-model-expect) first.

## How a tensor is built

A `TensorProto` is three things: a `dtype`, a `tensor_shape`, and the
elements, flattened in row-major order into the repeated field that matches
the dtype:

| dtype | field |
|---|---|
| `DT_FLOAT` | `float_val` |
| `DT_DOUBLE` | `double_val` |
| `DT_INT32`, `DT_INT16`, `DT_INT8`, `DT_UINT8` | `int_val` |
| `DT_INT64` | `int64_val` |
| `DT_UINT32` / `DT_UINT64` | `uint32_val` / `uint64_val` |
| `DT_BOOL` | `bool_val` |
| `DT_STRING` | `string_val` (bytes) |

The three converters (`python/json2tensor.py`, `go/tensorjson/tensorjson.go`,
`rust/src/json2tensor.rs`) follow the same steps:

1. **Unwrap** a spec object, keeping its `dtype` and `shape` if present.
2. **Flatten** in one depth-first walk. The first time the walk reaches a
   depth, it records that array's length as the dimension. Every later array
   at that depth must have the same length, or the tensor is ragged. Leaves
   are collected in visiting order, which *is* row-major order.
3. **Infer** the dtype from the kinds of leaves, unless one was given.
4. **Fill** the matching repeated field, range-checking integers.

The walk in Python, which the Go and Rust versions mirror:

```python
def _flatten(node, depth, shape, leaves, rank):
    if isinstance(node, list):
        if rank[0] is not None and depth >= rank[0]:
            raise TensorJSONError("ragged tensor: array found where a value was expected")
        if depth == len(shape):
            shape.append(len(node))            # first visit fixes this dimension
        elif shape[depth] != len(node):
            raise TensorJSONError("ragged tensor: ...")
        for child in node:
            _flatten(child, depth + 1, shape, leaves, rank)
        return
    if depth != len(shape):
        raise TensorJSONError("ragged tensor: value found where an array was expected")
    rank[0] = depth                            # first leaf fixes the rank
    leaves.append(node)
```

Each language needs one trick to tell `1` from `1.0`, because that decides
between int64 and float32:

- **Python**: `json` already returns `int` vs `float`. Check `bool` first,
  because it is a subclass of `int`.
- **Go**: decode with `json.Decoder.UseNumber()` and look at the literal's
  text. Decoding into `float64` would lose both the distinction and any
  integer above 2^53.
- **Rust**: `serde_json::Number::as_i64` / `as_u64` succeed only for integer
  literals.

For large numeric inputs (images, embeddings), packing the values into
`tensor_content` as one little-endian buffer is more compact than the
repeated fields, and the server accepts either form.

## tf.Example requests (classify, regress)

Classify and Regress take **`tf.Example`s** instead of tensors. An example
is a map from feature name to a typed list. There are exactly three list
types:

| feature list | holds |
|---|---|
| `float_list` | float32 |
| `int64_list` | int64 |
| `bytes_list` | byte strings |

The clients take one JSON object per example:

```json
{
  "examples": [
    {"petal_length": 1.4, "petal_width": 0.2, "color": "red", "ids": [3, 4]}
  ],
  "context": {"site": "lab-3"}
}
```

| JSON value | feature |
|---|---|
| `5.1`, `[5.1, 3.0]`, `[1, 2.5]` (any decimal) | `float_list` |
| `3`, `[3, 4]` (integers only) | `int64_list` |
| `"a"`, `["a", {"b64": "..."}]` | `bytes_list` |
| `{"float_list": [5, 3]}` | explicit type (the only way to send an empty list) |

A feature is always a flat list, and a scalar becomes a one-element list.
Booleans, `null` and nested arrays are rejected. The optional `context` is
one more example whose features apply to all the others, and is sent once
(`ExampleListWithContext`).

The same `1` vs `1.0` trap applies, and here the error comes from inside
the model, where `tf.io.parse_example` checks the type:

```console
$ echo '{"examples": [{"sepal_length": 5, "sepal_width": 3, "petal_length": 1, "petal_width": 0}]}' \
    | python python/client.py classify --model iris
input  1 examples (petal_length:int64, petal_width:int64, sepal_length:int64, sepal_width:int64)
error: Classify failed: INVALID_ARGUMENT: Name: <unknown>, Key: sepal_width, Index: 0.
  Data types don't match. Data type: int64 but expected type: float
```

The stderr line shows the inferred types before sending, so check it when a
request fails. REST has the same problem: it can't convert here either,
because the types are inside the serialized examples.

The converters are `python/json2example.py`, `go/examplejson/` and
`rust/src/json2example.rs`. They share their conformance cases in
`testdata/example_cases.json`.

## Reading tensor responses (gRPC)

A predict response is a map of output name → `TensorProto`. Rebuilding nested
arrays means reading the shape, getting the flat values, and nesting them
row-major. Getting the flat values has two cases:

**Typed fields** (`float_val`, `int64_val`, ...). TF Serving 2.21 answers
this way. The real `half_plus_two` output for a 2×2 input:

```
dtype: DT_FLOAT
tensor_shape { dim { size: 2 } dim { size: 2 } }
float_val: 2.5
float_val: 3
float_val: 3.5
float_val: 4
```

If the field holds **fewer** values than the shape has elements, the last
value repeats. `float_val: 0` with shape `[1000]` means a thousand zeros, and
no values at all means all zeros. TensorFlow uses this to shrink constant
tensors, so a client that ignores it reads a truncated array.

**`tensor_content`**: the raw little-endian bytes, in row-major order.
`tf.make_tensor_proto` produces this for numeric arrays, as do the mock
server and other servers built on TensorFlow. Strings never use it.

A client should accept both. All three here do, and they're tested against
both encodings.

A few more conventions in the clients' JSON output:

- float32 values print as the shortest decimal that round-trips the float32:
  `0.1`, not `0.10000000149011612`.
- Bytes that aren't valid UTF-8 print as `{"b64": "..."}`.
- NaN and ±Infinity print as the strings `"NaN"`, `"Infinity"` and
  `"-Infinity"`, because JSON has no literals for them.

The other responses are messages rather than tensors: `ClassificationResult`
(per example, a list of `{label, score}`), `RegressionResult` (per example,
one `value`), `GetModelMetadataResponse` (a `SignatureDefMap` packed in an
`Any`) and `GetModelStatusResponse`. `responses.py` / `responses.go` /
`responses.rs` render each one as JSON.

## REST request and response formats

REST accepts two layouts for predict, and the response layout follows the
request's:

**Row format**: `instances` is a list with one object per example, and you
get back `predictions`:

```console
$ curl -s localhost:8501/v1/models/demo:predict -d '{"instances": [
    {"age": 34, "income": 72000.0, "city": "Austin", "embedding": [1, 0, 0]},
    {"age": 51, "income": 38000.5, "city": "Dallas", "embedding": [0, 4, 0]}]}'
{"predictions": [{"score": 0.669074059, "label": "approve"},
                 {"score": 0.318864703, "label": "review"}]}
```

With a single input and a single output, the names disappear:

```console
$ curl -s localhost:8501/v1/models/demo:predict -d '{"signature_name": "half_plus_two", "instances": [1.0, 2.0, 5.0]}'
{"predictions": [2.5, 3.0, 4.5]}
```

**Columnar format**: `inputs` is one value per input name, batched, and you
get back `outputs`. This is the format the gRPC clients' JSON uses:

```console
$ curl -s localhost:8501/v1/models/demo:predict -d '{"inputs": {"age": [34, 51],
    "income": [72000.0, 38000.5], "city": ["Austin", "Dallas"], "embedding": [[1, 0, 0], [0, 4, 0]]}}'
{"outputs": {"label": ["approve", "review"], "score": [0.669074059, 0.318864703]}}
```

Columnar format doesn't force equal batch sizes the way rows do, but the
model usually does. With batching on, the server rejects the mismatch:
`Batching Run() input tensors must have equal 0th-dimension size`. Pick one
format per request; you can't mix them.

Things to know about REST:

- **Binary data** goes in as `{"b64": "..."}` (`"city": {"b64": "QXVzdGlu"}`
  is `"Austin"`). Outputs whose name ends in `_bytes` come back the same way.
- **Floats** are printed with about 9 significant digits (`0.669074059`),
  where gRPC gives the exact float32 (`0.66907406`). They are the same
  float32.
- **Errors** are `{"error": "..."}` with an HTTP status:
  `Missing named input: embedding in 'inputs' object.` (400),
  `Could not find any versions of model nope` (404).
- Classify and Regress answer with `results`: a list of `[label, score]`
  pairs per example for classify, one number per example for regress.

---
Next: [6. Monitoring](06-monitoring.md)
