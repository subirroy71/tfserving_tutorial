# tfserving_tutorial

Call a model in [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
over gRPC from **Python, Go and Rust**, building the request tensors
**on the fly from JSON**.

Most gRPC examples hard-code the model's inputs: one `TensorProto` per
feature, with the dtype and shape written into the client. Here the client
knows nothing about the model. It reads a JSON document, discovers each
tensor's shape by walking the nested arrays, infers the dtype from the JSON
literals, and sends whatever it found. The same binary can call any model and
any signature.

```console
$ echo '{"inputs": {"x": [[1.0, 2.0], [3.0, 4.0]]}}' | go run ./cmd/client --signature half_plus_two
input  x: DT_FLOAT [2, 2]
output y: DT_FLOAT [2, 2]
{
  "outputs": {
    "y": [[2.5, 3], [3.5, 4]]
  }
}
```

(The clients print one element per line; outputs in this README are
condensed.)

## What's in the repo

```
proto/        the 7 .proto files needed for Predict (TensorFlow + TF Serving)
model/        export_model.py - builds the demo SavedModel
models/demo/  the exported SavedModel, committed so you can skip TensorFlow
examples/     JSON requests
python/       json2tensor.py + client.py        (grpcio, no TensorFlow needed)
go/           tensorjson package + cmd/client   (generated stubs committed)
rust/         json2tensor.rs + main.rs          (tonic/prost, stubs built by build.rs)
testdata/     conversion cases shared by all three test suites
tools/        mock_server.py - Predict endpoint without Docker
scripts/      e2e.sh - runs all three clients and compares their outputs
```

## 1. Serve the model

The demo model has fixed weights and two signatures:

| signature | inputs | outputs |
|---|---|---|
| `serving_default` | `age` int64 `[batch]`, `income` float32 `[batch]`, `city` string `[batch]`, `embedding` float32 `[batch, 3]` | `score` float32 `[batch]`, `label` string `[batch]` |
| `half_plus_two` | `x` float32, **any shape** | `y` = x / 2 + 2, same shape |

It is already exported under `models/demo/1`. To re-export it yourself:

```bash
pip install -r model/requirements.txt
make model
```

Start TensorFlow Serving (gRPC on 8500, REST on 8501):

```bash
docker compose up
# or
docker run --rm -p 8500:8500 -p 8501:8501 \
  -v "$PWD/models/demo:/models/demo" -e MODEL_NAME=demo tensorflow/serving
```

The official image is built for x86-64. On an ARM machine (Apple Silicon) add
`--platform linux/amd64`, or use the stand-in server from
[section 6](#6-running-without-docker).

## 2. Run the clients

All three take the same flags: `--addr` (default `localhost:8500`), `--model`
(`demo`), `--signature` (`serving_default`), `--version`, `--input` (a file,
or `-` for stdin) and `--timeout`. They log the tensors they built to stderr
and print the outputs as JSON on stdout.

**Python**

```bash
pip install -r python/requirements.txt
./python/gen_protos.sh                      # writes python/gen/
python python/client.py --input examples/applicants.json
```

**Go** (1.24+)

```bash
cd go
go run ./cmd/client --input ../examples/applicants.json
```

**Rust**

```bash
cd rust
cargo run -- --input ../examples/applicants.json
```

Each prints:

```
input  age: DT_INT64 [3]
input  city: DT_STRING [3]
input  embedding: DT_FLOAT [3, 3]
input  income: DT_FLOAT [3]
output label: DT_STRING [3]
output score: DT_FLOAT [3]
{
  "outputs": {
    "label": ["approve", "review", "approve"],
    "score": [0.73885, 0.31216973, 0.8263534]
  }
}
```

## 3. The JSON format

A request is `{"inputs": {<name>: <tensor>, ...}}`. Each tensor is written in
one of two ways.

**Bare literal** - a scalar or nested arrays. Shape and dtype are inferred:

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
`null` are errors, reported with the input's name.

**Spec object** - when inference would pick the wrong thing, say it
explicitly. `dtype` and `shape` are each optional:

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
  case-insensitive and `DT_FLOAT`-style names work too. Integers are
  range-checked (`300` is rejected for `uint8`).
- `shape`: reshapes the flattened `values`; the element count must match.
- An empty tensor has no elements to infer from, so it needs a `dtype`.

> **The one thing that will bite you.** TensorFlow Serving's gRPC API does not
> convert dtypes. A float32 input written as `[1, 0, 0]` is inferred as int64
> and the server answers `INVALID_ARGUMENT`. Write `1.0` or wrap the input in
> `{"dtype": "float32", ...}` - `examples/applicants.json` does this for
> `embedding`.

Bare literals are the same "columnar" format TF Serving's REST API accepts, so
you can cross-check a request against port 8501 (REST *does* coerce dtypes,
because it reads the signature first):

```bash
curl -s localhost:8501/v1/models/demo:predict -d '{"inputs": {
  "age": [34, 51, 27], "income": [72000.0, 38000.5, 120000.0],
  "city": ["Austin", "Dallas", "austin"],
  "embedding": [[1, 0, 0], [0, 4, 0], [2, 2, 2]]}}'
```

## 4. How a tensor gets built

A `TensorProto` is three things: a `dtype`, a `tensor_shape`, and the elements
flattened in row-major order into the repeated field that matches the dtype.

| dtype | field |
|---|---|
| `DT_FLOAT` | `float_val` |
| `DT_DOUBLE` | `double_val` |
| `DT_INT32`, `DT_INT16`, `DT_INT8`, `DT_UINT8` | `int_val` |
| `DT_INT64` | `int64_val` |
| `DT_UINT32` / `DT_UINT64` | `uint32_val` / `uint64_val` |
| `DT_BOOL` | `bool_val` |
| `DT_STRING` | `string_val` (bytes) |

The three implementations (`python/json2tensor.py`,
`go/tensorjson/tensorjson.go`, `rust/src/json2tensor.rs`) follow the same
steps:

1. **Unwrap** a spec object, keeping its `dtype` and `shape` if present.
2. **Flatten** with one depth-first walk. The first time the walk reaches a
   depth it records that array's length as the dimension; every later array
   at that depth must have the same length, otherwise the tensor is ragged.
   Leaves are collected in visiting order, which *is* row-major order.
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

- **Python**: `json` already returns `int` vs `float` (check `bool` first -
  it is a subclass of `int`).
- **Go**: decode with `json.Decoder.UseNumber()` and look at the literal's
  text. Decoding into `float64` would lose both the distinction and any
  integer above 2^53.
- **Rust**: `serde_json::Number::as_i64` / `as_u64` succeed only for integer
  literals.

### Reading the response

Output tensors arrive in one of two encodings, and a client has to accept
both:

- typed fields (`float_val`, ...), which is what TF Serving sends by default.
  If a field holds fewer values than the shape has elements, the last value
  repeats; no values at all means zeros.
- `tensor_content`: the raw little-endian buffer. A model server can be
  configured to answer this way, and `tf.make_tensor_proto` produces it for
  numeric arrays.

All three clients decode either form, then rebuild the nested arrays from the
shape. float32 values are printed with their shortest round-trip decimal
(`0.1`, not `0.10000000149011612`). Bytes that are not valid UTF-8 come back
as `{"b64": ...}`, and NaN/Infinity as the strings `"NaN"`, `"Infinity"`,
`"-Infinity"`, since JSON has no literal for them.

## 5. The protos

`proto/` holds copies of the upstream files that `Predict` needs:

```
tensorflow/core/framework/{tensor,tensor_shape,types,resource_handle}.proto   (tensorflow v2.19.0)
tensorflow_serving/apis/{predict,model}.proto                                 (serving 2.19.0)
tensorflow_serving/apis/prediction_service.proto                              (trimmed)
```

`prediction_service.proto` is cut down to the `Predict` RPC. The upstream
service also declares `Classify`, `Regress`, `MultiInference` and
`GetModelMetadata`, which import about twenty more files. The package, service
and method names are unchanged, so the route is still
`/tensorflow.serving.PredictionService/Predict` and the stubs work against a
stock model server.

- Python: `python/gen_protos.sh` runs `grpc_tools.protoc`. The output goes to
  `python/gen/` (git-ignored) and the client puts that directory first on
  `sys.path`, so TensorFlow itself is not needed.
- Go: the generated code is committed in `go/gen/`. `go/gen_protos.sh`
  regenerates it; it uses `M` flags to fold upstream's one-package-per-file
  layout into two packages inside this module.
- Rust: `build.rs` runs `tonic-prost-build` with a vendored `protoc`, so
  `cargo build` needs nothing installed. Set `PROTOC` to use your own.

## 6. Running without Docker

`tools/mock_server.py` loads the SavedModel with TensorFlow and serves the
same `Predict` RPC. It is a test double for local runs and CI, not a model
server.

```bash
pip install tensorflow grpcio
pip install --no-deps tensorflow-serving-api
python tools/mock_server.py --port 8500              # answers with tensor_content
python tools/mock_server.py --port 8510 --as-fields  # answers with typed fields
```

Like the real gRPC endpoint, it rejects inputs whose dtype differs from the
signature.

## 7. Tests

```bash
make test        # unit tests in all three languages
make e2e         # all three clients against localhost:8500, outputs compared
```

`testdata/cases.json` lists JSON inputs with the expected dtype, shape and
values (or an expected error). The Python, Go and Rust suites all load that
one file, so the three converters cannot drift apart. Each suite also tests
decoding (`tensor_content`, repeated last value, bytes, non-finite floats)
and round trips.

## Where to go next

- **Ask the server for the signature.** `GetModelMetadata` returns each
  input's dtype and shape, which would let the client coerce `[1, 0, 0]` to
  float32 the way the REST API does. It needs `get_model_metadata.proto` and
  `meta_graph.proto` with their imports.
- **Send `tensor_content` for large tensors.** Packing the raw buffer is more
  compact than the repeated fields and avoids per-element varint encoding.
- **TLS and auth.** The clients use plaintext channels, which suits
  `localhost`. Use channel credentials for anything else.
- **Message size.** gRPC's default 4 MB receive limit is easy to hit with
  image batches; raise it on the channel.
