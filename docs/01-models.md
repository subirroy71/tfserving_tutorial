# 1. Models: export, inspect, warm up

TensorFlow Serving serves **SavedModels**. This chapter covers what one looks
like on disk, how its *signatures* define the API you call, and how to check a
model before handing it to the server.

The repo ships three ready-made versions under `models/`, so you can skip
straight to [chapter 2](02-loading.md) and come back later.

## The directory layout the server expects

```
models/
├── demo/                         one directory per model ("base path")
│   ├── 1/                        one numbered subdirectory per version
│   │   ├── saved_model.pb        graph + signatures
│   │   ├── fingerprint.pb        hash of the export (TF 2.12+)
│   │   ├── variables/            the weights
│   │   └── assets.extra/
│   │       └── tf_serving_warmup_requests   optional, see below
│   └── 2/
└── iris/
    └── 1/
```

The rules are simple and strict:

- The server is pointed at the **base path** (`models/demo`), never at a
  version directory.
- A version is a subdirectory whose name is a **positive integer**. Anything
  else (`latest/`, `v2/`, `.tmp-3/`) is ignored.
- Versions are compared numerically: `10` is newer than `9`.
- A version directory must be complete when it appears. Export to a temporary
  name and `mv` it into place; a half-copied directory can be picked up and
  fail to load (see [chapter 2](02-loading.md#rolling-out-a-new-version)).

## Signatures are the API

A SavedModel holds one or more **signatures** (`SignatureDef`s). Each one names
its input and output tensors, with a dtype and shape for each, plus a *method
name* that says which API can call it:

| method name | API | inputs | used for |
|---|---|---|---|
| `tensorflow/serving/predict` | Predict | any named tensors | almost everything |
| `tensorflow/serving/classify` | Classify | serialized `tf.Example`s | label + score per class |
| `tensorflow/serving/regress` | Regress | serialized `tf.Example`s | one float per example |

If a request doesn't name a signature, it gets `serving_default`.

The demo models:

| model | signature | method | inputs | outputs |
|---|---|---|---|---|
| `demo` | `serving_default` | predict | `age` int64 `[-1]`, `income` float `[-1]`, `city` string `[-1]`, `embedding` float `[-1, 3]` | `score` float `[-1]`, `label` string `[-1]` |
| `demo` | `half_plus_two` | predict | `x` float, any shape | `y` = x/2 + 2 |
| `iris` | `serving_default`, `classify` | classify | `inputs`: serialized `tf.Example` with 4 float features | `classes` string `[-1, 3]`, `scores` float `[-1, 3]` |
| `iris` | `regress` | regress | `inputs`: serialized `tf.Example` with `petal_length` | `outputs` float `[-1]` |
| `iris` | `predict` | predict | `features` float `[-1, 4]` | `probabilities` float `[-1, 3]`, `class_id` int64 `[-1]` |

`-1` is an unknown dimension, usually the batch. `demo` versions 1 and 2
have the same signatures but different weights, so responses show which
version answered.

## Inspecting a model with `saved_model_cli`

`saved_model_cli` comes with the `tensorflow` pip package. Run it on every
model before you deploy it: it shows you exactly what the server will
see.

```console
$ saved_model_cli show --dir models/demo/2 --tag_set serve --signature_def serving_default
The given SavedModel SignatureDef contains the following input(s):
  inputs['age'] tensor_info:
      dtype: DT_INT64
      shape: (-1)
      name: serving_default_age:0
  inputs['city'] tensor_info:
      dtype: DT_STRING
      shape: (-1)
      name: serving_default_city:0
  inputs['embedding'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 3)
      name: serving_default_embedding:0
  inputs['income'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1)
      name: serving_default_income:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['label'] tensor_info:
      dtype: DT_STRING
      shape: (-1)
      name: StatefulPartitionedCall:0
  outputs['score'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1)
      name: StatefulPartitionedCall:1
Method name is: tensorflow/serving/predict
```

`--all` prints every signature, and `run` executes one locally without a
server:

```console
$ saved_model_cli run --dir models/demo/2 --tag_set serve \
    --signature_def half_plus_two --input_exprs 'x=[[1.0,2.0]]'
Result for output key y:
[[2.5 3. ]]
```

The keys under `inputs[...]` and `outputs[...]` are the names you use in
requests and read from responses. The `name:` lines are internal graph tensor
names, and clients never see them.

The server reports the same information at runtime. See
[`metadata`](04-clients.md#metadata-what-does-the-model-expect).

## Exporting: TF2 (Predict signatures)

`model/export_model.py` builds all three versions. The TF2 part is the usual
pattern: a `tf.Module` with `tf.function`s whose `input_signature` fixes the
dtypes and shapes, passed to `tf.saved_model.save` as named signatures:

```python
class Demo(tf.Module):
    @tf.function(input_signature=[
        tf.TensorSpec([None], tf.int64, name="age"),
        tf.TensorSpec([None], tf.float32, name="income"),
        tf.TensorSpec([None], tf.string, name="city"),
        tf.TensorSpec([None, 3], tf.float32, name="embedding"),
    ])
    def score(self, age, income, city, embedding):
        ...
        return {"score": score, "label": label}     # dict keys = output names

tf.saved_model.save(m, "models/demo/2",
                    signatures={"serving_default": m.score, "half_plus_two": m.half_plus_two})
```

Things that matter for serving:

- **Return a dict.** The keys become output names. A bare tensor gets an
  unhelpful name like `output_0`.
- **Leave the batch dimension `None`.** Server-side batching
  ([chapter 3](03-serving.md#batching)) concatenates requests along dimension
  0. `half_plus_two` accepts any shape, including scalars, which is why a
  scalar request to it fails once batching is on.
- **Keras models** work the same way: `model.export("models/x/1")` (Keras 3)
  writes a SavedModel whose `serve` endpoint becomes `serving_default`.
- TF2 exports also list a pseudo-signature, `__saved_model_init_op`, that
  runs initialisers. It shows up in metadata and you never call it.

## Exporting: Classify and Regress signatures

Classify and Regress don't take arbitrary tensors. Their signatures follow a
fixed contract: one string input named `inputs` that holds a batch of
serialized `tf.Example` protos, and outputs named `classes` and/or `scores`
(classify) or `outputs` (regress). The model parses the examples itself.

**TF2.** A `tf.function` that follows the contract works:

```python
class M(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([None], tf.string, name="inputs")])
    def classify(self, inputs):
        x = tf.io.parse_example(inputs, {"x": tf.io.FixedLenFeature([], tf.float32)})["x"]
        scores = tf.stack([tf.sigmoid(-x), tf.sigmoid(x)], axis=1)
        classes = tf.tile(tf.constant([["neg", "pos"]]), [tf.shape(x)[0], 1])
        return {"classes": classes, "scores": scores}
```

`tf.saved_model.save` records it with the *predict* method name, which is
accepted because the server's `--enable_signature_method_name_check` is
`false` by default. If you turn that check on, these signatures stop working.

**TF1 builder.** `tf.compat.v1.saved_model` can write the real
`classify`/`regress` method names, which work with the check on or off. The
`iris` model is built this way, so its metadata shows the method names:

```python
v1 = tf.compat.v1
with tf.Graph().as_default() as graph, v1.Session(graph=graph) as sess:
    serialized = v1.placeholder(tf.string, [None], name="tf_example")
    parsed = tf.io.parse_example(serialized, {f: tf.io.FixedLenFeature([], tf.float32) for f in FEATURES})
    ...
    sig = v1.saved_model.signature_def_utils
    builder = v1.saved_model.Builder("models/iris/1")
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.SERVING], signature_def_map={
        "serving_default": sig.classification_signature_def(serialized, classes, scores),
        "regress": sig.regression_signature_def(reg_serialized, petal_width),
        "predict": sig.predict_signature_def({"features": dense}, {"probabilities": probs, "class_id": ids}),
    })
    builder.save()
```

Either way, `FixedLenFeature(..., tf.float32)` is part of the model's
contract. A client that sends `5` instead of `5.0` sends an `int64_list`, and
the server rejects it. The
[request chapter](05-requests-and-responses.md#tfexample-requests-classify-regress) covers
this.

Most new models only need Predict. Classify and Regress are here because
Estimator-era models use them, and because they come with a standard
response format.

## Warmup requests

The first request to a freshly loaded model is slow: TensorFlow builds and
optimises the graph lazily, and with a GPU it also allocates memory and
autotunes kernels. If that first request is a user's, they wait.

**Warmup** moves that cost into loading. Put a file named
`tf_serving_warmup_requests` under `assets.extra/` in the version directory.
It is a TFRecord file of `PredictionLog` protos, each holding one ordinary
request. The server replays them before marking the version `AVAILABLE`:

```python
from tensorflow_serving.apis import predict_pb2, prediction_log_pb2

request = predict_pb2.PredictRequest()
request.model_spec.name = "demo"
request.inputs["age"].CopyFrom(tf.make_tensor_proto([40], tf.int64))
...
log = prediction_log_pb2.PredictionLog(predict_log=prediction_log_pb2.PredictLog(request=request))
with tf.io.TFRecordWriter("models/demo/2/assets.extra/tf_serving_warmup_requests") as w:
    w.write(log.SerializeToString())
```

`ClassifyLog`, `RegressLog` and `MultiInferenceLog` work the same way; the
iris model warms up both its classify and regress signatures. The server
logs each replay:

```
saved_model_warmup_util.cc:240] Finished reading warmup data for model at
  /models/iris/1/assets.extra/tf_serving_warmup_requests.
  Number of warmup records read: 2. Elapsed time (microseconds): 650193.
```

That is 650 ms that the first user request no longer pays. The
`:tensorflow:serving:model_warmup_latency` metric records the same number
([chapter 6](06-monitoring.md)).

Warmup requests must be valid: if one fails, the version fails to load. Keep
them small (one or a few rows) and cover each signature and each batch size
you care about. The flag that turns warmup on is
`--enable_model_warmup` (default `true`).

## Re-exporting the models

```bash
pip install tensorflow==2.21.0
pip install --no-deps tensorflow-serving-api     # protos for the warmup file
make models                                      # rewrites models/demo/{1,2} and models/iris/1
```

Use the TensorFlow version that matches your model server (`2.21` here, see
`docker compose run --rm serving --version`). A model exported by a *newer*
TensorFlow can use ops that an older server does not have. Older exports
load fine in newer servers.

---
Next: [2. Loading models](02-loading.md)
