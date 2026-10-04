# tfserving_tutorial

A hands-on tutorial for [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving):
export models, load and version them, serve them, call every API from
**Python, Go and Rust**, read the responses, and monitor it all with
Prometheus and Grafana.

Every command and output in the docs was run against `tensorflow/serving:2.21.0`.

## Quick start

```bash
docker compose up -d                       # TF Serving: gRPC :8500, REST :8501
curl -s localhost:8501/v1/models/demo      # wait until versions 1 and 2 are AVAILABLE

python3 -m venv .venv && .venv/bin/pip install -r python/requirements.txt
PYTHON=.venv/bin/python ./python/gen_protos.sh
.venv/bin/python python/client.py predict --input examples/applicants.json
.venv/bin/python python/client.py classify --model iris --input examples/iris_classify.json

docker compose --profile monitoring up -d  # + Prometheus :9090, Grafana :3000
```

On Apple Silicon the server image runs under x86-64 emulation (set in the
compose file). If port 3000 or 9090 is taken, set `GRAFANA_PORT` /
`PROMETHEUS_PORT`.

## Chapters

1. [Models: export, inspect, warm up](docs/01-models.md): SavedModel layout,
   signatures, `saved_model_cli`, Predict vs Classify/Regress exports, warmup
   files
2. [Loading models](docs/02-loading.md): config files, version policies,
   labels, canary rollout, hot-loading and rollback, failed loads, reloading
   config at runtime
3. [Serving](docs/03-serving.md): gRPC vs REST, the flags that matter,
   server-side batching and what it changes for clients
4. [Calling the server](docs/04-clients.md): `predict`, `classify`,
   `regress`, `metadata`, `status` from three languages and curl; error codes
5. [Requests and responses](docs/05-requests-and-responses.md): building
   `TensorProto`s and `tf.Example`s from JSON, dtype traps, decoding
   responses, REST row and columnar formats
6. [Monitoring](docs/06-monitoring.md): the Prometheus endpoint, the
   metrics that matter, PromQL, the Grafana dashboard, health checks, logs
7. [Production](docs/07-production.md): one image per release, Kubernetes
   probes and config, gRPC load balancing, TLS, GPUs, troubleshooting

## The models

| model | versions | signatures | for |
|---|---|---|---|
| `demo` | 1, 2 (different weights) | `serving_default` (4 mixed-dtype inputs), `half_plus_two` | Predict, versions, labels |
| `iris` | 1 | `classify` (default), `regress`, `predict` | Classify, Regress |

They're committed under `models/` (84 KB), so TensorFlow is only needed to
re-export them (`make models`).

## What's in the repo

```
models/           SavedModels served by the tutorial (demo/1, demo/2, iris/1)
model/            export_model.py: builds them, including warmup files
config/           models.config, batching.config, monitoring.config
docker-compose.yml  the server, plus Prometheus + Grafana (profile "monitoring")
monitoring/       Prometheus scrape config, Grafana provisioning + dashboard
examples/         JSON requests for every API
python/           client.py, json2tensor.py, json2example.py, responses.py
go/               cmd/client, tensorjson/, examplejson/, responses/ (stubs committed in gen/)
rust/             main.rs, json2tensor.rs, json2example.rs, responses.rs (stubs built by build.rs)
proto/            the 38 upstream .proto files the clients need (TF / TF Serving 2.21.0)
testdata/         conversion cases shared by all three test suites
tools/            reload_config.py (gRPC config reload), mock_server.py (Predict-only stand-in)
deploy/           Dockerfile (models baked in), k8s/tfserving.yaml
scripts/e2e.sh    every command in all three clients against a live server
docs/             the chapters
```

## The protos

`proto/` holds unmodified copies of the upstream files for
`PredictionService` (Predict, Classify, Regress, MultiInference,
GetModelMetadata) and `ModelService` (GetModelStatus,
HandleReloadConfigRequest), with everything they import: 38 files from
`tensorflow/serving` 2.21.0 and `tensorflow/tensorflow` v2.21.0.

- **Python**: `python/gen_protos.sh` runs `grpc_tools.protoc` into
  `python/gen/` (git-ignored). The client puts that directory first on
  `sys.path`, so TensorFlow itself isn't needed.
- **Go**: the generated code is committed in `go/gen/`. `go/gen_protos.sh`
  regenerates it, folding upstream's one-package-per-file layout into one Go
  package per directory.
- **Rust**: `build.rs` runs `tonic-prost-build` with a vendored `protoc`, so
  `cargo build` needs nothing installed.

## Tests

```bash
make test        # unit tests: Python, Go, Rust
make e2e         # 15 cases x 3 clients against localhost:8500, outputs and exit codes compared
```

`testdata/cases.json` (tensors) and `testdata/example_cases.json`
(tf.Examples) list JSON inputs with the expected result or an expected
error. All three suites load the same files, so the converters can't drift
apart.

The Makefile uses `.venv/bin/python` when `.venv` exists. Make variables:
`PYTHON=...` picks another interpreter, `CARGO=...` the cargo
binary, and `ADDR=host:port` the server for `make e2e`.
