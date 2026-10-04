# 6. Monitoring

The model server can expose Prometheus metrics: request counts and
latencies per model, batching behaviour, and model loads. This chapter turns
them on, explains the ones that matter, and runs Prometheus and Grafana with
a ready-made dashboard.

## Turn on the metrics endpoint

Metrics are off by default. A `MonitoringConfig` text proto turns them on
(`config/monitoring.config`):

```
prometheus_config {
  enable: true
  path: "/monitoring/prometheus/metrics"
}
```

Pass it with `--monitoring_config_file=/config/monitoring.config`. The
endpoint is served on the **REST port**, so `--rest_api_port` must be set too:

```console
$ curl -s localhost:8501/monitoring/prometheus/metrics | grep 'request_count{'
:tensorflow:serving:request_count{model_name="demo",status="INVALID_ARGUMENT"} 2
:tensorflow:serving:request_count{model_name="demo",status="OK"} 7
:tensorflow:serving:request_count{model_name="iris",status="OK"} 4
```

Metric names start with a colon. That's legal in Prometheus and works in
PromQL as written. The endpoint lists nearly a thousand series, most of them
TensorFlow internals. The ones below are the ones to watch.

## The metrics that matter

All latencies are in **microseconds**, and all are histograms (`_bucket`,
`_sum`, `_count`).

### Traffic and errors

| metric | labels | meaning |
|---|---|---|
| `:tensorflow:serving:request_count` | `model_name`, `status` | requests by outcome (`OK`, `INVALID_ARGUMENT`, ...) |
| `:tensorflow:serving:request_latency` | `model_name`, `API`, `entrypoint` | time in the server for a whole request |
| `:tensorflow:serving:runtime_latency` | `model_name`, `API`, `runtime` | time running the model graph only |

`entrypoint` is `GRPC` or `REST`. `API` is `Predict` / `Classify` / `Regress`
for gRPC, lower-case `predict` / `classify` / `regress` for REST, and empty for
status and metadata calls. A big gap between `request_latency` and
`runtime_latency` means time spent outside the model: batching queues, or
encoding and decoding large payloads (especially JSON over REST).

### Batching

Batching metrics appear with `--enable_batching`:

| metric | meaning |
|---|---|
| `:tensorflow:serving:batching_session:queuing_latency` | time a request waited for its batch |
| `:tensorflow:serving:batching_session:input_batch_size` | rows per incoming request |
| `:tensorflow:serving:batching_session:processed_batch_size` | rows per executed batch, **including padding** |
| `:tensorflow:serving:batching_session:padding_size` | padding rows added, by `execution_batch_size` |

These tell you whether batching is paying off. During a burst of 100
concurrent two-row requests against this setup, executed batches averaged
about 8.4 rows (mostly the smallest allowed size, 8) and the p99 queueing
delay was about 3.7 ms. That's a little over the 2 ms timeout, because of
scheduling under emulation. If executed batches are mostly
padding, lower `allowed_batch_sizes`. If queueing delay is near
`batch_timeout_micros` while batches stay small, there isn't enough
concurrent traffic for batching to help.

### Model lifecycle

| metric | labels | meaning |
|---|---|---|
| `:tensorflow:cc:saved_model:load_attempt_count` | `model_path`, `status` (`success`/`fail`) | every load attempt |
| `:tensorflow:cc:saved_model:load_latency` | `model_path` | load time |
| `:tensorflow:serving:model_warmup_latency` | `model_path`, `status` | warmup time ([chapter 1](01-models.md#warmup-requests)) |

`model_path` names the version directory, e.g. `/models/iris/1`. A failed
load leaves the previous version serving ([chapter 2](02-loading.md#when-a-version-fails-to-load)),
so the failure count is often the only sign that anything went wrong.

## PromQL recipes

```promql
# requests per second, per model and outcome
sum by (model_name, status) (rate(:tensorflow:serving:request_count[1m]))

# error ratio
sum(rate(:tensorflow:serving:request_count{status!="OK"}[5m]))
  / sum(rate(:tensorflow:serving:request_count[5m]))

# p99 latency per model and API, in microseconds
histogram_quantile(0.99,
  sum by (le, model_name, API) (rate(:tensorflow:serving:request_latency_bucket[5m])))

# average executed batch size (includes padding)
sum(rate(:tensorflow:serving:batching_session:processed_batch_size_sum[5m]))
  / sum(rate(:tensorflow:serving:batching_session:processed_batch_size_count[5m]))

# any failed load in the last hour
increase(:tensorflow:cc:saved_model:load_attempt_count{status="fail"}[1h]) > 0
```

Two cautions. `request_count` counts errors caused by the *caller*
(`INVALID_ARGUMENT`, `NOT_FOUND`) along with server faults, so alert on
`UNAVAILABLE`, `INTERNAL` and `DEADLINE_EXCEEDED` separately from bad input.
And the latency histograms only measure time **inside** the server. Network
time and client-side queueing don't appear there, so measure those in the
client.

## Prometheus and Grafana

The compose file has both behind a profile:

```bash
docker compose --profile monitoring up -d
#   Prometheus  http://localhost:9090
#   Grafana     http://localhost:3000   (anonymous access, no login)
# Port taken? GRAFANA_PORT=3300 PROMETHEUS_PORT=9091 docker compose --profile monitoring up -d
```

- `monitoring/prometheus.yml` scrapes `serving:8501/monitoring/prometheus/metrics`
  every 5 s.
- `monitoring/grafana/provisioning/` adds Prometheus as the default data
  source and loads every dashboard in `monitoring/grafana/dashboards/`.
- `monitoring/grafana/dashboards/tfserving.json` is the **TensorFlow
  Serving** dashboard:

| row | panels |
|---|---|
| top | requests/s, error ratio, p99 request latency, failed model loads |
| traffic | requests/s by model and status; p50/p99 request latency by model, API and entrypoint |
| inside the server | p50/p99 model run time; batching queueing delay, executed batch size, rows per request |
| lifecycle | load attempts and warmup time per version directory |

Generate some traffic to watch it move:

```bash
for i in $(seq 200); do
  curl -s localhost:8501/v1/models/demo/labels/canary:predict \
    -d '{"signature_name": "half_plus_two", "inputs": {"x": [1.0, 2.0]}}' >/dev/null &
done; wait
```

To check that Prometheus is scraping, open *Status → Targets* at
`localhost:9090`. The `tfserving` target should be `UP`.

## Health checks

TF Serving has no dedicated health endpoint. The model status call does the
job:

- **Ready** (send traffic here): `GET /v1/models/<name>` returns 200 and
  shows the version you need as `AVAILABLE`. The server opens its ports only
  after the initial models have loaded, so a 200 from a just-started server
  means it's ready.
- **Alive**: the gRPC port accepts connections. Don't base liveness on model
  state, or a slow model load gets the container killed in a restart loop.

[Chapter 7](07-production.md#kubernetes) turns these into Kubernetes probes.

## Logs

The server logs to stderr in glog format (`I0000 <time> <thread> <file>:<line>] <message>`).
The lines worth searching for:

| log line | means |
|---|---|
| `Successfully loaded servable version {name: X version: N}` | version N is live |
| `Done unloading servable version` | an old version is gone |
| `Loading servable: {...} failed: ...` | a load failed (retries follow) |
| `Finished reading warmup data ... Number of warmup records read: N` | warmup ran |
| `Running gRPC ModelServer at 0.0.0.0:8500` / `Exporting HTTP/REST API` | ports are open |
| `Failed to start server. Error: ...` | bad config at startup; the process exits |

`TF_CPP_MIN_LOG_LEVEL=1` hides the info lines, and `TF_CPP_VMODULE` turns on
verbose logs per source file.

---
Next: [7. Production](07-production.md)
