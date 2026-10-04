# 3. Serving: ports, flags, batching

[Chapter 2](02-loading.md) decided *what* the server loads. This chapter is
about *how* it serves: the two APIs, the flags that matter, server-side
batching, and running it in Docker.

## Two APIs, two ports

| | gRPC | REST |
|---|---|---|
| flag | `--port` (8500) | `--rest_api_port` (0 = off; 8501 by convention) |
| wire format | protobuf `TensorProto`s | JSON |
| dtypes | **sent as-is; must match the signature exactly** | coerced to the signature's dtype |
| binary data | raw bytes | `{"b64": "..."}` |
| speed | faster: no JSON parsing, compact numbers | slower for large tensors |
| also serves | ModelService (status, config reload) | `/monitoring/prometheus/metrics` |
| best for | services calling services | curl, browsers, quick tests |

Both ports reach the same loaded models, and you can mix them freely: a
request over REST and one over gRPC can land in the same batch.

The clients in this repo use gRPC. Every call in [chapter 4](04-clients.md)
also has its curl equivalent.

## The compose setup

`docker-compose.yml` runs the server with everything this tutorial uses:

```yaml
serving:
  image: tensorflow/serving:2.21.0
  platform: linux/amd64
  ports: ["8500:8500", "8501:8501"]
  volumes:
    - ./models:/models:ro
    - ./config:/config:ro
  entrypoint: ["tensorflow_model_server"]
  command:
    - --port=8500
    - --rest_api_port=8501
    - --model_config_file=/config/models.config
    - --model_config_file_poll_wait_seconds=10
    - --monitoring_config_file=/config/monitoring.config
    - --enable_batching
    - --batching_parameters_file=/config/batching.config
    - --enable_model_warmup=true
    - --allow_version_labels_for_unavailable_models=true
```

Three things are worth knowing about the official image:

- **Its entrypoint adds `--model_name=$MODEL_NAME --model_base_path=...`** to
  whatever you pass. That suits the single-model `docker run -e MODEL_NAME=...`
  form. With a config file it's clearer to replace the entrypoint, as above,
  and write every flag yourself.
- **It is built for x86-64 only.** On Apple Silicon it runs under emulation
  (`platform: linux/amd64`). That works for development, but latency numbers
  measured that way mean nothing.
- **Pin the tag.** `tensorflow/serving:latest` moves, and a new server
  version can change behaviour. Pin it, and export models with the matching
  TensorFlow version.

`tensorflow/serving:2.21.0-gpu` is the CUDA build; see
[chapter 7](07-production.md#gpus).

## Flags worth knowing

Every flag is listed by `docker compose run --rm serving --help`. These are
the ones you will actually touch:

| flag | default | what it does |
|---|---|---|
| `--port` / `--rest_api_port` | 8500 / 0 | gRPC and REST ports. REST is **off** unless you set it. |
| `--model_config_file` | | the models to serve ([chapter 2](02-loading.md)) |
| `--model_config_file_poll_wait_seconds` | 0 | re-read the config file this often; 0 = read once |
| `--file_system_poll_wait_seconds` | 1 | how often to look for new version directories |
| `--allow_version_labels_for_unavailable_models` | false | let a label point at a version that is still loading |
| `--enable_batching` / `--batching_parameters_file` | off | server-side batching, below |
| `--enable_model_warmup` | true | replay `assets.extra/tf_serving_warmup_requests` ([chapter 1](01-models.md#warmup-requests)) |
| `--monitoring_config_file` | | Prometheus endpoint ([chapter 6](06-monitoring.md)) |
| `--rest_api_timeout_in_ms` | 30000 | REST deadline. gRPC clients set their own. |
| `--rest_api_num_threads` / `--grpc_max_threads` | 40 / 40 | request-handling threads |
| `--tensorflow_intra_op_parallelism` / `--tensorflow_inter_op_parallelism` | auto | TensorFlow thread pools; set them when you pin a container to N CPUs |
| `--num_load_threads` | 0 (serial) | load several models/versions in parallel at startup |
| `--max_num_load_retries` / `--load_retry_interval_micros` | 5 / 60 s | retry a failed load, e.g. a model on flaky storage |
| `--per_process_gpu_memory_fraction` | 0 (grow) | cap GPU memory |
| `--grpc_channel_arguments` | | e.g. `grpc.max_receive_message_length=67108864` for inputs over 4 MB |
| `--ssl_config_file` | | TLS for gRPC ([chapter 7](07-production.md#tls)) |
| `--enable_signature_method_name_check` | false | reject Classify/Regress on signatures without the matching method name |

## Batching

A model running on a batch of 32 rows usually takes about as long as it
does on one row. It costs barely more on a CPU, and almost nothing more on
a GPU. **Server-side batching** gathers concurrent requests to the same model
into one batch, runs it once, and splits the result back out. Clients don't
change: each still sends its own request and gets its own response.

Turn it on with `--enable_batching` and tune it with a
`BatchingParameters` text proto (`config/batching.config`):

```
max_batch_size { value: 32 }        # rows per batch, across requests
batch_timeout_micros { value: 2000 } # run a partial batch after 2 ms
max_enqueued_batches { value: 100 }  # queue limit; beyond it, UNAVAILABLE
num_batch_threads { value: 4 }       # batches run concurrently
allowed_batch_sizes: 8               # pad batches up to one of these
allowed_batch_sizes: 16
allowed_batch_sizes: 32              # the last one must equal max_batch_size
```

How to think about the knobs:

- **`batch_timeout_micros`** is the latency you add to a request that arrives
  at a quiet moment. Start around 1 to 5 ms. Under load, batches fill before
  the timeout, so it costs nothing.
- **`max_batch_size`** caps how many rows one batch holds. A single request
  larger than this is split across batches automatically.
- **`allowed_batch_sizes`** pads each batch up to the next allowed size, so
  the model sees only a few distinct shapes. That matters for XLA and GPU
  kernels that compile per shape.
- **`num_batch_threads`**: about the number of CPU cores, or 1 to 2 per GPU.
- **`max_enqueued_batches`** gives load shedding. When the queue is full, new
  requests fail fast with `UNAVAILABLE` instead of queueing forever.

Batching applies to every model the server loads. With
`--enable_per_model_batching_parameters`, a model can ship its own
`batching_params.pbtxt` inside its version directory.

### What batching changes for clients

1. **Every input needs a batch dimension.** The server concatenates requests
   along dimension 0, so a rank-0 (scalar) input can't be batched:

   ```console
   $ python python/client.py predict --signature half_plus_two --input examples/half_plus_two_scalar.json
   input  x: DT_FLOAT []
   error: Predict failed: INVALID_ARGUMENT: Batching Run() input tensors must have at least one dimension
   ```

   Without `--enable_batching` the same request succeeds. Send `[3.0]`, not
   `3.0`.

2. **All inputs of one request must agree on dimension 0.** `age` with 3 rows
   and `city` with 2 can't be batched.

3. **Results can differ in the last bits.** Padding a batch from 3 rows to 8
   can make TensorFlow pick a different kernel. Here is the same row from
   `demo` v1, sent as part of a 3-row request and then a 2-row one:

   ```
   0.31216973   (3-row request)
   0.31216979   (2-row request)
   ```

   Any floating-point model behaves like this. Compare outputs with a
   tolerance, not with `==`.

4. **Latency goes up slightly at low load and throughput goes up at high
   load.** The batching metrics in [chapter 6](06-monitoring.md#batching) show
   how full the batches really are.

## Running without Docker

`tools/mock_server.py` is a small Python stand-in for the **Predict** API
only. It loads the newest version of one model with TensorFlow, for CI
machines and laptops without Docker:

```bash
pip install tensorflow==2.21.0 grpcio
pip install --no-deps tensorflow-serving-api
python tools/mock_server.py --model-dir models/demo --port 8500               # answers with tensor_content
python tools/mock_server.py --model-dir models/demo --port 8510 --as-fields   # answers with typed fields
```

It answers `/tensorflow.serving.PredictionService/Predict` with the same
messages and the same dtype strictness. It does not implement
Classify, Regress, metadata, status, labels, batching or monitoring. For
those you need the real server.

---
Next: [4. Calling the server](04-clients.md)
