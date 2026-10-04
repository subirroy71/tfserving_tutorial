# 7. Production

This chapter covers packaging the server and models as one image, running it
on Kubernetes, TLS, GPUs, and a troubleshooting table.

What was tested: the image, the Kubernetes manifests (on a local `kind`
cluster) and TLS were run for this chapter. The GPU section was not, because
the test machine has no NVIDIA GPU.

## One image per release

`deploy/Dockerfile` copies the models and their config into the official
image:

```dockerfile
FROM tensorflow/serving:2.21.0
COPY models /models
COPY config /config
ENTRYPOINT ["tensorflow_model_server", "--port=8500", "--rest_api_port=8501",
            "--model_config_file=/config/models.config", ...]
```

```bash
docker build -f deploy/Dockerfile -t tfserving-tutorial:1 .
docker run --rm -p 8500:8500 -p 8501:8501 tfserving-tutorial:1
```

A tag then pins exactly which model versions and which server version run
together, and rolling back is redeploying the previous tag.

The alternative is to keep models out of the image and point `base_path` at
`gs://`, `s3://` or a mounted volume, so new versions appear without a
deploy ([chapter 2](02-loading.md#rolling-out-a-new-version)). That's faster
to ship, but harder to know what's running. Many teams bake the model in for
small models and use object storage for large ones.

## Kubernetes

`deploy/k8s/tfserving.yaml` has a ConfigMap, a Deployment and two Services.

```bash
kubectl apply -f deploy/k8s/tfserving.yaml
kubectl rollout status deploy/tfserving
kubectl port-forward svc/tfserving 8500:8500 8501:8501
python python/client.py status --model demo
```

### Probes

```yaml
startupProbe:                     # slow model loads must not count as failures
  httpGet: {path: /v1/models/demo, port: http}
  periodSeconds: 5
  failureThreshold: 60            # up to 5 minutes to load
readinessProbe:
  httpGet: {path: /v1/models/demo, port: http}
  periodSeconds: 10
livenessProbe:                    # process alive, independent of model state
  tcpSocket: {port: grpc}
  periodSeconds: 20
```

The server opens its ports only after the initial models have loaded, so
`GET /v1/models/demo` answering 200 means the pod can serve. The startup probe
gives large models time to load before liveness starts counting. Without
it, a model that takes longer than the liveness window to load is
restarted forever.

### Config without a rollout

The ConfigMap is mounted over `/config`, replacing the copy in the image, and
the server re-reads it every 30 s (`--model_config_file_poll_wait_seconds=30`).
On the test cluster, editing the ConfigMap to swap `stable` and `canary` took
effect on the running pods **75 seconds later**, with no restart: the
kubelet's sync delay plus the poll interval:

```
before:  model demo version 1 signature half_plus_two      (--label stable)
after:   model demo version 2 signature half_plus_two
```

Labels, policies and the list of models can all change this way. The
[label rules from chapter 2](02-loading.md#version-labels) still apply:
load a version, wait for it, then label it.

### gRPC load balancing

A gRPC client keeps one long-lived HTTP/2 connection. Behind a normal
`ClusterIP` Service, that connection goes to **one pod**, so all of that
client's requests go there no matter how many replicas exist. The manifest
adds a **headless** Service, `tfserving-headless`, whose DNS name resolves
to every pod IP (on the test cluster: `10.244.0.5` and `10.244.0.6`). Clients
then balance across pods themselves:

```python
channel = grpc.insecure_channel(
    "dns:///tfserving-headless:8500",
    options=[("grpc.lb_policy_name", "round_robin")])
```

The same works in Go (`grpc.WithDefaultServiceConfig` with `round_robin`) and
in tonic (`Channel::balance_list`). A service mesh or an L7 proxy (Envoy,
Linkerd, Istio) does the same without client changes. REST doesn't have this
problem.

### Resources

- **Memory**: a loaded version takes roughly its SavedModel size plus
  TensorFlow's working memory. During a version switch, two versions are
  loaded at once ([chapter 2](02-loading.md#rolling-out-a-new-version)), so
  set the limit for that peak.
- **CPU**: set `--tensorflow_intra_op_parallelism` and
  `--tensorflow_inter_op_parallelism` to the CPUs you request. Otherwise
  TensorFlow sizes its thread pools for the *node's* CPU count and the
  container gets throttled.
- **Scaling**: scale on request rate or latency. CPU utilisation is a poor
  signal when batching keeps the CPU idle between batches.

## TLS

`--ssl_config_file` takes an `SSLConfig` text proto, with PEM contents
inline (not file paths):

```
server_key: "-----BEGIN PRIVATE KEY-----\n..."
server_cert: "-----BEGIN CERTIFICATE-----\n..."
custom_ca: ""          # set, with client_verify: true, for mutual TLS
client_verify: false
```

It applies to the gRPC port. Tested with a self-signed certificate:

```python
creds = grpc.ssl_channel_credentials(open("server.crt", "rb").read())
stub = model_service_pb2_grpc.ModelServiceStub(grpc.secure_channel("localhost:8500", creds))
stub.GetModelStatus(req)        # works
# a plaintext channel to the same port fails with UNAVAILABLE
```

The repo's clients use plaintext channels, which suits `localhost`. In a
cluster it's common to leave the server on plaintext and let a mesh or
ingress handle TLS and authentication. TF Serving has no authentication of
its own, so never expose it directly to the internet.

## GPUs

*Not tested here.* The `tensorflow/serving:2.21.0-gpu` image is built with
CUDA. It needs the NVIDIA container toolkit on the host (`docker run --gpus all`)
or the NVIDIA device plugin on Kubernetes (`resources.limits: {nvidia.com/gpu: 1}`).
The settings that matter:

- `--per_process_gpu_memory_fraction`: by default TensorFlow grows its
  memory use as needed. Set a fraction to share a GPU between processes.
- Batching matters much more on GPU. Use larger `max_batch_size`, fewer
  `num_batch_threads` (1 to 2 per GPU), and `allowed_batch_sizes` to limit
  how many shapes get compiled.
- Warmup matters too: cuDNN autotuning and CUDA context setup happen on the
  first request otherwise.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `Failed to start server. Error: FAILED_PRECONDITION: Request to assign label to version N ... not currently available` | labels in the config at startup | `--allow_version_labels_for_unavailable_models=true` ([ch. 2](02-loading.md#version-labels)) |
| `Expects arg[2] to be float but int64 is provided` | `1` sent instead of `1.0` over gRPC | write decimals or use `{"dtype": "float32"}` ([ch. 5](05-requests-and-responses.md#tensor-requests-predict)) |
| `Data types don't match. Data type: int64 but expected type: float` | the same, inside a `tf.Example` | `5.0` or `{"float_list": [...]}` |
| `Batching Run() input tensors must have at least one dimension` | scalar input with batching on | send `[x]` ([ch. 3](03-serving.md#what-batching-changes-for-clients)) |
| `Batching Run() input tensors must have equal 0th-dimension size` | inputs with different batch sizes | same number of rows in every input |
| `Servable not found for request` / `Could not find any versions of model` | wrong name, version not loaded, or still loading | `status`; check `base_path` and the version policy |
| `Unrecognized servable version label: X` | label not in the config, or the config not reloaded yet | check the config; wait one poll interval |
| new version never appears | directory name isn't a number, or the policy excludes it | ([ch. 2](02-loading.md#version-policies-which-versions-stay-loaded)) |
| version stuck in `LOADING` | load failing and retrying | server log: `Loading servable: ... failed:` |
| config reload hangs, then every later reload hangs too | a new model's `base_path` doesn't exist | restart; check paths before reloading |
| `RESOURCE_EXHAUSTED: Received message larger than max` | message over 4 MB | raise the client's receive limit, and the server's `--grpc_channel_arguments` |
| first request after a deploy is slow | no warmup file | add `assets.extra/tf_serving_warmup_requests` ([ch. 1](01-models.md#warmup-requests)) |
| `exec format error`, or very slow on a Mac | x86-64 image on ARM | `platform: linux/amd64` (emulated; fine for development only) |
| works over curl, fails over gRPC | REST converts dtypes, gRPC doesn't | see the dtype rows above |
