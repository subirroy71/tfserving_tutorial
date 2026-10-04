# 2. Loading models: versions, policies, labels, reloads

This chapter covers how the server decides what to load: a single model, a
config file with several models, which versions stay loaded, how a new
version rolls out (and back), and how to change the config while the server
is running.

Every behaviour shown here was observed on `tensorflow/serving:2.21.0`.

## One model: the quick way

```bash
docker run --rm -p 8500:8500 -p 8501:8501 \
  -v "$PWD/models/demo:/models/demo" -e MODEL_NAME=demo \
  tensorflow/serving:2.21.0
```

The image's entrypoint turns `MODEL_NAME` into
`--model_name=demo --model_base_path=/models/demo`. The server loads the
highest version, here `2`, and checks every second for newer ones.

That's enough for one model. For anything more, use a config file.

## Several models: the config file

`config/models.config` is a text-format `ModelServerConfig`. It's the
same file `docker compose up` uses:

```
model_config_list {
  config {
    name: "demo"
    base_path: "/models/demo"
    model_platform: "tensorflow"
    model_version_policy {
      specific {
        versions: 1
        versions: 2
      }
    }
    version_labels { key: "stable" value: 1 }
    version_labels { key: "canary" value: 2 }
  }
  config {
    name: "iris"
    base_path: "/models/iris"
    model_platform: "tensorflow"
  }
}
```

Pass it with `--model_config_file=/config/models.config`. When that flag is
set, `--model_name` and `--model_base_path` are ignored. `base_path` is a path
**inside the container**. It can also be a `gs://` or `s3://` URL if the
server has credentials for it.

## Model status: is it loaded?

Ask before you send traffic:

```console
$ curl -s localhost:8501/v1/models/demo
{
 "model_version_status": [
  {"version": "2", "state": "AVAILABLE", "status": {"error_code": "OK", "error_message": ""}},
  {"version": "1", "state": "AVAILABLE", "status": {"error_code": "OK", "error_message": ""}}
 ]
}
$ python python/client.py status --model demo        # same over gRPC; also go/ and rust/
```

A version moves through these states:

```
START → LOADING → AVAILABLE → UNLOADING → END
              ↘ (load failed, retries exhausted) → END with an error_code
```

Only `AVAILABLE` versions serve requests. Versions that reached `END` stay in
the list until the server restarts, so the list is a history and not only
what's live. Asking for a model the server doesn't know gives HTTP 404
(`Could not find any versions of model nope`), and over gRPC `NOT_FOUND`.

## Version policies: which versions stay loaded

`model_version_policy` chooses which of the version directories under
`base_path` are loaded:

| policy | loads | typical use |
|---|---|---|
| *(none)* = `latest { num_versions: 1 }` | the highest version | most deployments |
| `latest { num_versions: 2 }` | the two highest | keep the previous version warm for fast rollback |
| `all {}` | every version directory | small models, debugging |
| `specific { versions: 1 versions: 2 }` | exactly these | pinned versions, canaries |

Observed while switching `demo` between policies (two versions on disk):

```
latest (default)       [('2', 'AVAILABLE'), ('1', 'END')]
all {}                 [('2', 'AVAILABLE'), ('1', 'AVAILABLE')]
specific { 1 }         [('2', 'END'),       ('1', 'AVAILABLE')]
```

Each loaded version costs its full memory, so `all {}` on a model with many
versions can exhaust RAM.

## Choosing a version in a request

Every API takes a `model_spec`. Leave out the version and you get the
**highest loaded** version. You can also ask for one by number or by label:

| | gRPC client | REST |
|---|---|---|
| latest | `predict --model demo` | `POST /v1/models/demo:predict` |
| by number | `predict --version 1` | `POST /v1/models/demo/versions/1:predict` |
| by label | `predict --label canary` | `POST /v1/models/demo/labels/canary:predict` |

Every response says which version answered, and the clients log it:

```console
$ python python/client.py predict --label stable --signature half_plus_two --input examples/half_plus_two_matrix.json
input  x: DT_FLOAT [2, 3]
model demo version 1 signature half_plus_two
output y: DT_FLOAT [2, 3]
...
```

Unknown versions and labels are errors, not a silent fallback:

```
Could not find version 7 of model demo           (REST 404 / gRPC NOT_FOUND)
Unrecognized servable version label: nope        (REST 400 / gRPC INVALID_ARGUMENT)
```

## Version labels

A label is a name for a version number, kept in the config:
`version_labels { key: "stable" value: 1 }`. Clients ask for `stable`, and
moving `stable` from 1 to 2 is a config change, with no client release. That
makes labels the tool for canaries and staged rollouts.

The server enforces one rule: **a label can only point at a version that is
already `AVAILABLE`.** That rule has two consequences.

**At startup, every label fails**, because no version has finished loading
when the config is first read:

```
Failed to start server. Error: FAILED_PRECONDITION: Request to assign label to
version 2 of model demo, which is not currently available for inference
```

Even a label on version 1 fails this way. Start the server with
`--allow_version_labels_for_unavailable_models=true`, which the compose file
does. With that flag, a label may point at a version that is still loading.

**At runtime, you can't load a version and label it in the same change.**
Without the flag, the server rejects that config change and keeps the old
one:

```
error: HandleReloadConfigRequest failed: FAILED_PRECONDITION: Request to assign
label to version 2 of model demo, which is not currently available for inference
```

### A canary rollout, step by step

This sequence was run against a live server. Each step is a config change
(see [reloading](#changing-the-config-while-running) below):

| step | `model_version_policy` | labels | result |
|---|---|---|---|
| 0 | `specific { 1 }` | stable → 1 | all traffic on v1 |
| 1 | `specific { 1, 2 }` | stable → 1 | v2 loads in the background (about 2 s here) |
| | *wait until `status` shows v2 `AVAILABLE`* | | |
| 2 | `specific { 1, 2 }` | stable → 1, **canary → 2** | some clients ask for `canary` |
| 3 | `specific { 1, 2 }` | **stable → 2** | promote; v1 still loaded for rollback |
| 4 | `specific { 2 }` | stable → 2 | retire v1 |

Rolling back is the same move in reverse: point `stable` back at 1, which
works as long as v1 is still loaded. That's why step 3 keeps it.

Labels don't split traffic by percentage. The *clients* (or a proxy in front
of the server) decide who asks for `canary`.

## Rolling out a new version

Under the default `latest` policy, a new version goes live when its directory
appears. The server polls `base_path` every `--file_system_poll_wait_seconds`
(default 1 s):

```console
$ cp -R models/iris/1 models/iris/.tmp-2 && mv models/iris/.tmp-2 models/iris/2
$ curl -s localhost:8501/v1/models/iris            # a few seconds later
[('2', 'AVAILABLE'), ('1', 'END')]
```

The server log shows the order:

```
Approving load for servable version {name: iris version: 2}
Successfully loaded servable version {name: iris version: 2}
Quiescing servable version {name: iris version: 1}
Done unloading servable version {name: iris version: 1}
```

**The new version loads (and warms up) before the old one unloads**, so
there's no moment without a model. In-flight requests on v1 finish first. The
cost is that both versions are in memory for a moment, so size the container
for two copies.

Copy to a hidden temporary name and `mv` it into place. `mv` within one
filesystem is atomic. A plain `cp -R models/iris/1 models/iris/2` can be
noticed half-copied.

### Rolling back

Delete the bad version's directory and the server goes back to the newest
remaining one:

```console
$ rm -rf models/iris/2
[('2', 'END'), ('1', 'AVAILABLE')]
```

The log shows v1 being loaded again from disk, so rollback takes as long as a
fresh load. `latest { num_versions: 2 }` or labels avoid that wait.

### When a version fails to load

A broken version doesn't take anything down. Here, `models/iris/3` holds a
corrupt `saved_model.pb`:

```
[('3', 'LOADING'), ('2', 'END'), ('1', 'AVAILABLE')]

loader.cc:471] SavedModel load for tags { serve }; Status: fail: OUT_OF_RANGE: Read less bytes than requested.
retrier.cc:40] Loading servable: {name: iris version: 3} failed: OUT_OF_RANGE: Read less bytes than requested
```

v1 keeps serving. v3 stays in `LOADING` while the server retries
(`--max_num_load_retries=5`, `--load_retry_interval_micros=60000000`, so for
about five minutes), then goes to `END` with the error in `status`. Alert on
it with the `load_attempt_count{status="fail"}` metric
([chapter 6](06-monitoring.md)), because the only other sign is that the old
version keeps answering.

## Changing the config while running

There are two ways to change models, policies or labels without restarting.

### 1. Let the server poll the file

`--model_config_file_poll_wait_seconds=10` (set in the compose file) re-reads
the config file every 10 s. Swapping the labels in `config/models.config` and
waiting 10 s:

```
before:  model demo version 1 signature half_plus_two     (--label stable)
after:   model demo version 2 signature half_plus_two
```

This is the simplest option, and it suits a ConfigMap in Kubernetes
([chapter 7](07-production.md)).

### 2. Push it over gRPC

`ModelService.HandleReloadConfigRequest` takes a whole `ModelServerConfig`.
`tools/reload_config.py` sends a text-format file:

```console
$ python tools/reload_config.py config/models.config
sending config for 2 models: demo, iris
reloaded
```

What to know before you script it:

- **It replaces the whole config.** A model missing from the request is
  unloaded.
- **File polling overrides it.** If the server also polls a config file, the
  next poll puts that file's contents back. Use one method or the other.
- **It returns before new *versions* load.** In step 1 of the canary above,
  the call came back while v2 was still loading, and v2 reached `AVAILABLE`
  about 2 s later. Poll `status` before the next step.
- **It waits for new *models* to load.** A model added with a `base_path`
  that doesn't exist makes the call hang until the client's deadline
  (`DEADLINE_EXCEEDED`). Worse, **later reloads queue behind it** and time
  out too, until the server restarts. Always set a deadline, and check paths
  first.
- Errors come back as gRPC errors (`FAILED_PRECONDITION: ...`), and the old
  config stays in effect.

---
Next: [3. Serving: ports, flags, batching](03-serving.md)
