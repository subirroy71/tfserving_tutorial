#!/usr/bin/env python3
"""Replace a running model server's model config over gRPC.

Reads a text-format ModelServerConfig (the same format as
--model_config_file) and sends it with ModelService.HandleReloadConfigRequest.
The server then loads, unloads and relabels versions to match, and answers
once the new config is in effect (or has failed).

    python tools/reload_config.py config/models.config
    python tools/reload_config.py --addr host:8500 my.config

The config *replaces* the current one: a model missing from the file is
unloaded. If the server also polls a config file
(--model_config_file_poll_wait_seconds), the next poll reapplies that file and
undoes this change.

Needs python/gen (./python/gen_protos.sh); TensorFlow is not required.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python", "gen"))

import grpc  # noqa: E402
from google.protobuf import text_format  # noqa: E402
from tensorflow_serving.apis import model_management_pb2, model_service_pb2_grpc  # noqa: E402
from tensorflow_serving.config import model_server_config_pb2  # noqa: E402
from xla.tsl.protobuf import error_codes_pb2  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="text-format ModelServerConfig file")
    p.add_argument("--addr", default="localhost:8500", help="host:port of the gRPC endpoint")
    p.add_argument("--timeout", type=float, default=60.0, help="seconds to wait for the reload")
    args = p.parse_args()

    request = model_management_pb2.ReloadConfigRequest()
    try:
        with open(args.config, encoding="utf-8") as f:
            text_format.Parse(f.read(), request.config)
    except (OSError, text_format.ParseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    names = [c.name for c in request.config.model_config_list.config]
    print(f"sending config for {len(names)} models: {', '.join(names)}", file=sys.stderr)

    with grpc.insecure_channel(args.addr) as channel:
        stub = model_service_pb2_grpc.ModelServiceStub(channel)
        try:
            response = stub.HandleReloadConfigRequest(request, timeout=args.timeout)
        except grpc.RpcError as exc:
            print(f"error: HandleReloadConfigRequest failed: {exc.code().name}: {exc.details()}", file=sys.stderr)
            return 1

    code = error_codes_pb2.Code.Name(response.status.error_code)
    if response.status.error_code != error_codes_pb2.OK:
        print(f"error: reload rejected: {code}: {response.status.error_message}", file=sys.stderr)
        return 1
    print("reloaded", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
