#!/usr/bin/env python3
"""TensorFlow Serving gRPC client that builds its request tensors from JSON.

    python client.py --model demo --input ../examples/applicants.json
    echo '{"inputs": {"x": [1.0, 2.0]}}' | python client.py --signature half_plus_two
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen"))

import grpc  # noqa: E402
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc  # noqa: E402

from json2tensor import TensorJSONError, describe, inputs_from_json, tensor_to_json  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--addr", default="localhost:8500", help="host:port of the gRPC endpoint")
    p.add_argument("--model", default="demo", help="model name")
    p.add_argument("--signature", default="serving_default", help="signature name")
    p.add_argument("--version", type=int, help="model version (default: latest)")
    p.add_argument("--input", default="-", help="JSON request file, or - for stdin")
    p.add_argument("--timeout", type=float, default=10.0, help="RPC deadline in seconds")
    args = p.parse_args()

    try:
        with (sys.stdin if args.input == "-" else open(args.input, encoding="utf-8")) as f:
            doc = json.load(f)
        inputs = inputs_from_json(doc)
    except (OSError, json.JSONDecodeError, TensorJSONError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    request = predict_pb2.PredictRequest()
    request.model_spec.name = args.model
    request.model_spec.signature_name = args.signature
    if args.version is not None:
        request.model_spec.version.value = args.version
    for name, tensor in sorted(inputs.items()):
        request.inputs[name].CopyFrom(tensor)
        print(f"input  {name}: {describe(tensor)}", file=sys.stderr)

    with grpc.insecure_channel(args.addr) as channel:
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        try:
            response = stub.Predict(request, timeout=args.timeout)
        except grpc.RpcError as exc:
            print(f"error: Predict failed: {exc.code().name}: {exc.details()}", file=sys.stderr)
            return 1

    outputs = {}
    for name, tensor in sorted(response.outputs.items()):
        print(f"output {name}: {describe(tensor)}", file=sys.stderr)
        outputs[name] = tensor_to_json(tensor)
    json.dump({"outputs": outputs}, sys.stdout, indent=2, sort_keys=True, ensure_ascii=False)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
