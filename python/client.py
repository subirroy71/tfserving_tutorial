#!/usr/bin/env python3
"""TensorFlow Serving gRPC client. Requests are built from JSON at runtime.

    python client.py predict  --model demo --input ../examples/applicants.json
    python client.py predict  --signature half_plus_two --label canary --input -
    python client.py classify --model iris --input ../examples/iris_classify.json
    python client.py regress  --model iris --signature regress --input ../examples/iris_regress.json
    python client.py metadata --model iris
    python client.py status   --model demo

Tensors and examples sent are logged to stderr, along with the servable that
answered; the result is printed as JSON on stdout. Exit status: 0 on success,
1 if the RPC failed, 2 for bad arguments or input.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen"))

import grpc  # noqa: E402
from tensorflow_serving.apis import (  # noqa: E402
    classification_pb2,
    get_model_metadata_pb2,
    get_model_status_pb2,
    model_service_pb2_grpc,
    predict_pb2,
    prediction_service_pb2_grpc,
    regression_pb2,
)

import json2example  # noqa: E402
import json2tensor  # noqa: E402
import responses  # noqa: E402


class UsageError(Exception):
    pass


def set_model_spec(spec, args, signature=True):
    spec.name = args.model
    if signature:
        spec.signature_name = args.signature
    if args.version is not None and args.label is not None:
        raise UsageError("--version and --label are mutually exclusive")
    if args.version is not None:
        spec.version.value = args.version
    if args.label is not None:
        spec.version_label = args.label


def read_json(path):
    with (sys.stdin if path == "-" else open(path, encoding="utf-8")) as f:
        return json.load(f)


def predict(args, channel):
    try:
        inputs = json2tensor.inputs_from_json(read_json(args.input))
    except (OSError, json.JSONDecodeError, json2tensor.TensorJSONError) as exc:
        raise UsageError(str(exc)) from None
    request = predict_pb2.PredictRequest()
    set_model_spec(request.model_spec, args)
    for name, tensor in sorted(inputs.items()):
        request.inputs[name].CopyFrom(tensor)
        print(f"input  {name}: {json2tensor.describe(tensor)}", file=sys.stderr)

    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    response = stub.Predict(request, timeout=args.timeout)
    print(responses.model_spec_line(response.model_spec), file=sys.stderr)
    outputs = {}
    for name, tensor in sorted(response.outputs.items()):
        print(f"output {name}: {json2tensor.describe(tensor)}", file=sys.stderr)
        outputs[name] = json2tensor.tensor_to_json(tensor)
    return {"outputs": outputs}


def examples_input(args):
    try:
        inp = json2example.input_from_json(read_json(args.input))
    except (OSError, json.JSONDecodeError, json2example.ExampleJSONError) as exc:
        raise UsageError(str(exc)) from None
    print(f"input  {json2example.describe(inp)}", file=sys.stderr)
    return inp


def classify(args, channel):
    request = classification_pb2.ClassificationRequest()
    set_model_spec(request.model_spec, args)
    request.input.CopyFrom(examples_input(args))
    response = prediction_service_pb2_grpc.PredictionServiceStub(channel).Classify(request, timeout=args.timeout)
    print(responses.model_spec_line(response.model_spec), file=sys.stderr)
    return responses.classification_to_json(response.result)


def regress(args, channel):
    request = regression_pb2.RegressionRequest()
    set_model_spec(request.model_spec, args)
    request.input.CopyFrom(examples_input(args))
    response = prediction_service_pb2_grpc.PredictionServiceStub(channel).Regress(request, timeout=args.timeout)
    print(responses.model_spec_line(response.model_spec), file=sys.stderr)
    return responses.regression_to_json(response.result)


def metadata(args, channel):
    request = get_model_metadata_pb2.GetModelMetadataRequest()
    set_model_spec(request.model_spec, args, signature=False)
    request.metadata_field.append("signature_def")  # the only field the server supports
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    response = stub.GetModelMetadata(request, timeout=args.timeout)
    print(responses.model_spec_line(response.model_spec), file=sys.stderr)
    return responses.metadata_to_json(response)


def status(args, channel):
    if args.label is not None:
        # The server would ignore it and report every version.
        raise UsageError("status does not support --label; use --version or neither")
    request = get_model_status_pb2.GetModelStatusRequest()
    set_model_spec(request.model_spec, args, signature=False)
    stub = model_service_pb2_grpc.ModelServiceStub(channel)
    return responses.status_to_json(stub.GetModelStatus(request, timeout=args.timeout))


COMMANDS = {
    # name: (handler, RPC name for errors, takes --signature/--input)
    "predict": (predict, "Predict", True),
    "classify": (classify, "Classify", True),
    "regress": (regress, "Regress", True),
    "metadata": (metadata, "GetModelMetadata", False),
    "status": (status, "GetModelStatus", False),
}


def parse_args(argv):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--addr", default="localhost:8500", help="host:port of the gRPC endpoint")
    common.add_argument("--model", default="demo", help="model name")
    common.add_argument("--version", type=int, help="model version (default: latest)")
    common.add_argument("--label", help="version label, e.g. stable or canary")
    common.add_argument("--timeout", type=float, default=10.0, help="RPC deadline in seconds")
    with_input = argparse.ArgumentParser(add_help=False)
    with_input.add_argument("--signature", default="serving_default", help="signature name")
    with_input.add_argument("--input", default="-", help="JSON request file, or - for stdin")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True, metavar="command")
    for name, (_, rpc, takes_input) in COMMANDS.items():
        parents = [common, with_input] if takes_input else [common]
        sub.add_parser(name, parents=parents, help=f"call {rpc}")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    handler, rpc, _ = COMMANDS[args.command]
    try:
        with grpc.insecure_channel(args.addr) as channel:
            result = handler(args, channel)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except grpc.RpcError as exc:
        print(f"error: {rpc} failed: {exc.code().name}: {exc.details()}", file=sys.stderr)
        return 1
    except ValueError as exc:  # a response we cannot render, e.g. unexpected metadata
        print(f"error: {exc}", file=sys.stderr)
        return 1
    json.dump(result, sys.stdout, indent=2, sort_keys=True, ensure_ascii=False)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
