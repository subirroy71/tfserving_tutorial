#!/usr/bin/env python3
"""Stand-in for TensorFlow Serving's gRPC Predict endpoint, for local tests/CI.

It loads the SavedModel with TensorFlow and answers
/tensorflow.serving.PredictionService/Predict with the same request and
response messages as the real model server, so the clients can be exercised
without Docker. It is NOT a replacement for tensorflow_model_server.

    pip install tensorflow tensorflow-serving-api grpcio
    python tools/mock_server.py --model-dir models/demo --port 8500

--as-fields makes it answer with typed fields (float_val, ...) the way
TensorFlow Serving does by default; without it, numeric tensors come back
packed in tensor_content. Clients must handle both.
"""
import argparse
import os
from concurrent import futures

import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc


class Predictor(prediction_service_pb2_grpc.PredictionServiceServicer):
    def __init__(self, name, model_dir, as_fields):
        self.name = name
        self.as_fields = as_fields
        self.version = max(int(d) for d in os.listdir(model_dir) if d.isdigit())
        self.model = tf.saved_model.load(os.path.join(model_dir, str(self.version)))

    def Predict(self, request, context):
        spec = request.model_spec
        if spec.name != self.name:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Servable not found for request: Latest({spec.name})")
        if spec.HasField("version") and spec.version.value != self.version:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Servable not found for request: Specific({spec.name}, {spec.version.value})")
        signature_name = spec.signature_name or "serving_default"
        if signature_name not in self.model.signatures:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Serving signature key \"{signature_name}\" not found.")
        fn = self.model.signatures[signature_name]
        _, wanted = fn.structured_input_signature
        if set(request.inputs) != set(wanted):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"input keys {sorted(request.inputs)} do not match signature inputs {sorted(wanted)}",
            )
        try:
            # tf.constant(TensorProto-as-ndarray) keeps the dtype the client
            # sent; the signature call then rejects mismatches, like TF Serving.
            feeds = {k: tf.constant(tf.make_ndarray(v)) for k, v in request.inputs.items()}
            for k, t in feeds.items():
                if t.dtype != wanted[k].dtype:
                    raise ValueError(f"input '{k}' expects {wanted[k].dtype.name}, got {t.dtype.name}")
            results = fn(**feeds)
        except Exception as exc:  # noqa: BLE001 - report every failure to the client
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc).splitlines()[0])

        response = predict_pb2.PredictResponse()
        response.model_spec.name = self.name
        response.model_spec.version.value = self.version
        response.model_spec.signature_name = signature_name
        for key, value in results.items():
            if request.output_filter and key not in request.output_filter:
                continue
            array = value.numpy()
            if self.as_fields:
                proto = tf.make_tensor_proto(array.ravel().tolist(), dtype=value.dtype, shape=array.shape)
            else:
                proto = tf.make_tensor_proto(array)
            response.outputs[key].CopyFrom(proto)
        return response


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default="models/demo", help="directory holding numbered versions")
    p.add_argument("--model-name", default="demo")
    p.add_argument("--port", type=int, default=8500)
    p.add_argument("--as-fields", action="store_true", help="answer with typed fields instead of tensor_content")
    args = p.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    prediction_service_pb2_grpc.add_PredictionServiceServicer_to_server(
        Predictor(args.model_name, args.model_dir, args.as_fields), server
    )
    server.add_insecure_port(f"127.0.0.1:{args.port}")
    server.start()
    print(f"mock PredictionService for '{args.model_name}' on 127.0.0.1:{args.port}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
