"""Turn TF Serving responses (other than Predict) into plain JSON.

Predict outputs are tensors and go through json2tensor.tensor_to_json. The
other APIs answer with their own messages; these functions give each one a
small, stable JSON shape that the Go and Rust clients print identically.
"""
from __future__ import annotations

from typing import Any

from tensorflow.core.framework import types_pb2
from tensorflow_serving.apis import (
    classification_pb2,
    get_model_metadata_pb2,
    get_model_status_pb2,
    model_pb2,
    regression_pb2,
)
from xla.tsl.protobuf import error_codes_pb2

from json2tensor import json_float


def model_spec_line(spec: model_pb2.ModelSpec) -> str:
    """Which servable answered, e.g. "model demo version 2 signature half_plus_two"."""
    line = f"model {spec.name}"
    if spec.HasField("version"):
        line += f" version {spec.version.value}"
    if spec.signature_name:
        line += f" signature {spec.signature_name}"
    return line


def classification_to_json(result: classification_pb2.ClassificationResult) -> dict[str, Any]:
    """{"classifications": [[{"label": ..., "score": ...}, ...] per example]}"""
    return {
        "classifications": [
            [{"label": c.label, "score": json_float(c.score, True)} for c in classification.classes]
            for classification in result.classifications
        ]
    }


def regression_to_json(result: regression_pb2.RegressionResult) -> dict[str, Any]:
    """{"regressions": [value per example]}"""
    return {"regressions": [json_float(r.value, True) for r in result.regressions]}


def status_to_json(response: get_model_status_pb2.GetModelStatusResponse) -> dict[str, Any]:
    """{"versions": [{"version", "state", "error_code", "error_message"}, ...]}"""
    state_name = get_model_status_pb2.ModelVersionStatus.State.Name
    return {
        "versions": [
            {
                "version": v.version,
                "state": state_name(v.state),
                "error_code": error_codes_pb2.Code.Name(v.status.error_code),
                "error_message": v.status.error_message,
            }
            for v in response.model_version_status
        ]
    }


def _tensor_info(info) -> dict[str, Any]:
    shape = info.tensor_shape
    return {
        "dtype": types_pb2.DataType.Name(info.dtype),
        # null for unknown rank; -1 for an unknown dimension (usually the batch).
        "shape": None if shape.unknown_rank else [d.size for d in shape.dim],
    }


def metadata_to_json(response: get_model_metadata_pb2.GetModelMetadataResponse) -> dict[str, Any]:
    """{"signatures": {name: {"method", "inputs": {name: {dtype, shape}}, "outputs": ...}}}"""
    any_msg = response.metadata["signature_def"]
    sigs = get_model_metadata_pb2.SignatureDefMap()
    if not any_msg.Unpack(sigs):
        raise ValueError(f"metadata has unexpected type {any_msg.type_url}")
    return {
        "signatures": {
            name: {
                "method": sig.method_name,
                "inputs": {k: _tensor_info(v) for k, v in sorted(sig.inputs.items())},
                "outputs": {k: _tensor_info(v) for k, v in sorted(sig.outputs.items())},
            }
            for name, sig in sorted(sigs.signature_def.items())
        }
    }
