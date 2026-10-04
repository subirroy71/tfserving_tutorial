"""Run:  python -m unittest discover -s python   (after python/gen_protos.sh)"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "gen"))
sys.path.insert(0, HERE)

from tensorflow.core.framework import tensor_shape_pb2, types_pb2  # noqa: E402
from tensorflow.core.protobuf import meta_graph_pb2  # noqa: E402
from tensorflow_serving.apis import (  # noqa: E402
    classification_pb2,
    get_model_metadata_pb2,
    get_model_status_pb2,
    model_pb2,
    regression_pb2,
)

import json2example as j2e  # noqa: E402
import responses  # noqa: E402


class SharedCases(unittest.TestCase):
    def test_cases(self):
        with open(os.path.join(HERE, "..", "testdata", "example_cases.json"), encoding="utf-8") as f:
            cases = json.load(f)["cases"]
        for case in cases:
            with self.subTest(case["name"]):
                if case.get("error"):
                    with self.assertRaises(j2e.ExampleJSONError):
                        j2e.input_from_json(case["input"])
                    continue
                self.assertEqual(j2e.input_to_json(j2e.input_from_json(case["input"])), case["expect"])

    def test_error_names_the_example_and_feature(self):
        with self.assertRaisesRegex(j2e.ExampleJSONError, r'^example 1: feature "b": '):
            j2e.input_from_json({"examples": [{"a": 1}, {"b": []}]})
        with self.assertRaisesRegex(j2e.ExampleJSONError, r'^context: feature "c": '):
            j2e.input_from_json({"examples": [{"a": 1}], "context": {"c": None}})


class Responses(unittest.TestCase):
    def test_model_spec_line(self):
        spec = model_pb2.ModelSpec(name="demo", signature_name="half_plus_two")
        spec.version.value = 2
        self.assertEqual(responses.model_spec_line(spec), "model demo version 2 signature half_plus_two")
        self.assertEqual(responses.model_spec_line(model_pb2.ModelSpec(name="demo")), "model demo")

    def test_classification(self):
        result = classification_pb2.ClassificationResult()
        c = result.classifications.add()
        c.classes.add(label="a", score=0.1)
        c.classes.add(label="b", score=0.9)
        self.assertEqual(
            responses.classification_to_json(result),
            {"classifications": [[{"label": "a", "score": 0.1}, {"label": "b", "score": 0.9}]]},
        )

    def test_regression(self):
        result = regression_pb2.RegressionResult()
        result.regressions.add(value=0.1)
        self.assertEqual(responses.regression_to_json(result), {"regressions": [0.1]})

    def test_status(self):
        resp = get_model_status_pb2.GetModelStatusResponse()
        v = resp.model_version_status.add(version=3, state=get_model_status_pb2.ModelVersionStatus.LOADING)
        v.status.error_message = ""
        self.assertEqual(
            responses.status_to_json(resp),
            {"versions": [{"version": 3, "state": "LOADING", "error_code": "OK", "error_message": ""}]},
        )

    def test_metadata(self):
        sig = meta_graph_pb2.SignatureDef(method_name="tensorflow/serving/predict")
        sig.inputs["x"].dtype = types_pb2.DT_FLOAT
        sig.inputs["x"].tensor_shape.CopyFrom(
            tensor_shape_pb2.TensorShapeProto(dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=-1),
                                                   tensor_shape_pb2.TensorShapeProto.Dim(size=3)])
        )
        sig.outputs["y"].dtype = types_pb2.DT_STRING
        sig.outputs["y"].tensor_shape.unknown_rank = True
        sigs = get_model_metadata_pb2.SignatureDefMap()
        sigs.signature_def["serving_default"].CopyFrom(sig)
        resp = get_model_metadata_pb2.GetModelMetadataResponse()
        resp.metadata["signature_def"].Pack(sigs)
        self.assertEqual(
            responses.metadata_to_json(resp),
            {"signatures": {"serving_default": {
                "method": "tensorflow/serving/predict",
                "inputs": {"x": {"dtype": "DT_FLOAT", "shape": [-1, 3]}},
                "outputs": {"y": {"dtype": "DT_STRING", "shape": None}},
            }}},
        )


if __name__ == "__main__":
    unittest.main()
