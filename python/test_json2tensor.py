"""Run:  python -m unittest discover -s python   (after python/gen_protos.sh)"""
import base64
import json
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "gen"))
sys.path.insert(0, HERE)

from tensorflow.core.framework import tensor_pb2, types_pb2  # noqa: E402

import json2tensor as j2t  # noqa: E402

_FIELDS = {
    "DT_FLOAT": "float_val", "DT_DOUBLE": "double_val", "DT_INT64": "int64_val",
    "DT_INT32": "int_val", "DT_INT16": "int_val", "DT_INT8": "int_val", "DT_UINT8": "int_val",
    "DT_UINT32": "uint32_val", "DT_UINT64": "uint64_val", "DT_BOOL": "bool_val", "DT_STRING": "string_val",
}


def shaped(dtype, dims, **fields):
    t = tensor_pb2.TensorProto(dtype=dtype, **fields)
    for d in dims:
        t.tensor_shape.dim.add(size=d)
    return t


class SharedCases(unittest.TestCase):
    def test_cases(self):
        with open(os.path.join(HERE, "..", "testdata", "cases.json"), encoding="utf-8") as f:
            cases = json.load(f)["cases"]
        for case in cases:
            with self.subTest(case["name"]):
                if case.get("error"):
                    with self.assertRaises(j2t.TensorJSONError):
                        j2t.tensor_from_json(case["input"])
                    continue
                want = case["expect"]
                got = j2t.tensor_from_json(case["input"])
                self.assertEqual(j2t.dtype_name(got.dtype), want["dtype"])
                self.assertEqual(j2t.tensor_shape(got), want["shape"])
                values = list(getattr(got, _FIELDS[want["dtype"]]))
                if "b64_values" in want:
                    self.assertEqual(values, [base64.b64decode(v) for v in want["b64_values"]])
                elif want["dtype"] == "DT_STRING":
                    self.assertEqual(values, [v.encode() for v in want["values"]])
                elif want["dtype"] in ("DT_FLOAT", "DT_DOUBLE"):
                    self.assertEqual(len(values), len(want["values"]))
                    for g, w in zip(values, want["values"]):
                        self.assertTrue(math.isclose(g, w, rel_tol=1e-6, abs_tol=1e-6))
                else:
                    self.assertEqual(values, want["values"])


class Decode(unittest.TestCase):
    def test_tensor_content(self):
        t = shaped(types_pb2.DT_FLOAT, [2, 2], tensor_content=struct.pack("<4f", 1, 2, 3, 4.5))
        self.assertEqual(j2t.tensor_to_json(t), [[1.0, 2.0], [3.0, 4.5]])
        t = shaped(types_pb2.DT_INT64, [2], tensor_content=struct.pack("<2q", -1, 2**40))
        self.assertEqual(j2t.tensor_to_json(t), [-1, 2**40])

    def test_typed_fields_and_scalar(self):
        self.assertEqual(j2t.tensor_to_json(shaped(types_pb2.DT_DOUBLE, [], double_val=[0.1])), 0.1)
        self.assertEqual(j2t.tensor_to_json(shaped(types_pb2.DT_BOOL, [2], bool_val=[True, False])), [True, False])

    def test_last_value_is_repeated(self):
        self.assertEqual(j2t.tensor_to_json(shaped(types_pb2.DT_INT32, [3], int_val=[7])), [7, 7, 7])
        self.assertEqual(j2t.tensor_to_json(shaped(types_pb2.DT_FLOAT, [2])), [0.0, 0.0])

    def test_strings(self):
        t = shaped(types_pb2.DT_STRING, [2], string_val=[b"ok", b"\xff\x00"])
        self.assertEqual(j2t.tensor_to_json(t), ["ok", {"b64": "/wA="}])

    def test_non_finite_and_empty(self):
        t = shaped(types_pb2.DT_FLOAT, [3], float_val=[float("nan"), float("inf"), float("-inf")])
        self.assertEqual(j2t.tensor_to_json(t), ["NaN", "Infinity", "-Infinity"])
        self.assertEqual(j2t.tensor_to_json(shaped(types_pb2.DT_FLOAT, [2, 0])), [[], []])

    def test_round_trip(self):
        for value in ([[1, 2], [3, 4]], ["a", "b"], [[True], [False]], 2.5, [[[0.5, 1.5]]]):
            self.assertEqual(j2t.tensor_to_json(j2t.tensor_from_json(value)), value)

    def test_bad_content_length(self):
        with self.assertRaises(j2t.TensorJSONError):
            j2t.tensor_to_json(shaped(types_pb2.DT_FLOAT, [2], tensor_content=b"\x00" * 7))


class Inputs(unittest.TestCase):
    def test_error_names_the_input(self):
        with self.assertRaisesRegex(j2t.TensorJSONError, 'input "x"'):
            j2t.inputs_from_json({"inputs": {"x": [[1], [2, 3]]}})
        with self.assertRaises(j2t.TensorJSONError):
            j2t.inputs_from_json({"x": [1]})


if __name__ == "__main__":
    unittest.main()
