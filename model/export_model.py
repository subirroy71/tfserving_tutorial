"""Export the SavedModels that the tutorial serves.

The weights are fixed constants, so every export is deterministic and the
outputs shown in the docs are reproducible.

  models/demo/1   TF2 model (tf.saved_model.save), two Predict signatures:
    serving_default   age int64 [batch], income float32 [batch],
                      city string [batch], embedding float32 [batch, 3]
                      -> score float32 [batch], label string [batch]
    half_plus_two     x float32, any shape -> y = x / 2 + 2

  models/demo/2   the same signatures with "retrained" weights, so a response
                  shows which version answered. Used for version policies,
                  labels, canaries and rollback.

  models/iris/1   TF1-style graph built with SavedModelBuilder, because only
                  that API can write Classify and Regress signatures. Its
                  inputs are serialized tf.Example protos.
    serving_default / classify   tf.Example(sepal_length, sepal_width,
                      petal_length, petal_width: float) -> classes, scores
    regress           tf.Example(petal_length: float) -> outputs (petal width)
    predict           features float32 [batch, 4] -> probabilities, class_id

Each version also gets assets.extra/tf_serving_warmup_requests, a TFRecord of
PredictionLog requests that the model server replays before marking the
version AVAILABLE.

Usage:  python model/export_model.py [models_dir]      (default: models)
Needs:  pip install -r model/requirements.txt
"""
import os
import shutil
import sys

import tensorflow as tf
from tensorflow_serving.apis import (
    classification_pb2,
    input_pb2,
    model_pb2,
    predict_pb2,
    prediction_log_pb2,
    regression_pb2,
)

# --- demo -------------------------------------------------------------------

DEMO_WEIGHTS = {
    # version: (age, income, austin, embedding, bias)
    1: (0.03, 0.00001, 0.5, [0.5, -0.25, 0.125], 0.0),
    2: (0.035, 0.000012, 0.25, [0.6, -0.2, 0.1], -0.2),
}


class Demo(tf.Module):
    def __init__(self, version):
        super().__init__()
        w_age, w_income, w_austin, w_embedding, bias = DEMO_WEIGHTS[version]
        self.w_age = tf.constant(w_age)
        self.w_income = tf.constant(w_income)
        self.w_austin = tf.constant(w_austin)
        self.bias = tf.constant(bias)
        self.w_embedding = tf.Variable(w_embedding, name="w_embedding")

    @tf.function(
        input_signature=[
            tf.TensorSpec([None], tf.int64, name="age"),
            tf.TensorSpec([None], tf.float32, name="income"),
            tf.TensorSpec([None], tf.string, name="city"),
            tf.TensorSpec([None, 3], tf.float32, name="embedding"),
        ]
    )
    def score(self, age, income, city, embedding):
        is_austin = tf.cast(tf.equal(tf.strings.lower(city), "austin"), tf.float32)
        logit = (
            self.w_age * (tf.cast(age, tf.float32) - 40.0)
            + self.w_income * (income - 50000.0)
            + self.w_austin * is_austin
            + tf.linalg.matvec(embedding, self.w_embedding)
            + self.bias
        )
        score = tf.sigmoid(logit)
        label = tf.where(score > 0.5, "approve", "review")
        return {"score": score, "label": label}

    @tf.function(input_signature=[tf.TensorSpec(None, tf.float32, name="x")])
    def half_plus_two(self, x):
        return {"y": x / 2.0 + 2.0}


def demo_warmup():
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "demo"
    request.model_spec.signature_name = "serving_default"
    for name, value in {
        "age": tf.constant([40], tf.int64),
        "income": tf.constant([50000.0]),
        "city": tf.constant(["Austin"]),
        "embedding": tf.constant([[0.0, 0.0, 0.0]]),
    }.items():
        request.inputs[name].CopyFrom(tf.make_tensor_proto(value))
    return [prediction_log_pb2.PredictionLog(predict_log=prediction_log_pb2.PredictLog(request=request))]


def export_demo(models_dir, version):
    path = os.path.join(models_dir, "demo", str(version))
    m = Demo(version)
    tf.saved_model.save(m, path, signatures={"serving_default": m.score, "half_plus_two": m.half_plus_two})
    write_warmup(path, demo_warmup())
    return path


# --- iris -------------------------------------------------------------------

IRIS_CLASSES = ["setosa", "versicolor", "virginica"]
IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
# Hand-set linear classifier: rows are features, columns are classes.
IRIS_W = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [-4.0, 0.0, 3.0],
    [0.0, 0.0, 2.0],
]
IRIS_B = [10.0, 0.0, -19.0]
# Regression: petal_width ~ 0.416 * petal_length - 0.363 (least squares on the
# classic data set).
IRIS_REG = (0.416, -0.363)


def export_iris(models_dir, version=1):
    path = os.path.join(models_dir, "iris", str(version))
    v1 = tf.compat.v1
    sig = v1.saved_model.signature_def_utils
    with tf.Graph().as_default() as graph, v1.Session(graph=graph) as sess:
        # Classify and Regress feed a batch of serialized tf.Example protos
        # into one string tensor; the graph parses them itself.
        serialized = v1.placeholder(tf.string, [None], name="tf_example")
        spec = {f: tf.io.FixedLenFeature([], tf.float32) for f in IRIS_FEATURES}
        parsed = tf.io.parse_example(serialized, spec)
        features = tf.stack([parsed[f] for f in IRIS_FEATURES], axis=1)

        w = tf.constant(IRIS_W)
        b = tf.constant(IRIS_B)

        def head(x):
            return tf.nn.softmax(tf.matmul(x, w) + b)

        scores = head(features)
        classes = tf.tile(tf.constant([IRIS_CLASSES]), [tf.shape(scores)[0], 1])

        # Regression only needs petal_length, so it parses its own examples
        # and does not require the other three features to be present.
        reg_serialized = v1.placeholder(tf.string, [None], name="tf_example_regress")
        reg_parsed = tf.io.parse_example(reg_serialized, {"petal_length": tf.io.FixedLenFeature([], tf.float32)})
        petal_width = IRIS_REG[0] * reg_parsed["petal_length"] + IRIS_REG[1]

        # A plain tensor signature on the same graph, for Predict.
        dense = v1.placeholder(tf.float32, [None, 4], name="features")
        probabilities = head(dense)
        class_id = tf.argmax(probabilities, axis=1)

        classify = sig.classification_signature_def(serialized, classes, scores)
        signatures = {
            "serving_default": classify,
            "classify": classify,
            "regress": sig.regression_signature_def(reg_serialized, petal_width),
            "predict": sig.predict_signature_def(
                {"features": dense}, {"probabilities": probabilities, "class_id": class_id}
            ),
        }
        builder = v1.saved_model.Builder(path)
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.SERVING], signature_def_map=signatures)
        builder.save()
    write_warmup(path, iris_warmup())
    return path


def iris_example(**features):
    ex = tf.train.Example()
    for name, value in features.items():
        ex.features.feature[name].float_list.value.append(value)
    return ex


def iris_warmup():
    spec = model_pb2.ModelSpec(name="iris", signature_name="classify")
    examples = input_pb2.Input()
    examples.example_list.examples.append(
        iris_example(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)
    )
    classify = classification_pb2.ClassificationRequest(model_spec=spec, input=examples)
    spec = model_pb2.ModelSpec(name="iris", signature_name="regress")
    examples = input_pb2.Input()
    examples.example_list.examples.append(iris_example(petal_length=1.4))
    regress = regression_pb2.RegressionRequest(model_spec=spec, input=examples)
    return [
        prediction_log_pb2.PredictionLog(classify_log=prediction_log_pb2.ClassifyLog(request=classify)),
        prediction_log_pb2.PredictionLog(regress_log=prediction_log_pb2.RegressLog(request=regress)),
    ]


# --- shared -----------------------------------------------------------------


def write_warmup(path, logs):
    # The model server looks for exactly this file name in this directory.
    extra = os.path.join(path, "assets.extra")
    os.makedirs(extra, exist_ok=True)
    with tf.io.TFRecordWriter(os.path.join(extra, "tf_serving_warmup_requests")) as w:
        for log in logs:
            w.write(log.SerializeToString())


def main():
    models_dir = sys.argv[1] if len(sys.argv) > 1 else "models"
    for name in ("demo", "iris"):
        shutil.rmtree(os.path.join(models_dir, name), ignore_errors=True)
    for path in (export_demo(models_dir, 1), export_demo(models_dir, 2), export_iris(models_dir)):
        print(f"exported {path}")


if __name__ == "__main__":
    main()
