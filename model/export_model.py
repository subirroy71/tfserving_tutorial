"""Export the demo SavedModel that the tutorial serves.

The weights are fixed constants, so the export is deterministic and the
outputs shown in the README are reproducible. Two signatures are exported:

  serving_default   four inputs of different dtypes and ranks
                      age        int64   [batch]
                      income     float32 [batch]
                      city       string  [batch]
                      embedding  float32 [batch, 3]
                    ->
                      score      float32 [batch]
                      label      string  [batch]

  half_plus_two     x: float32, ANY shape  ->  y = x / 2 + 2, same shape

Usage:  python model/export_model.py [export_dir]
"""
import sys

import tensorflow as tf


class Demo(tf.Module):
    def __init__(self):
        super().__init__()
        self.w_embedding = tf.Variable([0.5, -0.25, 0.125], name="w_embedding")

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
            0.03 * (tf.cast(age, tf.float32) - 40.0)
            + 0.00001 * (income - 50000.0)
            + 0.5 * is_austin
            + tf.linalg.matvec(embedding, self.w_embedding)
        )
        score = tf.sigmoid(logit)
        label = tf.where(score > 0.5, "approve", "review")
        return {"score": score, "label": label}

    @tf.function(input_signature=[tf.TensorSpec(None, tf.float32, name="x")])
    def half_plus_two(self, x):
        return {"y": x / 2.0 + 2.0}


def main():
    export_dir = sys.argv[1] if len(sys.argv) > 1 else "models/demo/1"
    m = Demo()
    tf.saved_model.save(
        m,
        export_dir,
        signatures={"serving_default": m.score, "half_plus_two": m.half_plus_two},
    )
    print(f"exported SavedModel to {export_dir}")


if __name__ == "__main__":
    main()
