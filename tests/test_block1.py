"""Regression tests for the Block 1 data and NumPy teaching library."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from common.my_dl_lib import BatchNorm1D, CrossEntropyLoss, Dropout, LayerNorm


ROOT = Path(__file__).resolve().parents[1]
CIRCLE_DIR = ROOT / "exercises/block_01_basics/task_01_circle_classifier"
MNIST_STARTER = ROOT / "exercises/block_01_basics/task_03_mnist_mlp/starter.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


circle_data = load_module("circle_data_for_tests", CIRCLE_DIR / "data_creater.py")
circle_model = load_module("circle_model_for_tests", CIRCLE_DIR / "Model.py")
mnist_starter = load_module("mnist_starter_for_tests", MNIST_STARTER)
mini_reference = load_module(
    "mini_network_reference_for_tests",
    ROOT / "solutions/block_01_basics/mini_network_reference.py",
)


class CircleDataTests(unittest.TestCase):
    def test_committed_files_are_radius_one_stratified_and_disjoint(self):
        train_path = CIRCLE_DIR / "train_data.csv"
        val_path = CIRCLE_DIR / "val_data.csv"
        self.assertTrue(circle_data.validate_data_splits(train_path, val_path))

        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)
        self.assertEqual(len(train), 800)
        self.assertEqual(len(val), 200)
        self.assertAlmostEqual(train["label"].mean(), val["label"].mean(), delta=0.005)

    def test_regeneration_uses_the_current_condition_instead_of_stale_csvs(self):
        square_condition = "(np.abs(x) <= 0.5) & (np.abs(y) <= 0.5)"
        with tempfile.TemporaryDirectory() as temp_dir:
            train_path = Path(temp_dir) / "train.csv"
            val_path = Path(temp_dir) / "val.csv"
            circle_data.create_data_splits(
                train_n=80,
                val_n=20,
                train_out_path=train_path,
                val_out_path=val_path,
                condition=circle_data.DEFAULT_CONDITION,
                seed=7,
            )
            old = pd.concat(
                [pd.read_csv(train_path), pd.read_csv(val_path)], ignore_index=True
            )
            old_labels = {
                (row.x, row.y): row.label for row in old.itertuples(index=False)
            }

            circle_data.create_data_splits(
                train_n=80,
                val_n=20,
                train_out_path=train_path,
                val_out_path=val_path,
                condition=square_condition,
                seed=7,
            )
            self.assertTrue(
                circle_data.validate_data_splits(train_path, val_path, square_condition)
            )
            new = pd.concat(
                [pd.read_csv(train_path), pd.read_csv(val_path)], ignore_index=True
            )
            changed = sum(
                old_labels[(row.x, row.y)] != row.label
                for row in new.itertuples(index=False)
            )
            self.assertGreater(changed, 0)

    def test_validation_rejects_a_wrong_label_and_train_val_leakage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            train, val = circle_data.create_data_splits(
                train_n=80,
                val_n=20,
                train_out_path=Path(temp_dir) / "train.csv",
                val_out_path=Path(temp_dir) / "val.csv",
                seed=11,
            )

        wrong = train.copy()
        wrong.loc[0, "label"] = 1 - wrong.loc[0, "label"]
        with self.assertRaisesRegex(ValueError, "labels that do not match"):
            circle_data.validate_data_splits(wrong, val)

        leaked = val.copy()
        leaked.iloc[0] = train.iloc[0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            circle_data.validate_data_splits(train, leaked)

    def test_model_records_a_real_validation_metric_each_epoch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            train, val = circle_data.create_data_splits(
                train_n=120,
                val_n=40,
                train_out_path=Path(temp_dir) / "train.csv",
                val_out_path=Path(temp_dir) / "val.csv",
                seed=5,
            )
        model = circle_model.MLPClassifier(
            train,
            val,
            Learning_rate=0.05,
            batch_size=20,
            epochs=3,
            seed=1,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit()
        self.assertEqual(len(model.loss), 3)
        self.assertEqual(len(model.train_accuracy), 3)
        self.assertEqual(len(model.val_loss), 3)
        self.assertEqual(len(model.val_accuracy), 3)
        self.assertTrue(np.isfinite(model.val_loss).all())
        self.assertTrue(all(0.0 <= value <= 1.0 for value in model.val_accuracy))
        val_loss, val_accuracy = model.evaluate(val)
        self.assertAlmostEqual(val_loss, model.val_loss[-1])
        self.assertAlmostEqual(val_accuracy, model.val_accuracy[-1])

    def test_prediction_accepts_new_points_without_known_labels(self):
        train = pd.read_csv(CIRCLE_DIR / "train_data.csv")
        val = pd.read_csv(CIRCLE_DIR / "val_data.csv")
        model = circle_model.MLPClassifier(train, val, seed=1)
        expected = model.predict(val)
        np.testing.assert_array_equal(model.predict(val[["x", "y"]]), expected)

    def test_training_loss_retains_extreme_logit_differences(self):
        train = pd.read_csv(CIRCLE_DIR / "train_data.csv")
        model = circle_model.MLPClassifier(train, train, seed=1)
        model.logits = np.array([[1000.0, 0.0]])
        model.y = np.array([[0.0, 1.0]])
        self.assertAlmostEqual(model.compute_loss(), 1000.0)


class CrossEntropyTests(unittest.TestCase):
    def test_confident_wrong_prediction_does_not_cap_the_loss(self):
        loss = CrossEntropyLoss()
        logits = np.array([[1000.0, 0.0], [0.0, 1000.0]])
        targets = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.assertAlmostEqual(loss.forward(logits, targets), 1000.0)
        np.testing.assert_allclose(loss.backward(), [[0.5, -0.5], [-0.5, 0.5]])
        self.assertAlmostEqual(loss.forward(logits + 1e6, targets), 1000.0)
        self.assertAlmostEqual(loss.forward(logits, 1 - targets), 0.0)

    def test_loss_and_gradient_agree_for_extreme_and_ordinary_logits(self):
        loss = CrossEntropyLoss()
        logits = np.array([[1000.0, 0.0, -1000.0], [0.4, -0.2, 0.8]])
        targets = np.eye(3)[[2, 1]]
        loss.forward(logits, targets)
        analytic = loss.backward().copy()
        epsilon = 1e-4
        numeric = np.zeros_like(logits)
        for index in np.ndindex(logits.shape):
            plus, minus = logits.copy(), logits.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            numeric[index] = (
                loss.forward(plus, targets) - loss.forward(minus, targets)
            ) / (2 * epsilon)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)


class BatchNormTests(unittest.TestCase):
    def test_train_and_eval_gradients_match_their_respective_forward(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(4, 3))
        dout = rng.normal(size=x.shape)
        epsilon = 1e-5
        for training in (True, False):
            with self.subTest(training=training):
                layer = BatchNorm1D(3)
                layer.gamma[...] = [[0.7, 1.2, -0.4]]
                layer.beta[...] = [[0.1, -0.2, 0.3]]
                layer.running_mean[...] = [[0.4, -0.3, 0.2]]
                layer.running_var[...] = [[0.5, 1.1, 0.8]]
                if not training:
                    layer.eval()
                layer.forward(x)
                analytic_dx = layer.backward(dout).copy()
                analytic_dgamma = layer.dgamma.copy()
                analytic_dbeta = layer.dbeta.copy()

                for value, analytic in (
                    (x, analytic_dx),
                    (layer.gamma, analytic_dgamma),
                    (layer.beta, analytic_dbeta),
                ):
                    numeric = np.zeros_like(value)
                    for index in np.ndindex(value.shape):
                        original = value[index]
                        value[index] = original + epsilon
                        plus = np.sum(layer.forward(x) * dout)
                        value[index] = original - epsilon
                        minus = np.sum(layer.forward(x) * dout)
                        value[index] = original
                        numeric[index] = (plus - minus) / (2 * epsilon)
                    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=1e-6)

    def test_eval_keeps_statistics_fixed_and_backward_uses_forward_mode(self):
        layer = BatchNorm1D(2)
        layer.eval()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        layer.forward(x)
        np.testing.assert_array_equal(layer.running_mean, np.zeros((1, 2)))
        np.testing.assert_array_equal(layer.running_var, np.ones((1, 2)))
        # A mode switch affects the next forward, not the cached computation.
        layer.train()
        np.testing.assert_allclose(
            layer.backward(np.ones_like(x)), np.ones_like(x) / np.sqrt(1 + layer.eps)
        )


class ReferenceOptimizerTests(unittest.TestCase):
    def test_adaptive_updates_apply_to_biases_as_well_as_weights(self):
        # For two identical x=1 rows, weight and bias receive the same gradient.
        # They must follow the same update in each advertised optimizer.
        for method in ("adam", "rmsprop", "adagrad"):
            with self.subTest(method=method):
                layer = mini_reference.Layer(1, 2, mini_reference.activation_ReLU)
                layer.weights[...] = 0.0
                inputs = np.ones((2, 1))
                update = getattr(layer, f"layer_backward_{method}")
                for demands in (
                    np.array([[2.0, -4.0], [4.0, -2.0]]),
                    np.array([[-1.0, 2.0], [-3.0, 4.0]]),
                ):
                    update(inputs, demands, learning_rate=0.01)
                    np.testing.assert_allclose(layer.weights[0], layer.biases)
                self.assertGreater(np.max(np.abs(layer.biases)), 0.0)

    def test_reference_training_loss_uses_logits_before_probabilities_underflow(self):
        logits = np.array([[1000.0, 0.0]])
        targets = np.array([[0.0, 1.0]])
        probs = mini_reference.activation_softmax(logits)
        self.assertAlmostEqual(
            mini_reference.precise_loss_function(probs, targets, logits=logits), 1000.0
        )


class LayerNormTests(unittest.TestCase):
    def test_arbitrary_leading_dimensions_preserve_shape(self):
        rng = np.random.default_rng(0)
        for shape in ((4,), (3, 4), (2, 3, 4), (2, 2, 3, 4)):
            layer = LayerNorm(4)
            x = rng.normal(size=shape)
            output = layer.forward(x)
            dx = layer.backward(rng.normal(size=shape))
            self.assertEqual(output.shape, shape)
            self.assertEqual(dx.shape, shape)
            self.assertEqual(layer.dgamma.shape, (1, 4))
            self.assertEqual(layer.dbeta.shape, (1, 4))

    def test_sequence_parameter_gradients_reduce_batch_and_time(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2, 3, 4))
        dout = rng.normal(size=x.shape)
        layer = LayerNorm(4)
        layer.forward(x)
        layer.backward(dout)
        np.testing.assert_allclose(
            layer.dbeta, np.sum(dout, axis=(0, 1), keepdims=False)[None, :]
        )
        np.testing.assert_allclose(
            layer.dgamma,
            np.sum(dout * layer.x_hat, axis=(0, 1), keepdims=False)[None, :],
        )

    def test_sequence_input_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(2, 2, 3))
        dout = rng.normal(size=x.shape)
        layer = LayerNorm(3)
        layer.gamma[...] = np.array([[0.7, 1.2, -0.4]])
        layer.beta[...] = np.array([[0.1, -0.2, 0.3]])

        epsilon = 1e-5
        numeric = np.zeros_like(x)
        for index in np.ndindex(x.shape):
            plus = x.copy()
            minus = x.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            plus_loss = np.sum(layer.forward(plus) * dout)
            minus_loss = np.sum(layer.forward(minus) * dout)
            numeric[index] = (plus_loss - minus_loss) / (2 * epsilon)

        layer.forward(x)
        analytic = layer.backward(dout)
        np.testing.assert_allclose(analytic, numeric, rtol=2e-4, atol=2e-5)


class ValidationAndDropoutTests(unittest.TestCase):
    def test_mnist_split_is_disjoint_and_stratified(self):
        labels = np.repeat(np.arange(10), 100)
        example_ids = np.arange(len(labels), dtype=np.int64)[:, None]
        x_train, train_labels, x_val, val_labels = (
            mnist_starter.stratified_train_val_split(
                example_ids, labels, train_size=600, val_size=200, seed=42
            )
        )
        self.assertFalse(set(x_train[:, 0]).intersection(set(x_val[:, 0])))
        np.testing.assert_array_equal(np.bincount(train_labels), np.full(10, 60))
        np.testing.assert_array_equal(np.bincount(val_labels), np.full(10, 20))

    def test_inverted_dropout_masks_activations_and_is_identity_in_eval(self):
        layer = Dropout(p=0.5, seed=0)
        activations = np.ones((32, 8))
        output = layer.forward(activations)
        self.assertTrue(set(np.unique(output)).issubset({0.0, 2.0}))
        np.testing.assert_array_equal(layer.backward(np.ones_like(output)), output)
        layer.eval()
        np.testing.assert_array_equal(layer.forward(activations), activations)


if __name__ == "__main__":
    unittest.main()
