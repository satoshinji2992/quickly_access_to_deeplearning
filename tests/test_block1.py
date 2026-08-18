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

from common.my_dl_lib import Dropout, LayerNorm


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
