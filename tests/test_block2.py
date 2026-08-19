import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from common.my_dl_lib import CrossEntropyLoss, Momentum, SGD
from exercises.block_02_resnet.task_10_image_data_pipeline.data_pipeline import (
    assert_disjoint_splits,
    load_cifar100_splits,
    stratified_train_val_split,
)
from exercises.block_02_resnet.task_11_conv2d_im2col.conv2d import (
    Conv2D,
    col2im,
    im2col,
)
from exercises.block_02_resnet.task_12_pooling_and_bn.layers import (
    BatchNorm2D,
    MaxPool2D,
)
from exercises.block_02_resnet.task_13_residual_block.residual_block import BasicBlock
from exercises.block_02_resnet.task_14_numpy_resnet_train.train_resnet import (
    SmallResNet,
    evaluate,
    one_hot,
    train_epoch,
)
from solutions.block_02_resnet.train_cifar100_solution import (
    load_checkpoint,
    read_checkpoint_config,
    restore_resume_config,
    save_checkpoint,
)


def _finite_difference(array, objective, index, epsilon=1e-5):
    original = array[index]
    array[index] = original + epsilon
    positive = objective()
    array[index] = original - epsilon
    negative = objective()
    array[index] = original
    return (positive - negative) / (2 * epsilon)


def _one_training_step(model, optimizer, x, labels):
    loss = CrossEntropyLoss()
    model.train()
    logits = model.forward(x)
    value = loss.forward(logits, one_hot(labels, model.num_classes))
    model.backward(loss.backward())
    optimizer.step()
    return value


class Block2Tests(unittest.TestCase):
    def test_im2col_rows_are_windows_and_col2im_accumulates_overlap(self):
        image = np.arange(1, 17, dtype=np.float64).reshape(1, 1, 4, 4)
        columns = im2col(image, kernel_size=3, stride=1, padding=0)
        expected = np.array(
            [
                [1, 2, 3, 5, 6, 7, 9, 10, 11],
                [2, 3, 4, 6, 7, 8, 10, 11, 12],
                [5, 6, 7, 9, 10, 11, 13, 14, 15],
                [6, 7, 8, 10, 11, 12, 14, 15, 16],
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(columns, expected)
        reconstructed = col2im(columns, image.shape, kernel_size=3)
        coverage = col2im(np.ones_like(columns), image.shape, kernel_size=3)
        np.testing.assert_array_equal(reconstructed, image * coverage)

    def test_conv2d_gradient_is_in_place_and_used_by_existing_optimizer(self):
        rng = np.random.default_rng(4)
        layer = Conv2D(1, 2, kernel_size=3, stride=1, padding=1)
        layer.W[...] = rng.normal(scale=0.2, size=layer.W.shape)
        x = rng.normal(size=(2, 1, 4, 4))
        upstream = rng.normal(size=(2, 2, 4, 4))
        optimizer = SGD(layer.parameters(), lr=0.05)
        gradient_identity = id(layer.dW)
        layer.forward(x)
        dx = layer.backward(upstream)
        self.assertEqual(id(layer.dW), gradient_identity)
        self.assertTrue(np.all(np.isfinite(dx)))

        analytic = layer.dW[0, 0, 1, 2]

        def objective():
            return float(np.sum(layer.forward(x) * upstream))

        numeric = _finite_difference(layer.W, objective, (0, 0, 1, 2))
        np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-6)
        before = layer.W.copy()
        # Finite differences replace the forward cache, so recompute first.
        layer.forward(x)
        layer.backward(upstream)
        optimizer.step()
        self.assertGreater(np.linalg.norm(layer.W - before), 0)

    def test_maxpool_routes_every_gradient_and_uses_first_maximum_on_ties(self):
        layer = MaxPool2D(kernel_size=2, stride=2)
        x = np.array(
            [[[[9.0, 9.0, 2.0, 3.0], [1.0, 0.0, 4.0, 1.0],
               [5.0, 2.0, 8.0, 1.0], [0.0, 6.0, 3.0, 7.0]]]]
        )
        output = layer.forward(x)
        np.testing.assert_array_equal(output, [[[[9.0, 4.0], [6.0, 8.0]]]])
        upstream = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
        dx = layer.backward(upstream)
        expected = np.zeros_like(x)
        expected[0, 0, 0, 0] = 1.0  # first of the tied 9s
        expected[0, 0, 1, 2] = 2.0
        expected[0, 0, 3, 1] = 3.0
        expected[0, 0, 2, 2] = 4.0
        np.testing.assert_array_equal(dx, expected)
        self.assertEqual(np.sum(dx), np.sum(upstream))

    def test_batchnorm_backward_and_running_buffer_references(self):
        rng = np.random.default_rng(7)
        layer = BatchNorm2D(2, momentum=0.7)
        x = rng.normal(size=(3, 2, 2, 2))
        upstream = rng.normal(size=x.shape)
        mean_identity = id(layer.running_mean)
        variance_identity = id(layer.running_var)
        layer.forward(x)
        dx = layer.backward(upstream)
        self.assertEqual(id(layer.running_mean), mean_identity)
        self.assertEqual(id(layer.running_var), variance_identity)

        def objective():
            return float(np.sum(layer.forward(x) * upstream))

        numeric = _finite_difference(x, objective, (1, 0, 1, 0))
        np.testing.assert_allclose(dx[1, 0, 1, 0], numeric, rtol=3e-4, atol=3e-5)
        layer.eval()
        running_before = (layer.running_mean.copy(), layer.running_var.copy())
        self.assertTrue(np.all(np.isfinite(layer.forward(x))))
        np.testing.assert_array_equal(layer.running_mean, running_before[0])
        np.testing.assert_array_equal(layer.running_var, running_before[1])

    def test_basic_block_projection_modes_and_named_state(self):
        rng = np.random.default_rng(11)
        block = BasicBlock(2, 4, stride=2)
        x = rng.normal(size=(2, 2, 6, 6))
        output = block.forward(x)
        self.assertEqual(output.shape, (2, 4, 3, 3))
        dx = block.backward(rng.normal(size=output.shape))
        self.assertEqual(dx.shape, x.shape)
        parameter_names = [name for name, _, _ in block.named_parameters("block")]
        buffer_names = [name for name, _ in block.named_buffers("block")]
        self.assertEqual(len(parameter_names), len(set(parameter_names)))
        self.assertIn("block.bn2.gamma", parameter_names)
        self.assertIn("block.proj_bn.running_var", buffer_names)
        block.eval()
        self.assertFalse(block.training)
        self.assertFalse(block.bn1.training)
        self.assertFalse(block.proj_bn.training)
        block.train()
        self.assertTrue(block.training)
        self.assertTrue(block.bn2.training)
        self.assertTrue(block.proj_bn.training)

    def test_stratified_split_is_independent_and_leakage_fails_loudly(self):
        images = np.arange(60 * 2, dtype=np.int64).reshape(60, 2)
        labels = np.repeat(np.arange(3), 20)
        train, validation, indices = stratified_train_val_split(
            images, labels, val_size=12, seed=3, return_indices=True
        )
        train_indices, validation_indices = indices
        self.assertEqual((len(train[0]), len(validation[0])), (48, 12))
        self.assertTrue(set(train_indices).isdisjoint(set(validation_indices)))
        np.testing.assert_array_equal(np.bincount(validation[1]), [4, 4, 4])
        assert_disjoint_splits(train, validation)
        with self.assertRaisesRegex(AssertionError, "data leakage"):
            assert_disjoint_splits(train, (train[0][:1], train[1][:1]))

    def test_cifar_loader_reserves_official_test_for_test_only(self):
        class FakeCIFAR100:
            def __init__(self, root, train, download):
                del root, download
                if train:
                    identifiers = np.arange(12, dtype=np.uint8)
                    self.targets = np.repeat(np.arange(3), 4).tolist()
                else:
                    identifiers = np.arange(100, 106, dtype=np.uint8)
                    self.targets = np.repeat(np.arange(3), 2).tolist()
                self.data = np.broadcast_to(
                    identifiers[:, None, None, None], (len(identifiers), 2, 2, 3)
                ).copy()

        torchvision = types.ModuleType("torchvision")
        datasets = types.ModuleType("torchvision.datasets")
        datasets.CIFAR100 = FakeCIFAR100
        torchvision.datasets = datasets
        modules = {"torchvision": torchvision, "torchvision.datasets": datasets}
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, modules
        ):
            train, validation, test = load_cifar100_splits(
                directory, val_size=3, seed=5, normalized=False
            )
        self.assertEqual(tuple(map(len, (train[1], validation[1], test[1]))), (9, 3, 6))
        np.testing.assert_array_equal(np.bincount(validation[1]), [1, 1, 1])
        train_ids = set(train[0][:, 0, 0, 0].tolist())
        validation_ids = set(validation[0][:, 0, 0, 0].tolist())
        test_ids = set(test[0][:, 0, 0, 0].tolist())
        self.assertTrue(train_ids.isdisjoint(validation_ids))
        self.assertEqual(test_ids, set(range(100, 106)))
        self.assertTrue(test_ids.isdisjoint(train_ids | validation_ids))

    def test_small_resnet_can_overfit_one_tiny_batch(self):
        rng = np.random.default_rng(5)
        images = rng.normal(size=(4, 3, 5, 5))
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        images[:2, 0] += 1.0
        images[2:, 1] += 1.0
        np.random.seed(3)
        model = SmallResNet(2, channels=(2,), blocks_per_stage=(1,))
        optimizer = Momentum(model.parameters(), lr=0.08, beta=0.8)
        losses = [
            _one_training_step(model, optimizer, images, labels)
            for _ in range(60)
        ]
        self.assertTrue(np.all(np.isfinite(losses)))
        self.assertLess(losses[-1], losses[0] * 0.25)

    def test_epoch_metrics_weight_the_short_final_batch_by_sample_count(self):
        class IdentityClassifier:
            num_classes = 2

            def train(self):
                return self

            def eval(self):
                return self

            def forward(self, values):
                return values

            def backward(self, gradient):
                return gradient

        class NoOpOptimizer:
            def step(self):
                pass

        # The last sample forms a short batch and deliberately has a very high
        # loss.  Averaging the two batch means would report about 10; the true
        # per-example mean is about 6.67.
        logits = np.array([[10.0, -10.0], [10.0, -10.0], [-10.0, 10.0]])
        labels = np.array([0, 0, 0], dtype=np.int64)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        expected_loss = float(
            -np.log(probabilities[np.arange(3), labels] + 1e-12).mean()
        )

        eval_loss, eval_accuracy = evaluate(
            IdentityClassifier(), CrossEntropyLoss(), logits, labels, batch_size=2
        )
        train_loss, train_accuracy = train_epoch(
            IdentityClassifier(),
            CrossEntropyLoss(),
            NoOpOptimizer(),
            logits,
            labels,
            batch_size=2,
            seed=4,
        )
        np.testing.assert_allclose(eval_loss, expected_loss, rtol=0, atol=1e-12)
        np.testing.assert_allclose(train_loss, expected_loss, rtol=0, atol=1e-12)
        self.assertEqual(eval_accuracy, 2 / 3)
        self.assertEqual(train_accuracy, 2 / 3)

    def test_checkpoint_round_trip_restores_everything_and_exact_logits(self):
        rng = np.random.default_rng(21)
        x = rng.normal(size=(4, 3, 5, 5))
        labels = np.array([0, 1, 2, 1], dtype=np.int64)
        np.random.seed(1)
        model = SmallResNet(3, channels=(3,), blocks_per_stage=(1,))
        optimizer = Momentum(model.parameters(), lr=0.07, beta=0.8)
        _one_training_step(model, optimizer, x, labels)
        model.eval()
        expected_logits = model.forward(x).copy()
        expected_buffers = {
            name: value.copy() for name, value in model.named_buffers()
        }
        expected_velocity = [value.copy() for value in optimizer.velocity]
        history = [{"epoch": 1, "val_loss": 1.25}]

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config = {"channels": [3], "seed": 1, "data_dir": directory}
            checkpoint = directory / "round_trip.npz"
            save_checkpoint(checkpoint, model, optimizer, 1, history, config)
            np.random.seed(99)
            restored = SmallResNet(3, channels=(3,), blocks_per_stage=(1,))
            restored_optimizer = Momentum(restored.parameters(), lr=999.0, beta=0.1)
            epoch, restored_history, restored_config = load_checkpoint(
                checkpoint, restored, restored_optimizer, return_config=True
            )

            self.assertEqual(epoch, 1)
            self.assertEqual(restored_history, history)
            self.assertEqual(restored_config["channels"], [3])
            self.assertEqual(restored_config["data_dir"], str(directory))
            self.assertEqual(restored_optimizer.lr, 0.07)
            self.assertEqual(restored_optimizer.beta, 0.8)
            for actual, expected in zip(restored_optimizer.velocity, expected_velocity):
                np.testing.assert_array_equal(actual, expected)
            for name, value in restored.named_buffers():
                np.testing.assert_array_equal(value, expected_buffers[name])
            self.assertFalse(restored.training)
            np.testing.assert_array_equal(restored.forward(x), expected_logits)

            # Continuation must also be faithful, not just immediate eval.
            _one_training_step(model, optimizer, x, labels)
            _one_training_step(restored, restored_optimizer, x, labels)
            for (expected, _), (actual, _) in zip(
                model.parameters(), restored.parameters()
            ):
                np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_resume_restores_trajectory_defining_run_config(self):
        from argparse import Namespace

        np.random.seed(8)
        model = SmallResNet(3, channels=(3,), blocks_per_stage=(1,))
        optimizer = Momentum(model.parameters(), lr=0.07, beta=0.8)
        saved = {
            "batch_size": 7,
            "lr": 0.07,
            "optimizer": "momentum",
            "weight_decay": 0.0,
            "train_limit": 21,
            "subset_size": 21,
            "val_size": 12,
            "val_limit": 6,
            "test_limit": 9,
            "no_augment": True,
            "seed": 17,
            "channels": [3],
            "blocks": [1],
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "resume.npz"
            save_checkpoint(checkpoint, model, optimizer, 2, [], saved)
            loaded = read_checkpoint_config(checkpoint)

        args = Namespace(
            batch_size=99,
            lr=99.0,
            optimizer="adamw",
            weight_decay=1.0,
            train_limit=None,
            subset_size=None,
            val_size=5000,
            val_limit=None,
            test_limit=None,
            no_augment=False,
            seed=0,
            channels=(16, 32, 64),
            blocks=(2, 2, 2),
        )
        restore_resume_config(args, loaded)
        self.assertEqual(args.batch_size, 7)
        self.assertEqual(args.train_limit, 21)
        self.assertIsNone(args.subset_size)
        self.assertTrue(args.no_augment)
        self.assertEqual(args.seed, 17)
        self.assertEqual(args.channels, (3,))
        self.assertEqual(args.blocks, (1,))


if __name__ == "__main__":
    unittest.main()
