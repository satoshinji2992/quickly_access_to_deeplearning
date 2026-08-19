"""Regression checks for the numeric content reviewed in generated figures."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_figure_content import (  # noqa: E402
    EXPECTED_CONV_OUTPUT,
    EXPECTED_IM2COL_OUTPUT,
    EXPECTED_IM2COL_ROWS,
    IM2COL_INPUT,
    KERNEL,
    flattened_kernel,
    im2col_rows,
    matrix_vector_product,
    valid_cross_correlation,
    CONV_INPUT,
)


TARGETS = (
    ROOT / "assets/images/im2col_explained.png",
    ROOT / "exercises/block_02_resnet/task_11_conv2d_im2col/assets/im2col_explained.png",
)


class Im2ColFigureTests(unittest.TestCase):
    def test_figure_uses_the_same_row_oriented_im2col_as_the_code(self):
        rows = im2col_rows(IM2COL_INPUT, kernel_size=3, stride=1)
        self.assertEqual(rows, EXPECTED_IM2COL_ROWS)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(len(row) == 9 for row in rows))

    def test_figure_convolution_values_are_correct(self):
        self.assertEqual(valid_cross_correlation(CONV_INPUT, KERNEL), EXPECTED_CONV_OUTPUT)
        self.assertEqual(
            matrix_vector_product(EXPECTED_IM2COL_ROWS, flattened_kernel(KERNEL)),
            EXPECTED_IM2COL_OUTPUT,
        )

    def test_generated_canonical_and_task_copy_exist_and_match(self):
        first, second = TARGETS
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with Image.open(first) as image:
            width, height = image.size
        self.assertGreaterEqual(width, 1200)
        self.assertGreaterEqual(height, 700)


if __name__ == "__main__":
    unittest.main()
