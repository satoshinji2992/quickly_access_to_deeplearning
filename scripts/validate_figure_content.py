"""Numeric reference checks for the convolution and im2col illustrations.

This module does not draw or modify images.  It keeps the arithmetic shown in
the tutorial small, executable, and independent of the PNG files.
"""

from __future__ import annotations


CONV_INPUT = (
    (1, 0, 2, 1, 0),
    (0, 1, 1, 0, 2),
    (2, 1, 0, 1, 1),
    (0, 2, 1, 0, 1),
    (1, 0, 1, 2, 0),
)

KERNEL = (
    (1, 0, -1),
    (1, 0, -1),
    (1, 0, -1),
)

EXPECTED_CONV_OUTPUT = (
    (0, 0, 0),
    (0, 3, -2),
    (1, 0, 0),
)

IM2COL_INPUT = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
)

EXPECTED_IM2COL_ROWS = (
    (1, 2, 3, 5, 6, 7, 9, 10, 11),
    (2, 3, 4, 6, 7, 8, 10, 11, 12),
    (5, 6, 7, 9, 10, 11, 13, 14, 15),
    (6, 7, 8, 10, 11, 12, 14, 15, 16),
)

EXPECTED_IM2COL_OUTPUT = (-6, -6, -6, -6)


def valid_cross_correlation(matrix, kernel):
    """Return a valid 2-D cross-correlation without flipping ``kernel``."""

    kernel_height = len(kernel)
    kernel_width = len(kernel[0])
    output_height = len(matrix) - kernel_height + 1
    output_width = len(matrix[0]) - kernel_width + 1
    return tuple(
        tuple(
            sum(
                matrix[top + row][left + column] * kernel[row][column]
                for row in range(kernel_height)
                for column in range(kernel_width)
            )
            for left in range(output_width)
        )
        for top in range(output_height)
    )


def im2col_rows(matrix, kernel_size=3, stride=1):
    """Return row-major flattened windows: ``(H_out*W_out, K*K)``."""

    height = len(matrix)
    width = len(matrix[0])
    output_height = (height - kernel_size) // stride + 1
    output_width = (width - kernel_size) // stride + 1
    return tuple(
        tuple(
            matrix[top * stride + row][left * stride + column]
            for row in range(kernel_size)
            for column in range(kernel_size)
        )
        for top in range(output_height)
        for left in range(output_width)
    )


def flattened_kernel(kernel):
    return tuple(value for row in kernel for value in row)


def matrix_vector_product(rows, vector):
    return tuple(sum(value * weight for value, weight in zip(row, vector)) for row in rows)


def validate_all():
    assert valid_cross_correlation(CONV_INPUT, KERNEL) == EXPECTED_CONV_OUTPUT
    rows = im2col_rows(IM2COL_INPUT)
    assert rows == EXPECTED_IM2COL_ROWS
    assert len(rows) == 4 and all(len(row) == 9 for row in rows)
    assert matrix_vector_product(rows, flattened_kernel(KERNEL)) == EXPECTED_IM2COL_OUTPUT


if __name__ == "__main__":
    validate_all()
    print("figure arithmetic validated")
