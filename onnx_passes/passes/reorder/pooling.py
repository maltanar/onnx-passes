"""Reorder elementwise operations past MaxPool.

MaxPool commutes with every elementwise function f which is non-decreasing and
identical on all elements of each pooling window: max_i f(x_i) = f(max_i x_i).
This holds exactly in floating-point arithmetic, as rounding preserves the
order of values. Adding a constant and multiplying by a positive constant are
such functions if the constant does not vary along the pooled (spatial) axes,
e.g., scalars or per-channel constants. Padded elements of MaxPool do not take
part in the maximum, so padding, strides, dilations and ceil_mode are
irrelevant. Negative scales turn the maximum into a minimum and do not commute.
"""

from onnx_passes.passes.base import RewriteRule
from onnx_passes.passes.verify import Verify

import onnx_ir as ir
import numpy as np


def _maxpool_operand(c: ir.Value, x: ir.Value, out: ir.Value):
    """Constant c reduced to singleton spatial axes, if it commutes with the
    MaxPool producing out, None otherwise."""

    node = out.producer()

    # The optional Indices output might change due to ties introduced by
    # rounding, only reorder if nobody uses the indices
    if len(node.outputs) > 1:
        indices = node.outputs[1]
        if indices.uses() or indices.is_graph_output():
            return None

    # Integer arithmetic could overflow, which does not preserve the order
    if x.dtype is None or not x.dtype.is_floating_point():
        return None

    if (value := ir.convenience.get_const_tensor(c)) is None:
        return None

    if x.shape is None or len(x.shape) < 3:
        return None

    value = value.numpy()

    # The constant must not broadcast the pooled input to a larger shape
    rank = len(x.shape)

    if value.ndim > rank:
        return None

    value = np.reshape(value, (*(1,) * (rank - value.ndim), *value.shape))

    for n, m in zip(value.shape, x.shape):
        if n != 1 and n != m:
            return None

    if not np.all(np.isfinite(value)):
        return None

    # Constant along the pooled axes: The same function for each window
    reduced = value[(slice(None), slice(None), *(slice(0, 1),) * (rank - 2))]

    if not np.all(value == reduced):
        return None

    return reduced


class MoveAddPastMaxPool_v1(RewriteRule, Verify):
    """Reorder addition of a scalar or per-channel constant past MaxPool.

    MaxPool(x + c) = MaxPool(x) + c if c does not vary along the pooled axes.
    """

    @property
    def commute(self) -> bool:
        return True

    @staticmethod
    def pattern(op, x, c):
        return op.MaxPool(op.Add(x, c), _outputs=["out"])

    @staticmethod
    def check(context, x, c, out):
        return _maxpool_operand(c, x, out) is not None

    @staticmethod
    def rewrite(op, x, c, out):
        value = _maxpool_operand(c, x, out)
        return op.Add(
            op.MaxPool(x, **out.producer().attributes),
            op.Constant(value=ir.tensor(np.ascontiguousarray(value)))
        )


class MoveMulPastMaxPool_v1(RewriteRule, Verify):
    """Reorder multiplication by a positive scalar or per-channel constant past
    MaxPool.

    MaxPool(s * x) = MaxPool(x) * s if s > 0 does not vary along the pooled
    axes. Negative scales turn the maximum into a minimum.
    """

    @property
    def commute(self) -> bool:
        return True

    @staticmethod
    def pattern(op, x, s):
        return op.MaxPool(op.Mul(x, s), _outputs=["out"])

    @staticmethod
    def check(context, x, s, out):
        if (value := _maxpool_operand(s, x, out)) is None:
            return False
        return bool(np.all(value > 0))

    @staticmethod
    def rewrite(op, x, s, out):
        value = _maxpool_operand(s, x, out)
        return op.Mul(
            op.MaxPool(x, **out.producer().attributes),
            op.Constant(value=ir.tensor(np.ascontiguousarray(value)))
        )
