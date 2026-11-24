from onnx_passes.passes.base import RewriteRule
from onnx_passes.passes.verify import Verify, tolerance

import onnx_ir as ir
import numpy as np


def _static_shape(x: ir.Value) -> tuple[int, ...] | None:
    if x.shape is not None and x.shape.is_static():
        return tuple(x.shape)
    return None


def _broadcast_to_operand(s: ir.Value, x: ir.Value) -> np.ndarray | None:
    """Constant s broadcast to the static shape of x if s does not expand x."""

    if (value := ir.convenience.get_const_tensor(s)) is None:
        return None

    if (shape := _static_shape(x)) is None:
        return None

    try:
        if np.broadcast_shapes(value.numpy().shape, shape) != shape:
            return None
    except ValueError:
        return None

    return np.broadcast_to(value.numpy(), shape)


def _is_scalar(s: np.ndarray) -> bool:
    return np.all(s == np.reshape(s, (-1,))[0])


def _per_output_channel(s: np.ndarray) -> bool:
    """True if s only varies along the leading (output channel) axis."""
    return bool(np.all(s == s[(slice(None), *(slice(0, 1),) * (s.ndim - 1))]))


@tolerance
class MoveChannelMulPastMatMulRhs_v1(RewriteRule, Verify):
    """Reorder per-column multiplication of the right hand side past MatMul.

    MatMul(x, s * y) = MatMul(x, y) * s if s is constant along the reduction
    axis, i.e., the second-to-last axis of y, such as per-output-channel scales
    of quantized weights. Effectively scalar s is covered by
    MoveMulPastMatMulRhs_v1.
    """

    @property
    def commute(self) -> bool:
        return True

    @staticmethod
    def pattern(op, x, y, s):
        return op.MatMul(x, op.Mul(s, y))

    @staticmethod
    def check(context, x, y, s):
        if (value := _broadcast_to_operand(s, y)) is None:
            return False

        # Vector right hand sides have no column axis
        if value.ndim < 2 or _is_scalar(value):
            return False

        return bool(np.all(value == value[..., :1, :]))

    @staticmethod
    def rewrite(op, x, y, s):
        scale = _broadcast_to_operand(s, y)[..., :1, :]  # noqa: Not None
        return op.Mul(op.MatMul(x, y), op.Constant(value=ir.tensor(scale)))


@tolerance
class MoveChannelMulPastConvWeights_v1(RewriteRule, Verify):
    """Reorder per-output-channel multiplication of Conv weights past Conv.

    Conv(x, s * w) = Conv(x, w) * s if s is constant along all but the output
    channel axis of the weights, such as per-output-channel scales of quantized
    weights. This holds for grouped convolutions as well.
    """

    @property
    def commute(self) -> bool:
        return True

    @staticmethod
    def pattern(op, x, w, s):
        return op.Conv(x, op.Mul(s, w), _outputs=["out"])

    @staticmethod
    def check(context, x, w, s, out):
        if (value := _broadcast_to_operand(s, w)) is None:
            return False

        return value.ndim >= 3 and _per_output_channel(value)

    @staticmethod
    def rewrite(op, x, w, s, out):
        value = _broadcast_to_operand(s, w)
        # Per-output-channel scale broadcasting over (N, C, *spatial) output
        scale = np.reshape(
            value[(slice(None), *(0,) * (value.ndim - 1))],
            (-1, *(1,) * (value.ndim - 2))
        )
        return op.Mul(
            op.Conv(x, w, **out.producer().attributes),
            op.Constant(value=ir.tensor(scale))
        )


@tolerance
class MoveChannelMulPastConvWeightsBias_v1(RewriteRule, Verify):
    """Reorder per-output-channel multiplication of Conv weights past Conv.

    Variant with bias, which is split off as an explicit Add following the
    multiplication: Conv(x, s * w, b) = Conv(x, w) * s + b
    """

    @property
    def commute(self) -> bool:
        return True

    @staticmethod
    def pattern(op, x, w, s, b):
        return op.Conv(x, op.Mul(s, w), b, _outputs=["out"])

    @staticmethod
    def check(context, x, w, s, b, out):
        if b is None or _static_shape(b) is None:
            return False
        return MoveChannelMulPastConvWeights_v1.check(context, x, w, s, out)

    @staticmethod
    def rewrite(op, x, w, s, b, out):
        rank = len(_static_shape(w))
        return op.Add(
            MoveChannelMulPastConvWeights_v1.rewrite(op, x, w, s, out),
            op.Reshape(
                b,
                op.Constant(value_ints=[-1, *(1,) * (rank - 2)])
            )
        )


@tolerance
class MoveScalarMulPastConvInput_v1(RewriteRule, Verify):
    """Reorder effectively scalar multiplication of the Conv input past Conv.

    Conv(s * x, w) = Conv(x, w) * s for scalar s: Zero padding is invariant to
    scaling. A bias is split off as an explicit Add following the
    multiplication: Conv(s * x, w, b) = Conv(x, w) * s + b
    """

    @property
    def commute(self) -> bool:
        return True

    @staticmethod
    def pattern(op, x, w, s):
        return op.Conv(
            op.Mul(s, x), w, _allow_other_inputs=True, _outputs=["out"]
        )

    @staticmethod
    def check(context, x, w, s, out):
        if (value := _broadcast_to_operand(s, x)) is None:
            return False

        if len(out.producer().inputs) > 2 and out.producer().inputs[2]:
            if _static_shape(out.producer().inputs[2]) is None:
                return False

        return _is_scalar(value)

    @staticmethod
    def rewrite(op, x, w, s, out):
        value = _broadcast_to_operand(s, x)
        scale = np.reshape(value, (-1,))[0]

        y = op.Mul(
            op.Conv(x, w, **out.producer().attributes),
            op.Constant(value=ir.tensor(scale))
        )

        if len(out.producer().inputs) > 2 and (b := out.producer().inputs[2]):
            rank = len(_static_shape(x))
            y = op.Add(
                y, op.Reshape(
                    b, op.Constant(value_ints=[-1, *(1,) * (rank - 2)])
                )
            )

        return y


@tolerance
class MoveMulPastPad_v1(RewriteRule, Verify):
    """Reorder multiplication to follow constant padding.

    Pad(s * x, c) = Pad(x, c / s) * s for scalar nonzero s. For zero padding,
    i.e., c = 0, the padding value is invariant to any s which is constant along
    the padded axes, e.g., per-channel scales.
    """

    @property
    def commute(self) -> bool:
        return True

    @staticmethod
    def pattern_v18(op, x, s, pads, const_value, axes, mode):
        return op.Pad(op.Mul(s, x), pads, const_value, axes, mode=mode)

    @staticmethod
    def check_v18(context, x, s, pads, const_value, axes, mode):
        if mode.as_string() != "constant":
            return False

        if (value := _broadcast_to_operand(s, x)) is None:
            return False

        if (pads := ir.convenience.get_const_tensor(pads)) is None:
            return False

        if const_value is not None:
            if (const_value := ir.convenience.get_const_tensor(const_value)) is None:
                return False
            const_value = const_value.numpy()
        else:
            const_value = np.zeros(())

        if axes is not None:
            if (axes := ir.convenience.get_const_tensor(axes)) is None:
                return False
            axes = [int(a) % value.ndim for a in axes.numpy().reshape(-1)]
        else:
            axes = list(range(value.ndim))

        # Scalar scales: the padding value can be rescaled unless zero scale
        if _is_scalar(value):
            return np.all(const_value == 0) or np.all(value != 0)

        # Non-scalar scales: Only for zero padding and scales constant along
        # all actually padded axes
        if not np.all(const_value == 0):
            return False

        pads = pads.numpy().reshape(2, -1)
        padded = [a for a, b, e in zip(axes, *pads) if b != 0 or e != 0]

        index = tuple(
            slice(0, 1) if a in padded else slice(None)
            for a in range(value.ndim)
        )

        return bool(np.all(value == value[index]))

    @staticmethod
    def rewrite_v18(op, x, s, pads, const_value, axes, mode):
        value = _broadcast_to_operand(s, x)

        if _is_scalar(value):
            scale = np.reshape(value, (-1,))[0]
            if const_value is not None:
                const_value = op.Div(
                    const_value,
                    op.CastLike(op.Constant(value=ir.tensor(scale)), const_value)
                )
            return op.Mul(
                op.Pad(x, pads, const_value, axes, mode=mode),
                op.Constant(value=ir.tensor(scale))
            )

        # Constant along all padded axes: Reduce these to singleton axes to
        # broadcast over the padded output
        pads_ = ir.convenience.get_const_tensor(pads).numpy().reshape(2, -1)

        if axes is not None:
            axes_ = ir.convenience.get_const_tensor(axes).numpy().reshape(-1)
            axes_ = [int(a) % value.ndim for a in axes_]
        else:
            axes_ = list(range(value.ndim))

        padded = [a for a, b, e in zip(axes_, *pads_) if b != 0 or e != 0]

        scale = value[tuple(
            slice(0, 1) if a in padded else slice(None)
            for a in range(value.ndim)
        )]

        return op.Mul(
            op.Pad(x, pads, const_value, axes, mode=mode),
            op.Constant(value=ir.tensor(np.ascontiguousarray(scale)))
        )
