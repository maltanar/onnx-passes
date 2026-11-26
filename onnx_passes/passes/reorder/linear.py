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


@tolerance
class MoveChannelMulPastDepthwiseConvInput_v1(RewriteRule, Verify):
    """Reorder per-channel multiplication of the input past depthwise Conv.

    For depthwise convolutions, i.e., group equals the number of input
    channels, every output channel only depends on a single input channel:
    Conv(s * x, w) = Conv(x, w) * s' where s' repeats the per-channel scale s
    for each of the m = C_out / C_in output channels of an input channel. Zero
    padding is invariant to scaling. A bias is split off as an explicit Add.
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
        node = out.producer()

        if (value := _broadcast_to_operand(s, x)) is None or value.ndim < 3:
            return False

        # Effectively scalar s is covered by MoveScalarMulPastConvInput_v1
        if _is_scalar(value):
            return False

        if (group := node.attributes.get("group")) is None:
            return False

        if (shape := _static_shape(w)) is None:
            return False

        channels = value.shape[1]
        if group.as_int() != channels or shape[0] % channels != 0:
            return False

        if len(node.inputs) > 2 and node.inputs[2]:
            if _static_shape(node.inputs[2]) is None:
                return False

        # Scale must only vary along the channel axis
        index = (slice(0, 1), slice(None), *(slice(0, 1),) * (value.ndim - 2))
        return bool(np.all(value == value[index]))

    @staticmethod
    def rewrite(op, x, w, s, out):
        node = out.producer()
        value = _broadcast_to_operand(s, x)
        rank = value.ndim

        # Per-channel scale repeated for each output channel of the group
        scale = value[(0, slice(None), *(0,) * (rank - 2))]
        scale = np.repeat(scale, _static_shape(w)[0] // scale.shape[0])
        scale = np.reshape(scale, (-1, *(1,) * (rank - 2)))

        y = op.Mul(
            op.Conv(x, w, **node.attributes),
            op.Constant(value=ir.tensor(np.ascontiguousarray(scale)))
        )

        if len(node.inputs) > 2 and (b := node.inputs[2]):
            y = op.Add(
                y, op.Reshape(
                    b, op.Constant(value_ints=[-1, *(1,) * (rank - 2)])
                )
            )

        return y


def _vector_along(value: np.ndarray, axis: int) -> np.ndarray | None:
    """The vector along axis if value only varies along this axis, else None."""
    # Move the axis last and check all other positions repeat the same vector
    vectors = np.reshape(np.moveaxis(value, axis, -1), (-1, value.shape[axis]))

    if not np.all(vectors == vectors[0]):
        return None

    return vectors[0]


def _uniform_magnitude_signs(
        value: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Splits s varying only along axis into signs and a common magnitude.

    Returns (signs along the axis, scalar magnitude) if s only varies along the
    axis, all magnitudes are identical and nonzero and s is not effectively
    scalar (covered by the scalar rules), None otherwise.
    """
    if _is_scalar(value) or (vector := _vector_along(value, axis)) is None:
        return None

    magnitude = np.abs(vector)

    if magnitude[0] == 0 or not np.all(magnitude == magnitude[0]):
        return None

    return np.sign(vector), magnitude[0]


def _constant_like(value: np.ndarray, like: ir.Value, op) -> ir.Value:
    """Constant from value, propagating a constant folding stop of like."""
    const = op.Constant(value=ir.tensor(np.ascontiguousarray(value)))

    if like.meta.get("constant_stop"):
        const.meta["constant_stop"] = True

    return const


@tolerance
class MoveSignedChannelMulPastMatMulLhs_v1(RewriteRule, Verify):
    """Reorder per-channel sign-scale multiplication of the input past MatMul.

    MatMul(s * x, w) = MatMul(x, sign(s) * w) * |s| for constant weights w if s
    only varies along the reduction axis, i.e., the last axis of x, and all
    channels share the same magnitude |s|. The signs are folded into the rows of
    the weights, which keeps integer weights integer, e.g., the per-channel
    +-(2x - 1) following bipolar MultiThresholds with sign-flipped channels.
    """

    @property
    def commute(self) -> bool:
        return True

    @staticmethod
    def pattern(op, x, s, w):
        return op.MatMul(op.Mul(s, x), w)

    @staticmethod
    def check(context, x, s, w):
        if (weights := ir.convenience.get_const_tensor(w)) is None:
            return False

        if weights.numpy().ndim < 2:
            return False

        if (value := _broadcast_to_operand(s, x)) is None:
            return False

        return _uniform_magnitude_signs(value, -1) is not None

    @staticmethod
    def rewrite(op, x, s, w):
        value = _broadcast_to_operand(s, x)
        signs, magnitude = _uniform_magnitude_signs(value, -1)  # noqa: checked
        weights = ir.convenience.get_const_tensor(w).numpy()

        # Signs per row of the weights, i.e., per reduction channel
        weights = weights * np.reshape(signs, (-1, 1)).astype(weights.dtype)

        return op.Mul(
            op.MatMul(x, _constant_like(weights, w, op)),
            op.Constant(value=ir.tensor(np.asarray(magnitude, value.dtype)))
        )


@tolerance
class MoveSignedChannelMulPastConvInput_v1(RewriteRule, Verify):
    """Reorder per-channel sign-scale multiplication of the input past Conv.

    Conv(s * x, w) = Conv(x, sign(s) * w) * |s| for constant weights w if s only
    varies along the channel axis of x and all channels share the same magnitude
    |s|. The signs are folded into the input channels of the weights (respecting
    groups), which keeps integer weights integer. Zero padding is invariant to
    scaling. A bias is split off as an explicit Add.
    """

    @property
    def commute(self) -> bool:
        return True

    @staticmethod
    def pattern(op, x, s, w):
        return op.Conv(
            op.Mul(s, x), w, _allow_other_inputs=True, _outputs=["out"]
        )

    @staticmethod
    def check(context, x, s, w, out):
        node = out.producer()

        if (weights := ir.convenience.get_const_tensor(w)) is None:
            return False

        if (value := _broadcast_to_operand(s, x)) is None or value.ndim < 3:
            return False

        group = node.attributes.get("group")
        group = group.as_int() if group is not None else 1

        shape = weights.numpy().shape
        channels = value.shape[1]

        # Input channels must split evenly into groups matching the weights
        if channels != group * shape[1] or shape[0] % group != 0:
            return False

        if len(node.inputs) > 2 and node.inputs[2]:
            if _static_shape(node.inputs[2]) is None:
                return False

        return _uniform_magnitude_signs(value, 1) is not None

    @staticmethod
    def rewrite(op, x, s, w, out):
        node = out.producer()
        value = _broadcast_to_operand(s, x)
        signs, magnitude = _uniform_magnitude_signs(value, 1)  # noqa: checked
        weights = ir.convenience.get_const_tensor(w).numpy()
        rank = weights.ndim

        group = node.attributes.get("group")
        group = group.as_int() if group is not None else 1

        # Signs of the input channels of each group, repeated for each output
        # channel of the group: (C_out, C_in / group, 1, ..., 1)
        signs = np.reshape(signs, (group, -1))
        signs = np.repeat(signs, weights.shape[0] // group, axis=0)
        signs = np.reshape(signs, (*signs.shape, *(1,) * (rank - 2)))
        weights = weights * signs.astype(weights.dtype)

        y = op.Mul(
            op.Conv(x, _constant_like(weights, w, op), **node.attributes),
            op.Constant(value=ir.tensor(np.asarray(magnitude, value.dtype)))
        )

        if len(node.inputs) > 2 and (b := node.inputs[2]):
            y = op.Add(
                y, op.Reshape(
                    b, op.Constant(value_ints=[-1, *(1,) * (rank - 2)])
                )
            )

        return y


def _conv_group(node: ir.Node) -> int:
    group = node.attributes.get("group")
    return group.as_int() if group is not None else 1


def _conv_is_unpadded(node: ir.Node) -> bool:
    """True if the Conv node does not pad its input (explicitly or auto)."""
    if (auto_pad := node.attributes.get("auto_pad")) is not None:
        if auto_pad.as_string() not in {"NOTSET", "VALID"}:
            return False

    if (pads := node.attributes.get("pads")) is not None:
        return not any(pads.as_ints())

    return True


@tolerance
class MoveAddPastConvInput_v1(RewriteRule, Verify):
    """Reorder per-channel addition of the input past unpadded Conv.

    Conv(x + c, w) = Conv(x, w) + Conv(c, w) for constant weights w, where c is
    constant along all but the channel axis. Without padding, every output
    element sees a full kernel window of c, so Conv(c, w) reduces to the
    per-output-channel bias sum_{j, k} w[m, j, k] * c[g * C_in / group + j] for
    output channel m of group g. An existing bias stays with the Conv.
    """

    @property
    def commute(self) -> bool:
        return True

    @staticmethod
    def pattern(op, x, c, w):
        return op.Conv(
            op.Add(x, c), w, _allow_other_inputs=True, _outputs=["out"]
        )

    @staticmethod
    def check(context, x, c, w, out):
        node = out.producer()

        if not _conv_is_unpadded(node):
            return False

        if (weights := ir.convenience.get_const_tensor(w)) is None:
            return False

        if (value := _broadcast_to_operand(c, x)) is None or value.ndim < 3:
            return False

        shape = weights.numpy().shape
        group = _conv_group(node)

        if value.shape[1] != group * shape[1] or shape[0] % group != 0:
            return False

        return _vector_along(value, 1) is not None

    @staticmethod
    def rewrite(op, x, c, w, out):
        node = out.producer()
        weights = ir.convenience.get_const_tensor(w).numpy()
        vector = _vector_along(_broadcast_to_operand(c, x), 1)
        group = _conv_group(node)

        # Input channel offsets of each group repeated for each output channel
        # of the group: (C_out, C_in / group)
        vector = np.reshape(vector, (group, -1))
        vector = np.repeat(vector, weights.shape[0] // group, axis=0)

        # Sum over input channels and the kernel window per output channel
        window = np.reshape(weights, (*weights.shape[:2], -1)).sum(axis=-1)
        bias = np.sum(
            window.astype(np.float64) * vector.astype(np.float64), axis=1
        )
        bias = np.reshape(bias, (-1, *(1,) * (weights.ndim - 2)))

        return op.Add(
            op.Conv(x, *node.inputs[1:], **node.attributes),
            op.Constant(value=ir.tensor(bias.astype(weights.dtype)))
        )
