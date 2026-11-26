from onnx_passes.passes.base import RewriteRule, Sequential, Transformation
from onnx_passes.passes.verify import Verify, tolerance

from onnxscript import GraphBuilder

from onnx_passes.ops import DOMAIN as PASSES_DOMAIN

import onnx_ir as ir
import numpy as np


@tolerance
class FuseScaleIntoMultiThreshold_v1(RewriteRule, Verify):
    """Fuse scale following MultiThreshold into the step weights."""

    @staticmethod
    def pattern(op, x, thresholds, weights, scale):
        return op.Mul(
            op.MultiThreshold(x, thresholds, weights, _domain=PASSES_DOMAIN),
            scale
        )

    @staticmethod
    def rewrite_v13(op, x, thresholds, weights, scale):
        return op.MultiThreshold(
            x,
            thresholds,
            op.Mul(
                weights,
                # Scale expands along the thresholding axis to match the weights
                op.Unsqueeze(
                    scale, op.Constant(value_ints=[-1])
                )
            ),
            _domain=PASSES_DOMAIN
        )

    @property
    def commute(self) -> bool:
        return True


@tolerance
class FuseAddedMultiThresholds_v1(RewriteRule, Verify):
    """Fuse two MultiThreshold operators joined via addition."""

    @staticmethod
    def pattern(op, x, thresholds1, weights1, thresholds2, weights2):
        return op.Add(
            op.MultiThreshold(x, thresholds1, weights1, _domain=PASSES_DOMAIN),
            op.MultiThreshold(x, thresholds2, weights2, _domain=PASSES_DOMAIN),
        )

    @staticmethod
    def rewrite_v15(op, x, thresholds1, weights1, thresholds2, weights2):
        # Extend thresholds to common rank by unsqueezing dimensions of size 1
        # from the left of both parameter tensors. Weights are extended when
        # thresholds and weights are broadcast against each other.
        thresholds1 = op.Unsqueeze(
            thresholds1,
            op.Range(
                op.Size(
                    op.Shape(thresholds1)
                ),
                common_rank := op.Max(
                    op.Size(
                        op.Shape(thresholds1)
                    ),
                    op.Size(
                        op.Shape(thresholds2)
                    ),
                    op.Constant(value_int=1)
                ),
                op.Constant(value_int=1),
            )
        )

        thresholds2 = op.Unsqueeze(
            thresholds2,
            op.Range(
                op.Size(
                    op.Shape(thresholds2)
                ),
                common_rank,
                op.Constant(value_int=1),
            )
        )

        # Expand thresholds and weights on either side to common shape such that
        # there is a 1:1 correspondence, required to stack matching amounts when
        # fusing the two operators.
        thresholds1 = op.Expand(
            thresholds1,
            common_shape1 := op.Shape(
                op.Max(
                    thresholds1,
                    op.CastLike(weights1, thresholds1)
                )
            )
        )

        weights1 = op.Expand(weights1, common_shape1)

        thresholds2 = op.Expand(
            thresholds2,
            common_shape2 := op.Shape(
                op.Max(
                    thresholds2,
                    op.CastLike(weights2, thresholds2)
                )
            )
        )

        weights2 = op.Expand(weights2, common_shape2)

        # Fuse the two operators by concatenating the thresholds and weights
        # along the thresholding axis after expanding all leading dimensions.
        thresholds = op.Concat(
            op.Expand(
                thresholds1,
                op.Concat(
                    # Common leading dimensions of thresholds and weights: after
                    # already expanding each side this is the same for all
                    common_shape := op.Max(
                        op.Shape(thresholds1, end=-1),
                        op.Shape(thresholds2, end=-1),
                    ),
                    op.Shape(thresholds1, start=-1),
                    axis=-1
                )
            ),
            op.Expand(
                thresholds2,
                op.Concat(
                    common_shape,
                    op.Shape(thresholds2, start=-1),
                    axis=-1
                )
            ),
            axis=-1
        )

        weights = op.Concat(
            op.Expand(
                weights1,
                op.Concat(
                    common_shape,
                    op.Shape(weights1, start=-1),
                    axis=-1
                )
            ),
            op.Expand(
                weights2,
                op.Concat(
                    common_shape,
                    op.Shape(weights2, start=-1),
                    axis=-1
                )
            ),
            axis=-1
        )

        return op.MultiThreshold(x, thresholds, weights, _domain=PASSES_DOMAIN)


@tolerance
class FuseConsecutiveMultiThresholds_v1(RewriteRule, Verify):
    """Fuses two consecutive MultiThreshold operations into a single operation.
    """

    @staticmethod
    def pattern_v18(op, x, thresholds1, weights1, thresholds2, weights2):
        return op.MultiThreshold(
            op.MultiThreshold(
                x, thresholds1, weights1, _domain=PASSES_DOMAIN
            ),
            thresholds2, weights2, _domain=PASSES_DOMAIN
        )

    @staticmethod
    def check_v18(context, x, thresholds1, weights1, thresholds2, weights2):
        # All thresholds must be sorted to make use of the cumulative sum of the
        # step weights, which only gives a correct sketch of the function if the
        # corresponding thresholds are in increasing order.
        if (t1 := ir.convenience.get_const_tensor(thresholds1)) is None:
            return False

        if np.any(np.sort(t1, axis=-1) != t1):
            return False

        if (t2 := ir.convenience.get_const_tensor(thresholds2)) is None:
            return False

        if np.any(np.sort(t2, axis=-1) != t2):
            return False

        # With non-constant step weights, this transformation gets messy rather
        # quickly - for now restrict this to the constant case...
        if ir.convenience.get_const_tensor(weights1) is not None:
            return ir.convenience.get_const_tensor(weights2) is not None

        return False

    @staticmethod
    def rewrite_v18(op, x, thresholds1, weights1, thresholds2, weights2):
        # The cumulative sum of the padded weights gives the output levels of
        # the first MultiThreshold function.
        levels = op.CumSum(
            op.Pad(
                # Expand the weights to match the thresholds at least along the
                # thresholding, i.e., final, axis.
                op.Expand(
                    weights1,
                    op.Shape(
                        thresholds1, start=-1
                    )
                ),
                # Add zero padding on the left such that the output levels start
                # counting at the zero level
                op.Constant(value_ints=[1, 0]),
                op.CastLike(
                    op.Constant(value_int=0),
                    weights1
                ),
                op.Constant(value_ints=[-1]),
            ),
            op.Constant(value_int=-1)
        )

        # Expand the second MultiThreshold function to accommodate the extra
        # axis introduced by the weights of the first.
        thresholds2 = op.Unsqueeze(
            op.Expand(
                thresholds2,
                op.Constant(value_ints=[1])
            ),
            op.Constant(value_ints=[-2])
        )

        weights2 = op.Unsqueeze(
            op.Expand(
                weights2,
                op.Constant(value_ints=[1])
            ),
            op.Constant(value_ints=[-2])
        )

        # Transform this sketch of the first function according to the second
        # MultiThreshold function.
        levels = op.MultiThreshold(
            levels, thresholds2, weights2, _domain=PASSES_DOMAIN
        )

        # Compute transformed step weights matching the thresholds of the first
        # MultiThreshold function by taking the "derivative" of the transformed
        # output levels, i.e., use a sliding [-1,+1] kernel.
        fused_weights = op.Sub(
            # Shift the levels one to the left, dropping the leftmost padded
            # level - this gives the +1 weight for the derivative kernel.
            op.Slice(
                levels,
                op.Constant(value_ints=[1]),
                op.Shape(
                    levels, start=-1
                ),
                op.Constant(value_ints=[-1]),
            ),
            # Slice of the final level, keeping the leftmost padded level - this
            # gives the -1 weight for the derivative kernel.
            op.Slice(
                levels,
                op.Constant(value_ints=[0]),
                op.Constant(value_ints=[-1]),
                op.Constant(value_ints=[-1]),
            )
        )

        # Bias the output by where the zero level of the first function gets
        # mapped by the second function, i.e., the leftmost transformed level.
        bias = op.Squeeze(
            op.Slice(
                levels,
                op.Constant(value_ints=[+0]),
                op.Constant(value_ints=[+1]),
                op.Constant(value_ints=[-1]),
            ),
            op.Constant(value_ints=[-1]),
        )

        return op.Add(
            op.MultiThreshold(
                x, thresholds1, fused_weights, _domain=PASSES_DOMAIN
            ),
            bias
        )


@tolerance
class FuseAddIntoMultiThreshold_v1(RewriteRule, Verify):
    """Fuse constant addition preceding MultiThreshold into the thresholds.

    This seems redundant, already covered by absorption passes, but additions
    might reappear when fusing chains of MultiThreshold operations.
    """

    @staticmethod
    def pattern(op, x, y, thresholds, weights):
        return op.MultiThreshold(
            op.Add(x, y), thresholds, weights, _domain=PASSES_DOMAIN
        )

    @staticmethod
    def check(context, x, y, thresholds, weights):
        return ir.convenience.get_const_tensor(y) is not None

    @staticmethod
    def rewrite(op, x, y, thresholds, weights):
        return op.MultiThreshold(
            x,
            op.Sub(
                thresholds,
                op.Unsqueeze(
                    y,
                    op.Constant(value_ints=[-1])
                )
            ),
            weights,
            _domain=PASSES_DOMAIN
        )


def _channel_matrix(value: np.ndarray, rank: int) -> np.ndarray | None:
    """MultiThreshold parameter (thresholds or weights) as (channels, steps).

    The parameters broadcast against the input of the given rank expanded by
    the trailing step axis. Returns None if the parameter varies along any axis
    but the channel axis (1) and the step axis.
    """
    if value.ndim > rank + 1:
        return None

    value = np.reshape(value, (*(1,) * (rank + 1 - value.ndim), *value.shape))
    index = tuple(
        slice(None) if axis in {1, rank} else slice(0, 1)
        for axis in range(rank + 1)
    )

    if not np.all(value == value[index]):
        return None

    return np.reshape(value[index], (value.shape[1], value.shape[-1]))


def _channel_vector(value: np.ndarray, rank: int) -> np.ndarray | None:
    """Constant broadcast against a tensor of the given rank as a vector along
    the channel axis (1), None if it varies along any other axis or expands."""
    if value.ndim > rank:
        return None

    value = np.reshape(value, (*(1,) * (rank - value.ndim), *value.shape))
    index = tuple(
        slice(None) if axis == 1 else slice(0, 1) for axis in range(rank)
    )

    if not np.all(value == value[index]):
        return None

    return np.reshape(value[index], (-1,))


def _negatable_producer(x: ir.Value) -> ir.Node | None:
    """Conv or MatMul with constant weights (and bias) producing x, if x is not
    used elsewhere, such that output channels can be negated in the weights."""
    if len(x.uses()) != 1 or x.is_graph_output():
        return None

    if (node := x.producer()) is None or node.domain not in {"", "ai.onnx"}:
        return None

    if node.op_type == "Conv":
        if ir.convenience.get_const_tensor(node.inputs[1]) is None:
            return None
        if len(node.inputs) > 2 and node.inputs[2] is not None:
            if ir.convenience.get_const_tensor(node.inputs[2]) is None:
                return None
        return node

    if node.op_type == "MatMul":
        if (weights := ir.convenience.get_const_tensor(node.inputs[1])) is None:
            return None
        if weights.numpy().ndim != 2 or x.shape is None or len(x.shape) != 2:
            return None
        return node

    return None


def _constant_like(op, value: np.ndarray, like: ir.Value) -> ir.Value:
    """Constant from value, propagating a constant folding stop of like."""
    const = op.Constant(value=ir.tensor(np.ascontiguousarray(value)))

    if like.meta.get("constant_stop"):
        const.meta["constant_stop"] = True

    return const


class AbsorbChannelOffsetIntoMultiThreshold_v1(Transformation, Verify):
    """Absorb per-channel constant offsets following MultiThreshold.

    Normalizing a quantized activation with per-channel scales of different
    signs or zero scales yields MultiThreshold channels with negative or zero
    step weights, followed by a per-channel constant offset restoring the
    output range. Per-channel offsets cannot move past padding and Conv, so
    absorb them into the MultiThreshold for these channels:

    1. Zero weights, i.e., constant channels: y = a for a > 0 is a single step
       of weight a at threshold -inf, all other steps are dead (+inf).
    2. Non-positive weights with a + sum(w) = 0: As w [x >= t] = w + |w| [x < t]
       and x < t <=> -x >= nextafter(-t, +inf), exact in floating-point, the
       channel counts up on the negated input. The negation is folded into the
       output channel of the Conv or MatMul producing x if x is used only here.

    The MultiThreshold might have other consumers, which receive the original
    output by subtracting the absorbed offsets, to be absorbed downstream.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        builder = GraphBuilder(model.graph)
        op = builder.op

        modified = False

        for node in list(model.graph):
            if node.op_type != "Add" or node.domain not in {"", "ai.onnx"}:
                continue

            for i in (0, 1):
                if self._absorb(op, node, node.inputs[i], node.inputs[1 - i]):
                    modified = True
                    break

        return ir.passes.PassResult(model, modified)

    @staticmethod
    def _absorb(op, add: ir.Node, y: ir.Value, c: ir.Value) -> bool:
        mt = y.producer()

        if mt is None or mt.op_type != "MultiThreshold":
            return False

        if mt.domain != PASSES_DOMAIN or len(mt.inputs) != 3:
            return False

        x, thresholds, weights = mt.inputs

        if x.shape is None or (rank := len(x.shape)) < 2:
            return False

        if x.dtype is None or not x.dtype.is_floating_point():
            return False

        if (offset := ir.convenience.get_const_tensor(c)) is None:
            return False

        if (t := ir.convenience.get_const_tensor(thresholds)) is None:
            return False

        if (w := ir.convenience.get_const_tensor(weights)) is None:
            return False

        offset = _channel_vector(offset.numpy(), rank)
        t = _channel_matrix(t.numpy(), rank)
        w = _channel_matrix(w.numpy(), rank)

        if offset is None or t is None or w is None:
            return False

        # Bring all parameters to (channels, steps), the offset to channels
        channels = max(len(offset), t.shape[0], w.shape[0])
        steps = max(t.shape[1], w.shape[1])

        if x.shape[1] != channels and channels != 1:
            return False

        try:
            t = np.broadcast_to(t, (channels, steps)).copy()
            w = np.broadcast_to(w, (channels, steps)).copy()
            offset = np.broadcast_to(offset, (channels,))
        except ValueError:
            return False

        if not (np.all(np.isfinite(offset)) and np.all(np.isfinite(w))):
            return False

        dead = np.all(w == 0, axis=-1) & (offset > 0)
        flip = (
            np.any(w != 0, axis=-1) & np.all(w <= 0, axis=-1)
            & (offset + np.sum(w, axis=-1) == 0) & (offset != 0)
        )

        producer = _negatable_producer(x)

        if producer is None or channels != x.shape[1]:
            flip[:] = False

        if not (np.any(dead) or np.any(flip)):
            return False

        absorbed = np.where(dead | flip, offset, 0).astype(offset.dtype)

        # 1. Constant channels: A single step of weight a at -inf
        t[dead] = np.inf
        w[dead] = 0
        t[dead, 0] = -np.inf
        w[dead, 0] = offset[dead]

        # 2. Channels counting down on x count up on -x
        t[flip] = np.nextafter(-t[flip], np.asarray(np.inf, t.dtype))
        w[flip] = -w[flip]

        if np.any(flip):
            x = AbsorbChannelOffsetIntoMultiThreshold_v1._negate(
                op, producer, np.where(flip, -1, 1)
            )

        shape = (channels, *(1,) * (rank - 2), steps)
        absorbed = np.reshape(absorbed, (channels, *(1,) * (rank - 2)))
        residual = np.reshape(offset, absorbed.shape) - absorbed

        out = op.MultiThreshold(
            x,
            op.Constant(value=ir.tensor(np.reshape(t, shape))),
            op.Constant(value=ir.tensor(np.reshape(w, shape))),
            _domain=PASSES_DOMAIN
        )

        # Other consumers of the MultiThreshold see the original output
        if len(y.uses()) > 1 or y.is_graph_output():
            ir.convenience.replace_all_uses_with(
                y, op.Sub(out, op.Constant(value=ir.tensor(absorbed))),
                replace_graph_outputs=True
            )

        if np.any(residual != 0):
            out = op.Add(out, op.Constant(value=ir.tensor(residual)))

        ir.convenience.replace_all_uses_with(
            add.outputs[0], out, replace_graph_outputs=True
        )

        return True

    @staticmethod
    def _negate(op, node: ir.Node, signs: np.ndarray) -> ir.Value:
        """Replacement for the Conv or MatMul node with output channels
        multiplied by the signs in {-1, +1}."""
        w = node.inputs[1]
        weights = ir.convenience.get_const_tensor(w).numpy()

        if node.op_type == "MatMul":
            signs = signs.astype(weights.dtype)
            return op.MatMul(
                node.inputs[0], _constant_like(op, weights * signs, w)
            )

        signs = signs.astype(weights.dtype)
        inputs = [
            node.inputs[0], _constant_like(
                op, weights * np.reshape(signs, (-1, *(1,) * (weights.ndim - 1))), w
            )
        ]

        if len(node.inputs) > 2 and (b := node.inputs[2]) is not None:
            bias = ir.convenience.get_const_tensor(b).numpy()
            inputs.append(_constant_like(op, bias * signs, b))

        return op.Conv(*inputs, **node.attributes)


from onnx_passes.passes import fold_constants
from onnx_passes.passes import multithreshold


class FuseMultiThresholdLoop_v1(Sequential):
    """Exhaustively apply multithreshold fusion transformations."""

    passes = [
        FuseScaleIntoMultiThreshold_v1,
        FuseAddedMultiThresholds_v1,
        FuseConsecutiveMultiThresholds_v1,
        FuseAddIntoMultiThreshold_v1,
        AbsorbChannelOffsetIntoMultiThreshold_v1,
        fold_constants,
        multithreshold.normalize,
    ]

    exhaustive = True
