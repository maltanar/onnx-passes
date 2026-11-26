"""Tests for pooling normalization and reordering past MaxPool."""

from __future__ import annotations

import itertools

import numpy as np
import onnx
import onnx.helper as oh
import onnx_ir as ir
import onnxruntime as ort
import pytest

from onnx_passes.passes.normalize.pooling import (
    RewritePaddedMaxPoolAsPadMaxPool_v1,
)


def _run(proto: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    """Runs on onnxruntime: The ONNX reference evaluator gets ceil_mode wrong."""
    session = ort.InferenceSession(
        proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return session.run(None, {session.get_inputs()[0].name: x})[0]


def _to_ir(proto: onnx.ModelProto) -> ir.Model:
    # Copy: IR initializers wrap the proto tensors, renaming them in place
    copy = onnx.ModelProto()
    copy.CopyFrom(proto)
    return ir.from_proto(copy)


def _apply(rule, proto: onnx.ModelProto) -> ir.passes.PassResult:
    return rule()(_to_ir(proto))


def _maxpool(dtype, x_shape, **attributes) -> onnx.ModelProto:
    elem = oh.np_dtype_to_tensor_dtype(np.dtype(dtype))
    graph = oh.make_graph(
        [oh.make_node("MaxPool", ["x"], ["y"], **attributes)], "g",
        [oh.make_tensor_value_info("x", elem, x_shape)],
        [oh.make_tensor_value_info("y", elem, None)],
    )
    return oh.make_model(
        graph, opset_imports=[oh.make_opsetid("", 19)], ir_version=9
    )


_CONFIGS = [
    config for config in itertools.product(
        [4, 5, 7],                                    # Input size
        [2, 3],                                       # Kernel
        [1, 2],                                       # Stride
        [1, 2],                                       # Dilation
        [[1, 1, 1, 1], [1, 0, 0, 1], [0, 1, 1, 0]],   # Pads (incl. asymmetric)
        [0, 1, None],                                 # ceil_mode (None: absent)
    )
]


@pytest.mark.parametrize("dtype", [np.float32, np.int8])
@pytest.mark.parametrize("size,k,s,d,pads,ceil_mode", _CONFIGS)
def test_padded_maxpool_rewrite_matches_onnxruntime(dtype, size, k, s, d, pads, ceil_mode):
    attributes = dict(kernel_shape=[k, k], strides=[s, s], dilations=[d, d], pads=pads)
    if ceil_mode is not None:
        attributes["ceil_mode"] = ceil_mode
    proto = _maxpool(dtype, [1, 2, size, size], **attributes)

    # Negative inputs: Padding must never win the maximum
    x = np.random.default_rng(size * k + s).integers(-9, -1, (1, 2, size, size))
    x = x.astype(dtype)

    result = _apply(RewritePaddedMaxPoolAsPadMaxPool_v1, proto)

    assert result.modified
    assert [n.op_type for n in result.model.graph] == ["Pad", "MaxPool"]
    np.testing.assert_array_equal(_run(ir.to_proto(result.model), x), _run(proto, x))


# ---------------------------------------------------------------------------
# Reordering past MaxPool and Pad
# ---------------------------------------------------------------------------

from onnx_passes.passes.reorder.linear import (  # noqa: E402
    MoveAddPastPad_v1,
    MoveMulPastPad_v1,
)
from onnx_passes.passes.reorder.pooling import (  # noqa: E402
    MoveAddPastMaxPool_v1,
    MoveMulPastMaxPool_v1,
)

X_SHAPE = [1, 3, 6, 6]
RNG = np.random.default_rng(0)


def _graph(nodes, inits: dict[str, np.ndarray], outputs=("y",), dtype=np.float32):
    elem = oh.np_dtype_to_tensor_dtype(np.dtype(dtype))
    graph = oh.make_graph(
        nodes, "g",
        [oh.make_tensor_value_info("x", elem, X_SHAPE)],
        [oh.make_tensor_value_info(o, onnx.TensorProto.INT64 if o == "indices" else elem, None)
         for o in outputs],
        [onnx.numpy_helper.from_array(np.asarray(v), k) for k, v in inits.items()],
    )
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 19)], ir_version=9)


def _check_moved(rule, proto, moved: str, dtype=np.float32):
    """The rule applies and the moved operation follows the other one."""
    result = _apply(rule, proto)
    assert result.modified
    assert [n.op_type for n in result.model.graph][-1] == moved
    x = RNG.uniform(-3, 3, X_SHAPE).astype(dtype)
    np.testing.assert_allclose(_run(ir.to_proto(result.model), x), _run(proto, x), rtol=1e-6)


def _channel(values) -> np.ndarray:
    return np.asarray(values, np.float32).reshape(1, -1, 1, 1)


_POOL = dict(kernel_shape=[3, 3], strides=[2, 2])


@pytest.mark.parametrize("c", [np.float32(-1.5), _channel([-1, 2, -128])])
@pytest.mark.parametrize("attrs", [
    {}, {"pads": [1, 1, 1, 1]}, {"pads": [1, 0, 0, 1], "ceil_mode": 1},
])
def test_add_moves_past_maxpool(c, attrs):
    proto = _graph(
        [oh.make_node("Add", ["x", "c"], ["xc"]),
         oh.make_node("MaxPool", ["xc"], ["y"], **_POOL, **attrs)],
        {"c": c},
    )
    _check_moved(MoveAddPastMaxPool_v1, proto, "Add")


def test_add_moves_past_maxpool_with_unused_indices():
    proto = _graph(
        [oh.make_node("Add", ["x", "c"], ["xc"]),
         oh.make_node("MaxPool", ["xc"], ["y", "indices"], **_POOL)],
        {"c": _channel([1, -1, 2])},
    )
    _check_moved(MoveAddPastMaxPool_v1, proto, "Add")


@pytest.mark.parametrize("s", [np.float32(0.5), _channel([2, 0.25, 3])])
def test_positive_mul_moves_past_maxpool(s):
    proto = _graph(
        [oh.make_node("Mul", ["x", "s"], ["xs"]),
         oh.make_node("MaxPool", ["xs"], ["y"], **_POOL, pads=[1, 1, 1, 1])],
        {"s": s},
    )
    _check_moved(MoveMulPastMaxPool_v1, proto, "Mul")


@pytest.mark.parametrize("rule,op,c,outputs,dtype", [
    # Negative scales turn the maximum into a minimum
    (MoveMulPastMaxPool_v1, "Mul", _channel([2, -2, 2]), ("y",), np.float32),
    # Varying along the pooled axes
    (MoveAddPastMaxPool_v1, "Add", np.arange(36, dtype=np.float32).reshape(1, 1, 6, 6), ("y",), np.float32),
    # Used indices might change due to ties
    (MoveAddPastMaxPool_v1, "Add", np.float32(1.0), ("y", "indices"), np.float32),
    # Integer overflow does not preserve the order
    (MoveAddPastMaxPool_v1, "Add", np.int8(100), ("y",), np.int8),
    # Non-finite constants
    (MoveAddPastMaxPool_v1, "Add", np.float32(np.inf), ("y",), np.float32),
])
def test_maxpool_rules_reject(rule, op, c, outputs, dtype):
    proto = _graph(
        [oh.make_node(op, ["x", "c"], ["xc"]),
         oh.make_node("MaxPool", ["xc"], list(outputs), **_POOL)],
        {"c": c}, outputs=outputs, dtype=dtype,
    )
    assert not _apply(rule, proto).modified


def _pad(op: str, c, value) -> onnx.ModelProto:
    return _graph(
        [oh.make_node(op, ["x", "c"], ["xc"]),
         oh.make_node("Pad", ["xc", "pads", "v"], ["y"], mode="constant")],
        {"c": c, "pads": np.array([0, 0, 1, 1, 0, 0, 1, 1]), "v": np.float32(value)},
    )


@pytest.mark.parametrize("c,value", [
    (_channel([-1, 2, -128]), -np.inf),   # Per-channel, infinite padding
    (np.float32(-1.5), 0.0),              # Scalar shifts finite padding
    (np.float32(2.5), 0.75),
])
def test_add_moves_past_pad(c, value):
    _check_moved(MoveAddPastPad_v1, _pad("Add", c, value), "Add")


def test_add_past_pad_rejects_per_channel_finite_padding():
    assert not _apply(MoveAddPastPad_v1, _pad("Add", _channel([1, 2, 3]), 0.5)).modified


def test_positive_channel_mul_moves_past_infinite_pad():
    _check_moved(MoveMulPastPad_v1, _pad("Mul", _channel([2, 0.5, 3]), -np.inf), "Mul")


def test_negative_channel_mul_does_not_move_past_infinite_pad():
    assert not _apply(MoveMulPastPad_v1, _pad("Mul", _channel([2, -2, 3]), -np.inf)).modified


def test_add_moves_past_padded_maxpool_end_to_end():
    """Padded MaxPool is normalized to Pad(-inf) + MaxPool, Add moves past both."""
    proto = _graph(
        [oh.make_node("Add", ["x", "c"], ["xc"]),
         oh.make_node("MaxPool", ["xc"], ["y"], **_POOL, pads=[1, 1, 1, 1], ceil_mode=1)],
        {"c": _channel([-1, 2, -128])},
    )
    # The padding rewrite needs the inferred type and shape of the Add output
    model = ir.passes.common.ShapeInferencePass()(_to_ir(proto)).model
    model = RewritePaddedMaxPoolAsPadMaxPool_v1()(model).model
    assert [n.op_type for n in model.graph] == ["Add", "Pad", "MaxPool"]
    for rule in (MoveAddPastPad_v1, MoveAddPastMaxPool_v1):
        model = rule()(model).model

    assert [n.op_type for n in model.graph] == ["Pad", "MaxPool", "Add"]
    x = RNG.uniform(-3, 3, X_SHAPE).astype(np.float32)
    np.testing.assert_allclose(_run(ir.to_proto(model), x), _run(proto, x), rtol=1e-6)
