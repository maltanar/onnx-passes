"""Tests for absorbing per-channel offsets into MultiThreshold."""

from __future__ import annotations

import numpy as np
import onnx
import onnx.helper as oh
import onnx_ir as ir
import pytest

from onnx_passes.ops import DOMAIN
from onnx_passes.passes.multithreshold.fuse import (
    AbsorbChannelOffsetIntoMultiThreshold_v1,
)

try:
    from tests.zoo import reference_session
except ImportError:  # pragma: no cover - running from within tests/
    from zoo import reference_session

RNG = np.random.default_rng(0)
X_SHAPE = [1, 4, 5, 5]

# Integer thresholds on an integer accumulator: Ties x == t are frequent
THRESHOLDS = np.array([-2.0, 0.0, 3.0], np.float32)


def _model(weights, offset, producer="Conv", other_consumer=False, bias=True):
    """x -> producer -> MultiThreshold -> Add(offset) -> y (and optionally z,
    another consumer of the MultiThreshold output)."""
    channels = len(weights)
    inits = {
        "t": np.tile(THRESHOLDS, (channels, 1)).reshape(channels, 1, 1, 3),
        "w": np.asarray(weights, np.float32).reshape(channels, 1, 1, -1),
        "c": np.asarray(offset, np.float32).reshape(channels, 1, 1),
    }
    if producer == "Conv":
        inits["k"] = RNG.integers(-1, 2, (channels, 4, 3, 3)).astype(np.float32)
        inputs = ["x", "k"]
        if bias:
            inits["b"] = RNG.integers(-2, 3, channels).astype(np.float32)
            inputs.append("b")
        nodes = [oh.make_node("Conv", inputs, ["acc"], pads=[1, 1, 1, 1])]
    else:
        nodes = [oh.make_node("Relu", ["x"], ["acc"])]

    nodes += [
        oh.make_node("MultiThreshold", ["acc", "t", "w"], ["m"], domain=DOMAIN),
        oh.make_node("Add", ["m", "c"], ["y"]),
    ]
    outputs = [oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, None)]
    if other_consumer:
        nodes.append(oh.make_node("Neg", ["m"], ["z"]))
        outputs.append(oh.make_tensor_value_info("z", onnx.TensorProto.FLOAT, None))

    graph = oh.make_graph(
        nodes, "g",
        [oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, X_SHAPE)],
        outputs,
        [onnx.numpy_helper.from_array(v, k) for k, v in inits.items()],
    )
    return oh.make_model(
        graph, ir_version=9,
        opset_imports=[oh.make_opsetid("", 19), oh.make_opsetid(DOMAIN, 1)],
    )


def _apply(proto: onnx.ModelProto):
    copy = onnx.ModelProto()
    copy.CopyFrom(proto)
    model = ir.from_proto(copy)
    ir.passes.common.ShapeInferencePass()(model)
    return AbsorbChannelOffsetIntoMultiThreshold_v1()(model)


def _run(model: ir.Model, x: np.ndarray) -> list[np.ndarray]:
    names = [i.name for i in model.graph.inputs]
    return reference_session(model).run(None, {names[0]: x})


def _assert_equivalent(proto: onnx.ModelProto, model: ir.Model):
    original = ir.from_proto(proto)
    for _ in range(3):
        x = RNG.integers(-2, 3, X_SHAPE).astype(np.float32)
        for produced, expected in zip(_run(model, x), _run(original, x)):
            np.testing.assert_array_equal(produced, expected)


def _ops(model: ir.Model) -> list[str]:
    return [n.op_type for n in model.graph]


def _multithreshold_weights(model: ir.Model) -> np.ndarray:
    node = next(n for n in model.graph if n.op_type == "MultiThreshold")
    return ir.convenience.get_const_tensor(node.inputs[2]).numpy()


@pytest.mark.parametrize("bias", [False, True])
def test_negative_channels_flip_into_conv(bias):
    """Channels counting down from the offset count up on the negated Conv."""
    proto = _model([1, -1, -2, 1], [0, 3, 6, 0], bias=bias)
    result = _apply(proto)

    assert result.modified
    assert _ops(result.model) == ["Conv", "MultiThreshold"]
    assert np.all(_multithreshold_weights(result.model) >= 0)
    _assert_equivalent(proto, result.model)


def test_constant_channel_becomes_single_step():
    """Zero-weighted channels with positive offset are a step at -inf."""
    proto = _model([1, 0, 1, 0], [0, 2, 0, 0.5])
    result = _apply(proto)

    assert result.modified
    assert _ops(result.model) == ["Conv", "MultiThreshold"]
    _assert_equivalent(proto, result.model)


def test_partial_absorption_keeps_residual_offset():
    """Channels which cannot be absorbed keep their offset."""
    proto = _model([-1, 1, 0, 1], [3, 1, 2, 0])
    result = _apply(proto)

    assert result.modified
    assert _ops(result.model) == ["Conv", "MultiThreshold", "Add"]
    offset = ir.convenience.get_const_tensor(
        next(n for n in result.model.graph if n.op_type == "Add").inputs[1]
    ).numpy()
    np.testing.assert_array_equal(offset.ravel(), [0, 1, 0, 0])
    _assert_equivalent(proto, result.model)


def test_other_consumers_see_the_original_output():
    proto = _model([1, -1, 0, 1], [0, 3, 2, 0], other_consumer=True)
    result = _apply(proto)

    assert result.modified
    assert "Sub" in _ops(result.model)
    _assert_equivalent(proto, result.model)


def test_flip_requires_negatable_producer():
    """Without a Conv or MatMul to negate, only constant channels absorb."""
    proto = _model([-1, 0, 1, 1], [3, 2, 0, 0], producer="Relu")
    result = _apply(proto)

    assert result.modified
    assert _ops(result.model)[-1] == "Add"
    _assert_equivalent(proto, result.model)


@pytest.mark.parametrize("weights,offset", [
    ([-1, -1, -1, -1], [2, 2, 2, 4]),        # Offset does not cancel the sum
    ([[-1, 1, 1]] * 4, [1, 1, 1, 1]),        # Mixed-sign steps in a channel
    ([1, 1, 1, 1], [1, 2, 3, 4]),            # Positive channels: Nothing to do
    ([0, 0, 0, 0], [-1, -1, -1, -1]),        # Negative constants
])
def test_rejects(weights, offset):
    proto = _model(weights, offset)
    assert not _apply(proto).modified
