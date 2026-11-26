"""Tests for reordering elementwise operations past MatMul and Conv."""

from __future__ import annotations

import numpy as np
import onnx
import onnx.helper as oh
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator

from onnx_passes.passes.reorder.linear import (
    MoveAddPastConvInput_v1,
    MoveSignedChannelMulPastConvInput_v1,
    MoveSignedChannelMulPastMatMulLhs_v1,
)

RNG = np.random.default_rng(0)


def _model(nodes, x_shape, y_shape, inits: dict[str, np.ndarray]) -> onnx.ModelProto:
    graph = oh.make_graph(
        nodes, "g",
        [oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, x_shape)],
        [oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, y_shape)],
        [onnx.numpy_helper.from_array(v.astype(np.float32), k) for k, v in inits.items()],
    )
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 19)], ir_version=9)


def _apply(rule, proto: onnx.ModelProto) -> tuple[ir.passes.PassResult, ir.Model]:
    # Copy: IR initializers wrap the proto tensors, renaming them in place
    copy = onnx.ModelProto()
    copy.CopyFrom(proto)
    result = rule()(ir.from_proto(copy))
    return result, result.model


def _assert_equivalent(proto: onnx.ModelProto, model: ir.Model, x_shape):
    x = RNG.integers(0, 2, x_shape).astype(np.float32)
    expected = ReferenceEvaluator(proto).run(None, {"x": x})[0]
    inputs = {model.graph.inputs[0].name: x}
    produced = ReferenceEvaluator(ir.to_proto(model)).run(None, inputs)[0]
    np.testing.assert_allclose(produced, expected, rtol=1e-6)


def _weights(model: ir.Model, op_type: str) -> np.ndarray:
    node = next(n for n in model.graph if n.op_type == op_type)
    return ir.convenience.get_const_tensor(node.inputs[1]).numpy()


@pytest.mark.parametrize("x_shape,s_shape", [
    ([1, 8], [8]),          # Vector along the reduction axis
    ([2, 3, 8], [1, 1, 8]),  # Batched input, broadcast scale
])
def test_matmul_signs_fold_into_integer_weights(x_shape, s_shape):
    s = np.where(RNG.random(8) < 0.5, -2.0, 2.0).reshape(s_shape)
    w = RNG.choice([-1.0, 1.0], (8, 5))
    proto = _model(
        [oh.make_node("Mul", ["x", "s"], ["xs"]), oh.make_node("MatMul", ["xs", "w"], ["y"])],
        x_shape, [*x_shape[:-1], 5], {"s": s, "w": w},
    )

    result, model = _apply(MoveSignedChannelMulPastMatMulLhs_v1, proto)

    assert result.modified
    assert [n.op_type for n in model.graph][:2] == ["MatMul", "Mul"]
    assert set(np.unique(_weights(model, "MatMul"))) <= {-1.0, 1.0}
    _assert_equivalent(proto, model, x_shape)


@pytest.mark.parametrize("s", [
    np.array([2.0, -2.0, 1.0, 2.0]),   # Non-uniform magnitude
    np.array([2.0, 2.0, 2.0, 2.0]),    # Scalar: covered by the scalar rules
    np.array([0.0, -0.0, 0.0, 0.0]),   # Zero magnitude
])
def test_matmul_rule_rejects(s):
    proto = _model(
        [oh.make_node("Mul", ["x", "s"], ["xs"]), oh.make_node("MatMul", ["xs", "w"], ["y"])],
        [1, 4], [1, 3], {"s": s, "w": RNG.choice([-1.0, 1.0], (4, 3))},
    )
    assert not _apply(MoveSignedChannelMulPastMatMulLhs_v1, proto)[0].modified


def test_matmul_rule_requires_constant_weights():
    proto = _model(
        [
            oh.make_node("Mul", ["x", "s"], ["xs"]),
            oh.make_node("Transpose", ["w"], ["wt"], perm=[1, 0]),
            oh.make_node("MatMul", ["xs", "wt"], ["y"]),
        ],
        [1, 4], [1, 3],
        {"s": np.array([2.0, -2.0, 2.0, -2.0]), "w": RNG.choice([-1.0, 1.0], (3, 4))},
    )
    assert not _apply(MoveSignedChannelMulPastMatMulLhs_v1, proto)[0].modified


@pytest.mark.parametrize("group,bias", [(1, False), (1, True), (2, True), (4, False)])
def test_conv_signs_fold_into_integer_weights(group, bias):
    c_in, c_out = 4, 8
    s = np.where(RNG.random(c_in) < 0.5, -0.5, 0.5).reshape(1, c_in, 1, 1)
    w = RNG.choice([-1.0, 1.0], (c_out, c_in // group, 3, 3))
    inits = {"s": s, "w": w}
    inputs = ["xs", "w"]
    if bias:
        inits["b"] = RNG.normal(size=c_out)
        inputs.append("b")
    proto = _model(
        [
            oh.make_node("Mul", ["x", "s"], ["xs"]),
            oh.make_node("Conv", inputs, ["y"], group=group, pads=[1, 1, 1, 1]),
        ],
        [1, c_in, 5, 5], [1, c_out, 5, 5], inits,
    )

    result, model = _apply(MoveSignedChannelMulPastConvInput_v1, proto)

    assert result.modified
    assert [n.op_type for n in model.graph][0] == "Conv"
    assert set(np.unique(_weights(model, "Conv"))) <= {-1.0, 1.0}
    _assert_equivalent(proto, model, [1, c_in, 5, 5])


def test_conv_rule_rejects_scale_varying_spatially():
    s = np.full((1, 2, 3, 3), 2.0)
    s[0, 0, 0, 0] = -2.0
    proto = _model(
        [oh.make_node("Mul", ["x", "s"], ["xs"]), oh.make_node("Conv", ["xs", "w"], ["y"])],
        [1, 2, 3, 3], [1, 4, 1, 1], {"s": s, "w": RNG.choice([-1.0, 1.0], (4, 2, 3, 3))},
    )
    assert not _apply(MoveSignedChannelMulPastConvInput_v1, proto)[0].modified


@pytest.mark.parametrize("c_shape,group,bias,attrs", [
    ([], 1, False, {}),                                  # Scalar offset
    ([1, 4, 1, 1], 1, True, {}),                         # Per-channel, bias
    ([4, 1, 1], 2, True, {"strides": [2, 2]}),           # Grouped, strided
    ([1, 4, 1, 1], 4, False, {"dilations": [2, 2]}),     # Depthwise, dilated
    ([1, 4, 1, 1], 1, False, {"pads": [0, 0, 0, 0]}),    # Explicit zero pads
])
def test_add_moves_past_unpadded_conv(c_shape, group, bias, attrs):
    c_in, c_out = 4, 8
    c = RNG.choice([-128.0, -1.0, 1.0], c_shape if c_shape else ())
    w = RNG.choice([-1.0, 1.0], (c_out, c_in // group, 3, 3))
    inits = {"c": np.asarray(c), "w": w}
    inputs = ["xc", "w"]
    if bias:
        inits["b"] = RNG.normal(size=c_out)
        inputs.append("b")
    proto = _model(
        [
            oh.make_node("Add", ["x", "c"], ["xc"]),
            oh.make_node("Conv", inputs, ["y"], group=group, **attrs),
        ],
        [1, c_in, 7, 7], None, inits,
    )

    result, model = _apply(MoveAddPastConvInput_v1, proto)

    assert result.modified
    assert [n.op_type for n in model.graph][0] == "Conv"
    _assert_equivalent(proto, model, [1, c_in, 7, 7])


@pytest.mark.parametrize("c,attrs", [
    (np.full((1, 2, 1, 1), 1.0), {"pads": [1, 1, 1, 1]}),  # Padding
    (np.full((1, 2, 1, 1), 1.0), {"auto_pad": "SAME_UPPER"}),
    (np.arange(18.0).reshape(1, 2, 3, 3), {}),              # Spatially varying
])
def test_add_past_conv_rejects(c, attrs):
    proto = _model(
        [oh.make_node("Add", ["x", "c"], ["xc"]), oh.make_node("Conv", ["xc", "w"], ["y"], **attrs)],
        [1, 2, 3, 3], None, {"c": c, "w": RNG.choice([-1.0, 1.0], (4, 2, 3, 3))},
    )
    assert not _apply(MoveAddPastConvInput_v1, proto)[0].modified
