"""Tests for the legalize passes preparing exported models for streamlining."""

from __future__ import annotations

import numpy as np
import onnx
import onnx.helper as oh
import onnx_ir as ir
from onnx.reference import ReferenceEvaluator

from onnx_passes.passes.legalize import (
    MIN_DEFAULT_OPSET,
    NormalizeDomains_v1,
    UpgradeOpset_v1,
    normalize_domains,
    upgrade_opset,
)
from onnx_passes.passes.streamline import StreamlineScales_v1, StreamlineThresholds_v1

try:
    from tests.zoo import reference_session
except ImportError:
    from zoo import reference_session


def _init(name: str, arr) -> onnx.TensorProto:
    return onnx.numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name)


def _quant_matmul_model(domain: str, opset: int, declare: bool) -> ir.Model:
    """x -> Quant(domain) -> MatMul -> y, mimicking an old Brevitas export."""
    rng = np.random.default_rng(0)
    nodes = [
        oh.make_node(
            "Quant", ["x", "s", "z", "bw"], ["q"], domain=domain,
            signed=1, narrow=0, rounding_mode="ROUND",
        ),
        oh.make_node("MatMul", ["q", "w"], ["y"]),
    ]
    graph = oh.make_graph(
        nodes, "g",
        [oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])],
        [oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3])],
        [
            _init("s", 0.25), _init("z", 0.0), _init("bw", 4.0),
            _init("w", rng.integers(-3, 4, (4, 3))),
        ],
    )
    opsets = [oh.make_opsetid("", opset)]
    if declare:
        opsets.append(oh.make_opsetid(domain, 1))
    return ir.from_proto(oh.make_model(graph, opset_imports=opsets, ir_version=7))


def test_normalize_domains_renames_and_declares():
    model = _quant_matmul_model("finn.custom_op.general", 9, declare=True)
    model.graph.node(0).domain = "finn.custom_op.general"
    # A second, undeclared custom-op domain
    model.graph.append(ir.node("Identity", [model.graph.outputs[0]], domain="onnx.brevitas"))

    assert normalize_domains(model)
    assert model.graph.node(0).domain == "qonnx.custom_op.general"
    assert dict(model.graph.opset_imports) == {
        "": 9, "qonnx.custom_op.general": 1, "onnx.brevitas": 1
    }
    # Idempotent: already legal models are not modified
    assert not normalize_domains(model)


def test_upgrade_opset_preserves_semantics():
    nodes = [
        oh.make_node("Clip", ["x"], ["c"], min=-1.0, max=1.0),  # attrs < opset 11
        oh.make_node("Relu", ["c"], ["y"]),
    ]
    graph = oh.make_graph(
        nodes, "g",
        [oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [8])],
        [oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [8])],
    )
    proto = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 9)], ir_version=7)
    model = ir.from_proto(proto)

    assert upgrade_opset(model)
    assert model.graph.opset_imports[""] == MIN_DEFAULT_OPSET

    x = np.linspace(-2, 2, 8, dtype=np.float32)
    expected = ReferenceEvaluator(proto).run(None, {"x": x})[0]
    produced = ReferenceEvaluator(ir.to_proto(model)).run(None, {"x": x})[0]
    np.testing.assert_array_equal(produced, expected)

    # Never downgrades or touches models at or above the target
    assert not upgrade_opset(model)
    model.graph.opset_imports[""] = MIN_DEFAULT_OPSET + 2
    assert not upgrade_opset(model)
    assert model.graph.opset_imports[""] == MIN_DEFAULT_OPSET + 2


def test_upgrade_opset_keeps_model_functions():
    model = _quant_matmul_model("custom", 13, declare=True)
    model.graph.node(0).op_type = "F"
    model.graph.node(0).attributes.clear()
    body = [oh.make_node("Mul", ["x", "s"], ["y"])]
    function = oh.make_function(
        "custom", "F", ["x", "s", "z", "bw"], ["y"], body,
        [oh.make_opsetid("", 13)],
    )
    model.functions[("custom", "F", "")] = ir.serde.deserialize_function(function)

    assert upgrade_opset(model)
    assert ("custom", "F", "") in model.functions
    assert model.graph.node(0).op_type == "F"


def test_legalize_passes_are_noops_on_legal_models():
    model = _quant_matmul_model("qonnx.custom_op.general", MIN_DEFAULT_OPSET, declare=True)
    assert not NormalizeDomains_v1()(model).modified
    assert not UpgradeOpset_v1()(model).modified


def test_streamline_legalizes_raw_brevitas_export():
    """Streamlining works directly on an opset 9 export with undeclared domain."""
    model = _quant_matmul_model("onnx.brevitas", 9, declare=False)

    reference = model.clone(deep_copy=True)
    normalize_domains(reference)
    upgrade_opset(reference)

    result = StreamlineThresholds_v1()(StreamlineScales_v1()(model).model).model
    assert result.graph.opset_imports[""] == MIN_DEFAULT_OPSET

    x = np.random.default_rng(1).uniform(-2, 2, (1, 4)).astype(np.float32)
    expected = reference_session(reference).run(None, {"x": x})[0]
    inputs = {result.graph.inputs[0].name: x}
    produced = reference_session(result).run(None, inputs)[0]
    np.testing.assert_allclose(produced, expected, rtol=1e-5, atol=1e-5)
