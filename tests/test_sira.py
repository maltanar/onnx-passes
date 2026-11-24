import os
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx.reference import ReferenceEvaluator

from onnx_passes.ops import link_ops
from onnx_passes.ops.qonnx import DOMAIN as QONNX_DOMAIN
from onnx_passes.passes.sira import (
    RangeAnalysis_v1,
    compute_accumulator_bitwidth,
    evaluate_ranges,
)


def _make_model(
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    nodes: list[ir.Node],
    initializers: list[ir.Value] | None = None,
    opset: int = 18,
    extra_opsets: dict[str, int] | None = None,
) -> ir.Model:
    opsets = {"": opset}
    if extra_opsets:
        opsets.update(extra_opsets)
    graph = ir.Graph(
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        initializers=initializers or [],
        opset_imports=opsets,
        name="test_graph",
    )
    return ir.Model(graph, ir_version=9)


def _const(name: str, arr: np.ndarray) -> ir.Value:
    return ir.Value(
        name=name,
        const_value=ir.tensor(arr),
        type=ir.TensorType(ir.DataType.from_numpy(arr.dtype)),
        shape=ir.Shape(arr.shape),
    )


def _run_sira(
    model: ir.Model,
    input_ranges: dict[str, tuple[np.ndarray, np.ndarray]],
    input_scales: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    backend: str = "onnxruntime",
):
    rm = RangeAnalysis_v1()(model.clone(deep_copy=True)).model
    return evaluate_ranges(rm, input_ranges, input_scales=input_scales, backend=backend, intermediates=True)


def _assert_sound(
    orig_model: ir.Model,
    ranges: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    input_ranges: dict[str, tuple[np.ndarray, np.ndarray]],
    input_scales: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    n_samples: int = 5,
    rng_seed: int = 42,
):
    """Sample inputs inside input_ranges and assert every tensor lies in its SIRA range."""
    rng = np.random.default_rng(rng_seed)
    m = link_ops(orig_model.clone(deep_copy=True))
    sess = ReferenceEvaluator(ir.to_proto(m))

    for _ in range(n_samples):
        feed = {}
        for name, (lo, hi) in input_ranges.items():
            if input_scales and name in input_scales:
                s, b = input_scales[name]
                q = rng.integers(lo.astype(np.int64), hi.astype(np.int64) + 1)
                feed[name] = (q * s + b).astype(np.float32)
            else:
                feed[name] = rng.uniform(lo, hi).astype(np.float32)
        ctx = sess.run(None, feed, intermediate=True)
        for tname, (vmin, vmax, vscale, vbias) in ranges.items():
            if tname not in ctx:
                continue
            val = np.asarray(ctx[tname], dtype=np.float32)
            if not (np.isnan(vscale).any() or np.isnan(vbias).any()):
                a = vmin * vscale + vbias
                b = vmax * vscale + vbias
                rmin = np.minimum(a, b)
                rmax = np.maximum(a, b)
                # Check integer property where scale != 0
                nz = vscale != 0
                if np.any(nz):
                    q = np.broadcast_to((val - vbias), val.shape)[np.broadcast_to(nz, val.shape)] / np.broadcast_to(vscale, val.shape)[np.broadcast_to(nz, val.shape)]
                    np.testing.assert_allclose(q, np.round(q), atol=1e-3, err_msg=f"Non-integer q for {tname}")
            else:
                rmin, rmax = vmin, vmax
            assert np.all(val >= rmin - 1e-4), f"Lower bound violated for {tname}: {val.min()} < {rmin.min()}"
            assert np.all(val <= rmax + 1e-4), f"Upper bound violated for {tname}: {val.max()} > {rmax.max()}"


def test_quant_signed_unsigned_bipolar():
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    s = _const("s", np.array([0.5, 0.1], dtype=np.float32))
    z = _const("z", np.array(0.0, dtype=np.float32))
    bw = _const("bw", np.array(4.0, dtype=np.float32))
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    node = ir.Node(
        QONNX_DOMAIN, "Quant", inputs=[x, s, z, bw], outputs=[y],
        attributes={
            "signed": ir.Attr("signed", ir.AttributeType.INT, 1),
            "narrow": ir.Attr("narrow", ir.AttributeType.INT, 0),
            "rounding_mode": ir.Attr("rounding_mode", ir.AttributeType.STRING, "ROUND"),
        },
    )
    model = _make_model([x], [y], [node], initializers=[s, z, bw], extra_opsets={QONNX_DOMAIN: 1})
    lo = np.array([[-3.5, -2.0]], dtype=np.float32)
    hi = np.array([[2.5, 2.0]], dtype=np.float32)
    ranges = _run_sira(model, {"x": (lo, hi)})
    ymin, ymax, ys, yb = ranges["y"]
    np.testing.assert_allclose(ymin, [[-7.0, -8.0]])
    np.testing.assert_allclose(ymax, [[5.0, 7.0]])
    np.testing.assert_allclose(ys, [0.5, 0.1])
    np.testing.assert_allclose(yb, [0.0, 0.0])
    _assert_sound(model, ranges, {"x": (lo, hi)})


def test_bipolar_quant():
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 3]))
    s = _const("s", np.array(0.25, dtype=np.float32))
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 3]))
    node = ir.Node(QONNX_DOMAIN, "BipolarQuant", inputs=[x, s], outputs=[y])
    model = _make_model([x], [y], [node], initializers=[s], extra_opsets={QONNX_DOMAIN: 1})
    lo = np.array([[-1.0, 0.1, -2.0]], dtype=np.float32)
    hi = np.array([[1.0, 2.0, -0.1]], dtype=np.float32)
    ranges = _run_sira(model, {"x": (lo, hi)})
    ymin, ymax, ys, yb = ranges["y"]
    np.testing.assert_allclose(ymin, [[-1.0, 1.0, -1.0]])
    np.testing.assert_allclose(ymax, [[1.0, 1.0, -1.0]])
    np.testing.assert_allclose(ys, 0.25)
    np.testing.assert_allclose(yb, 0.0)


def test_add_cases():
    # Case 1: scaled-int + constant
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    c = _const("c", np.array([1.5, -0.5], dtype=np.float32))
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    model = _make_model([x], [y], [ir.Node("", "Add", inputs=[x, c], outputs=[y])], initializers=[c])
    ranges = _run_sira(
        model,
        {"x": (np.array([[-4, -2]], dtype=np.float32), np.array([[3, 5]], dtype=np.float32))},
        input_scales={"x": (np.array(0.5, dtype=np.float32), np.array(0.0, dtype=np.float32))},
    )
    ymin, ymax, ys, yb = ranges["y"]
    np.testing.assert_allclose(ymin, [[-4, -2]])
    np.testing.assert_allclose(ymax, [[3, 5]])
    np.testing.assert_allclose(ys, 0.5)
    np.testing.assert_allclose(yb, [[1.5, -0.5]])

    # Case 2: scaled-int + scaled-int with integer ratio s1 = 2 * s0
    a = ir.Value(name="a", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    b = ir.Value(name="b", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    z = ir.Value(name="z", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    model2 = _make_model([a, b], [z], [ir.Node("", "Add", inputs=[a, b], outputs=[z])])
    r2 = _run_sira(
        model2,
        {
            "a": (np.array([[-2, 0]], dtype=np.float32), np.array([[3, 4]], dtype=np.float32)),
            "b": (np.array([[-1, 1]], dtype=np.float32), np.array([[2, 2]], dtype=np.float32)),
        },
        input_scales={
            "a": (np.array(0.25, dtype=np.float32), np.array(0.1, dtype=np.float32)),
            "b": (np.array(0.5, dtype=np.float32), np.array(-0.2, dtype=np.float32)),
        },
    )
    zmin, zmax, zs, zb = r2["z"]
    np.testing.assert_allclose(zmin, [[-4, 2]])
    np.testing.assert_allclose(zmax, [[7, 8]])
    np.testing.assert_allclose(zs, 0.25)
    np.testing.assert_allclose(zb, -0.1, atol=1e-6)

    # Case 3: non-integer ratio -> fallback to real-valued
    r3 = _run_sira(
        model2,
        {
            "a": (np.array([[-2, 0]], dtype=np.float32), np.array([[3, 4]], dtype=np.float32)),
            "b": (np.array([[-1, 1]], dtype=np.float32), np.array([[2, 2]], dtype=np.float32)),
        },
        input_scales={
            "a": (np.array(0.3, dtype=np.float32), np.array(0.0, dtype=np.float32)),
            "b": (np.array(0.5, dtype=np.float32), np.array(0.0, dtype=np.float32)),
        },
    )
    zmin3, zmax3, zs3, zb3 = r3["z"]
    assert np.isnan(zs3).all() and np.isnan(zb3).all()
    np.testing.assert_allclose(zmin3, [[-1.1, 0.5]], atol=1e-5)
    np.testing.assert_allclose(zmax3, [[1.9, 2.2]], atol=1e-5)


def test_mul_div_sub():
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    c = _const("c", np.array([-2.0, 3.0], dtype=np.float32))
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    model = _make_model([x], [y], [ir.Node("", "Mul", inputs=[x, c], outputs=[y])], initializers=[c])
    ranges = _run_sira(
        model,
        {"x": (np.array([[-4, 1]], dtype=np.float32), np.array([[2, 5]], dtype=np.float32))},
        input_scales={"x": (np.array(0.5, dtype=np.float32), np.array(1.0, dtype=np.float32))},
    )
    ymin, ymax, ys, yb = ranges["y"]
    np.testing.assert_allclose(ymin, [[-4, 1]])
    np.testing.assert_allclose(ymax, [[2, 5]])
    np.testing.assert_allclose(ys, [[-1.0, 1.5]])
    np.testing.assert_allclose(yb, [[-2.0, 3.0]])


def test_sira_paper_layer_example():
    """Verify SIRA paper Section 3.3 (Figure 5, Table 2 & Table 3) numerical example."""
    # X (1, 3) -> Quant -> qX; W (3, 3) -> Quant -> qW; MatMul(qX, qW) -> mm;
    # Add(mm, B) -> mm_b; Mul(mm_b, M) -> bn_m; Add(bn_m, N) -> bn_n; Relu -> r; Quant -> qy
    x = ir.Value(name="X", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 3]))
    qs_x = _const("qs_X", np.array(0.7, dtype=np.float32))
    z_0 = _const("z_0", np.array(0.0, dtype=np.float32))
    bw_4 = _const("bw_4", np.array(4.0, dtype=np.float32))

    # Construct quantized weights directly matching qW in Table 3 with qs_W = [0.2, 0.3, 0.1]
    w_int = np.array([
        [7, -8, 5],
        [-5, 6, 4],
        [4, -3, -8],
    ], dtype=np.float32)
    qs_w_vals = np.array([0.2, 0.3, 0.1], dtype=np.float32)
    w_const = _const("W", w_int * qs_w_vals)
    qs_w = _const("qs_W", qs_w_vals)

    b_const = _const("B", np.array([-3.30, -5.20, -6.10], dtype=np.float32))
    m_const = _const("M", np.array([0.60, 0.20, 0.40], dtype=np.float32))
    n_const = _const("N", np.array([-0.20, -0.40, 1.10], dtype=np.float32))
    qs_y = _const("qs_Y", np.array(0.10, dtype=np.float32))

    qx = ir.Value(name="qX")
    qw = ir.Value(name="qW")
    mm = ir.Value(name="mm")
    mm_b = ir.Value(name="mm_b")
    bn_m = ir.Value(name="bn_m")
    bn_n = ir.Value(name="bn_n")
    r = ir.Value(name="r")
    qy = ir.Value(name="qy", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 3]))

    q_attrs = {
        "signed": ir.Attr("signed", ir.AttributeType.INT, 1),
        "narrow": ir.Attr("narrow", ir.AttributeType.INT, 0),
        "rounding_mode": ir.Attr("rounding_mode", ir.AttributeType.STRING, "ROUND"),
    }
    qu_attrs = {
        "signed": ir.Attr("signed", ir.AttributeType.INT, 0),
        "narrow": ir.Attr("narrow", ir.AttributeType.INT, 0),
        "rounding_mode": ir.Attr("rounding_mode", ir.AttributeType.STRING, "ROUND"),
    }

    nodes = [
        ir.Node(QONNX_DOMAIN, "Quant", [x, qs_x, z_0, bw_4], attributes=q_attrs, outputs=[qx]),
        ir.Node(QONNX_DOMAIN, "Quant", [w_const, qs_w, z_0, bw_4], attributes=q_attrs, outputs=[qw]),
        ir.Node("", "MatMul", [qx, qw], outputs=[mm]),
        ir.Node("", "Add", [mm, b_const], outputs=[mm_b]),
        ir.Node("", "Mul", [mm_b, m_const], outputs=[bn_m]),
        ir.Node("", "Add", [bn_m, n_const], outputs=[bn_n]),
        ir.Node("", "Relu", [bn_n], outputs=[r]),
        ir.Node(QONNX_DOMAIN, "Quant", [r, qs_y, z_0, bw_4], attributes=qu_attrs, outputs=[qy]),
    ]
    model = _make_model(
        [x], [qy], nodes,
        initializers=[qs_x, z_0, bw_4, w_const, qs_w, b_const, m_const, n_const, qs_y],
        extra_opsets={QONNX_DOMAIN: 1},
    )

    x_lo = np.array([[-3.80, -2.10, -1.30]], dtype=np.float32)
    x_hi = np.array([[5.10, 5.00, 3.10]], dtype=np.float32)
    ranges = _run_sira(model, {"X": (x_lo, x_hi)})

    # Check qX, mm, bn_m, bn_n, r, qy are sound and scaled-integer where expected
    assert not np.isnan(ranges["qX"][2]).any()
    assert not np.isnan(ranges["mm"][2]).any()
    np.testing.assert_allclose(ranges["mm"][2], [[0.14, 0.21, 0.07]], atol=1e-5)
    np.testing.assert_allclose(ranges["bn_m"][2], [[0.084, 0.042, 0.028]], atol=1e-5)
    np.testing.assert_allclose(ranges["bn_n"][3], [[-2.18, -1.44, -1.34]], atol=1e-5)
    assert np.isnan(ranges["r"][2]).all()  # Relu breaks scaled-integer
    assert not np.isnan(ranges["qy"][2]).any()  # Quant restores scaled-integer
    assert compute_accumulator_bitwidth(ranges["mm"][0], ranges["mm"][1]) >= 8
    _assert_sound(model, ranges, {"X": (x_lo, x_hi)}, n_samples=10)


def test_conv_dense_and_depthwise():
    # Dense 2D Conv with padding and non-zero input bias
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2, 4, 4]))
    w = _const("w", np.array([[[[1, -1], [0, 1]], [[1, 0], [-1, 1]]], [[[-1, 1], [1, 0]], [[0, 1], [1, -1]]]], dtype=np.float32))
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2, 5, 5]))
    c_node = ir.Node("", "Conv", [x, w], attributes={"pads": ir.Attr("pads", ir.AttributeType.INTS, [1, 1, 1, 1])}, outputs=[y])
    model = _make_model([x], [y], [c_node], initializers=[w])
    lo = np.full((1, 2, 4, 4), -2.0, dtype=np.float32)
    hi = np.full((1, 2, 4, 4), 3.0, dtype=np.float32)
    x_sc = {"x": (np.array(0.5, dtype=np.float32), np.array(0.25, dtype=np.float32))}
    ranges = _run_sira(model, {"x": (lo, hi)}, input_scales=x_sc)
    _assert_sound(model, ranges, {"x": (lo, hi)}, input_scales=x_sc, n_samples=5)

    # Depthwise 2D Conv with per-channel input scale
    x_dw = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2, 4, 4]))
    w_dw = _const("w_dw", np.ones((2, 1, 3, 3), dtype=np.float32))
    y_dw = ir.Value(name="y_dw", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2, 2, 2]))
    dw_node = ir.Node("", "Conv", [x_dw, w_dw], attributes={"group": ir.Attr("group", ir.AttributeType.INT, 2)}, outputs=[y_dw])
    model_dw = _make_model([x_dw], [y_dw], [dw_node], initializers=[w_dw])
    dw_sc = {"x": (np.array([[[[0.25]], [[0.5]]]], dtype=np.float32), np.array([[[[0.1]], [[-0.2]]]], dtype=np.float32))}
    r_dw = _run_sira(model_dw, {"x": (lo, hi)}, input_scales=dw_sc)
    # Note: w_dw is not quantized here, so y_dw falls back to real-valued; check soundness
    _assert_sound(model_dw, r_dw, {"x": (lo, hi)}, input_scales=dw_sc, n_samples=5)


def test_softmax_and_pooling():
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 4]))
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 4]))
    model = _make_model([x], [y], [ir.Node("", "Softmax", [x], attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, 1)}, outputs=[y])])
    lo = np.array([[-1.0, 0.0, -2.0, 0.5]], dtype=np.float32)
    hi = np.array([[1.0, 2.0, 0.0, 1.5]], dtype=np.float32)
    ranges = _run_sira(model, {"x": (lo, hi)})
    ymin, ymax, ys, yb = ranges["y"]
    assert np.all(ymin >= 0.0) and np.all(ymax <= 1.0)
    assert np.isnan(ys).all()
    _assert_sound(model, ranges, {"x": (lo, hi)}, n_samples=10)


def _qonnx_zoo_models() -> list[Path]:
    """Collect .onnx models under $QONNX_MODEL_ZOO (empty if unset)."""
    root = os.environ.get("QONNX_MODEL_ZOO")
    if not root:
        return []
    return sorted(Path(root).expanduser().rglob("*.onnx"))


_QONNX_ZOO_MODELS = _qonnx_zoo_models()


@pytest.mark.skipif(
    not _QONNX_ZOO_MODELS,
    reason="set QONNX_MODEL_ZOO to the QONNX model zoo's models/ directory",
)
@pytest.mark.parametrize("model_path", _QONNX_ZOO_MODELS, ids=lambda p: p.stem)
def test_qonnx_model_zoo(model_path: Path):
    """Run SIRA RangeAnalysis_v1 on every model in the QONNX model zoo."""
    orig = ir.load(model_path)
    out_names = [o.name for o in orig.graph.outputs]
    dyn_inputs = [i for i in orig.graph.inputs if i.const_value is None]
    inp_ranges = {}
    for i in dyn_inputs:
        shape = [int(d) if isinstance(d, int) else 1 for d in i.shape.dims]
        inp_ranges[i.name] = (
            np.full(shape, -1.0, dtype=np.float32),
            np.full(shape, 1.0, dtype=np.float32),
        )

    rm = RangeAnalysis_v1()(orig).model

    ranges = evaluate_ranges(rm, inp_ranges, backend="onnxruntime", intermediates=True)
    assert len(ranges) > 0
    for out_name in out_names:
        vmin, vmax, vscale, vbias = ranges[out_name]
        assert vmin.shape == vmax.shape
        assert np.all(vmin <= vmax + 1e-5)
