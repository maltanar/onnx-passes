import json
import os
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun

from onnx_passes.cli.sira import main as sira_cli_main
from onnx_passes.ops.qonnx import DOMAIN as QONNX_DOMAIN
from onnx_passes.passes.sira import (
    RangeAnalysis_v1,
    analyze_accumulators,
    compute_accumulator_bitwidth,
    detect_stuck_channels,
    evaluate_ranges,
    summarize_ranges,
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
    return evaluate_ranges(
        rm, input_ranges, input_scales=input_scales, backend=backend, intermediates=True
    )


# ---------------------------------------------------------------------------
# Lightweight ReferenceEvaluator OpRuns for QONNX/Brevitas/FINN custom ops
# ---------------------------------------------------------------------------

def _round_mode(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "ROUND":
        return np.round(x)
    if mode == "CEIL":
        return np.ceil(x)
    if mode == "FLOOR":
        return np.floor(x)
    if mode == "UP":
        return np.ceil(np.abs(x)) * np.sign(x)
    if mode == "DOWN":
        return np.floor(np.abs(x)) * np.sign(x)
    if mode == "HALF_UP":
        return (np.floor(np.abs(x)) + 0.5) * np.sign(x)
    if mode == "HALF_DOWN":
        return (np.ceil(np.abs(x)) - 0.5) * np.sign(x)
    return np.round(x)


def _make_quant_op(domain: str):
    class Quant(OpRun):
        op_domain = domain

        def _run(self, x, scale, zeropoint, bitwidth, signed=1, narrow=0, rounding_mode="ROUND"):
            bw = np.asarray(bitwidth, dtype=np.float32)
            s = np.asarray(scale, dtype=np.float32)
            z = np.asarray(zeropoint, dtype=np.float32)
            x_f = np.asarray(x, dtype=np.float32)
            if int(signed) == 1 and float(np.max(bw)) == 1.0 and float(np.min(bw)) == 1.0:
                q_raw = x_f / s + z
                q = np.where(q_raw >= 0.0, 1.0, -1.0).astype(np.float32)
            else:
                qmin = (-2.0 ** (bw - signed) + narrow) * signed
                qmax = 2.0 ** (bw - signed) - 1.0 - narrow * (1 - signed)
                q = _round_mode(np.clip(x_f / s + z, qmin, qmax), rounding_mode)
            return (((q - z) * s).astype(np.float32),)

    return Quant


def _make_bipolar_op(domain: str):
    class BipolarQuant(OpRun):
        op_domain = domain

        def _run(self, x, scale):
            s = np.asarray(scale, dtype=np.float32)
            q = np.where(np.asarray(x, dtype=np.float32) >= 0.0, 1.0, -1.0).astype(np.float32)
            return ((q * s).astype(np.float32),)

    return BipolarQuant


def _make_trunc_op(domain: str):
    class Trunc(OpRun):
        op_domain = domain

        def _run(self, x, scale, zeropoint, in_bw, out_bw, rounding_mode="FLOOR"):
            s = np.asarray(scale, dtype=np.float32)
            z = np.asarray(zeropoint, dtype=np.float32)
            trunc_scale = 2.0 ** (np.asarray(in_bw, dtype=np.float32) - np.asarray(out_bw, dtype=np.float32))
            q = _round_mode(np.round(np.asarray(x, dtype=np.float32) / s + z) / trunc_scale, rounding_mode)
            return (((q - z) * s).astype(np.float32),)

    return Trunc


def _make_float_quant_op(domain: str):
    class FloatQuant(OpRun):
        op_domain = domain

        def _run(self, x, scale, exp_bw, man_bw, exp_bias, max_val, has_inf=0, has_nan=0, has_subnormal=1, rounding_mode="round", saturation=1):
            s = np.asarray(scale, dtype=np.float32)
            ebw = np.asarray(exp_bw, dtype=np.float32)
            mbw = np.asarray(man_bw, dtype=np.float32)
            ebias = np.asarray(exp_bias, dtype=np.float32)
            mval = np.asarray(max_val, dtype=np.float32)
            fmt_max = np.minimum((2.0 - 2.0 ** (-mbw)) * (2.0 ** (2.0 ** ebw - 1.0 - ebias)), mval)
            xs = np.asarray(x, dtype=np.float32) / s
            eps = np.float32(1.17549435e-38)
            fp_min_exp = 1.0 - ebias - mbw
            int_exp = np.maximum(np.floor(np.log(np.abs(xs) + eps) / np.log(np.float32(2.0))) - mbw, fp_min_exp)
            int_scale = 2.0 ** int_exp
            xq = int_scale * np.round(xs / int_scale)
            return ((np.clip(xq, -fmt_max, fmt_max) * s).astype(np.float32),)

    return FloatQuant


def _make_batchnorm_op():
    class BatchNormalization(OpRun):
        op_domain = ""

        def _run(self, x, scale, b, mean, var, epsilon=1e-5, momentum=0.9, spatial=1, training_mode=0):
            s = np.asarray(scale, dtype=np.float32)
            bias = np.asarray(b, dtype=np.float32)
            m = np.asarray(mean, dtype=np.float32)
            v = np.asarray(var, dtype=np.float32)
            shape = (1, -1) + (1,) * (x.ndim - 2)
            y = (np.asarray(x, dtype=np.float32) - m.reshape(shape)) / np.sqrt(v.reshape(shape) + np.float32(epsilon)) * s.reshape(shape) + bias.reshape(shape)
            return (y.astype(np.float32),)

    return BatchNormalization


_CUSTOM_OPS = [
    _make_batchnorm_op(),
    _make_quant_op("qonnx.custom_op.general"),
    _make_quant_op("onnx.brevitas"),
    _make_quant_op("finn.custom_op.general"),
    _make_bipolar_op("qonnx.custom_op.general"),
    _make_bipolar_op("onnx.brevitas"),
    _make_trunc_op("onnx.brevitas"),
    _make_trunc_op("qonnx.custom_op.general"),
    _make_float_quant_op("qonnx.custom_op.general"),
]


def _assert_sound(
    orig_model: ir.Model,
    ranges: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    input_ranges: dict[str, tuple[np.ndarray, np.ndarray]],
    input_scales: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    n_samples: int = 5,
    rng_seed: int = 42,
    atol: float = 1e-4,
):
    """Sample inputs (including corners) inside input_ranges and assert every tensor lies in its SIRA range."""
    rng = np.random.default_rng(rng_seed)
    m = orig_model.clone(deep_copy=True)
    for n in m.graph:
        if n.domain:
            m.graph.opset_imports.setdefault(n.domain, 1)
    sess = ReferenceEvaluator(ir.to_proto(m), new_ops=_CUSTOM_OPS)

    for sample_idx in range(n_samples):
        feed = {}
        for name, (lo, hi) in input_ranges.items():
            if input_scales and name in input_scales:
                s, b = input_scales[name]
                if sample_idx == 0:
                    q = lo.astype(np.int64)
                elif sample_idx == 1 and n_samples > 1:
                    q = hi.astype(np.int64)
                else:
                    q = rng.integers(lo.astype(np.int64), hi.astype(np.int64) + 1)
                feed[name] = (q * s + b).astype(np.float32)
            else:
                if sample_idx == 0:
                    feed[name] = lo.astype(np.float32)
                elif sample_idx == 1 and n_samples > 1:
                    feed[name] = hi.astype(np.float32)
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
                nz = np.broadcast_to(vscale != 0, val.shape)
                if np.any(nz):
                    q = np.broadcast_to(val - vbias, val.shape)[nz] / np.broadcast_to(vscale, val.shape)[nz]
                    recon = np.round(q) * np.broadcast_to(vscale, val.shape)[nz] + np.broadcast_to(vbias, val.shape)[nz]
                    np.testing.assert_allclose(
                        np.broadcast_to(val, val.shape)[nz], recon, atol=1e-4, rtol=1e-4,
                        err_msg=f"Non-integer q for {tname}",
                    )
            else:
                rmin, rmax = vmin, vmax
            assert np.all(val >= rmin - atol), f"Lower bound violated for {tname}: {val.min()} < {rmin.min()}"
            assert np.all(val <= rmax + atol), f"Upper bound violated for {tname}: {val.max()} > {rmax.max()}"


# ---------------------------------------------------------------------------
# Unit Tests for SIRA Handlers
# ---------------------------------------------------------------------------

def test_quant_signed_unsigned_bipolar_and_neg_scale():
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    s = _const("s", np.array([0.5, -0.1], dtype=np.float32))
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
    np.testing.assert_allclose(ys, [0.5, -0.1])
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
    _assert_sound(model, ranges, {"x": (lo, hi)})


def test_trunc_and_float_quant():
    # Trunc: input quantized with 8-bit, truncated to 4-bit
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    s = _const("s", np.array(0.25, dtype=np.float32))
    z = _const("z", np.array(0.0, dtype=np.float32))
    in_bw = _const("in_bw", np.array(8.0, dtype=np.float32))
    out_bw = _const("out_bw", np.array(4.0, dtype=np.float32))
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    t_node = ir.Node(
        "onnx.brevitas", "Trunc", inputs=[x, s, z, in_bw, out_bw], outputs=[y],
        attributes={"rounding_mode": ir.Attr("rounding_mode", ir.AttributeType.STRING, "FLOOR")},
    )
    model_t = _make_model([x], [y], [t_node], initializers=[s, z, in_bw, out_bw], extra_opsets={"onnx.brevitas": 1})
    lo = np.array([[-4.0, 0.0]], dtype=np.float32)
    hi = np.array([[4.0, 8.0]], dtype=np.float32)
    r_t = _run_sira(model_t, {"x": (lo, hi)})
    ymin, ymax, ys, yb = r_t["y"]
    np.testing.assert_allclose(ymin, [[-1.0, 0.0]])
    np.testing.assert_allclose(ymax, [[1.0, 2.0]])
    np.testing.assert_allclose(ys, 0.25)
    _assert_sound(model_t, r_t, {"x": (lo, hi)})

    # FloatQuant: E3M1 -> non-scaled-integer (NaN scale/bias)
    xf = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    sf = _const("s", np.array(1.0, dtype=np.float32))
    ebw = _const("ebw", np.array(3.0, dtype=np.float32))
    mbw = _const("mbw", np.array(1.0, dtype=np.float32))
    ebias = _const("ebias", np.array(3.0, dtype=np.float32))
    mval = _const("mval", np.array(24.0, dtype=np.float32))
    yf = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    fq_node = ir.Node(QONNX_DOMAIN, "FloatQuant", inputs=[xf, sf, ebw, mbw, ebias, mval], outputs=[yf])
    model_fq = _make_model([xf], [yf], [fq_node], initializers=[sf, ebw, mbw, ebias, mval], extra_opsets={QONNX_DOMAIN: 2})
    r_fq = _run_sira(model_fq, {"x": (lo, hi)})
    assert np.isnan(r_fq["y"][2]).all() and np.isnan(r_fq["y"][3]).all()
    _assert_sound(model_fq, r_fq, {"x": (lo, hi)})


def test_add_sub_neg_cases():
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

    # Sub & Neg
    a2 = ir.Value(name="a", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    b2 = ir.Value(name="b", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    sub_out = ir.Value(name="sub_out")
    neg_out = ir.Value(name="neg_out", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    model_sub = _make_model(
        [a2, b2], [neg_out],
        [ir.Node("", "Sub", [a2, b2], outputs=[sub_out]), ir.Node("", "Neg", [sub_out], outputs=[neg_out])],
    )
    r_sub = _run_sira(
        model_sub,
        {
            "a": (np.array([[-2, 0]], dtype=np.float32), np.array([[3, 4]], dtype=np.float32)),
            "b": (np.array([[-1, 1]], dtype=np.float32), np.array([[2, 2]], dtype=np.float32)),
        },
        input_scales={
            "a": (np.array(0.25, dtype=np.float32), np.array(0.1, dtype=np.float32)),
            "b": (np.array(0.5, dtype=np.float32), np.array(-0.2, dtype=np.float32)),
        },
    )
    assert not np.isnan(r_sub["neg_out"][2]).any()
    _assert_sound(
        model_sub, r_sub,
        {
            "a": (np.array([[-2, 0]], dtype=np.float32), np.array([[3, 4]], dtype=np.float32)),
            "b": (np.array([[-1, 1]], dtype=np.float32), np.array([[2, 2]], dtype=np.float32)),
        },
        input_scales={
            "a": (np.array(0.25, dtype=np.float32), np.array(0.1, dtype=np.float32)),
            "b": (np.array(0.5, dtype=np.float32), np.array(-0.2, dtype=np.float32)),
        },
    )


def test_mul_div_cases():
    # Mul with negative and positive constants
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    c = _const("c", np.array([-2.0, 3.0], dtype=np.float32))
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    model = _make_model([x], [y], [ir.Node("", "Mul", inputs=[x, c], outputs=[y])], initializers=[c])
    sc = {"x": (np.array(0.5, dtype=np.float32), np.array(1.0, dtype=np.float32))}
    lo = np.array([[-4, 1]], dtype=np.float32)
    hi = np.array([[2, 5]], dtype=np.float32)
    ranges = _run_sira(model, {"x": (lo, hi)}, input_scales=sc)
    ymin, ymax, ys, yb = ranges["y"]
    np.testing.assert_allclose(ymin, [[-4, 1]])
    np.testing.assert_allclose(ymax, [[2, 5]])
    np.testing.assert_allclose(ys, [[-1.0, 1.5]])
    np.testing.assert_allclose(yb, [[-2.0, 3.0]])
    _assert_sound(model, ranges, {"x": (lo, hi)}, input_scales=sc)

    # Div by constant
    xd = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    cd = _const("c", np.array([2.0, -4.0], dtype=np.float32))
    yd = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    model_d = _make_model([xd], [yd], [ir.Node("", "Div", inputs=[xd, cd], outputs=[yd])], initializers=[cd])
    rd = _run_sira(model_d, {"x": (lo, hi)}, input_scales=sc)
    np.testing.assert_allclose(rd["y"][2], [[0.25, -0.125]])
    np.testing.assert_allclose(rd["y"][3], [[0.5, -0.25]])
    _assert_sound(model_d, rd, {"x": (lo, hi)}, input_scales=sc)

    # Div by dynamic interval containing 0 -> [-inf, +inf]
    a = ir.Value(name="a", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    b = ir.Value(name="b", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    z = ir.Value(name="z", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    model_dz = _make_model([a, b], [z], [ir.Node("", "Div", inputs=[a, b], outputs=[z])])
    rdz = _run_sira(
        model_dz,
        {
            "a": (np.array([[1.0, 2.0]], dtype=np.float32), np.array([[3.0, 4.0]], dtype=np.float32)),
            "b": (np.array([[-1.0, -1.0]], dtype=np.float32), np.array([[1.0, 1.0]], dtype=np.float32)),
        },
    )
    assert np.all(np.isneginf(rdz["z"][0])) and np.all(np.isposinf(rdz["z"][1]))
    assert np.isnan(rdz["z"][2]).all()


def test_gemm_variants():
    # Gemm with transA=1, transB=1, alpha=0.5, beta=2.0, and bias C
    a = ir.Value(name="a", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([3, 1]))
    b = _const("b", np.array([[1.0, -2.0, 0.5], [-1.0, 1.0, 2.0]], dtype=np.float32))
    c = _const("c", np.array([0.5, -1.0], dtype=np.float32))
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    g_node = ir.Node(
        "", "Gemm", inputs=[a, b, c], outputs=[y],
        attributes={
            "alpha": ir.Attr("alpha", ir.AttributeType.FLOAT, 0.5),
            "beta": ir.Attr("beta", ir.AttributeType.FLOAT, 2.0),
            "transA": ir.Attr("transA", ir.AttributeType.INT, 1),
            "transB": ir.Attr("transB", ir.AttributeType.INT, 1),
        },
    )
    model = _make_model([a], [y], [g_node], initializers=[b, c])
    lo = np.array([[-1.0], [0.0], [-2.0]], dtype=np.float32)
    hi = np.array([[2.0], [1.0], [1.0]], dtype=np.float32)
    ranges = _run_sira(model, {"a": (lo, hi)})
    _assert_sound(model, ranges, {"a": (lo, hi)})


def test_sira_paper_layer_example():
    """Verify SIRA paper Section 3.3 (Figure 5, Table 2 & Table 3) numerical example."""
    x = ir.Value(name="X", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 3]))
    qs_x = _const("qs_X", np.array(0.7, dtype=np.float32))
    z_0 = _const("z_0", np.array(0.0, dtype=np.float32))
    bw_4 = _const("bw_4", np.array(4.0, dtype=np.float32))

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
    # Dense 2D Conv with quantized weights, padding, and non-zero input bias
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2, 4, 4]))
    w_raw = _const("w_raw", np.array([[[[0.5, -0.5], [0.0, 0.5]], [[0.5, 0.0], [-0.5, 0.5]]], [[[-0.5, 0.5], [0.5, 0.0]], [[0.0, 0.5], [0.5, -0.5]]]], dtype=np.float32))
    ws = _const("ws", np.array([[[[0.5]]], [[[0.25]]]], dtype=np.float32))
    wz = _const("wz", np.array(0.0, dtype=np.float32))
    wbw = _const("wbw", np.array(4.0, dtype=np.float32))
    wq = ir.Value(name="wq")
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2, 5, 5]))
    q_node = ir.Node(QONNX_DOMAIN, "Quant", [w_raw, ws, wz, wbw], outputs=[wq])
    c_node = ir.Node("", "Conv", [x, wq], attributes={"pads": ir.Attr("pads", ir.AttributeType.INTS, [1, 1, 1, 1])}, outputs=[y])
    model = _make_model([x], [y], [q_node, c_node], initializers=[w_raw, ws, wz, wbw], extra_opsets={QONNX_DOMAIN: 1})
    lo = np.full((1, 2, 4, 4), -2.0, dtype=np.float32)
    hi = np.full((1, 2, 4, 4), 3.0, dtype=np.float32)
    x_sc = {"x": (np.array(0.5, dtype=np.float32), np.array(0.25, dtype=np.float32))}
    ranges = _run_sira(model, {"x": (lo, hi)}, input_scales=x_sc)
    assert not np.isnan(ranges["y"][2]).any()
    _assert_sound(model, ranges, {"x": (lo, hi)}, input_scales=x_sc, n_samples=5)

    # Dense Conv with per-channel input scale -> must fall back to NaN scale/bias
    x_pc_sc = {"x": (np.array([[[[0.25]], [[0.5]]]], dtype=np.float32), np.array([[[[0.0]], [[0.0]]]], dtype=np.float32))}
    r_fb = _run_sira(model, {"x": (lo, hi)}, input_scales=x_pc_sc)
    assert np.isnan(r_fb["y"][2]).all()
    _assert_sound(model, r_fb, {"x": (lo, hi)}, input_scales=x_pc_sc, n_samples=5)

    # Depthwise 2D Conv with per-channel input scale & quantized weights -> propagates!
    x_dw = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2, 4, 4]))
    w_dw_raw = _const("w_dw_raw", np.ones((2, 1, 3, 3), dtype=np.float32))
    w_dw_s = _const("w_dw_s", np.array([[[[0.5]]], [[[0.25]]]], dtype=np.float32))
    w_dw_z = _const("w_dw_z", np.array(0.0, dtype=np.float32))
    w_dw_bw = _const("w_dw_bw", np.array(4.0, dtype=np.float32))
    w_dw_q = ir.Value(name="w_dw_q")
    y_dw = ir.Value(name="y_dw", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2, 2, 2]))
    q_dw_node = ir.Node(QONNX_DOMAIN, "Quant", [w_dw_raw, w_dw_s, w_dw_z, w_dw_bw], outputs=[w_dw_q])
    dw_node = ir.Node("", "Conv", [x_dw, w_dw_q], attributes={"group": ir.Attr("group", ir.AttributeType.INT, 2)}, outputs=[y_dw])
    model_dw = _make_model([x_dw], [y_dw], [q_dw_node, dw_node], initializers=[w_dw_raw, w_dw_s, w_dw_z, w_dw_bw], extra_opsets={QONNX_DOMAIN: 1})
    dw_sc = {"x": (np.array([[[[0.25]], [[0.5]]]], dtype=np.float32), np.array([[[[0.1]], [[-0.2]]]], dtype=np.float32))}
    r_dw = _run_sira(model_dw, {"x": (lo, hi)}, input_scales=dw_sc)
    assert not np.isnan(r_dw["y_dw"][2]).any()
    _assert_sound(model_dw, r_dw, {"x": (lo, hi)}, input_scales=dw_sc, n_samples=5)


def test_batchnorm_and_pooling_and_pad():
    # BatchNorm -> AveragePool (unpadded) -> MaxPool -> Pad
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2, 4, 4]))
    s = _const("s", np.array([0.5, -1.0], dtype=np.float32))
    b = _const("b", np.array([0.1, 0.2], dtype=np.float32))
    m = _const("m", np.array([0.0, 0.5], dtype=np.float32))
    v = _const("v", np.array([0.25, 1.0], dtype=np.float32))
    bn_out = ir.Value(name="bn_out", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2, 4, 4]))
    ap_out = ir.Value(name="ap_out", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2, 2, 2]))
    mp_out = ir.Value(name="mp_out", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2, 1, 1]))
    pads = _const("pads", np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64))
    pad_val = _const("pad_val", np.array(0.0, dtype=np.float32))
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2, 3, 3]))
    nodes = [
        ir.Node("", "BatchNormalization", [x, s, b, m, v], outputs=[bn_out]),
        ir.Node("", "AveragePool", [bn_out], attributes={"kernel_shape": ir.Attr("kernel_shape", ir.AttributeType.INTS, [2, 2]), "strides": ir.Attr("strides", ir.AttributeType.INTS, [2, 2])}, outputs=[ap_out]),
        ir.Node("", "MaxPool", [ap_out], attributes={"kernel_shape": ir.Attr("kernel_shape", ir.AttributeType.INTS, [2, 2]), "strides": ir.Attr("strides", ir.AttributeType.INTS, [2, 2])}, outputs=[mp_out]),
        ir.Node("", "Pad", [mp_out, pads, pad_val], outputs=[y]),
    ]
    model = _make_model([x], [y], nodes, initializers=[s, b, m, v, pads, pad_val])
    lo = np.full((1, 2, 4, 4), -2, dtype=np.float32)
    hi = np.full((1, 2, 4, 4), 3, dtype=np.float32)
    sc = {"x": (np.array(0.5, dtype=np.float32), np.array(0.0, dtype=np.float32))}
    ranges = _run_sira(model, {"x": (lo, hi)}, input_scales=sc)
    assert not np.isnan(ranges["bn_out"][2]).any()
    assert not np.isnan(ranges["ap_out"][2]).any()
    # Channel 1 has negative scale -> MaxPool falls back to real-valued
    assert np.isnan(ranges["mp_out"][2]).all()
    assert np.isnan(ranges["y"][2]).all()
    _assert_sound(model, ranges, {"x": (lo, hi)}, input_scales=sc, n_samples=5)


def test_elementwise_nonlinear_and_softmax():
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 4]))
    lo_c = _const("lo_c", np.array(-1.5, dtype=np.float32))
    hi_c = _const("hi_c", np.array(1.5, dtype=np.float32))
    p_c = _const("p_c", np.array(2.0, dtype=np.float32))

    c_out = ir.Value(name="c_out")
    a_out = ir.Value(name="a_out")
    p_out = ir.Value(name="p_out")
    s_out = ir.Value(name="s_out", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 4]))
    nodes = [
        ir.Node("", "Clip", [x, lo_c, hi_c], outputs=[c_out]),
        ir.Node("", "Abs", [c_out], outputs=[a_out]),
        ir.Node("", "Pow", [a_out, p_c], outputs=[p_out]),
        ir.Node("", "Softmax", [p_out], attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, 1)}, outputs=[s_out]),
    ]
    model = _make_model([x], [s_out], nodes, initializers=[lo_c, hi_c, p_c])
    lo = np.array([[-2.0, -0.5, 0.2, 0.5]], dtype=np.float32)
    hi = np.array([[1.0, 2.0, 0.8, 1.5]], dtype=np.float32)
    ranges = _run_sira(model, {"x": (lo, hi)})
    assert np.all(ranges["a_out"][0] >= 0.0)
    assert np.all(ranges["s_out"][0] >= 0.0) and np.all(ranges["s_out"][1] <= 1.0)
    assert np.isnan(ranges["s_out"][2]).all()
    _assert_sound(model, ranges, {"x": (lo, hi)}, n_samples=10)


def test_shape_layout_and_backend_parity():
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2, 3]))
    perm_shape = _const("new_shape", np.array([1, 6], dtype=np.int64))
    t_out = ir.Value(name="t_out")
    r_out = ir.Value(name="r_out")
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 6]))
    nodes = [
        ir.Node("", "Transpose", [x], attributes={"perm": ir.Attr("perm", ir.AttributeType.INTS, [0, 2, 1])}, outputs=[t_out]),
        ir.Node("", "Reshape", [t_out, perm_shape], outputs=[r_out]),
        ir.Node("", "Identity", [r_out], outputs=[y]),
    ]
    model = _make_model([x], [y], nodes, initializers=[perm_shape])
    lo = np.full((1, 2, 3), -3, dtype=np.float32)
    hi = np.full((1, 2, 3), 3, dtype=np.float32)
    sc = {"x": (np.array([[[0.5], [0.25]]], dtype=np.float32), np.array([[[0.1], [-0.1]]], dtype=np.float32))}
    r_ort = _run_sira(model, {"x": (lo, hi)}, input_scales=sc, backend="onnxruntime")
    r_ref = _run_sira(model, {"x": (lo, hi)}, input_scales=sc, backend="reference")
    for k in ("t_out", "r_out", "y"):
        assert not np.isnan(r_ort[k][2]).any()
        for idx in range(4):
            np.testing.assert_allclose(r_ort[k][idx], r_ref[k][idx], atol=1e-6)
    _assert_sound(model, r_ort, {"x": (lo, hi)}, input_scales=sc, n_samples=5)


def test_name_preservation_roundtrip(tmp_path: Path):
    """Verify original tensor and node names survive canonical renaming and .onnx save/reload."""
    x = ir.Value(name="orig_in", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    w = _const("orig_w", np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32))
    b = _const("orig_b", np.array([0.2, -0.3], dtype=np.float32))
    mm = ir.Value(name="orig_mm")
    y = ir.Value(name="orig_out", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    nodes = [
        ir.Node("", "Gemm", [x, w, b], outputs=[mm], name="orig_gemm_node"),
        ir.Node("", "Relu", [mm], outputs=[y], name="orig_relu_node"),
    ]
    model = _make_model([x], [y], nodes, initializers=[w, b])

    rm = RangeAnalysis_v1()(model).model
    onnx_path = tmp_path / "range_model.onnx"
    ir.save(rm, onnx_path)
    reloaded = ir.load(onnx_path)

    # Check node provenance on handler nodes and model-level sira_node_map
    node_origins = {
        n.name: n.metadata_props.get("sira_origin")
        for n in reloaded.graph
        if n.domain == "onnx_passes.ops.sira"
    }
    assert set(node_origins.values()) == {"orig_gemm_node", "orig_relu_node"}
    node_map = json.loads(reloaded.metadata_props["sira_node_map"])
    assert "orig_gemm_node" in node_map and len(node_map["orig_gemm_node"]) == 2  # MatMul + Add
    assert "orig_relu_node" in node_map and len(node_map["orig_relu_node"]) == 1

    # Check evaluate_ranges works on reloaded model using original tensor names
    lo = np.array([[-1.0, -2.0]], dtype=np.float32)
    hi = np.array([[2.0, 1.0]], dtype=np.float32)
    ranges = evaluate_ranges(reloaded, {"orig_in": (lo, hi)}, backend="onnxruntime", intermediates=True)
    for tname in ("orig_in", "orig_mm", "orig_out"):
        assert tname in ranges
        assert np.all(ranges[tname][0] <= ranges[tname][1] + 1e-5)


def test_accumulator_sizing_and_stuck_channels():
    # Channel 0 is active, Channel 1 has bias=-20 so Relu clamps it to 0 (stuck!)
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    w_raw = _const("w_raw", np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32))
    ws = _const("ws", np.array(1.0, dtype=np.float32))
    wz = _const("wz", np.array(0.0, dtype=np.float32))
    wbw = _const("wbw", np.array(4.0, dtype=np.float32))
    wq = ir.Value(name="wq", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 2]))
    b = _const("b", np.array([0.0, -20.0], dtype=np.float32))
    mm = ir.Value(name="mm", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    add_out = ir.Value(name="add_out", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    nodes = [
        ir.Node(QONNX_DOMAIN, "Quant", [w_raw, ws, wz, wbw], attributes={"signed": ir.Attr("signed", ir.AttributeType.INT, 1), "narrow": ir.Attr("narrow", ir.AttributeType.INT, 0)}, outputs=[wq], name="wq_node"),
        ir.Node("", "MatMul", [x, wq], outputs=[mm], name="mm_node"),
        ir.Node("", "Add", [mm, b], outputs=[add_out], name="add_node"),
        ir.Node("", "Relu", [add_out], outputs=[y], name="relu_node"),
    ]
    model = _make_model([x], [y], nodes, initializers=[w_raw, ws, wz, wbw, b], extra_opsets={QONNX_DOMAIN: 1})

    lo = np.array([[-3.0, -3.0]], dtype=np.float32)
    hi = np.array([[3.0, 3.0]], dtype=np.float32)
    sc = {"x": (np.array(0.5, dtype=np.float32), np.array(0.0, dtype=np.float32))}
    ranges = _run_sira(model, {"x": (lo, hi)}, input_scales=sc)

    acc = analyze_accumulators(model, ranges)
    assert len(acc) == 1
    assert acc[0]["node"] == "mm_node"
    assert acc[0]["is_scaled_int"] is True
    # [-6, 6] needs 4 bits ([-8, 7])
    assert acc[0]["bitwidth"] == 4
    assert acc[0]["per_channel_bitwidths"] == [4, 4]

    stuck = detect_stuck_channels(model, ranges)
    assert "mm" not in stuck
    assert stuck["y"]["stuck_channels"] == [1]
    assert stuck["y"]["total_channels"] == 2


def test_summarize_ranges_and_cli(tmp_path):
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    w_raw = _const("w_raw", np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32))
    ws = _const("ws", np.array(1.0, dtype=np.float32))
    wz = _const("wz", np.array(0.0, dtype=np.float32))
    wbw = _const("wbw", np.array(4.0, dtype=np.float32))
    wq = ir.Value(name="wq", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 2]))
    b = _const("b", np.array([0.0, -20.0], dtype=np.float32))
    mm = ir.Value(name="mm", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    add_out = ir.Value(name="add_out", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    nodes = [
        ir.Node(QONNX_DOMAIN, "Quant", [w_raw, ws, wz, wbw], attributes={"signed": ir.Attr("signed", ir.AttributeType.INT, 1), "narrow": ir.Attr("narrow", ir.AttributeType.INT, 0)}, outputs=[wq]),
        ir.Node("", "MatMul", [x, wq], outputs=[mm]),
        ir.Node("", "Add", [mm, b], outputs=[add_out]),
        ir.Node("", "Relu", [add_out], outputs=[y]),
    ]
    model = _make_model([x], [y], nodes, initializers=[w_raw, ws, wz, wbw, b], extra_opsets={QONNX_DOMAIN: 1})
    model_path = tmp_path / "model.onnx"
    ir.save(model, model_path)

    # Non-existent channel_axis=3 on 2D tensors must not error out
    ranges = _run_sira(model, {"x": (np.full((1, 2), -2.0, np.float32), np.full((1, 2), 2.0, np.float32))},
                       input_scales={"x": (np.array(0.5, np.float32), np.array(0.0, np.float32))})
    assert analyze_accumulators(model, ranges, channel_axis=3)[0]["per_channel_bitwidths"] is None
    assert detect_stuck_channels(model, ranges, channel_axis=3) == {}
    ch_fallback = summarize_ranges(ranges, granularity="channel", channel_axis=3)
    assert "channel_real_min" not in ch_fallback["y"]

    # CLI end-to-end with channel granularity + Markdown + JSON
    md_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"
    rc = sira_cli_main([
        str(model_path),
        "--min", "-3", "--max", "3",
        "--input-scale", "0.5", "--input-bias", "0.0",
        "--granularity", "channel",
        "--channel-axis", "1",
        "--report", str(md_path),
        "--json", str(json_path),
    ])
    assert rc == 0
    assert md_path.exists() and "Accumulator Sizing" in md_path.read_text()
    data = json.loads(json_path.read_text())
    assert data["accumulators"][0]["bitwidth"] == 4
    assert data["stuck_channels"]["y"]["stuck_channels"] == [1]
    assert data["ranges"]["y"]["channel_real_max"] == [3.0, 0.0]


def test_round_floor_ceil_integer_outputs():
    for op_type, exp_lo, exp_hi in [
        ("Round", -2.0, 2.0),
        ("Floor", -2.0, 2.0),
        ("Ceil", -1.0, 3.0),
    ]:
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 3]))
        y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 3]))
        model = _make_model([x], [y], [ir.Node("", op_type, [x], outputs=[y])])

        # Real-valued input -> scale=1, bias=0
        lo_f = np.full((1, 3), -1.7, dtype=np.float32)
        hi_f = np.full((1, 3), 2.3, dtype=np.float32)
        r_real = _run_sira(model, {"x": (lo_f, hi_f)})
        np.testing.assert_allclose(r_real["y"][0], exp_lo)
        np.testing.assert_allclose(r_real["y"][1], exp_hi)
        np.testing.assert_allclose(r_real["y"][2], 1.0)
        np.testing.assert_allclose(r_real["y"][3], 0.0)
        _assert_sound(model, r_real, {"x": (lo_f, hi_f)}, n_samples=6)

        # Scaled-int input with non-integer scale -> scale=1, bias=0
        lo_q = np.full((1, 3), -3.0, dtype=np.float32)
        hi_q = np.full((1, 3), 4.0, dtype=np.float32)
        sc = {"x": (np.array(0.35, dtype=np.float32), np.array(0.1, dtype=np.float32))}
        r_si = _run_sira(model, {"x": (lo_q, hi_q)}, input_scales=sc)
        np.testing.assert_allclose(r_si["y"][2], 1.0)
        np.testing.assert_allclose(r_si["y"][3], 0.0)
        _assert_sound(model, r_si, {"x": (lo_q, hi_q)}, input_scales=sc, n_samples=6)


def test_max_min_scaled_int_cases():
    # 1. Grid-aligned constant (either input position) -> scaled-int kept
    for const_first in (False, True):
        for op_type in ("Max", "Min"):
            x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
            c = _const("c", np.array(0.25, dtype=np.float32))  # 0.25 = 0.5 * 1 - 0.25 (on grid)
            y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
            ins = [c, x] if const_first else [x, c]
            m = _make_model([x], [y], [ir.Node("", op_type, ins, outputs=[y])], initializers=[c])
            lo = np.full((1, 2), -2.0, dtype=np.float32)
            hi = np.full((1, 2), 3.0, dtype=np.float32)
            sc = {"x": (np.array(0.5, dtype=np.float32), np.array(-0.25, dtype=np.float32))}
            r = _run_sira(m, {"x": (lo, hi)}, input_scales=sc)
            np.testing.assert_allclose(r["y"][2], 0.5)
            np.testing.assert_allclose(r["y"][3], -0.25)
            if op_type == "Max":
                np.testing.assert_allclose(r["y"][0], 1.0)
                np.testing.assert_allclose(r["y"][1], 3.0)
            else:
                np.testing.assert_allclose(r["y"][0], -2.0)
                np.testing.assert_allclose(r["y"][1], 1.0)
            _assert_sound(m, r, {"x": (lo, hi)}, input_scales=sc, n_samples=6)

    # 2. Off-grid active constant -> falls back to NaN
    x2 = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    c_off = _const("c_off", np.array(0.3, dtype=np.float32))
    y2 = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    m_off = _make_model([x2], [y2], [ir.Node("", "Max", [x2, c_off], outputs=[y2])], initializers=[c_off])
    lo = np.full((1, 2), -2.0, dtype=np.float32)
    hi = np.full((1, 2), 3.0, dtype=np.float32)
    sc = {"x": (np.array(0.5, dtype=np.float32), np.array(0.0, dtype=np.float32))}
    r_off = _run_sira(m_off, {"x": (lo, hi)}, input_scales=sc)
    assert np.isnan(r_off["y"][2]).all()
    _assert_sound(m_off, r_off, {"x": (lo, hi)}, input_scales=sc, n_samples=5)

    # 3. Off-grid inactive constant -> scaled-int kept
    x3 = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    y3 = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    c_inact = _const("c_inact", np.array(-10.3, dtype=np.float32))
    m_inact = _make_model([x3], [y3], [ir.Node("", "Max", [x3, c_inact], outputs=[y3])], initializers=[c_inact])
    r_inact = _run_sira(m_inact, {"x": (lo, hi)}, input_scales=sc)
    np.testing.assert_allclose(r_inact["y"][2], 0.5)
    np.testing.assert_allclose(r_inact["y"][0], -2.0)
    np.testing.assert_allclose(r_inact["y"][1], 3.0)
    _assert_sound(m_inact, r_inact, {"x": (lo, hi)}, input_scales=sc, n_samples=5)

    # 4. Negative scale and per-channel mixed-sign scales
    x4 = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    y4 = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    c_mix = _const("c_mix", np.array([1.0, 1.0], dtype=np.float32))
    m_mix = _make_model([x4], [y4], [ir.Node("", "Max", [x4, c_mix], outputs=[y4])], initializers=[c_mix])
    sc_neg = {"x": (np.array(-0.5, dtype=np.float32), np.array(0.0, dtype=np.float32))}
    r_neg = _run_sira(m_mix, {"x": (lo, hi)}, input_scales=sc_neg)
    np.testing.assert_allclose(r_neg["y"][2], -0.5)
    np.testing.assert_allclose(r_neg["y"][0], -2.0)
    np.testing.assert_allclose(r_neg["y"][1], -2.0)
    _assert_sound(m_mix, r_neg, {"x": (lo, hi)}, input_scales=sc_neg, n_samples=5)

    sc_mix = {"x": (np.array([[0.5, -0.5]], dtype=np.float32), np.array([[0.0, 0.0]], dtype=np.float32))}
    r_mix = _run_sira(m_mix, {"x": (lo, hi)}, input_scales=sc_mix)
    np.testing.assert_allclose(r_mix["y"][2], np.array([[0.5, -0.5]], dtype=np.float32))
    np.testing.assert_allclose(r_mix["y"][0], np.array([[2.0, -2.0]], dtype=np.float32))
    np.testing.assert_allclose(r_mix["y"][1], np.array([[3.0, -2.0]], dtype=np.float32))
    _assert_sound(m_mix, r_mix, {"x": (lo, hi)}, input_scales=sc_mix, n_samples=5)


def test_relu_and_clip_scaled_int():
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    m_relu = _make_model([x], [y], [ir.Node("", "Relu", [x], outputs=[y])])
    lo = np.full((1, 2), -3.0, dtype=np.float32)
    hi = np.full((1, 2), 4.0, dtype=np.float32)

    # Zero point on grid (bias = -0.5, scale = 0.5 -> k = 1) -> scaled-int kept
    sc_on = {"x": (np.array(0.5, dtype=np.float32), np.array(-0.5, dtype=np.float32))}
    r_on = _run_sira(m_relu, {"x": (lo, hi)}, input_scales=sc_on)
    np.testing.assert_allclose(r_on["y"][2], 0.5)
    np.testing.assert_allclose(r_on["y"][3], -0.5)
    np.testing.assert_allclose(r_on["y"][0], 1.0)
    np.testing.assert_allclose(r_on["y"][1], 4.0)
    _assert_sound(m_relu, r_on, {"x": (lo, hi)}, input_scales=sc_on, n_samples=5)

    # Zero point off grid (bias = 0.2, scale = 0.5) -> falls back to NaN
    sc_off = {"x": (np.array(0.5, dtype=np.float32), np.array(0.2, dtype=np.float32))}
    r_off = _run_sira(m_relu, {"x": (lo, hi)}, input_scales=sc_off)
    assert np.isnan(r_off["y"][2]).all()
    _assert_sound(m_relu, r_off, {"x": (lo, hi)}, input_scales=sc_off, n_samples=5)

    # Clip with grid-aligned bounds
    xc = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    yc = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    lo_c = _const("lo_c", np.array(-1.0, dtype=np.float32))
    hi_c = _const("hi_c", np.array(1.5, dtype=np.float32))
    m_clip = _make_model([xc], [yc], [ir.Node("", "Clip", [xc, lo_c, hi_c], outputs=[yc])], initializers=[lo_c, hi_c])
    sc_clip = {"x": (np.array(0.5, dtype=np.float32), np.array(0.0, dtype=np.float32))}
    r_clip = _run_sira(m_clip, {"x": (lo, hi)}, input_scales=sc_clip)
    np.testing.assert_allclose(r_clip["y"][2], 0.5)
    np.testing.assert_allclose(r_clip["y"][0], -2.0)
    np.testing.assert_allclose(r_clip["y"][1], 3.0)
    _assert_sound(m_clip, r_clip, {"x": (lo, hi)}, input_scales=sc_clip, n_samples=5)


def test_maxpool_positive_vs_negative_scale():
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2, 4, 4]))
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2, 2, 2]))
    mp = ir.Node("", "MaxPool", [x], attributes={"kernel_shape": ir.Attr("kernel_shape", ir.AttributeType.INTS, [2, 2]), "strides": ir.Attr("strides", ir.AttributeType.INTS, [2, 2])}, outputs=[y])
    model = _make_model([x], [y], [mp])
    lo = np.full((1, 2, 4, 4), -2.0, dtype=np.float32)
    hi = np.full((1, 2, 4, 4), 3.0, dtype=np.float32)

    # Per-channel positive scale -> scaled-int kept
    sc_pos = {"x": (np.array([[[[0.25]], [[0.5]]]], dtype=np.float32), np.array([[[[0.1]], [[-0.2]]]], dtype=np.float32))}
    r_pos = _run_sira(model, {"x": (lo, hi)}, input_scales=sc_pos)
    np.testing.assert_allclose(r_pos["y"][2], sc_pos["x"][0])
    np.testing.assert_allclose(r_pos["y"][3], sc_pos["x"][1])
    _assert_sound(model, r_pos, {"x": (lo, hi)}, input_scales=sc_pos, n_samples=4)

    # Negative scale -> falls back to NaN
    sc_neg = {"x": (np.array([[[[0.25]], [[-0.5]]]], dtype=np.float32), np.array(0.0, dtype=np.float32))}
    r_neg = _run_sira(model, {"x": (lo, hi)}, input_scales=sc_neg)
    assert np.isnan(r_neg["y"][2]).all()
    _assert_sound(model, r_neg, {"x": (lo, hi)}, input_scales=sc_neg, n_samples=4)


def test_decomposed_inlined_quantizer_end_to_end():
    # x -> Mul(1/s) -> Round -> Max(q_min) -> Min(q_max) -> MatMul(W_int)
    s_val = np.float32(0.25)
    inv_s = _const("inv_s", np.array(1.0 / s_val, dtype=np.float32))
    q_lo = _const("q_lo", np.array(-8.0, dtype=np.float32))
    q_hi = _const("q_hi", np.array(7.0, dtype=np.float32))
    w_int = _const("w_int", np.array([[1.0, -2.0], [3.0, 1.0]], dtype=np.float32))

    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    t1 = ir.Value(name="t1", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    t2 = ir.Value(name="t2", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    t3 = ir.Value(name="t3", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    t4 = ir.Value(name="t4", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    w_q = ir.Value(name="w_q", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 2]))
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    nodes = [
        ir.Node("", "Mul", [x, inv_s], outputs=[t1]),
        ir.Node("", "Round", [t1], outputs=[t2]),
        ir.Node("", "Max", [t2, q_lo], outputs=[t3]),
        ir.Node("", "Min", [t3, q_hi], outputs=[t4]),
        ir.Node("", "Round", [w_int], outputs=[w_q]),
        ir.Node("", "MatMul", [t4, w_q], outputs=[y]),
    ]
    model = _make_model([x], [y], nodes, initializers=[inv_s, q_lo, q_hi, w_int])

    lo = np.full((1, 2), -1.0, dtype=np.float32)
    hi = np.full((1, 2), 2.5, dtype=np.float32)
    r = _run_sira(model, {"x": (lo, hi)})
    # MatMul input t4 must be plain integer (scale=1, bias=0)
    np.testing.assert_allclose(r["t4"][2], 1.0)
    np.testing.assert_allclose(r["t4"][3], 0.0)
    # MatMul output y must be plain integer (scale=1, bias=0)
    np.testing.assert_allclose(r["y"][2], 1.0)
    np.testing.assert_allclose(r["y"][3], 0.0)
    _assert_sound(model, r, {"x": (lo, hi)}, n_samples=6)

    # Cross-check integer bounds of t4 against equivalent Quant node
    qx_in = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    qs = _const("qs", np.array(s_val, dtype=np.float32))
    qz = _const("qz", np.array(0.0, dtype=np.float32))
    qbw = _const("qbw", np.array(4.0, dtype=np.float32))
    qy = ir.Value(name="qy", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    qmodel = _make_model(
        [qx_in], [qy],
        [ir.Node(QONNX_DOMAIN, "Quant", [qx_in, qs, qz, qbw], attributes={"signed": ir.Attr("signed", ir.AttributeType.INT, 1), "narrow": ir.Attr("narrow", ir.AttributeType.INT, 0)}, outputs=[qy])],
        initializers=[qs, qz, qbw], extra_opsets={QONNX_DOMAIN: 1},
    )
    rq = _run_sira(qmodel, {"x": (lo, hi)})
    np.testing.assert_allclose(r["t4"][0], rq["qy"][0])
    np.testing.assert_allclose(r["t4"][1], rq["qy"][1])


def test_comparisons_where_and_inlined_bipolar():
    for op_type in ("GreaterOrEqual", "Greater", "LessOrEqual", "Less", "Equal"):
        a = ir.Value(name="a", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
        b = _const("b", np.array(0.0, dtype=np.float32))
        y = ir.Value(name="y", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([1, 2]))
        m = _make_model([a], [y], [ir.Node("", op_type, [a, b], outputs=[y])], initializers=[b])

        # Undecided interval [-1, 1] vs 0 -> [0, 1], scale=1, bias=0
        lo = np.full((1, 2), -1.0, dtype=np.float32)
        hi = np.full((1, 2), 1.0, dtype=np.float32)
        r = _run_sira(m, {"a": (lo, hi)})
        np.testing.assert_allclose(r["y"][0], 0.0)
        np.testing.assert_allclose(r["y"][1], 1.0)
        np.testing.assert_allclose(r["y"][2], 1.0)
        np.testing.assert_allclose(r["y"][3], 0.0)
        _assert_sound(m, r, {"a": (lo, hi)}, n_samples=5)

        # Decided positive interval [1, 2] vs 0
        lo_p = np.full((1, 2), 1.0, dtype=np.float32)
        hi_p = np.full((1, 2), 2.0, dtype=np.float32)
        rp = _run_sira(m, {"a": (lo_p, hi_p)})
        exp = 1.0 if op_type in ("GreaterOrEqual", "Greater") else 0.0
        np.testing.assert_allclose(rp["y"][0], exp)
        np.testing.assert_allclose(rp["y"][1], exp)

    # End-to-end inlined BipolarQuant: Mul(Where(GreaterOrEqual(x, 0), +1, -1), s)
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    zero = _const("zero", np.array(0.0, dtype=np.float32))
    pos1 = _const("pos1", np.array(1.0, dtype=np.float32))
    neg1 = _const("neg1", np.array(-1.0, dtype=np.float32))
    s_c = _const("s_c", np.array(0.25, dtype=np.float32))
    cond = ir.Value(name="cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([1, 2]))
    w_out = ir.Value(name="w_out", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    m_bp = _make_model(
        [x], [y],
        [
            ir.Node("", "GreaterOrEqual", [x, zero], outputs=[cond]),
            ir.Node("", "Where", [cond, pos1, neg1], outputs=[w_out]),
            ir.Node("", "Mul", [w_out, s_c], outputs=[y]),
        ],
        initializers=[zero, pos1, neg1, s_c],
    )
    lo = np.full((1, 2), -2.0, dtype=np.float32)
    hi = np.full((1, 2), 2.0, dtype=np.float32)
    r_bp = _run_sira(m_bp, {"x": (lo, hi)})
    np.testing.assert_allclose(r_bp["w_out"][0], -1.0)
    np.testing.assert_allclose(r_bp["w_out"][1], 1.0)
    np.testing.assert_allclose(r_bp["w_out"][2], 1.0)
    np.testing.assert_allclose(r_bp["w_out"][3], 0.0)
    np.testing.assert_allclose(r_bp["y"][0], -1.0)
    np.testing.assert_allclose(r_bp["y"][1], 1.0)
    np.testing.assert_allclose(r_bp["y"][2], 0.25)
    np.testing.assert_allclose(r_bp["y"][3], 0.0)
    _assert_sound(m_bp, r_bp, {"x": (lo, hi)}, n_samples=6)

    # Decided Where passes chosen branch through; non-int constants fall back to NaN
    r_dec = _run_sira(m_bp, {"x": (np.full((1, 2), 1.0, np.float32), np.full((1, 2), 2.0, np.float32))})
    np.testing.assert_allclose(r_dec["y"][0], 0.25)
    np.testing.assert_allclose(r_dec["y"][1], 0.25)


def test_cast_and_castlike():
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    c1 = ir.Value(name="c1", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    t_untyped = ir.Value(name="t_untyped")
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
    one_f = _const("one_f", np.array(1.0, dtype=np.float32))
    m = _make_model(
        [x, t_untyped], [y],
        [
            ir.Node("", "Cast", [x], attributes={"to": ir.Attr("to", ir.AttributeType.INT, 1)}, outputs=[c1]),
            ir.Node("", "CastLike", [c1, t_untyped], outputs=[y]),
        ],
        initializers=[],
    )
    # Give t_untyped a float type on graph input only (CastLike ignores target dtype)
    t_untyped.type = ir.TensorType(ir.DataType.FLOAT)
    t_untyped.shape = ir.Shape([1, 2])
    lo = np.full((1, 2), -2.0, dtype=np.float32)
    hi = np.full((1, 2), 3.0, dtype=np.float32)
    sc = {
        "x": (np.array(0.5, dtype=np.float32), np.array(0.0, dtype=np.float32)),
        "t_untyped": (np.array(1.0, dtype=np.float32), np.array(0.0, dtype=np.float32)),
    }
    r = _run_sira(m, {"x": (lo, hi), "t_untyped": (lo, hi)}, input_scales=sc)
    np.testing.assert_allclose(r["c1"][2], 0.5)
    np.testing.assert_allclose(r["y"][2], 0.5)
    np.testing.assert_allclose(r["y"][0], -2.0)
    np.testing.assert_allclose(r["y"][1], 3.0)
    _assert_sound(m, r, {"x": (lo, hi), "t_untyped": (lo, hi)}, input_scales=sc, n_samples=4)



