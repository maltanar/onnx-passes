"""Comprehensive SIRA instrumentation & soundness campaign for the QONNX Model Zoo.

Run via pytest:
    QONNX_MODEL_ZOO=/data/sandbox/qonnx_model_zoo/models pytest -v tests/test_sira_zoo.py

Or standalone for full campaign reporting:
    QONNX_MODEL_ZOO=/data/sandbox/qonnx_model_zoo/models python tests/test_sira_zoo.py \\
        --samples 16 --report sira_zoo_report.md --json sira_zoo_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator

from onnx_passes.passes.sira import RangeAnalysis_v1, evaluate_ranges

try:
    from tests.test_sira import _CUSTOM_OPS
except ImportError:
    from test_sira import _CUSTOM_OPS


# ---------------------------------------------------------------------------
# Zoo Discovery & Reasoned Input Specifications
# ---------------------------------------------------------------------------

def _qonnx_zoo_models() -> list[Path]:
    """Collect .onnx models under $QONNX_MODEL_ZOO (empty if unset)."""
    root = os.environ.get("QONNX_MODEL_ZOO")
    if not root:
        return []
    return sorted(Path(root).expanduser().rglob("*.onnx"))


_QONNX_ZOO_MODELS = _qonnx_zoo_models()


@dataclass(frozen=True)
class ZooInputSpec:
    """Input domain specification for a QONNX model zoo entry."""

    lo: float
    hi: float
    scale: float | None = None
    bias: float | None = None
    rationale: str = ""


def infer_zoo_input_spec(model_path: Path) -> ZooInputSpec:
    """Return a reasoned (possibly scaled-integer) input range for a zoo model."""
    stem = model_path.stem.lower()
    if stem.startswith("cnv_"):
        # Brevitas FINN CNV (CIFAR-10 / GTSRB): first ops are x * 2.0 - 1.0 -> Quant(8-bit)
        # Standard uint8/255.0 ToTensor() image input in [0, 1].
        return ZooInputSpec(
            lo=0.0, hi=255.0, scale=1.0 / 255.0, bias=0.0,
            rationale="uint8/255 image input ([0,1] -> x*2-1 -> 8-bit Quant)",
        )
    if stem.startswith(("lfc_", "tfc_")):
        # Brevitas FINN MNIST LFC/TFC: ToTensor() uint8/255.0 in [0, 1]
        return ZooInputSpec(
            lo=0.0, hi=255.0, scale=1.0 / 255.0, bias=0.0,
            rationale="MNIST uint8/255 image input in [0, 1]",
        )
    if "unsw_nb15" in stem:
        # First ops are (x + 1.0) / 2.0; README specifies 1-bit bipolar {-1, +1} input.
        # Represented as integer q in {0, 1} with scale=2.0, bias=-1.0.
        return ZooInputSpec(
            lo=0.0, hi=1.0, scale=2.0, bias=-1.0,
            rationale="1-bit bipolar {-1, +1} input ((x+1)/2 preprocessing)",
        )
    if stem in ("2w2a_rn8_pow2", "3w3a_rn8_fp"):
        # Input named quant_input; README specifies 8-bit quantized CIFAR-10 input.
        return ZooInputSpec(
            lo=-128.0, hi=127.0, scale=1.0 / 64.0, bias=0.0,
            rationale="8-bit quantized normalized CIFAR-10 input (quant_input)",
        )
    if "mobilenet" in stem:
        # README specifies 8-bit input quantization on normalized ImageNet.
        return ZooInputSpec(
            lo=-128.0, hi=127.0, scale=1.0 / 64.0, bias=0.0,
            rationale="8-bit quantized normalized ImageNet input",
        )
    if "resnet18" in stem:
        # Brevitas FloatQuant ResNet-18 on normalized CIFAR-10.
        return ZooInputSpec(
            lo=-2.5, hi=2.5,
            rationale="Normalized CIFAR-10 float input fed to FloatQuant",
        )
    if "cloudsatnet" in stem:
        # Landsat-8 EXSI fed to Quant(scale=0.005241, bw=8) -> max ~0.665.
        return ZooInputSpec(
            lo=-1.0, hi=1.0,
            rationale="Landsat-8 EXSI input covering Quant(bw=8, scale=0.005241)",
        )
    if "kwsmlp" in stem:
        # GSCV2 MFCC features fed to Quant(scale=0.82985, bw=8) -> max ~105.4.
        return ZooInputSpec(
            lo=-106.0, hi=106.0,
            rationale="GSCV2 MFCC features covering Quant(bw=8, scale=0.82985)",
        )
    if "jettagging" in stem:
        # Standardized LHC jet features.
        return ZooInputSpec(
            lo=-3.0, hi=3.0,
            rationale="Standardized LHC jet features",
        )
    return ZooInputSpec(lo=-1.0, hi=1.0, rationale="Default [-1, 1] range")


def build_input_dicts(
    model: ir.Model,
    spec: ZooInputSpec,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, tuple[np.ndarray, np.ndarray]] | None]:
    """Build input_ranges and optional input_scales dicts for dynamic model inputs."""
    dyn_inputs = [i for i in model.graph.inputs if i.const_value is None]
    inp_ranges: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    inp_scales: dict[str, tuple[np.ndarray, np.ndarray]] | None = (
        {} if spec.scale is not None and spec.bias is not None else None
    )
    for inp in dyn_inputs:
        shape = [int(d) if isinstance(d, int) else 1 for d in inp.shape.dims]
        inp_ranges[inp.name] = (
            np.full(shape, spec.lo, dtype=np.float32),
            np.full(shape, spec.hi, dtype=np.float32),
        )
        if inp_scales is not None:
            inp_scales[inp.name] = (
                np.array(spec.scale, dtype=np.float32),
                np.array(spec.bias, dtype=np.float32),
            )
    return inp_ranges, inp_scales


# ---------------------------------------------------------------------------
# Sample Generation (Corners, Midpoint, Rademacher, Weight-Aligned, Uniform)
# ---------------------------------------------------------------------------

def _first_linear_weight_sign(model: ir.Model, input_name: str, input_shape: list[int]) -> np.ndarray | None:
    """Heuristic: find first Conv/MatMul/Gemm weight sign aligned to input channels."""
    inits = {
        k: v.const_value.numpy()
        for k, v in model.graph.initializers.items()
        if v.const_value is not None
    }
    for node in model.graph:
        if node.op_type in ("Conv", "Gemm", "MatMul") and len(node.inputs) >= 2:
            w_name = node.inputs[1].name
            if w_name not in inits:
                continue
            w = inits[w_name]
            if node.op_type == "Conv" and w.ndim >= 2 and len(input_shape) >= 2 and w.shape[1] == input_shape[1]:
                ch_sign = np.sign(w.sum(axis=tuple(i for i in range(w.ndim) if i != 1)))
                bshape = [1, input_shape[1]] + [1] * (len(input_shape) - 2)
                return ch_sign.reshape(bshape)
            if node.op_type in ("Gemm", "MatMul") and w.ndim == 2 and len(input_shape) >= 2:
                feat = input_shape[-1]
                if w.shape[0] == feat:
                    return np.sign(w.sum(axis=1)).reshape([1] * (len(input_shape) - 1) + [feat])
                if w.shape[1] == feat:
                    return np.sign(w.sum(axis=0)).reshape([1] * (len(input_shape) - 1) + [feat])
    return None


def generate_samples(
    model: ir.Model,
    input_ranges: dict[str, tuple[np.ndarray, np.ndarray]],
    input_scales: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    n_samples: int,
    rng_seed: int = 42,
) -> list[dict[str, np.ndarray]]:
    """Generate diverse corner, Rademacher, weight-aligned, and uniform samples."""
    rng = np.random.default_rng(rng_seed)
    samples: list[dict[str, np.ndarray]] = []
    w_signs = {
        name: _first_linear_weight_sign(model, name, list(lo.shape))
        for name, (lo, _) in input_ranges.items()
    }

    for idx in range(n_samples):
        feed: dict[str, np.ndarray] = {}
        for name, (lo, hi) in input_ranges.items():
            is_si = input_scales is not None and name in input_scales
            if idx == 0:
                arr = lo
            elif idx == 1:
                arr = hi
            elif idx == 2:
                arr = np.round((lo + hi) * 0.5) if is_si else (lo + hi) * 0.5
            elif idx == 3 and w_signs[name] is not None:
                arr = np.where(w_signs[name] >= 0, hi, lo)
            elif idx == 4 and w_signs[name] is not None:
                arr = np.where(w_signs[name] >= 0, lo, hi)
            elif idx % 2 == 1 and idx < max(6, n_samples // 2):
                # Rademacher / mixed-corner sampling maximizes variance for accumulators
                mask = rng.integers(0, 2, size=lo.shape, dtype=bool)
                arr = np.where(mask, hi, lo)
            else:
                if is_si:
                    arr = rng.integers(lo.astype(np.int64), hi.astype(np.int64) + 1).astype(np.float32)
                else:
                    arr = rng.uniform(lo, hi).astype(np.float32)

            if is_si:
                s, b = input_scales[name]
                feed[name] = (arr.astype(np.float32) * s + b).astype(np.float32)
            else:
                feed[name] = arr.astype(np.float32)
        samples.append(feed)
    return samples


# ---------------------------------------------------------------------------
# Campaign Instrumentation & Verification
# ---------------------------------------------------------------------------

@dataclass
class ZooModelReport:
    model: str
    input_rationale: str
    scaled_int_input: bool
    intermediates_checked: bool
    n_samples: int
    total_tensors: int
    scaled_int_tensors: int
    scaled_int_pct: float
    max_sira_bitwidth: int | None
    max_emp_bitwidth: int | None
    median_tightness: float | None
    min_safety_margin: float
    worst_margin_tensor: str
    fallback_ops: dict[str, int] = field(default_factory=dict)
    has_per_channel_variation: bool = False
    reload_parity_ok: bool = True
    eval_seconds: float = 0.0


def _fallback_op_audit(orig: ir.Model, ranges: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> dict[str, int]:
    """Count original op types whose outputs fell back to non-scaled-integer."""
    counts: dict[str, int] = {}
    for node in orig.graph:
        for out in node.outputs:
            if out.name in ranges:
                _, _, vs, vb = ranges[out.name]
                if np.isnan(vs).any() or np.isnan(vb).any():
                    counts[node.op_type] = counts.get(node.op_type, 0) + 1
                    break
    return counts


def run_zoo_campaign_model(
    model_path: Path,
    n_samples: int = 8,
    intermediates: bool = True,
    check_reload_parity: bool = True,
    atol: float = 1e-3,
) -> ZooModelReport:
    """Run SIRA analysis, structural checks, and empirical campaign on one zoo model."""
    t0 = time.perf_counter()
    orig = ir.load(model_path)
    out_names = [o.name for o in orig.graph.outputs]
    spec = infer_zoo_input_spec(model_path)
    inp_ranges, inp_scales = build_input_dicts(orig, spec)

    rm = RangeAnalysis_v1()(orig.clone(deep_copy=True)).model
    ranges = evaluate_ranges(
        rm, inp_ranges, input_scales=inp_scales, backend="onnxruntime", intermediates=intermediates,
    )
    assert len(ranges) > 0

    # 1. Bound validity & integer-bound checks
    si_count = 0
    max_sira_bw: int | None = None
    has_per_channel = False
    for tname, (vmin, vmax, vscale, vbias) in ranges.items():
        if np.issubdtype(vmin.dtype, np.floating):
            assert not np.isnan(vmin).any(), f"NaN in vmin of {tname}"
            assert not np.isnan(vmax).any(), f"NaN in vmax of {tname}"
        assert np.all(vmin <= vmax + 1e-5), f"Inverted range on {tname}"

        if vmin.size > 1 and (np.ptp(vmin) > 1e-6 or np.ptp(vmax) > 1e-6):
            has_per_channel = True

        if not (np.isnan(vscale).any() or np.isnan(vbias).any()):
            si_count += 1
            np.testing.assert_allclose(
                vmin, np.round(vmin), atol=1e-4, rtol=1e-4,
                err_msg=f"Non-integer vmin on scaled-int tensor {tname}",
            )
            np.testing.assert_allclose(
                vmax, np.round(vmax), atol=1e-4, rtol=1e-4,
                err_msg=f"Non-integer vmax on scaled-int tensor {tname}",
            )
            span = float(np.max(vmax - vmin))
            bw = int(np.ceil(np.log2(max(span + 1.0, 1.0))))
            max_sira_bw = bw if max_sira_bw is None else max(max_sira_bw, bw)

    # 2. Propagation coverage assertions
    stem = model_path.stem.lower()
    if intermediates:
        if "float" not in stem and "jettagging" not in stem:
            assert si_count > 0, f"Expected scaled-integer propagation in {model_path.name}"
            for out_name in out_names:
                assert not np.isnan(ranges[out_name][2]).any(), f"Expected scaled-int output on {model_path.name}"
        elif "jettagging" in stem:
            assert si_count > 0
            assert np.isnan(ranges[out_names[0]][2]).all()

    # 3. Save/reload parity check (fast output-only check)
    reload_ok = True
    if check_reload_parity:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "range_model.onnx"
            ir.save(rm, p)
            rm_reloaded = ir.load(p)
            r_orig = evaluate_ranges(rm, inp_ranges, input_scales=inp_scales, backend="onnxruntime", intermediates=False)
            r_rel = evaluate_ranges(rm_reloaded, inp_ranges, input_scales=inp_scales, backend="onnxruntime", intermediates=False)
            assert set(r_orig.keys()) == set(r_rel.keys())
            for k in r_orig:
                for arr_a, arr_b in zip(r_orig[k], r_rel[k]):
                    np.testing.assert_allclose(arr_a, arr_b, equal_nan=True, atol=1e-6, rtol=1e-6)

    # 4. Empirical soundness + tightness/margin/integer-grid instrumentation
    m_ref = orig.clone(deep_copy=True)
    for n in m_ref.graph:
        if n.domain:
            m_ref.graph.opset_imports.setdefault(n.domain, 1)
    sess = ReferenceEvaluator(ir.to_proto(m_ref), new_ops=_CUSTOM_OPS)

    samples = generate_samples(orig, inp_ranges, inp_scales, n_samples=n_samples)
    emp_mins: dict[str, np.ndarray] = {}
    emp_maxs: dict[str, np.ndarray] = {}

    for feed in samples:
        res = sess.run(None, feed, intermediate=intermediates)
        ctx = res if intermediates else {name: val for name, val in zip(out_names, res)}
        for tname, (vmin, vmax, vscale, vbias) in ranges.items():
            if tname not in ctx:
                continue
            val = np.asarray(ctx[tname], dtype=np.float32)
            emp_mins[tname] = val.copy() if tname not in emp_mins else np.minimum(emp_mins[tname], val)
            emp_maxs[tname] = val.copy() if tname not in emp_maxs else np.maximum(emp_maxs[tname], val)

            if not (np.isnan(vscale).any() or np.isnan(vbias).any()):
                a = vmin * vscale + vbias
                b = vmax * vscale + vbias
                rmin = np.minimum(a, b)
                rmax = np.maximum(a, b)
                nz = np.broadcast_to(vscale != 0, val.shape)
                if np.any(nz):
                    q = np.broadcast_to(val - vbias, val.shape)[nz] / np.broadcast_to(vscale, val.shape)[nz]
                    q_round = np.round(q)
                    recon = q_round * np.broadcast_to(vscale, val.shape)[nz] + np.broadcast_to(vbias, val.shape)[nz]
                    np.testing.assert_allclose(
                        np.broadcast_to(val, val.shape)[nz], recon, atol=1e-4, rtol=1e-4,
                        err_msg=f"Non-integer q for {tname}",
                    )
                    q_lo = np.broadcast_to(vmin, val.shape)[nz]
                    q_hi = np.broadcast_to(vmax, val.shape)[nz]
                    assert np.all(q_round >= q_lo - 1.0) and np.all(q_round <= q_hi + 1.0), (
                        f"Integer q out of [vmin, vmax] for {tname}"
                    )
            else:
                rmin, rmax = vmin, vmax

            assert np.all(val >= rmin - atol), f"Lower bound violated for {tname}: {val.min()} < {rmin.min()}"
            assert np.all(val <= rmax + atol), f"Upper bound violated for {tname}: {val.max()} > {rmax.max()}"

    # 5. Aggregate tightness & safety margins
    tightness_vals: list[float] = []
    min_margin = float("inf")
    worst_tensor = ""
    max_emp_bw: int | None = None

    for tname, (vmin, vmax, vscale, vbias) in ranges.items():
        if tname not in emp_mins:
            continue
        is_si = not (np.isnan(vscale).any() or np.isnan(vbias).any())
        if is_si:
            a = vmin * vscale + vbias
            b = vmax * vscale + vbias
            rmin, rmax = np.minimum(a, b), np.maximum(a, b)
            nz = np.broadcast_to(vscale != 0, emp_mins[tname].shape)
            if np.any(nz):
                q_min = np.round(np.broadcast_to(emp_mins[tname] - vbias, emp_mins[tname].shape)[nz] / np.broadcast_to(vscale, emp_mins[tname].shape)[nz])
                q_max = np.round(np.broadcast_to(emp_maxs[tname] - vbias, emp_maxs[tname].shape)[nz] / np.broadcast_to(vscale, emp_maxs[tname].shape)[nz])
                emp_span = float(np.max(np.maximum(q_max - q_min, np.abs(q_max - q_min))))
                ebw = int(np.ceil(np.log2(max(emp_span + 1.0, 1.0))))
                max_emp_bw = ebw if max_emp_bw is None else max(max_emp_bw, ebw)
        else:
            rmin, rmax = vmin, vmax

        margin = float(min(np.min(emp_mins[tname] - rmin), np.min(rmax - emp_maxs[tname])))
        if margin < min_margin:
            min_margin = margin
            worst_tensor = tname

        width = rmax - rmin
        emp_width = emp_maxs[tname] - emp_mins[tname]
        pos = width > 1e-6
        if np.any(pos):
            ratio = float(np.median(np.broadcast_to(emp_width, width.shape)[pos] / width[pos]))
            tightness_vals.append(min(ratio, 1.0))

    elapsed = time.perf_counter() - t0
    return ZooModelReport(
        model=model_path.name,
        input_rationale=spec.rationale,
        scaled_int_input=inp_scales is not None,
        intermediates_checked=intermediates,
        n_samples=n_samples,
        total_tensors=len(ranges),
        scaled_int_tensors=si_count,
        scaled_int_pct=round(100.0 * si_count / len(ranges), 1),
        max_sira_bitwidth=max_sira_bw,
        max_emp_bitwidth=max_emp_bw,
        median_tightness=round(float(np.median(tightness_vals)), 4) if tightness_vals else None,
        min_safety_margin=round(min_margin, 6) if min_margin != float("inf") else 0.0,
        worst_margin_tensor=worst_tensor,
        fallback_ops=_fallback_op_audit(orig, ranges),
        has_per_channel_variation=has_per_channel,
        reload_parity_ok=reload_ok,
        eval_seconds=round(elapsed, 2),
    )


@pytest.mark.skipif(
    not _QONNX_ZOO_MODELS,
    reason="set QONNX_MODEL_ZOO to the QONNX model zoo's models/ directory",
)
@pytest.mark.parametrize("model_path", _QONNX_ZOO_MODELS, ids=lambda p: p.stem)
def test_qonnx_model_zoo(model_path: Path):
    """Run SIRA instrumentation campaign on every QONNX model zoo entry."""
    stem = model_path.stem.lower()
    is_large = any(k in stem for k in ("resnet18", "mobilenet", "cloudsatnet"))
    intermediates = os.environ.get("SIRA_ZOO_INTERMEDIATES", "1") != "0"
    default_samples = 3 if is_large else 8
    n_samples = int(os.environ.get("SIRA_ZOO_SAMPLES", str(default_samples)))
    run_zoo_campaign_model(
        model_path,
        n_samples=n_samples,
        intermediates=intermediates,
        check_reload_parity=True,
    )


def _format_markdown_report(reports: list[ZooModelReport]) -> str:
    lines = [
        "# SIRA QONNX Model Zoo Instrumentation Report",
        "",
        "| Model | SI Input | Intermediates | Samples | SI Tensors | Max SIRA BW | Max Emp BW | Median Tightness | Min Margin | Time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        sira_bw = str(r.max_sira_bitwidth) if r.max_sira_bitwidth is not None else "-"
        emp_bw = str(r.max_emp_bitwidth) if r.max_emp_bitwidth is not None else "-"
        tight = f"{r.median_tightness:.3f}" if r.median_tightness is not None else "-"
        lines.append(
            f"| `{r.model}` | {'yes' if r.scaled_int_input else 'no'} | "
            f"{'yes' if r.intermediates_checked else 'no'} | {r.n_samples} | "
            f"{r.scaled_int_tensors}/{r.total_tensors} ({r.scaled_int_pct:.1f}%) | "
            f"{sira_bw} | {emp_bw} | {tight} | {r.min_safety_margin:.4g} | {r.eval_seconds:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SIRA QONNX Model Zoo instrumentation campaign.")
    parser.add_argument("--zoo-dir", type=Path, default=None, help="Path to QONNX model zoo models/ dir (defaults to $QONNX_MODEL_ZOO)")
    parser.add_argument("--samples", type=int, default=12, help="Number of samples per model")
    parser.add_argument("--no-intermediates", action="store_true", help="Evaluate graph outputs only (faster for large networks)")
    parser.add_argument("--no-reload-check", action="store_true", help="Skip ir.save/ir.load parity check")
    parser.add_argument("--report", type=Path, default=None, help="Optional Markdown report output path")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON report output path")
    args = parser.parse_args(argv)

    zoo_dir = args.zoo_dir or (Path(os.environ["QONNX_MODEL_ZOO"]) if "QONNX_MODEL_ZOO" in os.environ else None)
    if zoo_dir is None or not zoo_dir.exists():
        raise SystemExit("Set QONNX_MODEL_ZOO or pass --zoo-dir pointing to QONNX model zoo models/")

    models = sorted(zoo_dir.expanduser().rglob("*.onnx"))
    reports: list[ZooModelReport] = []
    for p in models:
        rep = run_zoo_campaign_model(
            p,
            n_samples=args.samples,
            intermediates=not args.no_intermediates,
            check_reload_parity=not args.no_reload_check,
        )
        reports.append(rep)
        print(
            f"{rep.model:36s} SI={rep.scaled_int_tensors}/{rep.total_tensors} ({rep.scaled_int_pct:5.1f}%) "
            f"BW={rep.max_sira_bitwidth}/{rep.max_emp_bitwidth} tight={rep.median_tightness} "
            f"margin={rep.min_safety_margin:.4g} ({rep.eval_seconds:.2f}s)"
        )

    md = _format_markdown_report(reports)
    if args.report:
        args.report.write_text(md)
    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in reports], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
