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
    from tests.zoo import QONNX_ZOO_MODELS, build_input_dicts, generate_samples, infer_zoo_input_spec
except ImportError:
    from test_sira import _CUSTOM_OPS
    from zoo import QONNX_ZOO_MODELS, build_input_dicts, generate_samples, infer_zoo_input_spec


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
    not QONNX_ZOO_MODELS,
    reason="set QONNX_MODEL_ZOO to the QONNX model zoo's models/ directory",
)
@pytest.mark.parametrize("model_path", QONNX_ZOO_MODELS, ids=lambda p: p.stem)
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
