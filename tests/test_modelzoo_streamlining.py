"""Two-phase streamlining campaign on the QONNX Model Zoo.

See docs/modelzoo_streamlining.md for the definitions of the phases, metrics and
pass criteria.

Run via pytest:
    QONNX_MODEL_ZOO=~/sandbox/qonnx_model_zoo/models pytest -v tests/test_modelzoo_streamlining.py

Or standalone for the campaign report:
    QONNX_MODEL_ZOO=~/sandbox/qonnx_model_zoo/models python tests/test_modelzoo_streamlining.py \\
        --report streamline_report.md --json streamline_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest

from onnx_passes.passes.sira import RangeAnalysis_v1, evaluate_ranges
from onnx_passes.passes.streamline import StreamlineScales_v1, StreamlineThresholds_v1

try:
    from tests.zoo import (
        QONNX_ZOO_MODELS, build_input_dicts, generate_samples, infer_zoo_input_spec,
        load_zoo_model, reference_session, requires_zoo,
    )
except ImportError:
    from zoo import (
        QONNX_ZOO_MODELS, build_input_dicts, generate_samples, infer_zoo_input_spec,
        load_zoo_model, reference_session, requires_zoo,
    )

# onnx-passes operator functions need a recent default opset
TARGET_OPSET = 19

LINEAR_OPS = {"Conv", "ConvInteger", "MatMul", "MatMulInteger", "Gemm"}
QUANTIZER_OPS = {"Quant", "BipolarQuant", "Trunc", "FloatQuant"}
# Quantizer cores remaining once quantizers are inlined (or not inlined at all)
RESIDUAL_QUANTIZER_OPS = QUANTIZER_OPS | {"Round", "Sign"}
# Comparisons not absorbed into a MultiThreshold are residual step quantizers,
# e.g., the inlined bipolar quantizer core Where(x >= 0, +1, -1)
RESIDUAL_STEP_OPS = {"GreaterOrEqual", "Greater", "LessOrEqual", "Less"}

# Equivalence: elementwise tolerance and the fraction of output elements which
# may mismatch (float reordering can flip a Round near a .5 tie) before failing
ATOL = 1e-3
RTOL = 1e-4
MAX_MISMATCH_FRACTION = 0.01

BUCKETS = ("int", "scaled-int", "non-int")


def streamlining_models() -> list[Path]:
    """Zoo models in scope: minifloat models are excluded (integer-only spec)."""
    return [p for p in QONNX_ZOO_MODELS if "FLOAT" not in p.stem.upper()]


def _is_large(model_path: Path) -> bool:
    stem = model_path.stem.lower()
    return any(k in stem for k in ("mobilenet", "cloudsatnet", "rn8"))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _sira_ranges(model: ir.Model, model_path: Path) -> dict:
    spec = infer_zoo_input_spec(model_path)
    inp_ranges, inp_scales = build_input_dicts(model, spec)
    rm = RangeAnalysis_v1()(model.clone(deep_copy=True)).model
    return evaluate_ranges(rm, inp_ranges, input_scales=inp_scales, intermediates=True)


def _value_bucket(value: ir.Value, ranges: dict) -> str:
    """Classify a tensor as int / scaled-int / non-int from SIRA ranges."""
    if value.name in ranges:
        vmin, vmax, vscale, vbias = ranges[value.name]
        if not (np.isnan(vscale).any() or np.isnan(vbias).any()):
            return "int" if np.all(vscale == 1) and np.all(vbias == 0) else "scaled-int"
        # Constants are (C, C, NaN, NaN): integer-valued constants are integers
        if np.array_equal(vmin, vmax) and np.all(vmin == np.round(vmin)):
            return "int"
        return "non-int"
    if value.const_value is not None:
        w = value.const_value.numpy()
        return "int" if np.all(w == np.round(w)) else "non-int"
    return "non-int"


def linear_op_buckets(model: ir.Model, ranges: dict) -> dict[str, int]:
    """Count linear operators by the worst bucket over their two data inputs."""
    counts = dict.fromkeys(BUCKETS, 0)
    for node in model.graph:
        if node.op_type not in LINEAR_OPS:
            continue
        kinds = [_value_bucket(v, ranges) for v in node.inputs[:2] if v is not None]
        counts[max(kinds, key=BUCKETS.index)] += 1
    return counts


def _const_derived(model: ir.Model) -> set[ir.Value]:
    """Values computed from initializers/constants only (e.g., weight paths)."""
    const = {v for v in model.graph.initializers.values()}
    for node in model.graph:  # topologically sorted
        ins = [v for v in node.inputs if v is not None]
        if node.op_type == "Constant" or (ins and all(v in const for v in ins)):
            const.update(node.outputs)
    return const


def activation_quantizers(model: ir.Model) -> int:
    """Quantizer nodes acting on activations (not on weights/constants)."""
    const = _const_derived(model)
    return sum(
        1 for n in model.graph
        if n.op_type in QUANTIZER_OPS and n.inputs[0] not in const
    )


def residual_quantizers(model: ir.Model) -> int:
    """Quantizers or quantizer cores still acting on activations."""
    const = _const_derived(model)
    return sum(
        1 for n in model.graph
        if n.op_type in RESIDUAL_QUANTIZER_OPS | RESIDUAL_STEP_OPS
        and n.inputs[0] not in const
    )


def op_histogram(model: ir.Model) -> dict[str, int]:
    return dict(Counter(n.op_type for n in model.graph))


# ---------------------------------------------------------------------------
# Equivalence
# ---------------------------------------------------------------------------

def _run(model: ir.Model, samples: list[dict[str, np.ndarray]]) -> list[list[np.ndarray]]:
    """Run samples keyed by original input order (passes rename graph inputs)."""
    sess = reference_session(model)
    names = [i.name for i in model.graph.inputs if i.const_value is None]
    return [sess.run(None, dict(zip(names, feed.values()))) for feed in samples]


def compare_outputs(expected, produced) -> tuple[float, float]:
    """Return (max abs error, mismatch fraction) over all outputs and samples."""
    max_err, mismatched, total = 0.0, 0, 0
    for exp_outs, out_outs in zip(expected, produced):
        for e, o in zip(exp_outs, out_outs):
            e, o = np.asarray(e, dtype=np.float64), np.asarray(o, dtype=np.float64)
            max_err = max(max_err, float(np.max(np.abs(e - o), initial=0.0)))
            mismatched += int(np.sum(~np.isclose(o, e, atol=ATOL, rtol=RTOL)))
            total += e.size
    return max_err, mismatched / max(total, 1)


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------

@dataclass
class StreamlineReport:
    model: str
    n_samples: int
    errors: dict[str, str] = field(default_factory=dict)
    linear_before: dict[str, int] | None = None
    linear_after_p1: dict[str, int] | None = None
    act_quantizers_before: int | None = None
    residual_after_p2: int | None = None
    multithresholds_after_p2: int | None = None
    max_abs_err_p1: float | None = None
    mismatch_p1: float | None = None
    max_abs_err_p2: float | None = None
    mismatch_p2: float | None = None
    ops_after_p1: dict[str, int] = field(default_factory=dict)
    ops_after_p2: dict[str, int] = field(default_factory=dict)
    seconds_p1: float | None = None
    seconds_p2: float | None = None


def _error(e: BaseException) -> str:
    while e.__cause__ is not None:
        e = e.__cause__
    return f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"


def run_streamlining_model(model_path: Path, n_samples: int) -> StreamlineReport:
    """Run both streamlining phases on one model, recording stage errors."""
    rep = StreamlineReport(model=model_path.name, n_samples=n_samples)

    def stage(name, fn):
        try:
            return fn()
        except Exception as e:  # noqa: record any failure per stage
            rep.errors[name] = _error(e)
            if os.environ.get("STREAMLINE_ZOO_TRACEBACK"):
                traceback.print_exc()
            return None

    orig = load_zoo_model(model_path, target_opset=TARGET_OPSET)
    spec = infer_zoo_input_spec(model_path)
    inp_ranges, inp_scales = build_input_dicts(orig, spec)
    samples = generate_samples(orig, inp_ranges, inp_scales, n_samples=n_samples)

    rep.act_quantizers_before = activation_quantizers(orig)
    ranges = stage("sira_before", lambda: _sira_ranges(orig, model_path))
    if ranges is not None:
        rep.linear_before = linear_op_buckets(orig, ranges)
    expected = stage("eval_original", lambda: _run(orig, samples))

    # Phase 1: Move scales and biases into the layer tails
    t0 = time.perf_counter()
    p1 = stage("phase1", lambda: StreamlineScales_v1()(orig.clone(deep_copy=True)).model)
    rep.seconds_p1 = round(time.perf_counter() - t0, 2)
    if p1 is None:
        return rep
    rep.ops_after_p1 = op_histogram(p1)
    ranges = stage("sira_after_p1", lambda: _sira_ranges(p1, model_path))
    if ranges is not None:
        rep.linear_after_p1 = linear_op_buckets(p1, ranges)
    if expected is not None and (out := stage("eval_p1", lambda: _run(p1, samples))) is not None:
        rep.max_abs_err_p1, rep.mismatch_p1 = compare_outputs(expected, out)

    # Phase 2: Layer tails to thresholds
    t0 = time.perf_counter()
    p2 = stage("phase2", lambda: StreamlineThresholds_v1()(p1.clone(deep_copy=True)).model)
    rep.seconds_p2 = round(time.perf_counter() - t0, 2)
    if p2 is None:
        return rep
    rep.ops_after_p2 = op_histogram(p2)
    rep.residual_after_p2 = residual_quantizers(p2)
    rep.multithresholds_after_p2 = rep.ops_after_p2.get("MultiThreshold", 0)
    if expected is not None and (out := stage("eval_p2", lambda: _run(p2, samples))) is not None:
        rep.max_abs_err_p2, rep.mismatch_p2 = compare_outputs(expected, out)

    return rep


# ---------------------------------------------------------------------------
# Pytest
# ---------------------------------------------------------------------------

@requires_zoo
@pytest.mark.parametrize("model_path", streamlining_models(), ids=lambda p: p.stem)
def test_modelzoo_streamlining(model_path: Path):
    """Both phases run, preserve the outputs and do not regress the metrics."""
    default = 1 if _is_large(model_path) else 3
    n_samples = int(os.environ.get("STREAMLINE_ZOO_SAMPLES", str(default)))
    rep = run_streamlining_model(model_path, n_samples)

    # 1. No stage raised
    assert not rep.errors, f"Stage errors: {rep.errors}"

    # 2. Numerical equivalence, tolerating rare rounding flips with a warning
    for phase, err, frac in (("phase1", rep.max_abs_err_p1, rep.mismatch_p1),
                             ("phase2", rep.max_abs_err_p2, rep.mismatch_p2)):
        assert frac <= MAX_MISMATCH_FRACTION, (
            f"{phase}: {frac:.2%} of output elements mismatch (max abs err {err:.4g})"
        )
        if frac > 0:
            warnings.warn(f"{model_path.name} {phase}: {frac:.2%} mismatch, max abs err {err:.4g}")

    # 3. No regression of the streamlining metrics
    assert rep.linear_after_p1["int"] >= rep.linear_before["int"], (
        f"Fewer int-only linear ops after phase 1: {rep.linear_before} -> {rep.linear_after_p1}"
    )
    assert rep.residual_after_p2 <= rep.act_quantizers_before, (
        f"More residual quantizers after phase 2: {rep.act_quantizers_before} -> {rep.residual_after_p2}"
    )


# ---------------------------------------------------------------------------
# Standalone campaign report
# ---------------------------------------------------------------------------

def _fmt_buckets(b: dict[str, int] | None) -> str:
    return "-" if b is None else f"{b['int']}/{b['scaled-int']}/{b['non-int']}"


def _fmt(x, spec=".3g") -> str:
    return "-" if x is None else format(x, spec)


def format_markdown_report(reports: list[StreamlineReport]) -> str:
    lines = [
        "# QONNX Model Zoo Streamlining Report",
        "",
        "Linear ops are counted as int/scaled-int/non-int (SIRA). Residual quantizers count"
        " Quant/BipolarQuant/Trunc/Round/Sign and unabsorbed comparisons left after phase 2.",
        "",
        "| Model | Linear before | Linear after P1 | Act. quant before | Residual after P2 | MultiThreshold |"
        " Max err P1 | Mismatch P1 | Max err P2 | Mismatch P2 | Time P1/P2 (s) | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in reports:
        errors = "; ".join(f"{k}: {v}" for k, v in r.errors.items()).replace("|", "\\|")
        lines.append(
            f"| `{r.model}` | {_fmt_buckets(r.linear_before)} | {_fmt_buckets(r.linear_after_p1)} | "
            f"{_fmt(r.act_quantizers_before, 'd')} | {_fmt(r.residual_after_p2, 'd')} | "
            f"{_fmt(r.multithresholds_after_p2, 'd')} | {_fmt(r.max_abs_err_p1)} | "
            f"{_fmt(r.mismatch_p1, '.2%')} | {_fmt(r.max_abs_err_p2)} | {_fmt(r.mismatch_p2, '.2%')} | "
            f"{_fmt(r.seconds_p1, '.1f')}/{_fmt(r.seconds_p2, '.1f')} | {errors or '-'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the QONNX Model Zoo streamlining campaign.")
    parser.add_argument("--samples", type=int, default=None, help="Samples per model (default: 3, 1 for large models)")
    parser.add_argument("--filter", default="", help="Only run models whose file name contains this substring")
    parser.add_argument("--report", type=Path, default=None, help="Optional Markdown report output path")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON report output path")
    args = parser.parse_args(argv)

    models = [p for p in streamlining_models() if args.filter.lower() in p.name.lower()]
    if not models:
        raise SystemExit("No models: set QONNX_MODEL_ZOO to the QONNX model zoo's models/ directory")

    reports = []
    for p in models:
        n = args.samples or (1 if _is_large(p) else 3)
        rep = run_streamlining_model(p, n)
        reports.append(rep)
        print(
            f"{rep.model:36s} linear {_fmt_buckets(rep.linear_before)} -> {_fmt_buckets(rep.linear_after_p1)}"
            f"  quant {rep.act_quantizers_before} -> {_fmt(rep.residual_after_p2, 'd')}"
            f"  err {_fmt(rep.max_abs_err_p1)}/{_fmt(rep.max_abs_err_p2)}"
            f"  {rep.errors or ''}",
            flush=True,
        )

    md = format_markdown_report(reports)
    print(md)
    if args.report:
        args.report.write_text(md)
    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in reports], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
