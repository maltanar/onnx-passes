"""Command-line entry point for SIRA range analysis, accumulator sizing, and stuck-channel detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx_ir as ir

from onnx_passes.passes.sira import (
    RangeAnalysis_v1,
    analyze_accumulators,
    detect_stuck_channels,
    evaluate_ranges,
    summarize_ranges,
)


def _format_markdown(
    model_path: Path,
    summary: dict[str, dict[str, Any]],
    accumulators: list[dict[str, Any]],
    stuck: dict[str, dict[str, Any]],
    granularity: str,
) -> str:
    lines = [
        f"# SIRA Analysis Report: `{model_path.name}`",
        "",
        "## Accumulator Sizing (`Conv` / `MatMul` / `Gemm`)",
        "",
        "| Node | Op | Output Tensor | Scaled-Int | Bitwidth | Per-Channel Bitwidths (min..max) |",
        "|---|---|---|---|---:|---|",
    ]
    for a in accumulators:
        pc = a["per_channel_bitwidths"]
        pc_str = f"{min(pc)}..{max(pc)} ({len(pc)} ch)" if pc else "-"
        bw = str(a["bitwidth"]) if a["bitwidth"] is not None else "-"
        lines.append(
            f"| `{a['node']}` | `{a['op_type']}` | `{a['tensor']}` | "
            f"{'yes' if a['is_scaled_int'] else 'no'} | {bw} | {pc_str} |"
        )

    lines += [
        "",
        "## Stuck Activation Channels",
        "",
    ]
    if stuck:
        lines += [
            "| Tensor | Stuck / Total | Channel Indices |",
            "|---|---:|---|",
        ]
        for tname, info in stuck.items():
            idxs = info["stuck_channels"]
            preview = str(idxs[:10]) + (" ..." if len(idxs) > 10 else "")
            lines.append(f"| `{tname}` | {len(idxs)}/{info['total_channels']} | `{preview}` |")
    else:
        lines.append("No stuck activation channels detected.")

    lines += [
        "",
        f"## Tensor Ranges (`granularity={granularity}`)",
        "",
        "| Tensor | Shape | Scaled-Int | Real Min | Real Max | Int Min | Int Max |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for tname, info in summary.items():
        imin = f"{info['int_min']:.6g}" if "int_min" in info else "-"
        imax = f"{info['int_max']:.6g}" if "int_max" in info else "-"
        lines.append(
            f"| `{tname}` | `{info['shape']}` | {'yes' if info['is_scaled_int'] else 'no'} | "
            f"{info['real_min']:.6g} | {info['real_max']:.6g} | {imin} | {imax} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for SIRA analysis."""
    parser = argparse.ArgumentParser(
        prog="onnx-passes-sira",
        description="Run SIRA range analysis, accumulator sizing, and stuck-channel detection on an ONNX/QONNX model.",
    )
    parser.add_argument("model", type=Path, help="Path to input .onnx model")
    parser.add_argument("--min", dest="input_min", type=float, default=-1.0, help="Uniform input lower bound (default: -1.0)")
    parser.add_argument("--max", dest="input_max", type=float, default=1.0, help="Uniform input upper bound (default: 1.0)")
    parser.add_argument("--input-scale", type=float, default=None, help="Optional input scale (enables scaled-integer input)")
    parser.add_argument("--input-bias", type=float, default=0.0, help="Optional input bias (default: 0.0 when --input-scale is set)")
    parser.add_argument(
        "--granularity",
        choices=("tensor", "channel", "element"),
        default="tensor",
        help="Range summary granularity (default: tensor)",
    )
    parser.add_argument("--channel-axis", type=int, default=1, help="Channel axis for channel summaries/stuck detection (default: 1)")
    parser.add_argument("--backend", choices=("onnxruntime", "reference"), default="onnxruntime", help="Execution backend (default: onnxruntime)")
    parser.add_argument("--no-intermediates", action="store_true", help="Evaluate graph outputs only")
    parser.add_argument("--save-range-model", type=Path, default=None, help="Optional path to save the RangeAnalysis_v1 ONNX model")
    parser.add_argument("--report", type=Path, default=None, help="Optional Markdown report output path")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON report output path")
    args = parser.parse_args(argv)

    orig = ir.load(args.model)
    dyn_inputs = [i for i in orig.graph.inputs if i.const_value is None]
    inp_ranges: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    inp_scales: dict[str, tuple[np.ndarray, np.ndarray]] | None = (
        {} if args.input_scale is not None else None
    )
    for inp in dyn_inputs:
        shape = [int(d) if isinstance(d, int) else 1 for d in inp.shape.dims] if inp.shape else [1]
        inp_ranges[inp.name] = (
            np.full(shape, args.input_min, dtype=np.float32),
            np.full(shape, args.input_max, dtype=np.float32),
        )
        if inp_scales is not None:
            inp_scales[inp.name] = (
                np.array(args.input_scale, dtype=np.float32),
                np.array(args.input_bias, dtype=np.float32),
            )

    rm = RangeAnalysis_v1()(orig.clone(deep_copy=True)).model
    if args.save_range_model:
        ir.save(rm, args.save_range_model)

    intermediates = not args.no_intermediates
    ranges = evaluate_ranges(
        rm, inp_ranges, input_scales=inp_scales, backend=args.backend, intermediates=intermediates,
    )
    summary = summarize_ranges(ranges, granularity=args.granularity, channel_axis=args.channel_axis)
    accumulators = analyze_accumulators(orig, ranges, channel_axis=args.channel_axis)
    stuck = detect_stuck_channels(orig, ranges, channel_axis=args.channel_axis)

    si_count = sum(1 for v in summary.values() if v["is_scaled_int"])
    print(f"Model: {args.model.name} | Tensors: {len(summary)} ({si_count} scaled-int)")
    if accumulators:
        bws = [a["bitwidth"] for a in accumulators]
        print(f"Accumulators ({len(accumulators)} linear layers): bitwidths={bws}")
    print(f"Stuck activation tensors: {len(stuck)}")
    for tname, info in list(stuck.items())[:10]:
        print(f"  {tname}: {len(info['stuck_channels'])}/{info['total_channels']} stuck channels")

    if args.report:
        args.report.write_text(_format_markdown(args.model, summary, accumulators, stuck, args.granularity))
    if args.json:
        payload = {
            "model": str(args.model),
            "granularity": args.granularity,
            "channel_axis": args.channel_axis,
            "accumulators": accumulators,
            "stuck_channels": stuck,
            "ranges": summary,
        }
        args.json.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
