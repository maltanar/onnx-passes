"""Shared QONNX model zoo helpers for the zoo test campaigns.

Used by ``tests/test_sira_zoo.py`` (SIRA instrumentation) and
``tests/test_modelzoo_streamlining.py`` (streamlining). Point the environment
variable ``QONNX_MODEL_ZOO`` at the zoo's ``models/`` directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnx.version_converter
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator

from onnx_passes.ops import link_ops

try:
    from tests.test_sira import _CUSTOM_OPS
except ImportError:
    from test_sira import _CUSTOM_OPS


# ---------------------------------------------------------------------------
# Zoo Discovery & Reasoned Input Specifications
# ---------------------------------------------------------------------------

def qonnx_zoo_models() -> list[Path]:
    """Collect .onnx models under $QONNX_MODEL_ZOO (empty if unset)."""
    root = os.environ.get("QONNX_MODEL_ZOO")
    if not root:
        return []
    return sorted(Path(root).expanduser().rglob("*.onnx"))


QONNX_ZOO_MODELS = qonnx_zoo_models()

requires_zoo = pytest.mark.skipif(
    not QONNX_ZOO_MODELS,
    reason="set QONNX_MODEL_ZOO to the QONNX model zoo's models/ directory",
)


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
# Model Import (Opset Upgrade) & Reference Execution
# ---------------------------------------------------------------------------

# Custom-op domains used across the zoo; the legacy FINN domain hosts the same
# Quant operator as the QONNX domain
_LEGACY_DOMAINS = {"finn.custom_op.general": "qonnx.custom_op.general"}


def load_zoo_model(model_path: Path, target_opset: int | None = None) -> ir.Model:
    """Load a zoo model, declaring custom-op domains and optionally upgrading.

    Many zoo models omit opset imports for their custom-op domains and use the
    legacy ``finn.custom_op.general`` domain. Both are normalized here. If
    ``target_opset`` is given, the default domain is upgraded via the ONNX
    version converter (onnx-passes needs a recent opset, e.g. for
    ``Constant(value_int=...)`` in its operator functions).
    """
    proto = onnx.load(model_path)
    for node in proto.graph.node:
        node.domain = _LEGACY_DOMAINS.get(node.domain, node.domain)
    declared = {o.domain for o in proto.opset_import}
    for domain in sorted({n.domain for n in proto.graph.node if n.domain} - declared):
        proto.opset_import.append(onnx.helper.make_opsetid(domain, 1))
    if target_opset is not None:
        current = next(o.version for o in proto.opset_import if o.domain in ("", "ai.onnx"))
        if current < target_opset:
            proto = onnx.version_converter.convert_version(proto, target_opset)
    return ir.from_proto(proto)


def reference_session(model: ir.Model) -> ReferenceEvaluator:
    """ReferenceEvaluator for zoo models and their onnx-passes transformations.

    onnx-passes operators (Quant, MultiThreshold, ...) are linked in as model
    functions; QONNX ops without an onnx-passes definition (BipolarQuant, Trunc)
    fall back to the Python implementations from ``test_sira``.
    """
    m = link_ops(model.clone(deep_copy=True))
    for node in m.graph:
        if node.domain:
            m.graph.opset_imports.setdefault(node.domain, 1)
    linked = {(f.domain, f.name) for f in m.functions.values()}
    new_ops = [op for op in _CUSTOM_OPS if (op.op_domain, op.__name__) not in linked]
    return ReferenceEvaluator(ir.to_proto(m), new_ops=new_ops)
