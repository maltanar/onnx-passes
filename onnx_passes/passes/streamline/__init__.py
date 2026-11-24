"""Two-phase streamlining of quantized neural networks.

Following the scaled-integer view of the SIRA paper, streamlining is split into
two phases which can be applied (and measured) separately:

1. ``StreamlineScales_v1``: Inline quantizers and batch normalization and move
   scales and biases into the layer tails so that the linear operators (Conv,
   MatMul, Gemm) operate on plain integers. Layer tails remain chains of
   elementwise Mul/Add, nonlinearities and Round/Clip quantizer cores.
2. ``StreamlineThresholds_v1``: Convert the layer tails to MultiThreshold
   operators by absorbing (approximate inverses of) the elementwise functions
   into the threshold comparisons, then unbroadcast parameter tensors.

Resolving the ``streamline`` module from the command line applies both phases
in order. Decomposition into unit-step thresholds for the FINN backend
(``multithreshold.decompose``) is a backend lowering concern and not included.
"""

from onnx_passes.passes.base import Sequential

from onnx_passes.passes import canonicalize
from onnx_passes.passes import cleanup
from onnx_passes.passes import eliminate
from onnx_passes.passes import reorder
from onnx_passes.passes import unbroadcast
from onnx_passes.passes.inline import quantizer
from onnx_passes.passes.lower import batchnorm
from onnx_passes.passes.lower import gemm
from onnx_passes.passes.multithreshold import CompileToMultiThresholds_v1
from onnx_passes.passes.multithreshold.single import (
    InferSingleStepMultiThreshold_v1
)


class StreamlineScales_v1(Sequential):
    """Phase 1: Move scales and biases into the layer tails."""

    passes = [
        canonicalize,
        quantizer,
        batchnorm,
        gemm,
        reorder,
        eliminate,
        cleanup,
    ]

    exhaustive = True


class StreamlineThresholds_v1(Sequential):
    """Phase 2: Convert layer tails to MultiThreshold operators."""

    passes = [
        CompileToMultiThresholds_v1,
        # Single steps (bipolar quantizers) lose their threshold reduction
        # during compilation and are only recovered afterward
        InferSingleStepMultiThreshold_v1,
        unbroadcast,
        cleanup,
    ]
