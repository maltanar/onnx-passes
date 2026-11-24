from onnx_passes.passes.base import RewriteRule
from onnx_passes.passes.verify import Verify, tolerance

from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN

import onnx_ir as ir


@tolerance
class InferSingleStepMultiThreshold_v1(RewriteRule, Verify):
    """Infer single-threshold MultiThreshold from a bare unit step.

    Single steps, e.g., from bipolar quantizers, end up as Cast(x >= t) once
    the singleton threshold reduction has been eliminated, which no longer
    matches InferMultiThreshold_v1.

    Note: Not part of the multithreshold.normalize module, as this would also
    match the steps inside not yet normalized multi-threshold subgraphs. Apply
    this only after the MultiThreshold compilation has been completed.
    """

    @staticmethod
    def pattern(op, x, thresholds):
        return op.Cast(
            op.GreaterOrEqual(x, thresholds), to=ir.DataType.FLOAT
        )

    @staticmethod
    def check(context, x, thresholds):
        if ir.convenience.get_const_tensor(thresholds) is None:
            return False

        if ir.convenience.get_const_tensor(x) is not None:
            return False

        # Thresholds must not broadcast the input to a larger shape, which is
        # the case for steps inside unnormalized multi-threshold subgraphs
        if x.shape is None or x.shape.is_dynamic():
            return False

        output = context.root.outputs[0]
        if output.shape is None or output.shape.is_dynamic():
            return False

        return tuple(output.shape) == tuple(x.shape)

    @staticmethod
    def rewrite(op, x, thresholds):
        return op.MultiThreshold(
            x,
            # Append the thresholding axis holding a single threshold
            op.Unsqueeze(thresholds, op.Constant(value_ints=[-1])),
            # Unit step weight
            op.Constant(value_float=1.0),
            _domain=CUSTOM_DOMAIN
        )
