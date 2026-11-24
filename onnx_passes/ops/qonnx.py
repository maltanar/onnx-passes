from onnx_passes.ops import OnnxOperator, Opset

import onnx_ir as ir
import numpy as np

DOMAIN = "qonnx.custom_op.general"


class Quant_v1(OnnxOperator, domain=DOMAIN):
    """QONNX arbitrary bitwidth integer quantization custom operator.

    See https://github.com/fastmachinelearning/qonnx for details.
    """

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing Quant."""

        def quant(
                x,
                scale,
                zeropoint,
                bitwidth,
                signed: int = 1,
                narrow: int = 0,
                rounding_mode: str = "ROUND"
        ):
            # Convert signed and narrow mode attributes to tensors matching the
            # input type to be used in dynamic calculations.
            signed = op.CastLike(
                op.Constant(value_int=signed),
                x
            )

            narrow = op.CastLike(
                op.Constant(value_int=narrow),
                x
            )

            # Minimum clipping range of the integer quantization:
            #   (-2 ** (bitwidth - signed) + narrow) * signed
            minimum = op.Mul(
                op.Add(
                    op.Neg(
                        op.Pow(
                            op.CastLike(
                                op.Constant(value_float=2.0),
                                x
                            ),
                            op.Sub(
                                op.CastLike(
                                    bitwidth,
                                    x
                                ),
                                signed
                            )
                        )
                    ),
                    narrow
                ),
                signed
            )

            # Maximum clipping range of the integer quantization:
            #   2 ** (bitwidth - signed) - 1 - narrow * (1 - signed)
            maximum = op.Sub(
                op.Sub(
                    op.Pow(
                        op.CastLike(
                            op.Constant(value_float=2.0),
                            x
                        ),
                        op.Sub(
                            op.CastLike(
                                bitwidth,
                                x
                            ),
                            signed
                        )
                    ),
                    op.Mul(
                        narrow,
                        op.Sub(
                            op.CastLike(
                                op.Constant(value_float=1.0),
                                x
                            ),
                            signed
                        )
                    )
                ),
                op.CastLike(
                    op.Constant(value_float=1.0),
                    x
                )
            )

            # Divide input by scale and shift by zeropoint: Float to integer
            # quantization, still with fractional part and can be out of range.
            q = op.Add(op.Div(x, scale), zeropoint)

            # Clipping to the integer range, used in all rounding mode branches
            # below - avoid inserting this into all if-then subgraphs.
            c = op.Clip(q, minimum, maximum)

            # Rounding mode "ROUND" - round to even, matches standard ONNX Round
            # operator behavior
            if rounding_mode == op.Constant(value_string="ROUND"):
                y = op.Round(c)

            # Rounding mode "CEIL" -  matches standard ONNX Ceil operator
            # behavior
            elif rounding_mode == op.Constant(value_string="CEIL"):
                y = op.Ceil(c)

            # Rounding mode "FLOOR" - matches standard ONNX Floor operator
            # behavior
            elif rounding_mode == op.Constant(value_string="FLOOR"):
                y = op.Floor(c)

            # Rounding mode "UP" - combination of standard ONNX operator
            # behaviors
            elif rounding_mode == op.Constant(value_string="UP"):
                y = op.Mul(
                    op.Ceil(
                        op.Abs(
                            c
                        )
                    ),
                    op.Sign(
                        c
                    )
                )

            # Rounding mode "DOWN" - combination of standard ONNX operator
            # behaviors
            elif rounding_mode == op.Constant(value_string="DOWN"):
                y = op.Mul(
                    op.Floor(
                        op.Abs(
                            c
                        )
                    ),
                    op.Sign(
                        c
                    )
                )

            # Rounding mode "HALF_UP" - combination of standard ONNX operator
            # behaviors
            elif rounding_mode == op.Constant(value_string="HALF_UP"):
                y = op.Mul(
                    op.Add(
                        op.Floor(
                            op.Abs(
                                c
                            )
                        ),
                        op.CastLike(
                            op.Constant(
                                value_float=0.5
                            ),
                            c
                        )
                    ),
                    op.Sign(
                        c
                    )
                )

            # Rounding mode "HALF_DOWN" - combination of standard ONNX operator
            # behaviors
            elif rounding_mode == op.Constant(value_string="HALF_DOWN"):
                y = op.Mul(
                    op.Sub(
                        op.Ceil(
                            op.Abs(
                                c
                            )
                        ),
                        op.CastLike(
                            op.Constant(
                                value_float=0.5
                            ),
                            c
                        )
                    ),
                    op.Sign(
                        c
                    )
                )

            # Else branch cannot be omitted, and it is not possible to raise
            # exceptions or assertions - fallback to NaN...
            else:
                y = op.CastLike(
                    op.Constant(value_float=np.nan),
                    c
                )

            # Clipping and rounding according to the selected rounding mode or
            # special treatment for signed 1-bit quantization.
            q = op.Where(
                # Signed 1-bit quantization: Convert to bipolar encoding
                # where q >= 0 is encoded as +1 and q < 0 as -1
                op.And(
                    op.Equal(
                        bitwidth,
                        op.CastLike(
                            op.Constant(value_int=1),
                            bitwidth
                        )
                    ),
                    op.Cast(
                        signed, to=ir.DataType.BOOL
                    )
                ),
                op.Where(
                    op.GreaterOrEqual(
                        q,
                        op.CastLike(
                            op.Constant(value_float=0.0),
                            q
                        )
                    ),
                    op.CastLike(
                        op.Constant(value_float=+1.0),
                        q
                    ),
                    op.CastLike(
                        op.Constant(value_float=-1.0),
                        q
                    )
                ),
                # Non-bipolar encodings: Clip to integer bounds and round
                # according to the rounding mode
                y
            )

            # Multiply quantized value by scale and shift by zeropoint: Integer
            # to float dequantization.
            return op.Mul(op.Sub(q, zeropoint), scale)

        return quant


class Trunc_v1(OnnxOperator, domain=DOMAIN):
    """QONNX truncation custom operator (opset version 1).

    Rounds the input to the integer grid of the input scale and zero point,
    then drops the in_bitwidth - out_bitwidth least significant bits. Note that
    the output is rescaled by the *input* scale, as in the specification.

    See https://github.com/fastmachinelearning/qonnx for details.
    """

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing Trunc."""

        def trunc(
                x,
                scale,
                zeropt,
                in_bitwidth,
                out_bitwidth,
                rounding_mode: str = "FLOOR"
        ):
            # Parameters might be of integer type: Match the input type
            scale = op.CastLike(scale, x)
            zeropt = op.CastLike(zeropt, x)

            # Scale to the integer grid of the input and round
            q = op.Round(op.Add(op.Div(x, scale), zeropt))

            # Truncation scale: Drop the least significant bits
            q = op.Div(
                q,
                op.Pow(
                    op.CastLike(op.Constant(value_float=2.0), x),
                    op.Sub(
                        op.CastLike(in_bitwidth, x),
                        op.CastLike(out_bitwidth, x)
                    )
                )
            )

            # Rounding of the truncated bits according to the rounding mode,
            # lowercase variants are allowed by the specification
            if rounding_mode == op.Constant(value_string="FLOOR"):
                y = op.Floor(q)
            elif rounding_mode == op.Constant(value_string="floor"):
                y = op.Floor(q)
            elif rounding_mode == op.Constant(value_string="ROUND"):
                y = op.Round(q)
            elif rounding_mode == op.Constant(value_string="round"):
                y = op.Round(q)
            # Note: The ONNX reference evaluator refuses comparing strings of
            # different length, thus test the 4-character modes last
            elif rounding_mode == op.Constant(value_string="CEIL"):
                y = op.Ceil(q)
            elif rounding_mode == op.Constant(value_string="ceil"):
                y = op.Ceil(q)
            # Else branch cannot be omitted, and it is not possible to raise
            # exceptions or assertions - fallback to NaN...
            else:
                y = op.CastLike(op.Constant(value_float=np.nan), q)

            # Integer to float: Note the input scale as in the specification
            return op.Mul(op.Sub(y, zeropt), scale)

        return trunc


class Trunc_v2(OnnxOperator, domain=DOMAIN):
    """QONNX truncation custom operator (opset version 2).

    Rounds the input to the integer grid of the input scale and zero point,
    divides by the power-of-two ratio of output and input scale, clamps to the
    output bitwidth and rounds, then rescales by the output scale.

    See https://github.com/fastmachinelearning/qonnx for details.
    """

    @staticmethod
    def script(op: Opset):
        """Generate an ONNX Script function implementing Trunc."""

        def trunc(
                x,
                scale,
                zeropt,
                in_bitwidth,
                out_scale,
                out_bitwidth,
                rounding_mode: str = "FLOOR",
                signed: int = 1,
                narrow: int = 0
        ):
            # Parameters might be of integer type: Match the input type
            scale = op.CastLike(scale, x)
            zeropt = op.CastLike(zeropt, x)
            out_scale = op.CastLike(out_scale, x)
            bitwidth = op.CastLike(out_bitwidth, x)
            signed = op.CastLike(op.Constant(value_int=signed), x)
            narrow = op.CastLike(op.Constant(value_int=narrow), x)
            _1 = op.CastLike(op.Constant(value_float=1.0), x)
            _2 = op.CastLike(op.Constant(value_float=2.0), x)

            # Scale to the integer grid of the input and round
            q = op.Round(op.Add(op.Div(x, scale), zeropt))

            # Truncation scale: Power-of-two ratio of output and input scales
            trunc_scale = op.Pow(
                _2, op.Round(op.Div(op.Log(op.Div(out_scale, scale)), op.Log(_2)))
            )
            q = op.Div(q, trunc_scale)

            # Clamp to the output integer range:
            #   [(-2 ** (bitwidth - signed) + narrow) * signed,
            #    2 ** (bitwidth - signed) - 1 - narrow * (1 - signed)]
            minimum = op.Mul(
                op.Add(op.Neg(op.Pow(_2, op.Sub(bitwidth, signed))), narrow),
                signed
            )
            maximum = op.Sub(
                op.Sub(
                    op.Pow(_2, op.Sub(bitwidth, signed)),
                    op.Mul(narrow, op.Sub(_1, signed))
                ),
                _1
            )
            q = op.Min(op.Max(q, minimum), maximum)

            # Rounding of the truncated bits according to the rounding mode,
            # lowercase variants are allowed by the specification
            if rounding_mode == op.Constant(value_string="FLOOR"):
                y = op.Floor(q)
            elif rounding_mode == op.Constant(value_string="floor"):
                y = op.Floor(q)
            elif rounding_mode == op.Constant(value_string="ROUND"):
                y = op.Round(q)
            elif rounding_mode == op.Constant(value_string="round"):
                y = op.Round(q)
            # Note: The ONNX reference evaluator refuses comparing strings of
            # different length, thus test the 4-character modes last
            elif rounding_mode == op.Constant(value_string="CEIL"):
                y = op.Ceil(q)
            elif rounding_mode == op.Constant(value_string="ceil"):
                y = op.Ceil(q)
            else:
                y = op.CastLike(op.Constant(value_float=np.nan), q)

            # Integer to float with the zero point rescaled to the output grid
            return op.Mul(op.Sub(y, op.Div(zeropt, trunc_scale)), out_scale)

        return trunc
