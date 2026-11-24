from typing import Sequence

import numpy as np
import onnx_ir as ir
from onnxscript.values import Opset

from onnx_passes.ops import OnnxOperator

DOMAIN = "onnx_passes.ops.sira"

INTS = Sequence[int]


class IsEqual_v1(OnnxOperator, domain=DOMAIN, op_type="IsEqual", version=1):
    """Elementwise equality check compatible with opset 9+."""

    @staticmethod
    def script(op: Opset):
        if op.version >= 11:
            def is_equal(a, b):
                return op.Equal(a, b)
        else:
            def is_equal(a, b):
                return op.Not(op.Or(op.Less(a, b), op.Greater(a, b)))
        return is_equal


class AllTrue_v1(OnnxOperator, domain=DOMAIN, op_type="AllTrue", version=1):
    """Scalar bool: True iff all elements of cond are True."""

    @staticmethod
    def script(op: Opset):
        def all_true(cond):
            return op.Equal(
                op.ReduceSum(op.Cast(op.Not(cond), to=ir.DataType.INT64), keepdims=0),
                op.Constant(value_int=0),
            )
        return all_true


class IsScaledInt_v1(OnnxOperator, domain=DOMAIN, op_type="IsScaledInt", version=1):
    """Scalar bool: True iff neither scale nor bias contains NaN."""

    @staticmethod
    def script(op: Opset):
        def is_scaled_int(scale, bias):
            nan_s = op.ReduceSum(
                op.Cast(op.IsNaN(scale), to=ir.DataType.INT64), keepdims=0
            )
            nan_b = op.ReduceSum(
                op.Cast(op.IsNaN(bias), to=ir.DataType.INT64), keepdims=0
            )
            zero = op.Constant(value_int=0)
            return op.And(op.Equal(nan_s, zero), op.Equal(nan_b, zero))
        return is_scaled_int


class IsConstant_v1(OnnxOperator, domain=DOMAIN, op_type="IsConstant", version=1):
    """Scalar bool: True iff v_min == v_max everywhere."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def is_constant(v_min, v_max):
            return sira.AllTrue(sira.IsEqual(v_min, v_max))
        return is_constant


class IsZero_v1(OnnxOperator, domain=DOMAIN, op_type="IsZero", version=1):
    """Scalar bool: True iff all elements of x are 0."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def is_zero(x):
            return sira.AllTrue(sira.IsEqual(x, op.CastLike(op.Constant(value_float=0.0), x)))
        return is_zero


class IsScalarValued_v1(OnnxOperator, domain=DOMAIN, op_type="IsScalarValued", version=1):
    """Scalar bool: True iff all elements of x are equal."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def is_scalar_valued(x):
            return sira.IsEqual(op.ReduceMin(x, keepdims=0), op.ReduceMax(x, keepdims=0))
        return is_scalar_valued


class ToRealRange_v1(OnnxOperator, domain=DOMAIN, op_type="ToRealRange", version=1):
    """Convert (min, max, scale, bias) to real-valued (r_min, r_max)."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def to_real_range(v_min, v_max, v_scale, v_bias):
            is_si = sira.IsScaledInt(v_scale, v_bias)
            s_safe = op.CastLike(
                op.Where(is_si, v_scale, op.CastLike(op.Constant(value_float=1.0), v_scale)),
                v_min,
            )
            b_safe = op.CastLike(
                op.Where(is_si, v_bias, op.CastLike(op.Constant(value_float=0.0), v_bias)),
                v_min,
            )
            a = op.Add(op.Mul(v_min, s_safe), b_safe)
            b = op.Add(op.Mul(v_max, s_safe), b_safe)
            r_min = op.Where(is_si, op.Min(a, b), v_min)
            r_max = op.Where(is_si, op.Max(a, b), v_max)
            return r_min, r_max
        return to_real_range


class QuantEval_v1(OnnxOperator, domain=DOMAIN, op_type="QuantEval", version=1):
    """Evaluate integer part (g o f^-1)(x) of QONNX Quant."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def quant_eval(
            x, scale, zeropoint, bitwidth,
            signed: int = 1,
            narrow: int = 0,
            rounding_mode: int = 0,
        ):
            signed_t = op.CastLike(op.Constant(value_int=signed), x)
            narrow_t = op.CastLike(op.Constant(value_int=narrow), x)
            bw = op.CastLike(bitwidth, x)
            two = op.CastLike(op.Constant(value_float=2.0), x)
            one = op.CastLike(op.Constant(value_float=1.0), x)

            exp = op.Sub(bw, signed_t)
            pow2 = op.Pow(two, exp)
            minimum = op.Mul(op.Add(op.Neg(pow2), narrow_t), signed_t)
            maximum = op.Sub(
                op.Sub(pow2, op.Mul(narrow_t, op.Sub(one, signed_t))),
                one,
            )

            q = op.Add(op.Div(x, scale), zeropoint)
            c = op.Clip(q, minimum, maximum)

            if rounding_mode == op.Constant(value_int=0):
                y = op.Round(c)
            elif rounding_mode == op.Constant(value_int=1):
                y = op.Ceil(c)
            elif rounding_mode == op.Constant(value_int=2):
                y = op.Floor(c)
            elif rounding_mode == op.Constant(value_int=3):
                y = op.Mul(op.Ceil(op.Abs(c)), op.Sign(c))
            elif rounding_mode == op.Constant(value_int=4):
                y = op.Mul(op.Floor(op.Abs(c)), op.Sign(c))
            elif rounding_mode == op.Constant(value_int=5):
                y = op.Mul(
                    op.Add(op.Floor(op.Abs(c)), op.CastLike(op.Constant(value_float=0.5), c)),
                    op.Sign(c),
                )
            elif rounding_mode == op.Constant(value_int=6):
                y = op.Mul(
                    op.Sub(op.Ceil(op.Abs(c)), op.CastLike(op.Constant(value_float=0.5), c)),
                    op.Sign(c),
                )
            else:
                y = op.CastLike(op.Constant(value_float=float("nan")), c)

            is_bipolar = op.And(
                sira.IsEqual(bw, op.CastLike(op.Constant(value_float=1.0), bw)),
                op.Cast(signed_t, to=ir.DataType.BOOL),
            )
            bipolar_val = op.Where(
                op.GreaterOrEqual(q, op.CastLike(op.Constant(value_float=0.0), q)),
                op.CastLike(op.Constant(value_float=1.0), q),
                op.CastLike(op.Constant(value_float=-1.0), q),
            )
            return op.Where(is_bipolar, bipolar_val, y)

        return quant_eval


class SiraQuant_v1(OnnxOperator, domain=DOMAIN, op_type="Quant", version=1):
    """SIRA range analysis handler for Quant operator."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_quant(
            x_min, x_max, x_scale, x_bias,
            s_min, s_max, s_scale, s_bias,
            z_min, z_max, z_scale, z_bias,
            bw_min, bw_max, bw_scale, bw_bias,
            signed: int = 1,
            narrow: int = 0,
            rounding_mode: int = 0,
        ):
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            s, _ = sira.ToRealRange(s_min, s_max, s_scale, s_bias)
            z, _ = sira.ToRealRange(z_min, z_max, z_scale, z_bias)
            bw, _ = sira.ToRealRange(bw_min, bw_max, bw_scale, bw_bias)

            s = op.CastLike(s, x_rmin)
            z = op.CastLike(z, x_rmin)

            q_low = sira.QuantEval(x_rmin, s, z, bw, signed=signed, narrow=narrow, rounding_mode=rounding_mode)
            q_high = sira.QuantEval(x_rmax, s, z, bw, signed=signed, narrow=narrow, rounding_mode=rounding_mode)

            y_min = op.Min(q_low, q_high)
            y_max = op.Max(q_low, q_high)
            y_scale = s
            y_bias = op.Neg(op.Mul(s, z))
            return y_min, y_max, y_scale, y_bias

        return sira_quant


class SiraBipolarQuant_v1(OnnxOperator, domain=DOMAIN, op_type="BipolarQuant", version=1):
    """SIRA range analysis handler for BipolarQuant operator."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_bipolar_quant(
            x_min, x_max, x_scale, x_bias,
            s_min, s_max, s_scale, s_bias,
        ):
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            s, _ = sira.ToRealRange(s_min, s_max, s_scale, s_bias)
            s = op.CastLike(s, x_rmin)

            zero = op.CastLike(op.Constant(value_float=0.0), x_rmin)
            pos1 = op.CastLike(op.Constant(value_float=1.0), x_rmin)
            neg1 = op.CastLike(op.Constant(value_float=-1.0), x_rmin)

            q_low = op.Where(op.GreaterOrEqual(x_rmin, zero), pos1, neg1)
            q_high = op.Where(op.GreaterOrEqual(x_rmax, zero), pos1, neg1)

            y_min = op.Min(q_low, q_high)
            y_max = op.Max(q_low, q_high)
            y_scale = s
            y_bias = op.CastLike(op.Constant(value_float=0.0), s)
            return y_min, y_max, y_scale, y_bias

        return sira_bipolar_quant


class SiraTrunc_v1(OnnxOperator, domain=DOMAIN, op_type="Trunc", version=1):
    """SIRA range analysis handler for Trunc (v1, 5-input) operator."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_trunc(
            x_min, x_max, x_scale, x_bias,
            s_min, s_max, s_scale, s_bias,
            z_min, z_max, z_scale, z_bias,
            in_bw_min, in_bw_max, in_bw_scale, in_bw_bias,
            out_bw_min, out_bw_max, out_bw_scale, out_bw_bias,
            rounding_mode: int = 2,
        ):
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            s, _ = sira.ToRealRange(s_min, s_max, s_scale, s_bias)
            z, _ = sira.ToRealRange(z_min, z_max, z_scale, z_bias)
            in_bw, _ = sira.ToRealRange(in_bw_min, in_bw_max, in_bw_scale, in_bw_bias)
            out_bw, _ = sira.ToRealRange(out_bw_min, out_bw_max, out_bw_scale, out_bw_bias)

            s = op.CastLike(s, x_rmin)
            z = op.CastLike(z, x_rmin)
            trunc_bw = op.CastLike(op.Sub(in_bw, out_bw), x_rmin)
            trunc_scale = op.Pow(op.CastLike(op.Constant(value_float=2.0), x_rmin), trunc_bw)

            v_low = op.Div(op.Round(op.Add(op.Div(x_rmin, s), z)), trunc_scale)
            v_high = op.Div(op.Round(op.Add(op.Div(x_rmax, s), z)), trunc_scale)

            if rounding_mode == op.Constant(value_int=2):
                q_low = op.Floor(v_low)
                q_high = op.Floor(v_high)
            elif rounding_mode == op.Constant(value_int=1):
                q_low = op.Ceil(v_low)
                q_high = op.Ceil(v_high)
            elif rounding_mode == op.Constant(value_int=0):
                q_low = op.Round(v_low)
                q_high = op.Round(v_high)
            else:
                q_low = op.Floor(v_low)
                q_high = op.Floor(v_high)

            y_min = op.Min(q_low, q_high)
            y_max = op.Max(q_low, q_high)
            y_scale = s
            y_bias = op.Neg(op.Mul(s, z))
            return y_min, y_max, y_scale, y_bias

        return sira_trunc


class FloatQuantEval_v1(OnnxOperator, domain=DOMAIN, op_type="FloatQuantEval", version=1):
    """Evaluate FloatQuant on a single tensor endpoint."""

    @staticmethod
    def script(op: Opset):
        def fq_eval(val, s, exp_bw, man_bw, exp_bias, max_val):
            two = op.CastLike(op.Constant(value_float=2.0), val)
            one = op.CastLike(op.Constant(value_float=1.0), val)
            max_exp = op.Sub(op.Sub(op.Pow(two, exp_bw), one), exp_bias)
            max_mant = op.Sub(two, op.Pow(two, op.Neg(man_bw)))
            fmt_max = op.Min(op.Mul(max_mant, op.Pow(two, max_exp)), max_val)
            fmt_min = op.Neg(fmt_max)

            xs = op.Div(val, s)
            eps = op.CastLike(op.Constant(value_float=1.17549435e-38), xs)
            fp_min_exp = op.Sub(op.Sub(one, exp_bias), man_bw)
            int_exp = op.Max(
                op.Sub(op.Floor(op.Div(op.Log(op.Add(op.Abs(xs), eps)), op.Log(two))), man_bw),
                fp_min_exp,
            )
            int_scale = op.Pow(two, int_exp)
            xq = op.Mul(int_scale, op.Round(op.Div(xs, int_scale)))
            return op.Mul(op.Clip(xq, fmt_min, fmt_max), s)

        return fq_eval


class SiraFloatQuant_v1(OnnxOperator, domain=DOMAIN, op_type="FloatQuant", version=1):
    """SIRA range analysis handler for FloatQuant operator (non-integer)."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_float_quant(
            x_min, x_max, x_scale, x_bias,
            s_min, s_max, s_scale, s_bias,
            exp_bw_min, exp_bw_max, exp_bw_scale, exp_bw_bias,
            man_bw_min, man_bw_max, man_bw_scale, man_bw_bias,
            exp_bias_min, exp_bias_max, exp_bias_scale, exp_bias_bias,
            max_val_min, max_val_max, max_val_scale, max_val_bias,
            has_inf: int = 0,
            has_nan: int = 0,
            has_subnormal: int = 1,
            rounding_mode: str = "round",
            saturation: int = 1,
        ):
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            s, _ = sira.ToRealRange(s_min, s_max, s_scale, s_bias)
            exp_bw, _ = sira.ToRealRange(exp_bw_min, exp_bw_max, exp_bw_scale, exp_bw_bias)
            man_bw, _ = sira.ToRealRange(man_bw_min, man_bw_max, man_bw_scale, man_bw_bias)
            exp_bias, _ = sira.ToRealRange(exp_bias_min, exp_bias_max, exp_bias_scale, exp_bias_bias)
            max_val, _ = sira.ToRealRange(max_val_min, max_val_max, max_val_scale, max_val_bias)

            s = op.CastLike(s, x_rmin)
            exp_bw = op.CastLike(exp_bw, x_rmin)
            man_bw = op.CastLike(man_bw, x_rmin)
            exp_bias = op.CastLike(exp_bias, x_rmin)
            max_val = op.CastLike(max_val, x_rmin)

            y_low = sira.FloatQuantEval(x_rmin, s, exp_bw, man_bw, exp_bias, max_val)
            y_high = sira.FloatQuantEval(x_rmax, s, exp_bw, man_bw, exp_bias, max_val)
            y_min = op.Min(y_low, y_high)
            y_max = op.Max(y_low, y_high)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return y_min, y_max, nan_s, nan_b

        return sira_float_quant


class SiraAdd_v1(OnnxOperator, domain=DOMAIN, op_type="Add", version=1):
    """SIRA range analysis handler for Add operator."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_add(
            a_min, a_max, a_scale, a_bias,
            b_min, b_max, b_scale, b_bias,
        ):
            a_si = sira.IsScaledInt(a_scale, a_bias)
            b_si = sira.IsScaledInt(b_scale, b_bias)
            a_rmin, a_rmax = sira.ToRealRange(a_min, a_max, a_scale, a_bias)
            b_rmin, b_rmax = sira.ToRealRange(b_min, b_max, b_scale, b_bias)

            a_const = sira.IsConstant(a_rmin, a_rmax)
            b_const = sira.IsConstant(b_rmin, b_rmax)

            c1a = op.And(a_si, b_const)
            c1b = op.And(b_si, a_const)

            both_si = op.And(a_si, b_si)
            a_s_safe = op.Where(a_si, a_scale, op.CastLike(op.Constant(value_float=1.0), a_scale))
            b_s_safe = op.Where(b_si, b_scale, op.CastLike(op.Constant(value_float=1.0), b_scale))
            a_b_safe = op.Where(a_si, a_bias, op.CastLike(op.Constant(value_float=0.0), a_bias))
            b_b_safe = op.Where(b_si, b_bias, op.CastLike(op.Constant(value_float=0.0), b_bias))

            a_nz = op.Not(sira.IsEqual(a_s_safe, op.CastLike(op.Constant(value_float=0.0), a_s_safe)))
            r_ba = op.Div(b_s_safe, op.Where(a_nz, a_s_safe, op.CastLike(op.Constant(value_float=1.0), a_s_safe)))
            k_ba = op.Round(r_ba)
            tol_ba = op.Mul(op.CastLike(op.Constant(value_float=1e-5), r_ba), op.Max(op.CastLike(op.Constant(value_float=1.0), r_ba), op.Abs(k_ba)))
            ok_ba = op.And(both_si, sira.AllTrue(op.And(a_nz, op.LessOrEqual(op.Abs(op.Sub(r_ba, k_ba)), tol_ba))))

            b_nz = op.Not(sira.IsEqual(b_s_safe, op.CastLike(op.Constant(value_float=0.0), b_s_safe)))
            r_ab = op.Div(a_s_safe, op.Where(b_nz, b_s_safe, op.CastLike(op.Constant(value_float=1.0), b_s_safe)))
            k_ab = op.Round(r_ab)
            tol_ab = op.Mul(op.CastLike(op.Constant(value_float=1e-5), r_ab), op.Max(op.CastLike(op.Constant(value_float=1.0), r_ab), op.Abs(k_ab)))
            ok_ab = op.And(both_si, sira.AllTrue(op.And(b_nz, op.LessOrEqual(op.Abs(op.Sub(r_ab, k_ab)), tol_ab))))

            r_min = op.Add(a_rmin, b_rmin)
            r_max = op.Add(a_rmax, b_rmax)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))

            kb_min = op.Mul(op.CastLike(k_ba, b_min), b_min)
            kb_max = op.Mul(op.CastLike(k_ba, b_max), b_max)
            q2a_min = op.Add(a_min, op.Min(kb_min, kb_max))
            q2a_max = op.Add(a_max, op.Max(kb_min, kb_max))

            ka_min = op.Mul(op.CastLike(k_ab, a_min), a_min)
            ka_max = op.Mul(op.CastLike(k_ab, a_max), a_max)
            q2b_min = op.Add(b_min, op.Min(ka_min, ka_max))
            q2b_max = op.Add(b_max, op.Max(ka_min, ka_max))

            sum_bias = op.Add(a_b_safe, b_b_safe)
            zero_b = op.Sub(b_rmin, b_rmin)
            zero_a = op.Sub(a_rmin, a_rmin)

            y_min = op.Where(
                c1a, op.Add(a_min, zero_b),
                op.Where(
                    c1b, op.Add(zero_a, b_min),
                    op.Where(ok_ba, q2a_min, op.Where(ok_ab, q2b_min, r_min)),
                ),
            )
            y_max = op.Where(
                c1a, op.Add(a_max, zero_b),
                op.Where(
                    c1b, op.Add(zero_a, b_max),
                    op.Where(ok_ba, q2a_max, op.Where(ok_ab, q2b_max, r_max)),
                ),
            )
            y_scale = op.Where(
                c1a, a_s_safe,
                op.Where(
                    c1b, b_s_safe,
                    op.Where(ok_ba, a_s_safe, op.Where(ok_ab, b_s_safe, nan_s)),
                ),
            )
            y_bias = op.Where(
                c1a, op.Add(a_b_safe, op.CastLike(b_rmin, a_b_safe)),
                op.Where(
                    c1b, op.Add(b_b_safe, op.CastLike(a_rmin, b_b_safe)),
                    op.Where(op.Or(ok_ba, ok_ab), sum_bias, nan_b),
                ),
            )
            return y_min, y_max, y_scale, y_bias

        return sira_add


class SiraNeg_v1(OnnxOperator, domain=DOMAIN, op_type="Neg", version=1):
    """SIRA range analysis handler for Neg operator."""

    @staticmethod
    def script(op: Opset):
        def sira_neg(x_min, x_max, x_scale, x_bias):
            return op.Neg(x_max), op.Neg(x_min), x_scale, op.Neg(x_bias)

        return sira_neg


class SiraSub_v1(OnnxOperator, domain=DOMAIN, op_type="Sub", version=1):
    """SIRA range analysis handler for Sub operator."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_sub(
            a_min, a_max, a_scale, a_bias,
            b_min, b_max, b_scale, b_bias,
        ):
            nb_min, nb_max, nb_scale, nb_bias = sira.Neg(b_min, b_max, b_scale, b_bias)
            y_min, y_max, y_scale, y_bias = sira.Add(
                a_min, a_max, a_scale, a_bias,
                nb_min, nb_max, nb_scale, nb_bias,
            )
            return y_min, y_max, y_scale, y_bias

        return sira_sub


class SiraMul_v1(OnnxOperator, domain=DOMAIN, op_type="Mul", version=1):
    """SIRA range analysis handler for Mul operator."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_mul(
            a_min, a_max, a_scale, a_bias,
            b_min, b_max, b_scale, b_bias,
        ):
            a_si = sira.IsScaledInt(a_scale, a_bias)
            b_si = sira.IsScaledInt(b_scale, b_bias)
            a_rmin, a_rmax = sira.ToRealRange(a_min, a_max, a_scale, a_bias)
            b_rmin, b_rmax = sira.ToRealRange(b_min, b_max, b_scale, b_bias)

            a_const = sira.IsConstant(a_rmin, a_rmax)
            b_const = sira.IsConstant(b_rmin, b_rmax)

            c1a = op.And(a_si, b_const)
            c1b = op.And(b_si, a_const)

            p1 = op.Mul(a_rmin, b_rmin)
            p2 = op.Mul(a_rmin, b_rmax)
            p3 = op.Mul(a_rmax, b_rmin)
            p4 = op.Mul(a_rmax, b_rmax)
            r_min = op.Min(op.Min(p1, p2), op.Min(p3, p4))
            r_max = op.Max(op.Max(p1, p2), op.Max(p3, p4))

            zero_b = op.Sub(b_rmin, b_rmin)
            zero_a = op.Sub(a_rmin, a_rmin)

            a_s_safe = op.Where(a_si, a_scale, op.CastLike(op.Constant(value_float=1.0), a_scale))
            b_s_safe = op.Where(b_si, b_scale, op.CastLike(op.Constant(value_float=1.0), b_scale))
            a_b_safe = op.Where(a_si, a_bias, op.CastLike(op.Constant(value_float=0.0), a_bias))
            b_b_safe = op.Where(b_si, b_bias, op.CastLike(op.Constant(value_float=0.0), b_bias))

            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))

            y_min = op.Where(c1a, op.Add(a_min, zero_b), op.Where(c1b, op.Add(zero_a, b_min), r_min))
            y_max = op.Where(c1a, op.Add(a_max, zero_b), op.Where(c1b, op.Add(zero_a, b_max), r_max))
            y_scale = op.Where(
                c1a, op.Mul(a_s_safe, op.CastLike(b_rmin, a_s_safe)),
                op.Where(c1b, op.Mul(b_s_safe, op.CastLike(a_rmin, b_s_safe)), nan_s),
            )
            y_bias = op.Where(
                c1a, op.Mul(a_b_safe, op.CastLike(b_rmin, a_b_safe)),
                op.Where(c1b, op.Mul(b_b_safe, op.CastLike(a_rmin, b_b_safe)), nan_b),
            )
            return y_min, y_max, y_scale, y_bias

        return sira_mul


class SiraDiv_v1(OnnxOperator, domain=DOMAIN, op_type="Div", version=1):
    """SIRA range analysis handler for Div operator."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_div(
            a_min, a_max, a_scale, a_bias,
            b_min, b_max, b_scale, b_bias,
        ):
            a_si = sira.IsScaledInt(a_scale, a_bias)
            a_rmin, a_rmax = sira.ToRealRange(a_min, a_max, a_scale, a_bias)
            b_rmin, b_rmax = sira.ToRealRange(b_min, b_max, b_scale, b_bias)

            b_const = sira.IsConstant(b_rmin, b_rmax)
            zero = op.CastLike(op.Constant(value_float=0.0), b_rmin)
            b_no_zero = sira.AllTrue(op.Or(op.Greater(b_rmin, zero), op.Less(b_rmax, zero)))

            c1 = op.And(op.And(a_si, b_const), b_no_zero)

            b_safe_min = op.Where(b_no_zero, b_rmin, op.CastLike(op.Constant(value_float=1.0), b_rmin))
            b_safe_max = op.Where(b_no_zero, b_rmax, op.CastLike(op.Constant(value_float=1.0), b_rmax))
            p1 = op.Div(a_rmin, b_safe_min)
            p2 = op.Div(a_rmin, b_safe_max)
            p3 = op.Div(a_rmax, b_safe_min)
            p4 = op.Div(a_rmax, b_safe_max)

            inf = op.CastLike(op.Constant(value_float=float("inf")), a_rmin)
            ninf = op.CastLike(op.Constant(value_float=float("-inf")), a_rmin)
            r_min = op.Where(b_no_zero, op.Min(op.Min(p1, p2), op.Min(p3, p4)), ninf)
            r_max = op.Where(b_no_zero, op.Max(op.Max(p1, p2), op.Max(p3, p4)), inf)

            zero_b = op.Sub(b_rmin, b_rmin)
            a_s_safe = op.Where(a_si, a_scale, op.CastLike(op.Constant(value_float=1.0), a_scale))
            a_b_safe = op.Where(a_si, a_bias, op.CastLike(op.Constant(value_float=0.0), a_bias))
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))

            y_min = op.Where(c1, op.Add(a_min, zero_b), r_min)
            y_max = op.Where(c1, op.Add(a_max, zero_b), r_max)
            y_scale = op.Where(c1, op.Div(a_s_safe, op.CastLike(b_safe_min, a_s_safe)), nan_s)
            y_bias = op.Where(c1, op.Div(a_b_safe, op.CastLike(b_safe_min, a_b_safe)), nan_b)
            return y_min, y_max, y_scale, y_bias

        return sira_div


class SiraMatMul_v1(OnnxOperator, domain=DOMAIN, op_type="MatMul", version=1):
    """SIRA range analysis handler for MatMul operator."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)
        use_v18 = op.version >= 18

        def sira_matmul_v18(
            a_min, a_max, a_scale, a_bias,
            b_min, b_max, b_scale, b_bias,
        ):
            a_si = sira.IsScaledInt(a_scale, a_bias)
            b_si = sira.IsScaledInt(b_scale, b_bias)
            a_rmin, a_rmax = sira.ToRealRange(a_min, a_max, a_scale, a_bias)
            b_rmin, b_rmax = sira.ToRealRange(b_min, b_max, b_scale, b_bias)

            a_const = sira.IsConstant(a_rmin, a_rmax)
            b_const = sira.IsConstant(b_rmin, b_rmax)
            both_si = op.And(a_si, b_si)

            zero_a = op.CastLike(op.Constant(value_float=0.0), a_rmin)
            zero_b = op.CastLike(op.Constant(value_float=0.0), b_rmin)

            b_zero_bias = op.And(b_si, sira.IsZero(op.Where(b_si, b_bias, op.CastLike(op.Constant(value_float=1.0), b_bias))))
            a_zero_bias = op.And(a_si, sira.IsZero(op.Where(a_si, a_bias, op.CastLike(op.Constant(value_float=1.0), a_bias))))

            a_s_safe = op.Where(a_si, a_scale, op.CastLike(op.Constant(value_float=1.0), a_scale))
            b_s_safe = op.Where(b_si, b_scale, op.CastLike(op.Constant(value_float=1.0), b_scale))
            a_b_safe = op.Where(a_si, a_bias, op.CastLike(op.Constant(value_float=0.0), a_bias))
            b_b_safe = op.Where(b_si, b_bias, op.CastLike(op.Constant(value_float=0.0), b_bias))

            a_s_full = op.Add(op.Sub(a_min, a_min), op.CastLike(a_s_safe, a_min))
            b_s_full = op.Add(op.Sub(b_min, b_min), op.CastLike(b_s_safe, b_min))

            ax_m1 = op.Constant(value_ints=[-1])
            ax_m2 = op.Constant(value_ints=[-2])
            a_s_red = op.ReduceMin(a_s_full, ax_m1, keepdims=1)
            b_s_red = op.ReduceMin(b_s_full, ax_m2, keepdims=1)
            a_s_k_ok = sira.AllTrue(sira.IsEqual(a_s_red, op.ReduceMax(a_s_full, ax_m1, keepdims=1)))
            b_s_k_ok = sira.AllTrue(sira.IsEqual(b_s_red, op.ReduceMax(b_s_full, ax_m2, keepdims=1)))
            k_ok = op.And(a_s_k_ok, b_s_k_ok)

            case_bw = op.And(op.And(both_si, b_const), op.And(b_zero_bias, k_ok))
            case_aw = op.And(op.And(both_si, a_const), op.And(a_zero_bias, k_ok))

            qb_pos = op.Max(b_min, zero_b)
            qb_neg = op.Min(b_min, zero_b)
            q_bw_min = op.Add(op.MatMul(a_min, qb_pos), op.MatMul(a_max, qb_neg))
            q_bw_max = op.Add(op.MatMul(a_max, qb_pos), op.MatMul(a_min, qb_neg))

            qa_pos = op.Max(a_min, zero_a)
            qa_neg = op.Min(a_min, zero_a)
            q_aw_min = op.Add(op.MatMul(qa_pos, b_min), op.MatMul(qa_neg, b_max))
            q_aw_max = op.Add(op.MatMul(qa_pos, b_max), op.MatMul(qa_neg, b_min))

            wb_pos = op.Max(b_rmin, zero_b)
            wb_neg = op.Min(b_rmin, zero_b)
            r_bw_min = op.Add(op.MatMul(a_rmin, wb_pos), op.MatMul(a_rmax, wb_neg))
            r_bw_max = op.Add(op.MatMul(a_rmax, wb_pos), op.MatMul(a_rmin, wb_neg))

            wa_pos = op.Max(a_rmin, zero_a)
            wa_neg = op.Min(a_rmin, zero_a)
            r_aw_min = op.Add(op.MatMul(wa_pos, b_rmin), op.MatMul(wa_neg, b_rmax))
            r_aw_max = op.Add(op.MatMul(wa_pos, b_rmax), op.MatMul(wa_neg, b_rmin))

            r_min = op.Where(b_const, r_bw_min, r_aw_min)
            r_max = op.Where(b_const, r_bw_max, r_aw_max)

            s_out = op.Mul(op.CastLike(a_s_red, a_s_safe), op.CastLike(b_s_red, b_s_safe))
            b_bw_out = op.MatMul(op.Add(op.Sub(a_min, a_min), op.CastLike(a_b_safe, b_rmin)), b_rmin)
            b_aw_out = op.MatMul(a_rmin, op.Add(op.Sub(b_min, b_min), op.CastLike(b_b_safe, a_rmin)))
            b_out = op.Where(case_bw, b_bw_out, b_aw_out)

            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            si_ok = op.Or(case_bw, case_aw)
            y_min = op.Where(case_bw, q_bw_min, op.Where(case_aw, q_aw_min, r_min))
            y_max = op.Where(case_bw, q_bw_max, op.Where(case_aw, q_aw_max, r_max))
            y_scale = op.Where(si_ok, s_out, nan_s)
            y_bias = op.Where(si_ok, b_out, nan_b)
            return y_min, y_max, y_scale, y_bias

        def sira_matmul_v9(
            a_min, a_max, a_scale, a_bias,
            b_min, b_max, b_scale, b_bias,
        ):
            a_si = sira.IsScaledInt(a_scale, a_bias)
            b_si = sira.IsScaledInt(b_scale, b_bias)
            a_rmin, a_rmax = sira.ToRealRange(a_min, a_max, a_scale, a_bias)
            b_rmin, b_rmax = sira.ToRealRange(b_min, b_max, b_scale, b_bias)

            a_const = sira.IsConstant(a_rmin, a_rmax)
            b_const = sira.IsConstant(b_rmin, b_rmax)
            both_si = op.And(a_si, b_si)

            zero_a = op.CastLike(op.Constant(value_float=0.0), a_rmin)
            zero_b = op.CastLike(op.Constant(value_float=0.0), b_rmin)

            b_zero_bias = op.And(b_si, sira.IsZero(op.Where(b_si, b_bias, op.CastLike(op.Constant(value_float=1.0), b_bias))))
            a_zero_bias = op.And(a_si, sira.IsZero(op.Where(a_si, a_bias, op.CastLike(op.Constant(value_float=1.0), a_bias))))

            a_s_safe = op.Where(a_si, a_scale, op.CastLike(op.Constant(value_float=1.0), a_scale))
            b_s_safe = op.Where(b_si, b_scale, op.CastLike(op.Constant(value_float=1.0), b_scale))
            a_b_safe = op.Where(a_si, a_bias, op.CastLike(op.Constant(value_float=0.0), a_bias))
            b_b_safe = op.Where(b_si, b_bias, op.CastLike(op.Constant(value_float=0.0), b_bias))

            a_s_full = op.Add(op.Sub(a_min, a_min), op.CastLike(a_s_safe, a_min))
            b_s_full = op.Add(op.Sub(b_min, b_min), op.CastLike(b_s_safe, b_min))

            a_s_red = op.ReduceMin(a_s_full, axes=[-1], keepdims=1)
            b_s_red = op.ReduceMin(b_s_full, axes=[-2], keepdims=1)
            a_s_k_ok = sira.AllTrue(sira.IsEqual(a_s_red, op.ReduceMax(a_s_full, axes=[-1], keepdims=1)))
            b_s_k_ok = sira.AllTrue(sira.IsEqual(b_s_red, op.ReduceMax(b_s_full, axes=[-2], keepdims=1)))
            k_ok = op.And(a_s_k_ok, b_s_k_ok)

            case_bw = op.And(op.And(both_si, b_const), op.And(b_zero_bias, k_ok))
            case_aw = op.And(op.And(both_si, a_const), op.And(a_zero_bias, k_ok))

            qb_pos = op.Max(b_min, zero_b)
            qb_neg = op.Min(b_min, zero_b)
            q_bw_min = op.Add(op.MatMul(a_min, qb_pos), op.MatMul(a_max, qb_neg))
            q_bw_max = op.Add(op.MatMul(a_max, qb_pos), op.MatMul(a_min, qb_neg))

            qa_pos = op.Max(a_min, zero_a)
            qa_neg = op.Min(a_min, zero_a)
            q_aw_min = op.Add(op.MatMul(qa_pos, b_min), op.MatMul(qa_neg, b_max))
            q_aw_max = op.Add(op.MatMul(qa_pos, b_max), op.MatMul(qa_neg, b_min))

            wb_pos = op.Max(b_rmin, zero_b)
            wb_neg = op.Min(b_rmin, zero_b)
            r_bw_min = op.Add(op.MatMul(a_rmin, wb_pos), op.MatMul(a_rmax, wb_neg))
            r_bw_max = op.Add(op.MatMul(a_rmax, wb_pos), op.MatMul(a_rmin, wb_neg))

            wa_pos = op.Max(a_rmin, zero_a)
            wa_neg = op.Min(a_rmin, zero_a)
            r_aw_min = op.Add(op.MatMul(wa_pos, b_rmin), op.MatMul(wa_neg, b_rmax))
            r_aw_max = op.Add(op.MatMul(wa_pos, b_rmax), op.MatMul(wa_neg, b_rmin))

            r_min = op.Where(b_const, r_bw_min, r_aw_min)
            r_max = op.Where(b_const, r_bw_max, r_aw_max)

            s_out = op.Mul(op.CastLike(a_s_red, a_s_safe), op.CastLike(b_s_red, b_s_safe))
            b_bw_out = op.MatMul(op.Add(op.Sub(a_min, a_min), op.CastLike(a_b_safe, b_rmin)), b_rmin)
            b_aw_out = op.MatMul(a_rmin, op.Add(op.Sub(b_min, b_min), op.CastLike(b_b_safe, a_rmin)))
            b_out = op.Where(case_bw, b_bw_out, b_aw_out)

            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            si_ok = op.Or(case_bw, case_aw)
            y_min = op.Where(case_bw, q_bw_min, op.Where(case_aw, q_aw_min, r_min))
            y_max = op.Where(case_bw, q_bw_max, op.Where(case_aw, q_aw_max, r_max))
            y_scale = op.Where(si_ok, s_out, nan_s)
            y_bias = op.Where(si_ok, b_out, nan_b)
            return y_min, y_max, y_scale, y_bias

        return sira_matmul_v18 if use_v18 else sira_matmul_v9


class SiraGemm_v1(OnnxOperator, domain=DOMAIN, op_type="Gemm", version=1):
    """SIRA range analysis handler for Gemm (2-input: A, B)."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_gemm(
            a_min, a_max, a_scale, a_bias,
            b_min, b_max, b_scale, b_bias,
            alpha: float = 1.0,
            transA: int = 0,
            transB: int = 0,
        ):
            if transA != 0:
                a_min, a_max, a_scale, a_bias = sira.Transpose(a_min, a_max, a_scale, a_bias, perm=[1, 0])
            if transB != 0:
                b_min, b_max, b_scale, b_bias = sira.Transpose(b_min, b_max, b_scale, b_bias, perm=[1, 0])
            m_min, m_max, m_scale, m_bias = sira.MatMul(a_min, a_max, a_scale, a_bias, b_min, b_max, b_scale, b_bias)
            alpha_t = op.CastLike(op.Constant(value_float=alpha), m_min)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            y_min, y_max, y_scale, y_bias = sira.Mul(
                m_min, m_max, m_scale, m_bias, alpha_t, alpha_t, nan_s, nan_b
            )
            return y_min, y_max, y_scale, y_bias

        return sira_gemm


class SiraConv_v1(OnnxOperator, domain=DOMAIN, op_type="Conv", version=1):
    """SIRA range analysis handler for Conv operator (2-input: X, W)."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)
        use_v18 = op.version >= 18

        def sira_conv_v18(
            x_min, x_max, x_scale, x_bias,
            w_min, w_max, w_scale, w_bias,
            dilations: INTS = [1, 1],
            group: int = 1,
            kernel_shape: INTS = [1, 1],
            pads: INTS = [0, 0, 0, 0],
            strides: INTS = [1, 1],
            spatial_axes: INTS = [2, 3],
            weight_axes: INTS = [1, 2, 3],
        ):
            x_si = sira.IsScaledInt(x_scale, x_bias)
            w_si = sira.IsScaledInt(w_scale, w_bias)
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            w_rmin, w_rmax = sira.ToRealRange(w_min, w_max, w_scale, w_bias)

            w_const = sira.IsConstant(w_rmin, w_rmax)
            w_zero_bias = op.And(w_si, sira.IsZero(op.Where(w_si, w_bias, op.CastLike(op.Constant(value_float=1.0), w_bias))))

            x_s_safe = op.Where(x_si, x_scale, op.CastLike(op.Constant(value_float=1.0), x_scale))
            w_s_safe = op.Where(w_si, w_scale, op.CastLike(op.Constant(value_float=1.0), w_scale))
            x_b_safe = op.Where(x_si, x_bias, op.CastLike(op.Constant(value_float=0.0), x_bias))

            w_s_full = op.Add(op.Sub(w_min, w_min), op.CastLike(w_s_safe, w_min))
            w_ax = op.Constant(value_ints=weight_axes)
            w_s_red = op.ReduceMin(w_s_full, w_ax, keepdims=1)
            w_s_ok = sira.AllTrue(sira.IsEqual(w_s_red, op.ReduceMax(w_s_full, w_ax, keepdims=1)))
            w_s_out = op.Unsqueeze(
                op.Squeeze(w_s_red, op.Constant(value_ints=[1])),
                op.Constant(value_ints=[0]),
            )

            x_s_ok = sira.IsScalarValued(x_s_safe)

            si_ok = op.And(
                op.And(op.And(x_si, w_si), op.And(w_const, w_zero_bias)),
                op.And(w_s_ok, x_s_ok),
            )

            zero_w = op.CastLike(op.Constant(value_float=0.0), w_min)
            qw_pos = op.Max(w_min, zero_w)
            qw_neg = op.Min(w_min, zero_w)
            q_min = op.Add(
                op.Conv(x_min, qw_pos, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
                op.Conv(x_max, qw_neg, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
            )
            q_max = op.Add(
                op.Conv(x_max, qw_pos, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
                op.Conv(x_min, qw_neg, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
            )

            w_pos = op.Max(w_rmin, zero_w)
            w_neg = op.Min(w_rmin, zero_w)
            r_min = op.Add(
                op.Conv(x_rmin, w_pos, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
                op.Conv(x_rmax, w_neg, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
            )
            r_max = op.Add(
                op.Conv(x_rmax, w_pos, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
                op.Conv(x_rmin, w_neg, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
            )

            s_out = op.Mul(op.CastLike(op.ReduceMin(x_s_safe, keepdims=0), w_s_out), w_s_out)
            b_out = op.Conv(
                op.Add(op.Sub(x_min, x_min), op.CastLike(x_b_safe, w_rmin)),
                w_rmin,
                dilations=dilations,
                group=group,
                kernel_shape=kernel_shape,
                pads=pads,
                strides=strides,
            )

            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            y_min = op.Where(si_ok, q_min, r_min)
            y_max = op.Where(si_ok, q_max, r_max)
            y_scale = op.Where(si_ok, s_out, nan_s)
            y_bias = op.Where(si_ok, b_out, nan_b)
            return y_min, y_max, y_scale, y_bias

        def sira_conv_v9(
            x_min, x_max, x_scale, x_bias,
            w_min, w_max, w_scale, w_bias,
            dilations: INTS = [1, 1],
            group: int = 1,
            kernel_shape: INTS = [1, 1],
            pads: INTS = [0, 0, 0, 0],
            strides: INTS = [1, 1],
        ):
            x_si = sira.IsScaledInt(x_scale, x_bias)
            w_si = sira.IsScaledInt(w_scale, w_bias)
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            w_rmin, w_rmax = sira.ToRealRange(w_min, w_max, w_scale, w_bias)

            w_const = sira.IsConstant(w_rmin, w_rmax)
            w_zero_bias = op.And(w_si, sira.IsZero(op.Where(w_si, w_bias, op.CastLike(op.Constant(value_float=1.0), w_bias))))

            x_s_safe = op.Where(x_si, x_scale, op.CastLike(op.Constant(value_float=1.0), x_scale))
            w_s_safe = op.Where(w_si, w_scale, op.CastLike(op.Constant(value_float=1.0), w_scale))
            x_b_safe = op.Where(x_si, x_bias, op.CastLike(op.Constant(value_float=0.0), x_bias))

            w_s_full = op.Add(op.Sub(w_min, w_min), op.CastLike(w_s_safe, w_min))
            w_m = op.Shape(w_min, start=0, end=1)
            w_s_2d = op.Reshape(w_s_full, op.Concat(w_m, op.Constant(value_ints=[-1]), axis=0))
            w_s_1d = op.ReduceMin(w_s_2d, axes=[1], keepdims=0)
            w_s_ok = sira.AllTrue(sira.IsEqual(op.ReduceMin(w_s_2d, axes=[1], keepdims=1), op.ReduceMax(w_s_2d, axes=[1], keepdims=1)))

            x_spatial_ones = op.Add(
                op.Cast(op.Equal(op.Shape(x_min, start=2), op.Constant(value_int=-1)), to=ir.DataType.INT64),
                op.Constant(value_int=1),
            )
            out_scale_shape = op.Concat(op.Constant(value_ints=[1]), w_m, x_spatial_ones, axis=0)
            w_s_out = op.Reshape(w_s_1d, out_scale_shape)

            c_in = op.Shape(x_min, start=1, end=2)
            is_dw = op.And(
                op.Equal(op.Constant(value_int=group), op.ReduceSum(c_in, keepdims=0)),
                op.Equal(op.ReduceSum(c_in, keepdims=0), op.ReduceSum(w_m, keepdims=0)),
            )

            x_s_full = op.Add(op.Sub(x_min, x_min), op.CastLike(x_s_safe, x_min))
            x_s_nc_flat = op.Reshape(
                x_s_full,
                op.Concat(op.Shape(x_min, start=0, end=2), op.Constant(value_ints=[-1]), axis=0),
            )
            x_s_nc = op.ReduceMin(x_s_nc_flat, axes=[2], keepdims=0)
            x_s_spatial_ok = sira.AllTrue(sira.IsEqual(op.ReduceMin(x_s_nc_flat, axes=[2], keepdims=1), op.ReduceMax(x_s_nc_flat, axes=[2], keepdims=1)))

            x_s_out = op.Reshape(x_s_nc, op.Concat(op.Shape(x_min, start=0, end=2), x_spatial_ones, axis=0))
            x_s_ok = op.Where(is_dw, x_s_spatial_ok, sira.IsScalarValued(x_s_safe))

            si_ok = op.And(
                op.And(op.And(x_si, w_si), op.And(w_const, w_zero_bias)),
                op.And(w_s_ok, x_s_ok),
            )

            zero_w = op.CastLike(op.Constant(value_float=0.0), w_min)
            qw_pos = op.Max(w_min, zero_w)
            qw_neg = op.Min(w_min, zero_w)
            q_min = op.Add(
                op.Conv(x_min, qw_pos, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
                op.Conv(x_max, qw_neg, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
            )
            q_max = op.Add(
                op.Conv(x_max, qw_pos, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
                op.Conv(x_min, qw_neg, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
            )

            w_pos = op.Max(w_rmin, zero_w)
            w_neg = op.Min(w_rmin, zero_w)
            r_min = op.Add(
                op.Conv(x_rmin, w_pos, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
                op.Conv(x_rmax, w_neg, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
            )
            r_max = op.Add(
                op.Conv(x_rmax, w_pos, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
                op.Conv(x_rmin, w_neg, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
            )

            s_out = op.Mul(
                op.CastLike(op.Where(is_dw, x_s_out, op.ReduceMin(x_s_safe, keepdims=0)), w_s_out),
                w_s_out,
            )
            b_out = op.Conv(
                op.Add(op.Sub(x_min, x_min), op.CastLike(x_b_safe, w_rmin)),
                w_rmin,
                dilations=dilations,
                group=group,
                kernel_shape=kernel_shape,
                pads=pads,
                strides=strides,
            )

            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            y_min = op.Where(si_ok, q_min, r_min)
            y_max = op.Where(si_ok, q_max, r_max)
            y_scale = op.Where(si_ok, s_out, nan_s)
            y_bias = op.Where(si_ok, b_out, nan_b)
            return y_min, y_max, y_scale, y_bias

        return sira_conv_v18 if use_v18 else sira_conv_v9


class SiraDepthwiseConv_v1(OnnxOperator, domain=DOMAIN, op_type="DepthwiseConv", version=1):
    """SIRA range analysis handler for depthwise Conv (group == C_in == M)."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_dw_conv(
            x_min, x_max, x_scale, x_bias,
            w_min, w_max, w_scale, w_bias,
            dilations: INTS = [1, 1],
            group: int = 1,
            kernel_shape: INTS = [1, 1],
            pads: INTS = [0, 0, 0, 0],
            strides: INTS = [1, 1],
            spatial_axes: INTS = [2, 3],
            weight_axes: INTS = [1, 2, 3],
        ):
            x_si = sira.IsScaledInt(x_scale, x_bias)
            w_si = sira.IsScaledInt(w_scale, w_bias)
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            w_rmin, w_rmax = sira.ToRealRange(w_min, w_max, w_scale, w_bias)

            w_const = sira.IsConstant(w_rmin, w_rmax)
            w_zero_bias = op.And(w_si, sira.IsZero(op.Where(w_si, w_bias, op.CastLike(op.Constant(value_float=1.0), w_bias))))

            x_s_safe = op.Where(x_si, x_scale, op.CastLike(op.Constant(value_float=1.0), x_scale))
            w_s_safe = op.Where(w_si, w_scale, op.CastLike(op.Constant(value_float=1.0), w_scale))
            x_b_safe = op.Where(x_si, x_bias, op.CastLike(op.Constant(value_float=0.0), x_bias))

            w_s_full = op.Add(op.Sub(w_min, w_min), op.CastLike(w_s_safe, w_min))
            w_ax = op.Constant(value_ints=weight_axes)
            w_s_red = op.ReduceMin(w_s_full, w_ax, keepdims=1)
            w_s_ok = sira.AllTrue(sira.IsEqual(w_s_red, op.ReduceMax(w_s_full, w_ax, keepdims=1)))
            w_s_out = op.Unsqueeze(
                op.Squeeze(w_s_red, op.Constant(value_ints=[1])),
                op.Constant(value_ints=[0]),
            )

            x_s_full = op.Add(op.Sub(x_min, x_min), op.CastLike(x_s_safe, x_min))
            s_ax = op.Constant(value_ints=spatial_axes)
            x_s_out = op.ReduceMin(x_s_full, s_ax, keepdims=1)
            x_s_ok = sira.AllTrue(sira.IsEqual(x_s_out, op.ReduceMax(x_s_full, s_ax, keepdims=1)))

            si_ok = op.And(
                op.And(op.And(x_si, w_si), op.And(w_const, w_zero_bias)),
                op.And(w_s_ok, x_s_ok),
            )

            zero_w = op.CastLike(op.Constant(value_float=0.0), w_min)
            qw_pos = op.Max(w_min, zero_w)
            qw_neg = op.Min(w_min, zero_w)
            q_min = op.Add(
                op.Conv(x_min, qw_pos, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
                op.Conv(x_max, qw_neg, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
            )
            q_max = op.Add(
                op.Conv(x_max, qw_pos, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
                op.Conv(x_min, qw_neg, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
            )

            w_pos = op.Max(w_rmin, zero_w)
            w_neg = op.Min(w_rmin, zero_w)
            r_min = op.Add(
                op.Conv(x_rmin, w_pos, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
                op.Conv(x_rmax, w_neg, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
            )
            r_max = op.Add(
                op.Conv(x_rmax, w_pos, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
                op.Conv(x_rmin, w_neg, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides),
            )

            s_out = op.Mul(op.CastLike(x_s_out, w_s_out), w_s_out)
            b_out = op.Conv(
                op.Add(op.Sub(x_min, x_min), op.CastLike(x_b_safe, w_rmin)),
                w_rmin,
                dilations=dilations,
                group=group,
                kernel_shape=kernel_shape,
                pads=pads,
                strides=strides,
            )

            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            y_min = op.Where(si_ok, q_min, r_min)
            y_max = op.Where(si_ok, q_max, r_max)
            y_scale = op.Where(si_ok, s_out, nan_s)
            y_bias = op.Where(si_ok, b_out, nan_b)
            return y_min, y_max, y_scale, y_bias

        return sira_dw_conv


class SiraBatchNormalization_v1(OnnxOperator, domain=DOMAIN, op_type="BatchNormalization", version=1):
    """SIRA range analysis handler for inference-mode BatchNormalization."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_batchnorm(
            x_min, x_max, x_scale, x_bias,
            s_min, s_max, s_scale, s_bias,
            b_min, b_max, b_scale, b_bias,
            m_min, m_max, m_scale, m_bias,
            v_min, v_max, v_scale, v_bias,
            epsilon: float = 1e-5,
            momentum: float = 0.9,
        ):
            s_val, _ = sira.ToRealRange(s_min, s_max, s_scale, s_bias)
            b_val, _ = sira.ToRealRange(b_min, b_max, b_scale, b_bias)
            m_val, _ = sira.ToRealRange(m_min, m_max, m_scale, m_bias)
            v_val, _ = sira.ToRealRange(v_min, v_max, v_scale, v_bias)

            spatial_ones = op.Add(
                op.Cast(op.Equal(op.Shape(x_min, start=2), op.Constant(value_int=-1)), to=ir.DataType.INT64),
                op.Constant(value_int=1),
            )
            param_shape = op.Concat(
                op.Constant(value_ints=[1]),
                op.Shape(x_min, start=1, end=2),
                spatial_ones,
                axis=0,
            )
            eps_t = op.CastLike(op.Constant(value_float=epsilon), v_val)
            inv_std = op.Div(s_val, op.Sqrt(op.Add(v_val, eps_t)))
            bn_s = op.Reshape(inv_std, param_shape)
            bn_b = op.Reshape(op.Sub(b_val, op.Mul(m_val, inv_std)), param_shape)

            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            m_out_min, m_out_max, m_out_scale, m_out_bias = sira.Mul(
                x_min, x_max, x_scale, x_bias,
                bn_s, bn_s, nan_s, nan_b,
            )
            y_min, y_max, y_scale, y_bias = sira.Add(
                m_out_min, m_out_max, m_out_scale, m_out_bias,
                bn_b, bn_b, nan_s, nan_b,
            )
            return y_min, y_max, y_scale, y_bias

        return sira_batchnorm


class SiraRelu_v1(OnnxOperator, domain=DOMAIN, op_type="Relu", version=1):
    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_relu(x_min, x_max, x_scale, x_bias):
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            zero = op.CastLike(op.Constant(value_float=0.0), x_rmin)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return op.Max(x_rmin, zero), op.Max(x_rmax, zero), nan_s, nan_b

        return sira_relu


class SiraClip_v1(OnnxOperator, domain=DOMAIN, op_type="Clip", version=1):
    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_clip(
            x_min, x_max, x_scale, x_bias,
            lo_min, lo_max, lo_scale, lo_bias,
            hi_min, hi_max, hi_scale, hi_bias,
        ):
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            lo, _ = sira.ToRealRange(lo_min, lo_max, lo_scale, lo_bias)
            hi, _ = sira.ToRealRange(hi_min, hi_max, hi_scale, hi_bias)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return op.Clip(x_rmin, lo, hi), op.Clip(x_rmax, lo, hi), nan_s, nan_b

        return sira_clip


class SiraCeil_v1(OnnxOperator, domain=DOMAIN, op_type="Ceil", version=1):
    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_ceil(x_min, x_max, x_scale, x_bias):
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return op.Ceil(x_rmin), op.Ceil(x_rmax), nan_s, nan_b

        return sira_ceil


class SiraFloor_v1(OnnxOperator, domain=DOMAIN, op_type="Floor", version=1):
    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_floor(x_min, x_max, x_scale, x_bias):
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return op.Floor(x_rmin), op.Floor(x_rmax), nan_s, nan_b

        return sira_floor


class SiraRound_v1(OnnxOperator, domain=DOMAIN, op_type="Round", version=1):
    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_round(x_min, x_max, x_scale, x_bias):
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return op.Round(x_rmin), op.Round(x_rmax), nan_s, nan_b

        return sira_round


class SiraLog_v1(OnnxOperator, domain=DOMAIN, op_type="Log", version=1):
    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_log(x_min, x_max, x_scale, x_bias):
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return op.Log(x_rmin), op.Log(x_rmax), nan_s, nan_b

        return sira_log


class SiraExp_v1(OnnxOperator, domain=DOMAIN, op_type="Exp", version=1):
    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_exp(x_min, x_max, x_scale, x_bias):
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return op.Exp(x_rmin), op.Exp(x_rmax), nan_s, nan_b

        return sira_exp


class SiraSqrt_v1(OnnxOperator, domain=DOMAIN, op_type="Sqrt", version=1):
    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_sqrt(x_min, x_max, x_scale, x_bias):
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return op.Sqrt(x_rmin), op.Sqrt(x_rmax), nan_s, nan_b

        return sira_sqrt


class SiraSigmoid_v1(OnnxOperator, domain=DOMAIN, op_type="Sigmoid", version=1):
    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_sigmoid(x_min, x_max, x_scale, x_bias):
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return op.Sigmoid(x_rmin), op.Sigmoid(x_rmax), nan_s, nan_b

        return sira_sigmoid


class SiraTanh_v1(OnnxOperator, domain=DOMAIN, op_type="Tanh", version=1):
    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_tanh(x_min, x_max, x_scale, x_bias):
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return op.Tanh(x_rmin), op.Tanh(x_rmax), nan_s, nan_b

        return sira_tanh


class SiraAbs_v1(OnnxOperator, domain=DOMAIN, op_type="Abs", version=1):
    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_abs(x_min, x_max, x_scale, x_bias):
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            zero = op.CastLike(op.Constant(value_float=0.0), x_rmin)
            has_zero = op.And(op.LessOrEqual(x_rmin, zero), op.GreaterOrEqual(x_rmax, zero))
            y_min = op.Where(has_zero, zero, op.Min(op.Abs(x_rmin), op.Abs(x_rmax)))
            y_max = op.Max(op.Abs(x_rmin), op.Abs(x_rmax))
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return y_min, y_max, nan_s, nan_b

        return sira_abs


class SiraPow_v1(OnnxOperator, domain=DOMAIN, op_type="Pow", version=1):
    """SIRA range analysis handler for Pow(a, b)."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_pow(
            a_min, a_max, a_scale, a_bias,
            b_min, b_max, b_scale, b_bias,
        ):
            a_rmin, a_rmax = sira.ToRealRange(a_min, a_max, a_scale, a_bias)
            b_rmin, b_rmax = sira.ToRealRange(b_min, b_max, b_scale, b_bias)
            zero = op.CastLike(op.Constant(value_float=0.0), a_rmin)
            p1 = op.Pow(a_rmin, b_rmin)
            p2 = op.Pow(a_rmin, b_rmax)
            p3 = op.Pow(a_rmax, b_rmin)
            p4 = op.Pow(a_rmax, b_rmax)
            cand_min = op.Min(op.Min(p1, p2), op.Min(p3, p4))
            cand_max = op.Max(op.Max(p1, p2), op.Max(p3, p4))
            has_zero = op.And(
                op.And(op.LessOrEqual(a_rmin, zero), op.GreaterOrEqual(a_rmax, zero)),
                op.Greater(b_rmin, op.CastLike(op.Constant(value_float=0.0), b_rmin)),
            )
            y_min = op.Where(has_zero, op.Min(cand_min, zero), cand_min)
            y_max = op.Where(has_zero, op.Max(cand_max, zero), cand_max)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return y_min, y_max, nan_s, nan_b

        return sira_pow


class SiraSoftmax_v1(OnnxOperator, domain=DOMAIN, op_type="Softmax", version=1):
    """SIRA range analysis handler for Softmax."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_softmax(x_min, x_max, x_scale, x_bias, axes: INTS = [-1]):
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            ax = op.Constant(value_ints=axes)
            shift = op.ReduceMax(x_rmax, ax, keepdims=1)
            e_min = op.Exp(op.Sub(x_rmin, shift))
            e_max = op.Exp(op.Sub(x_rmax, shift))
            sum_min = op.ReduceSum(e_min, ax, keepdims=1)
            sum_max = op.ReduceSum(e_max, ax, keepdims=1)
            denom_for_min = op.Add(e_min, op.Sub(sum_max, e_max))
            denom_for_max = op.Add(e_max, op.Sub(sum_min, e_min))
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return op.Div(e_min, denom_for_min), op.Div(e_max, denom_for_max), nan_s, nan_b

        return sira_softmax


class SiraMaxPool_v1(OnnxOperator, domain=DOMAIN, op_type="MaxPool", version=1):
    """SIRA range analysis handler for MaxPool."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_maxpool(
            x_min, x_max, x_scale, x_bias,
            ceil_mode: int = 0,
            dilations: INTS = [1, 1],
            kernel_shape: INTS = [1, 1],
            pads: INTS = [0, 0, 0, 0],
            storage_order: int = 0,
            strides: INTS = [1, 1],
        ):
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            y_min = op.MaxPool(
                x_rmin,
                ceil_mode=ceil_mode,
                dilations=dilations,
                kernel_shape=kernel_shape,
                pads=pads,
                storage_order=storage_order,
                strides=strides,
            )
            y_max = op.MaxPool(
                x_rmax,
                ceil_mode=ceil_mode,
                dilations=dilations,
                kernel_shape=kernel_shape,
                pads=pads,
                storage_order=storage_order,
                strides=strides,
            )
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return y_min, y_max, nan_s, nan_b

        return sira_maxpool


class SiraAveragePool_v1(OnnxOperator, domain=DOMAIN, op_type="AveragePool", version=1):
    """SIRA range analysis handler for AveragePool."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)
        use_v18 = op.version >= 18

        def sira_avgpool_v18(
            x_min, x_max, x_scale, x_bias,
            ceil_mode: int = 0,
            count_include_pad: int = 0,
            kernel_shape: INTS = [1, 1],
            pads: INTS = [0, 0, 0, 0],
            strides: INTS = [1, 1],
            spatial_axes: INTS = [2, 3],
        ):
            x_si = sira.IsScaledInt(x_scale, x_bias)
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)

            r_min = op.AveragePool(
                x_rmin,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                kernel_shape=kernel_shape,
                pads=pads,
                strides=strides,
            )
            r_max = op.AveragePool(
                x_rmax,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                kernel_shape=kernel_shape,
                pads=pads,
                strides=strides,
            )

            pad_sum = op.ReduceSum(op.Constant(value_ints=pads), keepdims=0)
            no_pad = op.Or(
                op.Equal(pad_sum, op.Constant(value_int=0)),
                op.Equal(op.Constant(value_int=count_include_pad), op.Constant(value_int=1)),
            )

            k_prod = op.CastLike(
                op.ReduceProd(op.Constant(value_ints=kernel_shape), keepdims=0),
                x_min,
            )

            x_s_safe = op.Where(x_si, x_scale, op.CastLike(op.Constant(value_float=1.0), x_scale))
            x_b_safe = op.Where(x_si, x_bias, op.CastLike(op.Constant(value_float=0.0), x_bias))

            x_s_full = op.Add(op.Sub(x_min, x_min), op.CastLike(x_s_safe, x_min))
            s_ax = op.Constant(value_ints=spatial_axes)
            x_s_out = op.ReduceMin(x_s_full, s_ax, keepdims=1)
            x_s_spatial_ok = sira.AllTrue(sira.IsEqual(x_s_out, op.ReduceMax(x_s_full, s_ax, keepdims=1)))

            si_ok = op.And(op.And(x_si, x_s_spatial_ok), no_pad)
            q_min = op.Round(op.Mul(op.AveragePool(x_min, ceil_mode=ceil_mode, count_include_pad=count_include_pad, kernel_shape=kernel_shape, pads=pads, strides=strides), k_prod))
            q_max = op.Round(op.Mul(op.AveragePool(x_max, ceil_mode=ceil_mode, count_include_pad=count_include_pad, kernel_shape=kernel_shape, pads=pads, strides=strides), k_prod))

            s_out = op.Div(x_s_out, op.CastLike(k_prod, x_s_out))
            b_out = op.AveragePool(
                op.Add(op.Sub(x_min, x_min), op.CastLike(x_b_safe, x_min)),
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                kernel_shape=kernel_shape,
                pads=pads,
                strides=strides,
            )

            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return (
                op.Where(si_ok, q_min, r_min),
                op.Where(si_ok, q_max, r_max),
                op.Where(si_ok, s_out, nan_s),
                op.Where(si_ok, b_out, nan_b),
            )

        def sira_avgpool_v9(
            x_min, x_max, x_scale, x_bias,
            ceil_mode: int = 0,
            count_include_pad: int = 0,
            kernel_shape: INTS = [1, 1],
            pads: INTS = [0, 0, 0, 0],
            strides: INTS = [1, 1],
        ):
            x_si = sira.IsScaledInt(x_scale, x_bias)
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)

            r_min = op.AveragePool(
                x_rmin,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                kernel_shape=kernel_shape,
                pads=pads,
                strides=strides,
            )
            r_max = op.AveragePool(
                x_rmax,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                kernel_shape=kernel_shape,
                pads=pads,
                strides=strides,
            )

            pad_sum = op.ReduceSum(op.Constant(value_ints=pads), keepdims=0)
            no_pad = op.Or(
                op.Equal(pad_sum, op.Constant(value_int=0)),
                op.Equal(op.Constant(value_int=count_include_pad), op.Constant(value_int=1)),
            )

            k_prod = op.CastLike(
                op.ReduceProd(op.Constant(value_ints=kernel_shape), keepdims=0),
                x_min,
            )

            x_s_safe = op.Where(x_si, x_scale, op.CastLike(op.Constant(value_float=1.0), x_scale))
            x_b_safe = op.Where(x_si, x_bias, op.CastLike(op.Constant(value_float=0.0), x_bias))

            x_s_full = op.Add(op.Sub(x_min, x_min), op.CastLike(x_s_safe, x_min))
            x_s_nc_flat = op.Reshape(
                x_s_full,
                op.Concat(op.Shape(x_min, start=0, end=2), op.Constant(value_ints=[-1]), axis=0),
            )
            x_s_nc = op.ReduceMin(x_s_nc_flat, axes=[2], keepdims=0)
            x_s_spatial_ok = sira.AllTrue(sira.IsEqual(op.ReduceMin(x_s_nc_flat, axes=[2], keepdims=1), op.ReduceMax(x_s_nc_flat, axes=[2], keepdims=1)))

            spatial_ones = op.Add(
                op.Cast(op.Equal(op.Shape(x_min, start=2), op.Constant(value_int=-1)), to=ir.DataType.INT64),
                op.Constant(value_int=1),
            )
            x_s_out = op.Reshape(
                x_s_nc,
                op.Concat(op.Shape(x_min, start=0, end=2), spatial_ones, axis=0),
            )

            si_ok = op.And(op.And(x_si, x_s_spatial_ok), no_pad)
            q_min = op.Round(op.Mul(op.AveragePool(x_min, ceil_mode=ceil_mode, count_include_pad=count_include_pad, kernel_shape=kernel_shape, pads=pads, strides=strides), k_prod))
            q_max = op.Round(op.Mul(op.AveragePool(x_max, ceil_mode=ceil_mode, count_include_pad=count_include_pad, kernel_shape=kernel_shape, pads=pads, strides=strides), k_prod))

            s_out = op.Div(x_s_out, op.CastLike(k_prod, x_s_out))
            b_out = op.AveragePool(
                op.Add(op.Sub(x_min, x_min), op.CastLike(x_b_safe, x_min)),
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                kernel_shape=kernel_shape,
                pads=pads,
                strides=strides,
            )

            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return (
                op.Where(si_ok, q_min, r_min),
                op.Where(si_ok, q_max, r_max),
                op.Where(si_ok, s_out, nan_s),
                op.Where(si_ok, b_out, nan_b),
            )

        return sira_avgpool_v18 if use_v18 else sira_avgpool_v9


class SiraPad_v1(OnnxOperator, domain=DOMAIN, op_type="Pad", version=1):
    """SIRA range analysis handler for Pad (opset < 11 attribute-based)."""

    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)
        use_v11 = op.version >= 11

        def sira_pad_v11(
            x_min, x_max, x_scale, x_bias,
            pads_min, pads_max, pads_scale, pads_bias,
            val_min, val_max, val_scale, val_bias,
            mode: str = "constant",
        ):
            x_si = sira.IsScaledInt(x_scale, x_bias)
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)
            r_min = op.Pad(x_rmin, pads_min, val_min, mode=mode)
            r_max = op.Pad(x_rmax, pads_min, val_max, mode=mode)

            x_b_safe = op.Where(x_si, x_bias, op.CastLike(op.Constant(value_float=0.0), x_bias))
            bias_eq_val = sira.AllTrue(sira.IsEqual(x_b_safe, op.CastLike(val_min, x_b_safe)))
            si_ok = op.And(x_si, bias_eq_val)

            q_min = op.Pad(x_min, pads_min, op.CastLike(op.Constant(value_float=0.0), x_min), mode=mode)
            q_max = op.Pad(x_max, pads_min, op.CastLike(op.Constant(value_float=0.0), x_max), mode=mode)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return (
                op.Where(si_ok, q_min, r_min),
                op.Where(si_ok, q_max, r_max),
                op.Where(si_ok, x_scale, nan_s),
                op.Where(si_ok, x_bias, nan_b),
            )

        def sira_pad_v9(
            x_min, x_max, x_scale, x_bias,
            mode: str = "constant",
            pads: INTS = [0, 0, 0, 0, 0, 0, 0, 0],
            value: float = 0.0,
        ):
            x_si = sira.IsScaledInt(x_scale, x_bias)
            x_rmin, x_rmax = sira.ToRealRange(x_min, x_max, x_scale, x_bias)

            r_min = op.Pad(x_rmin, mode=mode, pads=pads, value=value)
            r_max = op.Pad(x_rmax, mode=mode, pads=pads, value=value)

            x_b_safe = op.Where(x_si, x_bias, op.CastLike(op.Constant(value_float=0.0), x_bias))
            bias_eq_val = sira.AllTrue(sira.IsEqual(x_b_safe, op.CastLike(op.Constant(value_float=value), x_b_safe)))
            si_ok = op.And(x_si, bias_eq_val)

            q_min = op.Pad(x_min, mode=mode, pads=pads, value=0.0)
            q_max = op.Pad(x_max, mode=mode, pads=pads, value=0.0)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return (
                op.Where(si_ok, q_min, r_min),
                op.Where(si_ok, q_max, r_max),
                op.Where(si_ok, x_scale, nan_s),
                op.Where(si_ok, x_bias, nan_b),
            )

        return sira_pad_v11 if use_v11 else sira_pad_v9


class SiraReshape_v1(OnnxOperator, domain=DOMAIN, op_type="Reshape", version=1):
    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_reshape(
            x_min, x_max, x_scale, x_bias,
            s_min, s_max, s_scale, s_bias,
        ):
            x_si = sira.IsScaledInt(x_scale, x_bias)
            y_min = op.Reshape(x_min, s_min)
            y_max = op.Reshape(x_max, s_min)

            s_safe = op.Where(x_si, x_scale, op.CastLike(op.Constant(value_float=1.0), x_scale))
            b_safe = op.Where(x_si, x_bias, op.CastLike(op.Constant(value_float=0.0), x_bias))
            s_scalar = sira.IsScalarValued(s_safe)
            b_scalar = sira.IsScalarValued(b_safe)

            s_exp = op.Reshape(op.Add(op.Sub(x_min, x_min), op.CastLike(s_safe, x_min)), s_min)
            b_exp = op.Reshape(op.Add(op.Sub(x_min, x_min), op.CastLike(b_safe, x_min)), s_min)

            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            y_scale = op.Where(x_si, op.Where(s_scalar, op.ReduceMin(s_safe, keepdims=0), s_exp), nan_s)
            y_bias = op.Where(x_si, op.Where(b_scalar, op.ReduceMin(b_safe, keepdims=0), b_exp), nan_b)
            return y_min, y_max, y_scale, y_bias

        return sira_reshape


class SiraFlatten_v1(OnnxOperator, domain=DOMAIN, op_type="Flatten", version=1):
    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_flatten(x_min, x_max, x_scale, x_bias, axis: int = 1):
            x_si = sira.IsScaledInt(x_scale, x_bias)
            y_min = op.Flatten(x_min, axis=axis)
            y_max = op.Flatten(x_max, axis=axis)

            s_safe = op.Where(x_si, x_scale, op.CastLike(op.Constant(value_float=1.0), x_scale))
            b_safe = op.Where(x_si, x_bias, op.CastLike(op.Constant(value_float=0.0), x_bias))
            s_scalar = sira.IsScalarValued(s_safe)
            b_scalar = sira.IsScalarValued(b_safe)

            s_exp = op.Flatten(op.Add(op.Sub(x_min, x_min), op.CastLike(s_safe, x_min)), axis=axis)
            b_exp = op.Flatten(op.Add(op.Sub(x_min, x_min), op.CastLike(b_safe, x_min)), axis=axis)

            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            y_scale = op.Where(x_si, op.Where(s_scalar, op.ReduceMin(s_safe, keepdims=0), s_exp), nan_s)
            y_bias = op.Where(x_si, op.Where(b_scalar, op.ReduceMin(b_safe, keepdims=0), b_exp), nan_b)
            return y_min, y_max, y_scale, y_bias

        return sira_flatten


class SiraTranspose_v1(OnnxOperator, domain=DOMAIN, op_type="Transpose", version=1):
    @staticmethod
    def script(op: Opset):
        sira = Opset(DOMAIN, 1)

        def sira_transpose(x_min, x_max, x_scale, x_bias, perm: INTS = []):
            x_si = sira.IsScaledInt(x_scale, x_bias)
            y_min = op.Transpose(x_min, perm=perm)
            y_max = op.Transpose(x_max, perm=perm)

            s_safe = op.Where(x_si, x_scale, op.CastLike(op.Constant(value_float=1.0), x_scale))
            b_safe = op.Where(x_si, x_bias, op.CastLike(op.Constant(value_float=0.0), x_bias))
            s_scalar = sira.IsScalarValued(s_safe)
            b_scalar = sira.IsScalarValued(b_safe)

            s_exp = op.Transpose(op.Add(op.Sub(x_min, x_min), op.CastLike(s_safe, x_min)), perm=perm)
            b_exp = op.Transpose(op.Add(op.Sub(x_min, x_min), op.CastLike(b_safe, x_min)), perm=perm)

            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            y_scale = op.Where(x_si, op.Where(s_scalar, op.ReduceMin(s_safe, keepdims=0), s_exp), nan_s)
            y_bias = op.Where(x_si, op.Where(b_scalar, op.ReduceMin(b_safe, keepdims=0), b_exp), nan_b)
            return y_min, y_max, y_scale, y_bias

        return sira_transpose


class SiraShape_v1(OnnxOperator, domain=DOMAIN, op_type="Shape", version=1):
    @staticmethod
    def script(op: Opset):
        def sira_shape(x_min, x_max, x_scale, x_bias):
            s_min = op.Shape(x_min)
            s_max = op.Shape(x_max)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return s_min, s_max, nan_s, nan_b

        return sira_shape


class SiraGather_v1(OnnxOperator, domain=DOMAIN, op_type="Gather", version=1):
    @staticmethod
    def script(op: Opset):
        def sira_gather(
            x_min, x_max, x_scale, x_bias,
            idx_min, idx_max, idx_scale, idx_bias,
            axis: int = 0,
        ):
            y_min = op.Gather(x_min, idx_min, axis=axis)
            y_max = op.Gather(x_max, idx_min, axis=axis)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return y_min, y_max, nan_s, nan_b

        return sira_gather


class SiraUnsqueeze_v1(OnnxOperator, domain=DOMAIN, op_type="Unsqueeze", version=1):
    @staticmethod
    def script(op: Opset):
        use_v13 = op.version >= 13

        def sira_unsqueeze_v13(x_min, x_max, x_scale, x_bias, axes: INTS = [0]):
            ax = op.Constant(value_ints=axes)
            y_min = op.Unsqueeze(x_min, ax)
            y_max = op.Unsqueeze(x_max, ax)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return y_min, y_max, nan_s, nan_b

        def sira_unsqueeze_v9(x_min, x_max, x_scale, x_bias, axes: INTS = [0]):
            y_min = op.Unsqueeze(x_min, axes=axes)
            y_max = op.Unsqueeze(x_max, axes=axes)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return y_min, y_max, nan_s, nan_b

        return sira_unsqueeze_v13 if use_v13 else sira_unsqueeze_v9


class SiraConcat2_v1(OnnxOperator, domain=DOMAIN, op_type="Concat2", version=1):
    """SIRA range analysis handler for 2-input Concat."""

    @staticmethod
    def script(op: Opset):
        def sira_concat2(
            a_min, a_max, a_scale, a_bias,
            b_min, b_max, b_scale, b_bias,
            axis: int = 0,
        ):
            y_min = op.Concat(a_min, b_min, axis=axis)
            y_max = op.Concat(a_max, b_max, axis=axis)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return y_min, y_max, nan_s, nan_b

        return sira_concat2


class SiraIdentity_v1(OnnxOperator, domain=DOMAIN, op_type="Identity", version=1):
    @staticmethod
    def script(op: Opset):
        def sira_identity(x_min, x_max, x_scale, x_bias):
            return x_min, x_max, x_scale, x_bias

        return sira_identity


class SiraCast_v1(OnnxOperator, domain=DOMAIN, op_type="Cast", version=1):
    @staticmethod
    def script(op: Opset):
        def sira_cast(x_min, x_max, x_scale, x_bias, to: int = 1):
            y_min = op.Cast(x_min, to=to)
            y_max = op.Cast(x_max, to=to)
            nan_s = op.Constant(value_float=float("nan"))
            nan_b = op.Constant(value_float=float("nan"))
            return y_min, y_max, nan_s, nan_b

        return sira_cast
