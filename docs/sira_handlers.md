# SIRA Range Analysis Handler Specification

This document specifies the Scaled-Integer Range Analysis (SIRA) operator handlers implemented in [`onnx_passes/ops/sira.py`](../onnx_passes/ops/sira.py) (domain `onnx_passes.ops.sira`) and orchestrated by [`RangeAnalysis_v1`](../onnx_passes/passes/sira/__init__.py).

## 1. Tensor Representation

Every tensor $v$ in the original ONNX graph is represented in the range analysis graph by a 4-tuple of tensors:
$$(v_{\min}, v_{\max}, v_{\text{scale}}, v_{\text{bias}})$$

Two modes exist depending on whether $v$ has an underlying scaled-integer representation:

1. **Scaled-Integer Tensor**:
   - $v_{\text{scale}}$ and $v_{\text{bias}}$ are finite (no `NaN` entries) and have shapes broadcastable to $v$.
   - $[v_{\min}, v_{\max}]$ represents the **integer** interval $[q_v, \bar{q}_v]$.
   - The real-valued interval is recovered as:
     $$\underline{v} = \min(v_{\text{scale}} \cdot v_{\min} + v_{\text{bias}},\; v_{\text{scale}} \cdot v_{\max} + v_{\text{bias}})$$
     $$\bar{v} = \max(v_{\text{scale}} \cdot v_{\min} + v_{\text{bias}},\; v_{\text{scale}} \cdot v_{\max} + v_{\text{bias}})$$

2. **Non-Scaled-Integer (Real-Valued) Tensor**:
   - $v_{\text{scale}}$ and $v_{\text{bias}}$ are `NaN`.
   - $[v_{\min}, v_{\max}]$ represents the **real-valued** interval $[\underline{v}, \bar{v}]$.

3. **Constant Tensor** $C$:
   - Represented as $(C, C, \text{NaN}, \text{NaN})$ (a point range with $v_{\min} = v_{\max}$).

4. **Shape / Granularity Invariant**:
   - $v_{\min}$ and $v_{\max}$ are produced with per-element shapes whenever possible (matching $v$), or broadcastable shapes to $v$.
   - $v_{\text{scale}}$ and $v_{\text{bias}}$ have broadcastable shapes to $v$ (e.g., scalar `[]` for per-tensor, `[1, C, 1, 1]` for per-channel, or full shape when spatially varying), except at graph outputs where they are broadcast to the output tensor shape for static ONNX shape compatibility.

---

## 2. Helper Functions (`onnx_passes.ops.sira`)

| Helper | Signature | Operation |
| --- | --- | --- |
| `IsEqual` | `(a, b) -> bool` | Elementwise equality (`Equal` in opset $\ge 11$, `Not(Or(Less, Greater))` in opset $< 11$). |
| `AllTrue` | `(cond) -> scalar bool` | Returns `True` iff all elements of `cond` are `True`. |
| `IsScaledInt` | `(scale, bias) -> scalar bool` | Returns `True` iff neither `scale` nor `bias` contains `NaN`. |
| `IsConstant` | `(v_min, v_max) -> scalar bool` | Returns `True` iff `v_min == v_max` everywhere. |
| `IsZero` | `(x) -> scalar bool` | Returns `True` iff `x == 0` everywhere. |
| `IsScalarValued` | `(x) -> scalar bool` | Returns `True` iff `ReduceMin(x) == ReduceMax(x)`. |
| `ToRealRange` | `(v_min, v_max, v_scale, v_bias) -> (r_min, r_max)` | Converts any 4-tuple to real-valued `[r_min, r_max]`. |
| `QuantEval` | `(x, scale, zeropoint, bitwidth) -> q` | Evaluates the integer quantization map $(g \circ f^{-1})(x)$. |
| `FloatQuantEval` | `(val, s, exp_bw, man_bw, exp_bias, max_val) -> y` | Evaluates saturating round-to-even mini-float quantization on a single tensor. |

---

## 3. Quantization Operators

### 3.1 `Quant` (`SiraQuant_v1`)
- **Inputs**: `x`, `s` (scale), `z` (zero-point), `bw` (bitwidth), each as a 4-tuple.
- **Attributes**: `signed` (int, default `1`), `narrow` (int, default `0`), `rounding_mode` (int enum `0=ROUND, 1=CEIL, 2=FLOOR, 3=UP, 4=DOWN, 5=HALF_UP, 6=HALF_DOWN`).
- **Constraints**:
  - Always produces a scaled-integer output regardless of whether `x` is scaled-integer.
  - `s`, `z`, `bw` are expected to be constant parameters (broadcastable to `x`).
- **Operation**:
  1. Convert `x` to real-valued $[x_{\text{rmin}}, x_{\text{rmax}}]$ via `ToRealRange`.
  2. Evaluate the monotonic integer quantizer $(g \circ f^{-1})(v) = \text{round}(\text{clip}(v / s + z, q_{\text{lo}}, q_{\text{hi}}))$ at both endpoints $x_{\text{rmin}}$ and $x_{\text{rmax}}$ (handling signed 1-bit bipolar mapping $q \ge 0 \mapsto +1, q < 0 \mapsto -1$).
  3. Output $y_{\min} = \min(q_{\text{low}}, q_{\text{high}})$, $y_{\max} = \max(q_{\text{low}}, q_{\text{high}})$, $y_{\text{scale}} = s$, $y_{\text{bias}} = -s \cdot z$.

### 3.2 `BipolarQuant` (`SiraBipolarQuant_v1`)
- **Inputs**: `x`, `s` (scale).
- **Constraints**: Always produces a scaled-integer output; `s` is constant.
- **Operation**:
  1. Convert `x` to real-valued $[x_{\text{rmin}}, x_{\text{rmax}}]$.
  2. Map $v \mapsto +1$ if $v \ge 0$ else $-1$ at both endpoints.
  3. Output $y_{\min} = \min(q_{\text{low}}, q_{\text{high}})$, $y_{\max} = \max(q_{\text{low}}, q_{\text{high}})$, $y_{\text{scale}} = s$, $y_{\text{bias}} = 0$.

### 3.3 `Trunc` (`SiraTrunc_v1`)
- **Inputs**: `x`, `s` (scale), `z` (zero-point), `in_bw`, `out_bw` (5-input QONNX `Trunc` v1 signature).
- **Attributes**: `rounding_mode` (int enum, default `2=FLOOR`; explicitly handles `0=ROUND`, `1=CEIL`, `2=FLOOR`, defaulting other values to `FLOOR`).
- **Constraints**: Always produces a scaled-integer output; `s`, `z`, `in_bw`, `out_bw` are constant.
- **Operation**:
  1. Convert `x` to real-valued $[x_{\text{rmin}}, x_{\text{rmax}}]$.
  2. Compute $\text{trunc\_scale} = 2^{\text{in\_bw} - \text{out\_bw}}$ and evaluate $\text{round\_mode}(\text{Round}(v / s + z) / \text{trunc\_scale})$ at both endpoints.
  3. Output $y_{\min} = \min(q_{\text{low}}, q_{\text{high}})$, $y_{\max} = \max(q_{\text{low}}, q_{\text{high}})$, $y_{\text{scale}} = s$, $y_{\text{bias}} = -s \cdot z$.

### 3.4 `FloatQuant` (`SiraFloatQuant_v1`)
- **Inputs**: `x`, `s`, `exp_bw`, `man_bw`, `exp_bias`, `max_val`.
- **Attributes**: `has_inf`, `has_nan`, `has_subnormal`, `rounding_mode`, `saturation` (accepted for schema compatibility; `FloatQuantEval_v1` currently implements saturating round-to-even, i.e., `saturation=1`, `rounding_mode="round"`).
- **Constraints**: Floating-point quantization does **not** produce an integer component; output `scale` and `bias` are always `NaN`.
- **Operation**:
  1. Convert `x` to real-valued $[x_{\text{rmin}}, x_{\text{rmax}}]$.
  2. Evaluate `FloatQuantEval` at both endpoints and output $[\min(y_{\text{low}}, y_{\text{high}}), \max(y_{\text{low}}, y_{\text{high}}), \text{NaN}, \text{NaN}]$.

---

## 4. Elementwise Affine & Arithmetic Operators

### 4.1 `Add` (`SiraAdd_v1`)
- **Inputs**: `a`, `b`.
- **Constraints for Scaled-Integer Propagation**:
  - **Case 1 (Scaled-Integer + Constant)**: One input (say `a`) is scaled-integer and the other (`b`) is constant ($b_{\text{rmin}} = b_{\text{rmax}}$).
  - **Case 2 (Integer-Related Scales)**: Both `a` and `b` are scaled-integers and either $s_b = k \cdot s_a$ or $s_a = k \cdot s_b$ for integer $k \in \mathbb{Z}$ (checked within relative tolerance $10^{-5}$ with nonzero divisor).
  - **Fallback**: Any other case falls back to real-valued interval addition.
- **Operation**:
  - **Case 1**: $q_y = q_a$, $s_y = s_a$, $b_y = b_a + b_{\text{real}}$ (with $q_a$ broadcast to output shape).
  - **Case 2** (for $s_b = k \cdot s_a$): $q_{y,\min} = q_{a,\min} + \min(k \cdot q_{b,\min}, k \cdot q_{b,\max})$, $q_{y,\max} = q_{a,\max} + \max(k \cdot q_{b,\min}, k \cdot q_{b,\max})$, $s_y = s_a$, $b_y = b_a + b_b$ (symmetric for $s_a = k \cdot s_b$).
  - **Fallback**: $[a_{\text{rmin}} + b_{\text{rmin}}, a_{\text{rmax}} + b_{\text{rmax}}, \text{NaN}, \text{NaN}]$.

### 4.2 `Neg` (`SiraNeg_v1`)
- **Inputs**: `x`.
- **Constraints**: Always preserves scaled-integer status.
- **Operation**: $(-x_{\max}, -x_{\min}, x_{\text{scale}}, -x_{\text{bias}})$.

### 4.3 `Sub` (`SiraSub_v1`)
- **Inputs**: `a`, `b`.
- **Constraints & Operation**: Implemented as `SiraAdd(a, SiraNeg(b))`.

### 4.4 `Mul` (`SiraMul_v1`)
- **Inputs**: `a`, `b`.
- **Constraints for Scaled-Integer Propagation**: One input is scaled-integer and the other is constant ($v_{\min} = v_{\max}$).
- **Operation**:
  - **Scaled-Integer**: $q_y = q_a$ (broadcast to output shape), $s_y = s_a \cdot b_{\text{real}}$, $b_y = b_a \cdot b_{\text{real}}$.
  - **Fallback**: Real-valued interval product $[\min(P), \max(P), \text{NaN}, \text{NaN}]$ over $P = \{a_{\text{rmin}} b_{\text{rmin}}, a_{\text{rmin}} b_{\text{rmax}}, a_{\text{rmax}} b_{\text{rmin}}, a_{\text{rmax}} b_{\text{rmax}}\}$.

### 4.5 `Div` (`SiraDiv_v1`)
- **Inputs**: `a`, `b`.
- **Constraints for Scaled-Integer Propagation**: `a` is scaled-integer, `b` is constant, and $0 \notin [b_{\text{rmin}}, b_{\text{rmax}}]$.
- **Operation**:
  - **Scaled-Integer**: $q_y = q_a$, $s_y = s_a / b_{\text{real}}$, $b_y = b_a / b_{\text{real}}$.
  - **Fallback**: If $0 \notin [b_{\text{rmin}}, b_{\text{rmax}}]$, $[\min(Q), \max(Q), \text{NaN}, \text{NaN}]$ over the four endpoint quotients; otherwise $[-\infty, +\infty, \text{NaN}, \text{NaN}]$.

---

## 5. Linear, Matrix, Convolution, and Normalization Operators

### 5.1 `MatMul` (`SiraMatMul_v1`)
- **Inputs**: `a`, `b`.
- **Constraints for Scaled-Integer Propagation**:
  1. Both `a` and `b` are scaled-integers.
  2. One operand is constant weights $W$ with **zero bias** ($b_W = 0$ everywhere).
  3. Scales do not vary along the contraction axis $K$ (axis `-1` of `a`, axis `-2` of `b`).
- **Operation**:
  - **Scaled-Integer** (for $B = W$ constant weights):
    - Split $q_W^+ = \max(q_W, 0)$, $q_W^- = \min(q_W, 0)$.
    - $q_{Y,\min} = q_{A,\min} q_W^+ + q_{A,\max} q_W^-$, $q_{Y,\max} = q_{A,\max} q_W^+ + q_{A,\min} q_W^-$.
    - $s_Y = s_A \cdot s_B$ (reduced along $K$), $b_Y = b_A^{\text{full}} \cdot W_{\text{real}}$.
  - **Fallback**: Real-valued dot-product bounds using $W^+ = \max(W_{\text{real}}, 0)$, $W^- = \min(W_{\text{real}}, 0)$, and `NaN` scale/bias.

### 5.2 `Gemm` (`SiraGemm_v1` / decomposed in `RangeAnalysis_v1`)
- **Inputs**: `a`, `b`, optional `c`.
- **Attributes**: `alpha`, `beta`, `transA`, `transB`.
- **Operation**: `RangeAnalysis_v1` decomposes `Gemm` directly into `SiraTranspose` (if `transA`/`transB`), `SiraMatMul`, `SiraMul` (if `alpha != 1` or `beta != 1`), and `SiraAdd` (if `c` is present), avoiding dynamic `If` subgraphs during ONNX shape inference. (`SiraGemm_v1` is also provided in `onnx_passes.ops.sira` for 2-input `Gemm`.)

### 5.3 `Conv` (`SiraConv_v1`) and `DepthwiseConv` (`SiraDepthwiseConv_v1`)
- **Inputs**: `x`, `w` (optional bias `b` is added via `SiraReshape` + `SiraAdd`).
- **Attributes**: `dilations`, `group`, `kernel_shape`, `pads`, `strides`, `spatial_axes`, `weight_axes`.
- **Constraints for Scaled-Integer Propagation**:
  1. Both `x` and `w` are scaled-integers.
  2. `w` is constant with **zero bias** ($b_W = 0$).
  3. `w` has at most per-output-channel scaling (constant over input-channel and spatial axes `weight_axes`).
  4. For dense/grouped `Conv`: `x` has per-tensor (scalar-valued) scaling. For `DepthwiseConv` (`group == C_in == M`): `x` may have per-channel scaling (constant over `spatial_axes`).
- **Operation**:
  - **Scaled-Integer**:
    - Split $q_W^+ = \max(q_W, 0)$, $q_W^- = \min(q_W, 0)$.
    - $q_{Y,\min} = \text{Conv}(q_{X,\min}, q_W^+) + \text{Conv}(q_{X,\max}, q_W^-)$, $q_{Y,\max} = \text{Conv}(q_{X,\max}, q_W^+) + \text{Conv}(q_{X,\min}, q_W^-)$.
    - $s_Y = s_X \cdot s_W$ (per-channel $[1, M, 1, \dots]$), $b_Y = \text{Conv}(b_X^{\text{full}}, W_{\text{real}})$.
  - **Fallback**: Real-valued `Conv` bounds with $W^+, W^-$, and `NaN` scale/bias.

### 5.4 `BatchNormalization` (`SiraBatchNormalization_v1`)
- **Inputs**: `x`, `scale`, `b`, `mean`, `var`.
- **Attributes**: `epsilon`, `momentum`.
- **Constraints**: Inference mode (single output); parameters are constants. Propagates scaled-integers whenever `x` is scaled-integer.
- **Operation**:
  - **Scaled-Integer**: Computes per-channel affine parameters $\gamma' = \text{scale} / \sqrt{\text{var} + \epsilon}$ and $\beta' = b - \text{mean} \cdot \gamma'$, then applies `SiraAdd(SiraMul(x, \gamma'), \beta')`.
  - **Non-Scaled-Integer Fallback**: Evaluates $(x_{\text{rmin}} - \text{mean}) \cdot \gamma' + b$ and $(x_{\text{rmax}} - \text{mean}) \cdot \gamma' + b$ directly (taking elementwise $\min/\max$ to handle $\gamma' < 0$) and outputs `NaN` scale/bias. Evaluating $(x - \text{mean}) \cdot \gamma' + b$ directly avoids float32 catastrophic cancellation when `var` is near zero.

---

## 6. Pooling and Padding Operators

### 6.1 `MaxPool` (`SiraMaxPool_v1`)
- **Constraints for Scaled-Integer Propagation**: `x` is scaled-integer, `x_scale > 0` everywhere, and both `x_scale` and `x_bias` are spatially constant across `spatial_axes`.
- **Operation**:
  - **Scaled-Integer**: $[\text{MaxPool}(q_{X,\min}),\; \text{MaxPool}(q_{X,\max}),\; s_X,\; b_X]$.
  - **Fallback**: $[\text{MaxPool}(x_{\text{rmin}}), \text{MaxPool}(x_{\text{rmax}}), \text{NaN}, \text{NaN}]$.

### 6.2 `AveragePool` (`SiraAveragePool_v1`)
- **Constraints for Scaled-Integer Propagation**: `x` is scaled-integer, `x_scale` is spatially constant, and window divisor $K = \prod \text{kernel\_shape}$ is constant ($\sum \text{pads} = 0$ or `count_include_pad == 1`).
- **Operation**:
  - **Scaled-Integer**: $q_{Y,\min} = \text{Round}(\text{AvgPool}(q_{X,\min}) \cdot K)$, $q_{Y,\max} = \text{Round}(\text{AvgPool}(q_{X,\max}) \cdot K)$, $s_Y = s_X / K$, $b_Y = \text{AvgPool}(b_X^{\text{full}})$.
  - **Fallback**: $[\text{AvgPool}(x_{\text{rmin}}), \text{AvgPool}(x_{\text{rmax}}), \text{NaN}, \text{NaN}]$.

### 6.3 `Pad` (`SiraPad_v1`)
- **Inputs**: `x`, `pads`, `constant_value` (`RangeAnalysis_v1` normalizes opset `< 11` attribute-based `Pad` into this 3-input form).
- **Constraints for Scaled-Integer Propagation**: `x` is scaled-integer and `constant_value == x_bias` everywhere (checked regardless of `mode`).
- **Operation**: Pads $q_X$ with `0` (preserving $s_X, b_X$) when constraints hold; otherwise pads $[x_{\text{rmin}}, x_{\text{rmax}}]$ with `constant_value` and outputs `NaN` scale/bias.

---

## 7. Non-Linear & Monotonic Elementwise Operators

### 7.1 `Max` and `Min` (`SiraMax_v1`, `SiraMin_v1`)
- **Variadic Dispatch**: Binary inputs use `SiraMax_v1`/`SiraMin_v1` directly; $> 2$ inputs are chained pairwise; single-input nodes pass through via `SiraIdentity_v1`.
- **Constraints for Scaled-Integer Propagation**:
  1. **Inactive constant**: If one operand is a constant $c$ that never becomes active ($c \le a_{\text{rmin}}$ everywhere for `Max`; $c \ge a_{\text{rmax}}$ everywhere for `Min`) and the other operand is scaled-integer, the active operand's 4-tuple is passed through (broadcast to the output shape).
  2. **Grid-aligned constant**: If one operand is scaled-integer $(q, s, b)$ with $s \ne 0$ everywhere and the other is a constant $c$ such that $k = (c - b) / s$ is an integer everywhere within relative tolerance $10^{-5} \cdot \max(1, |k|)$, the output retains $(s, b)$ with per-element sign-aware clamping: for `Max`, `Where(s > 0, max(q, k), min(q, k))` applied to both endpoints and reordered so $q_{\min} \le q_{\max}$ (`Min` is symmetric). Either argument position can be the constant.
  3. **Identical scaled-int operands**: If both operands are scaled-integer with equal `scale` and `bias` everywhere, `Max`/`Min` is applied directly to $q$ (with the same sign-aware reordering), preserving $(s, b)$.
- **Fallback**: Real-valued $[\max(a_{\text{rmin}}, b_{\text{rmin}}), \max(a_{\text{rmax}}, b_{\text{rmax}})]$ (or $\min$) with `NaN` scale/bias.

### 7.2 `Relu` and `Clip` (`SiraRelu_v1`, `SiraClip_v1`)
- **`Relu` (`SiraRelu_v1`)**: Equivalent to `SiraMax_v1(x, 0)`, preserving $(s_X, b_X)$ when $0$ is on $x$'s grid or inactive ($x_{\text{rmin}} \ge 0$), and falling back to `NaN` scale/bias otherwise.
- **`Clip` (`SiraClip_v1`)**: Composes `SiraMin_v1(SiraMax_v1(x, lo), hi)` (missing `lo`/`hi` default to $-\infty$ / $+\infty$, which are always inactive).

### 7.3 Integer-Producing Rounding / Sign Operators (`Round`, `Floor`, `Ceil`, `Sign`)
- **Operation**: Convert input to real-valued $[x_{\text{rmin}}, x_{\text{rmax}}]$ via `ToRealRange`, apply the monotonic operator $R \in \{\text{Round}, \text{Floor}, \text{Ceil}, \text{Sign}\}$ to both endpoints, and always output integer `scale = 1.0`, `bias = 0.0` (whether the input was scaled-integer or real-valued).

### 7.4 Real-Valued Non-Linear Operators (Always `NaN` Scale/Bias)
All remaining non-linear operators below convert inputs to real-valued $[x_{\text{rmin}}, x_{\text{rmax}}]$ and output `NaN` for `scale` and `bias`:

| Operator | Handler | Interval Formula |
| --- | --- | --- |
| `Log` | `SiraLog_v1` | $[\ln(x_{\text{rmin}}),\; \ln(x_{\text{rmax}})]$ |
| `Exp` | `SiraExp_v1` | $[\exp(x_{\text{rmin}}),\; \exp(x_{\text{rmax}})]$ |
| `Sqrt` | `SiraSqrt_v1` | $[\sqrt{x_{\text{rmin}}},\; \sqrt{x_{\text{rmax}}}]$ |
| `Sigmoid` | `SiraSigmoid_v1` | $[\sigma(x_{\text{rmin}}),\; \sigma(x_{\text{rmax}})]$ |
| `Tanh` | `SiraTanh_v1` | $[\tanh(x_{\text{rmin}}),\; \tanh(x_{\text{rmax}})]$ |
| `Abs` | `SiraAbs_v1` | $[0 \text{ if } 0 \in [x_{\text{rmin}}, x_{\text{rmax}}] \text{ else } \min(|x_{\text{rmin}}|, |x_{\text{rmax}}|),\; \max(|x_{\text{rmin}}|, |x_{\text{rmax}}|)]$ |
| `Pow` | `SiraPow_v1` | $\min/\max$ over $\{a_{\text{rmin}}^{b_{\text{rmin}}}, a_{\text{rmin}}^{b_{\text{rmax}}}, a_{\text{rmax}}^{b_{\text{rmin}}}, a_{\text{rmax}}^{b_{\text{rmax}}}\}$ (including $0$ if $0 \in [a_{\text{rmin}}, a_{\text{rmax}}]$ and $b_{\text{rmin}} > 0$; assumes real-valued power at endpoints) |
| `Softmax` | `SiraSoftmax_v1` | $y_{\min, i} = \frac{e^{x_{\text{rmin}, i}}}{e^{x_{\text{rmin}, i}} + \sum_{j \neq i} e^{x_{\text{rmax}, j}}}$, $y_{\max, i} = \frac{e^{x_{\text{rmax}, i}}}{e^{x_{\text{rmax}, i}} + \sum_{j \neq i} e^{x_{\text{rmin}, j}}}$ (shifted by $\max(x_{\text{rmax}})$ for stability; takes `axes: INTS` from `axis`) |

---

## 8. Shape, Layout, and Indexing Operators

- **`Reshape` (`SiraReshape_v1`), `Flatten` (`SiraFlatten_v1`), `Transpose` (`SiraTranspose_v1`)**:
  - Apply the layout/shape transform to $x_{\min}$ and $x_{\max}$.
  - If `x` is scaled-integer: scalar-valued `scale`/`bias` are kept scalar; non-scalar `scale`/`bias` are broadcast to `x` and transformed identically.
- **`Identity` (`SiraIdentity_v1`)**: Passes `(x_min, x_max, x_scale, x_bias)` through unchanged.
- **`Shape` (`SiraShape_v1`)**: Outputs `(Shape(x_min), Shape(x_max), NaN, NaN)`.
- **`Gather` (`SiraGather_v1`)**: Assumes constant indices `idx_min`; gathers `x_min` and `x_max` along `axis`, outputting `NaN` scale/bias.
- **`Unsqueeze` (`SiraUnsqueeze_v1`)**: Unsqueezes `x_min` and `x_max` along `axes`, outputting `NaN` scale/bias.
- **`Concat` (`SiraConcat2_v1`, pairwise chained)**: Concatenates `min` and `max` along `axis`, outputting `NaN` scale/bias.
- **`Cast` (`SiraCast_v1`)**: Casts `x_min` and `x_max` to target dtype `to`, outputting `NaN` scale/bias.

---

## 9. Graph-Level Notes

- **Opset Target**: `RangeAnalysis_v1` upgrades the range-analysis graph's default ONNX opset to $\ge 18$ so all `onnx_passes.ops.sira` ONNX functions use modern ONNX operator signatures.
- **Scope**: Only single-output operators in the set above are currently supported; unsupported multi-output or composite ops should be lowered prior to `RangeAnalysis_v1`.
