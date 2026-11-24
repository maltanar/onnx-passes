# Handoff: SIRA scaled-integer propagation through inlined quantizers

## Goal

Make SIRA (`RangeAnalysis_v1` + `evaluate_ranges`) recognize integer and
scaled-integer tensors in graphs where QONNX quantizers have been **inlined**
into plain ONNX ops. Once this works, SIRA can serve as the oracle that checks
whether streamlining succeeded in making Conv/MatMul/Gemm operate on plain
integers.

### Why

The model zoo streamlining campaign (see `docs/modelzoo_streamlining.md`)
measures phase 1 of streamlining by bucketing every Conv/MatMul/Gemm by the
SIRA scale/bias of its data inputs:

- `int`: scale == 1 and bias == 0
- `scaled-int`: finite scale/bias other than that
- `non-int`: NaN scale/bias

Phase 1 (`onnx_passes.passes.streamline.StreamlineScales_v1`) inlines each
`Quant` into a chain like this:

```
x -> Mul(1/s) -> Round -> Max(q_min) -> Min(q_max) -> Mul(s) -> ...
```

It then reorders the `Mul(s)` past the next linear operator. Today SIRA only
creates scaled-integers at `Quant`, `BipolarQuant` and `Trunc` nodes:

- `Round`, `Floor`, `Ceil`, `Relu`, `Clip` and `MaxPool` all output NaN
  scale/bias.
- `Max`/`Min` have no handler, so `RangeAnalysis_v1` raises
  `NotImplementedError: No SIRA range analysis handler for operator: :Max`.

As a result every phase 1 model is reported as `non-int`, or the analysis
crashes.

These op types appear in phase 1 outputs across the zoo:
`Mul, Add, Max, Min, Round, Pad, Conv, MatMul, Gemm, AveragePool, MaxPool,
Reshape, Trunc, BipolarQuant`.

## Where things live

- `onnx_passes/ops/sira.py`: handler ONNX-script functions (domain
  `onnx_passes.ops.sira`). It has helpers such as `IsScaledInt`, `ToRealRange`,
  `IsConstant`, `IsEqual` and `AllTrue`. `SiraAdd_v1` contains the
  "integer within relative tolerance 1e-5" check; reuse that pattern.
- `onnx_passes/passes/sira/__init__.py`: `RangeAnalysis_v1` dispatch.
  - `_DIRECT_OPS` (top of file) lists ops whose handler takes 4 inputs per
    original input.
  - A per-op output-shape inference block lists unary ops (`{"Quant", ...,
    "Round", ...}`) and binary ops (`{"Add", "Sub", "Mul", "Div", "Pow"}`).
  - `Clip` and `MaxPool` have dedicated branches.
- `docs/sira_handlers.md`: the handler specification. Section 1 defines the
  4-tuple representation; section 6.1 covers MaxPool; section 7 covers the
  non-linear ops. **Update it alongside the code.**
- `tests/test_sira.py`: unit tests. It provides `_make_model`, `_const`,
  `_run_sira` and `_assert_sound`. `_assert_sound` samples inputs and asserts
  that every tensor lies inside its SIRA range and, for scaled-int tensors,
  on its integer grid.
- `tests/test_sira_zoo.py`: the zoo campaign. It needs
  `QONNX_MODEL_ZOO=~/sandbox/qonnx_model_zoo/models`.

Representation reminder: a scaled-integer tensor is `(q_min, q_max, scale,
bias)` with **integer** `q` bounds, and the real value is `scale * q + bias`.
A real-valued tensor is `(r_min, r_max, NaN, NaN)`. A constant is
`(C, C, NaN, NaN)`. Scale and bias broadcast to the tensor (scalar,
per-channel or full shape).

## Required changes

All handlers must stay **sound**: every value that can occur must lie in
`[real_min, real_max]`, and for scaled-int outputs it must lie on the grid
`scale * q + bias` with `q` an integer in `[q_min, q_max]`. If a condition for
the scaled-integer case does not hold, fall back to today's real-valued
behavior.

### 1. `Round`, `Floor`, `Ceil`: output is always an integer

- Output `(R(x_rmin), R(x_rmax), 1, 0)`, where `R` is round, floor or ceil
  applied to the real range from `ToRealRange`. This holds whether the input
  is real-valued or scaled-int.
- Scale/bias are scalar `1.0` / `0.0` constants (float), not NaN.
- ONNX `Round` rounds half to even. It is monotonic, so endpoint rounding is
  sound.

### 2. `Max` and `Min` (new handlers)

- ONNX `Max`/`Min` are variadic.
  - Support the 2-input case directly.
  - Chain pairwise for more than 2 inputs, as `Concat` does with
    `SiraConcat2_v1`.
  - Raise for a single input, or pass it through as Identity.
- Add them to `_DIRECT_OPS`, and to the binary broadcast branch of the
  shape-inference block.
- **Real-valued fallback:** `Max` returns `[max(a_rmin, b_rmin),
  max(a_rmax, b_rmax)]` with NaN scale/bias; `Min` likewise.
- **Scaled-integer case 1** (`a` scaled-int `(qa, s, b)`, other input
  constant `c`):
  - Let `k = (c - b) / s` elementwise.
  - If `s != 0` everywhere and `k` is an integer everywhere (within the
    tolerance used in `SiraAdd_v1`), the output is scaled-int with the same
    `s, b`.
  - For `Max`: `q_y = Where(s > 0, max(qa, k), min(qa, k))` applied to both
    bounds. Then reorder so `q_min <= q_max`. A negative scale turns a real
    `Max` into a `Min` on `q`.
  - `Min` is symmetric.
  - Per-element mixed signs must work, because scales can be per-channel.
- **Scaled-integer case 2** (constant never active): for `Max`, if
  `c <= a_rmin` everywhere, the output is exactly `a`, so pass `a`'s 4-tuple
  through. For `Min`, use `c >= a_rmax`. This keeps a scaled-int status when
  `c` is off-grid but irrelevant.
- **Optional:** both inputs scaled-int with identical scale and bias gives
  elementwise max/min on `q` (with the same sign handling).
- Either argument position can be the constant one.

### 3. `Relu` and `Clip`

- `Relu(x)` is equivalent to `Max(x, 0)`, so apply the same logic as `Max`
  with constant 0.
- `Clip(x, lo, hi)` is equivalent to `Min(Max(x, lo), hi)`, so compose the
  same logic. Missing `lo`/`hi` mean ±inf, which is never active.
- It is fine to implement both by calling the new Max/Min functions from
  inside the handler script.

### 4. `MaxPool`: preserve scaled-int under a positive, spatially constant scale

- Conditions:
  - `x` is scaled-int;
  - `scale > 0` everywhere;
  - scale and bias are constant over the spatial axes (per-tensor or
    per-channel is fine).
- When they hold, output `(MaxPool(q_min), MaxPool(q_max), s, b)`.
  Otherwise use the current real-valued behavior.
- MaxPool padding is -inf, so padding needs no grid check.
- See `SiraAveragePool_v1` for an existing "spatially constant scale" check
  you can reuse.

### 5. `Sign` (optional, nice to have)

- Output `(sign(x_rmin), sign(x_rmax), 1, 0)`.
- Not needed today, because onnx-passes does not inline `BipolarQuant` yet.

## Tests to add (in `tests/test_sira.py`)

For every case, check the expected 4-tuple **and** run `_assert_sound`:

1. `Round`/`Floor`/`Ceil`:
   - On a real-valued input, the output is scale 1, bias 0, with integer
     bounds.
   - On a scaled-int input with a non-integer scale, the output is also
     scale 1, bias 0.
2. `Max`/`Min` with:
   - a grid-aligned constant (scaled-int is kept);
   - an off-grid but active constant (falls back to NaN);
   - an off-grid inactive constant (scaled-int is kept);
   - a negative scale;
   - per-channel scales with mixed signs;
   - the constant as either the first or the second input.
3. `Relu` on a scaled-int input:
   - zero point on the grid: scaled-int is kept;
   - zero point off the grid: falls back.
4. `Clip` with grid-aligned bounds.
5. `MaxPool`:
   - per-channel positive scale: scaled-int is kept;
   - a negative scale: falls back.
6. **End-to-end decomposed quantizer:**
   - The graph is `x -> Mul(1/s) -> Round -> Max(q_min) -> Min(q_max) ->
     MatMul(W_int)`, with real input `x` and integer-valued `W`.
   - Assert that the MatMul input is scale 1, bias 0, and that the MatMul
     output is integer.
   - Cross-check that the integer bounds of `Min` match those of the
     equivalent `Quant` node analyzed by the existing handler.

## Acceptance criteria

- `pytest tests/test_sira.py` passes, including all the new tests.
- `QONNX_MODEL_ZOO=... pytest tests/test_sira_zoo.py` still passes.
  - The original zoo models only contain `Max`/`Min`/`Round` rarely, so
    behavior there should be essentially unchanged.
  - If a coverage assertion changes, explain why in the commit.
- `docs/sira_handlers.md` has been updated:
  - new sections for `Max`/`Min`;
  - section 6.1 (MaxPool) updated;
  - the `Round`/`Floor`/`Ceil`/`Relu`/`Clip` rows moved out of the "always
    NaN" table in section 7 into their own descriptions.
- **Smoke check on a streamlined model.** After phase 1, every MatMul in
  `kwsmlp_w3a3` should have an `int` activation input. This uses
  `tests/zoo.py` and `onnx_passes/passes/streamline/`, which exist in the
  working tree of branch `feature/modelzoo-streamlining`:

  ```python
  import sys; sys.path.insert(0, "tests")
  import numpy as np
  from pathlib import Path
  from zoo import load_zoo_model, infer_zoo_input_spec, build_input_dicts
  from onnx_passes.passes.sira import RangeAnalysis_v1, evaluate_ranges
  from onnx_passes.passes.streamline import StreamlineScales_v1

  p = Path("~/sandbox/qonnx_model_zoo/models/GSCV2/kwsmlp_w3a3/kwsmlp_w3a3.onnx").expanduser()
  m = StreamlineScales_v1()(load_zoo_model(p, target_opset=19)).model
  r, s = build_input_dicts(m, infer_zoo_input_spec(p))
  ranges = evaluate_ranges(RangeAnalysis_v1()(m.clone(deep_copy=True)).model, r, input_scales=s)
  for n in m.graph:
      if n.op_type == "MatMul":
          _, _, sc, b = ranges[n.inputs[0].name]
          print(n.name, sc.ravel()[:4], b.ravel()[:4])   # expect 1.0 / 0.0
  ```

  The first MatMul's input may legitimately stay `scaled-int` if phase 1 has
  not moved the input scale out. Report what you see rather than tuning the
  streamlining passes. Streamlining is out of scope for this task.

## Out of scope

- Do not change the streamlining passes, `tests/zoo.py`, or the
  model zoo streamlining test.
- Inlining `BipolarQuant`/`Trunc` in onnx-passes is a separate known gap.
- Accumulator bitwidth analysis is unchanged.

## Follow-up (2026-11-24): gaps found after the first round

The changes above landed in `feature/sira` d3d4a68. With BipolarQuant inlining
and the gap B reorder rules on `feature/modelzoo-streamlining`, three
SIRA-side gaps remain. They block the phase 1 measurement only; the
streamlining itself verifies.

### 6. Comparisons, `Where` and `Cast` (bipolar quantizer cores)

Phase 1 now inlines `BipolarQuant(x, s)` as
`Mul(Where(GreaterOrEqual(x, 0), +1, -1), s)`. `RangeAnalysis_v1` raises
`NotImplementedError: No SIRA range analysis handler for operator: :GreaterOrEqual`.
This affects LFC_1W1A, TFC_1W1A, CNV_1W1A, cnv_1w1a_gtsrb and UNSW.

- `GreaterOrEqual`/`Greater`/`LessOrEqual`/`Less`/`Equal`: boolean output.
  Represent it as integer `{0, 1}`, i.e. `(lo, hi, 1, 0)`. Use `lo = hi` when
  the comparison is decided by the input ranges.
- `Where(c, a, b)` with constant `a`, `b`: the output is the union of `a`
  and `b`.
  - If `a` and `b` are integers, it is scaled-int `(min(a,b), max(a,b), 1, 0)`.
    That covers bipolar ±1.
  - Otherwise it is real-valued.
  - When `c` is decided, pass the chosen branch through.
- `Cast`/`CastLike` to float of an integer-range input: pass through unchanged.
  This is also needed in item 7.

### 7. `CastLike` and `Trunc` in MobileNet's global pool head

MobileNet's head is `AveragePool -> Mul(49) -> Trunc -> Reshape -> MatMul`.
`Trunc` is not inlined by onnx-passes and has no type inference, so a
`CastLike(1.0, <untyped value>)` from Gemm lowering stays unfolded. Phase 1
measurement then fails with `No SIRA range analysis handler for operator:
:CastLike`. A `CastLike` handler (identity on the range, with the target
dtype ignored) is enough to unblock the measurement.

### Gap C (Conv weight `Mul(const, const)`)

This no longer occurs: phase 1 now moves weight scales past Conv. The shape
inference failure in `ToRealRange` for a Conv weight produced by an
elementwise op is still a latent robustness issue, but no zoo model hits it.
