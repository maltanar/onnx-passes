# QONNX Model Zoo Streamlining Campaign

Goal: measure (and later improve) how far the QONNX model zoo can be streamlined
with onnx-passes transformations. Streamlining is an optimization, not
pass-or-die: models are expected to improve, not necessarily to become fully
integer.

## Phases

Following the SIRA paper, streamlining is split into two phases, each packaged
as a reusable `Sequential` pass in `onnx_passes/passes/streamline/` (also
resolvable from the CLI).

1. **Phase 1 – move scales and biases into the layer tail.** Inline quantizers,
   lower BatchNorm, and reorder/eliminate/fold/cleanup exhaustively so that
   Conv/MatMul/Gemm operate on plain integers. The layer tail remains a chain of
   Mul/Add/nonlinearity/Round/Clip ops.
2. **Phase 2 – layer tails to thresholds.** Convert layer tails to
   `MultiThreshold` using the onnx-passes `multithreshold` passes (absorbing
   approximate inverses, not the SIRA emulation approach), followed by
   `unbroadcast` and `cleanup`. `multithreshold.decompose` is a FINN backend
   lowering step and is **excluded**.

The exact phase 1 pass list is determined empirically against the zoo.

## Metrics

SIRA (`RangeAnalysis_v1` + `evaluate_ranges`) is used purely as a measurement
oracle; it does not drive any transformation. Input domains come from
`infer_zoo_input_spec` (including input scales where known).

- **Phase 1:** every Conv/ConvInteger/MatMul/Gemm is bucketed by the SIRA
  scale/bias of its data inputs (activation and weight):
  - `int`: scale == 1 and bias == 0 everywhere,
  - `scaled-int`: finite scale/bias, but not (1, 0),
  - `non-int`: NaN scale/bias.
  Counts are reported before and after phase 1.
- **Phase 2:** number of activation `Quant`/`BipolarQuant` nodes in the original
  model vs. residual quantizer cores (`Round` + `Sign`) remaining after phase 2.
  Success is zero residual.
- **Stray ops after phase 2:** operators outside the canonical streamlined
  structure, by type. Canonical are `MultiThreshold`, linear operators, `Pad`,
  pooling, data movement (`Reshape`, `Transpose`, ...), residual `Add`s joining
  two activations, and the output tail (the final output scale and bias, data
  movement and the classification head). Anything else, e.g., a per-channel
  `Mul` or `Add` stuck between two layers, is stray. Success is zero stray ops.
  This catches what the quantizer counts miss: the stuck `Add` and channel
  masks in 2W2A_rn8_pow2 changed none of the other metrics.

Accumulator bitwidth is out of scope.

## Verification (hard asserts, per model)

1. The pipeline runs without raising.
2. Numerical equivalence vs. the original model (ReferenceEvaluator with custom
   QONNX ops) on samples from `generate_samples`, `atol=1e-3`. Up to 1% of
   output elements may mismatch (float reordering can flip a `Round` near a .5
   tie); that is reported as a warning, not a failure.
3. No regression against the unstreamlined model: `int` count after phase 1
   >= before; residual quantizers after phase 2 <= before.
4. No regression against the baseline (`tests/zoo_baseline.json`), see below.

## Baseline ratchet

`tests/zoo_baseline.json` records the metrics of every model at the pinned
zoo commit. A run fails if any model gets worse than its entry:

- new stage errors,
- fewer `int` or more `non-int` linear ops after phase 1,
- more residual quantizers or stray ops after phase 2,
- a higher max abs error (beyond 1e-6 of float noise) or mismatch fraction.

`MultiThreshold` counts and timings are reported but not compared. Models
without an entry are reported as new and not compared.

When a change improves a model, the report says so. Lock the improvement in by
updating the baseline in the same commit, so the review diff shows what
improved:

```text
QONNX_MODEL_ZOO=<clean zoo checkout at the pinned commit>/models \
    python tests/test_modelzoo_streamlining.py --update-baseline
```

Use a clean checkout (`git clone` of the zoo at the commit named in
`.github/workflows/tests.yml`): stray local files in the zoo directory would
end up in the baseline otherwise. `--filter` updates only the matching
entries.

## CI

`.github/workflows/tests.yml` runs on every push and pull request:

- **Unit tests:** `pytest`, zoo tests skip without `QONNX_MODEL_ZOO`.
- **Model zoo streamlining:** checks out the zoo at `QONNX_MODEL_ZOO_REF` and
  runs the campaign against the baseline. The Markdown report appears on the
  run's summary page and is uploaded as the `zoo-streamlining-report`
  artifact (Markdown and JSON). Then runs the SIRA zoo tests.

The zoo commit and the baseline belong together: update both at once.

## Model scope

- Minifloat ResNet18 models (`FLOAT` in the path) are skipped: the streamlining
  spec only covers integer quantization.
- JetTagging ends in Softmax; it is fine for everything except the final
  Softmax to become integer.

## Infrastructure

- `tests/zoo.py`: shared zoo helpers (model discovery, `ZooInputSpec`,
  `infer_zoo_input_spec`, `build_input_dicts`, `generate_samples`, custom-op
  ReferenceEvaluator setup), used by `tests/test_sira_zoo.py` and
  `tests/test_modelzoo_streamlining.py`.
- `tests/test_modelzoo_streamlining.py`: one parametrized pytest per model plus
  a `main()` that prints a markdown report, mirroring `test_sira_zoo.py`. The
  standalone run exits nonzero on regressions or stage errors.
- Set `QONNX_MODEL_ZOO` to the zoo's `models/` directory.

## Results

### Checkpoint 1 (2026-11-24, on top of feature/sira d3d4a68)

Linear = int/scaled-int/non-int Conv/MatMul/Gemm count (SIRA); quant =
activation quantizers before -> residual after phase 2.

| Model | Linear before | Linear after P1 | Quant before -> after P2 | MultiThreshold | Max err P1/P2 | Blocker |
|---|---:|---:|---:|---:|---:|---|
| CNV_1W1A / 1W2A / 2W2A, cnv_1w1a_gtsrb | 8/1/0 | - | 9 -> - | - | - | phase 1 crash (A) |
| TFC_1W1A / 1W2A / 2W2A | 4/0/0 | - | 4 -> - | - | - | phase 1 crash (A) |
| mobilenet_4W4A | 0/28/0 | - | 28 -> - | - | - | phase 1 crash (A) |
| 2W2A_rn8_pow2 | 0/10/0 | - (C) | 13 -> 1 (Trunc) | 13 | 0 / 0 | |
| 3W3A_rn8_fp | 0/10/0 | - (C) | 12 -> 1 (Trunc) | 12 | 0 / 0 | |
| cloudsatnet_qat_4bit_exsi | 0/12/0 | - (C) | 12 -> 0 | 12 | 0 / 9.5e-7 | |
| kwsmlp_w3a3 | 0/4/0 | 1/0/3 (B) | 4 -> 0 | 4 | 4.8e-7 | |
| LFC_1W1A | 4/0/0 | 4/0/0 | 4 -> 4 (BipolarQuant) | 0 | 2.4e-7 | |
| LFC_1W2A | 4/0/0 | 4/0/0 | 4 -> 0 | 4 | 4.8e-7 | |
| unsw_nb15-mlp-w2a2 | 0/4/0 | 0/0/4 (B) | 4 -> 1 (BipolarQuant) | 3 | 0 / 0 | |
| qkeras_jettagging | 0/3/1 | - | 3 -> - | - | - | int64 Quant params (D) |

No equivalence mismatches anywhere a phase completed.

Gaps found:

- **(A)** `RewriteUnsqueezeAsReshape_v1`
  ([normalize/reshape.py](../onnx_passes/passes/normalize/reshape.py)) gathers
  index -1 from `Shape(x)` for each inserted dimension. That fails when `x` is a
  scalar, which is what the `Shape -> Gather -> Unsqueeze` flatten idiom
  produces. Constant folding then raises
  `IndexError: cannot do a non-empty take from an empty axes`.
- **(B)** Phase 1 folds weight quantizers to integer weights, but leaves the
  weight scale as a constant `Mul` on the weight input of the linear op and
  never moves it past the linear op. Weights therefore count as non-int.
- **(C)** SIRA's Conv handler fails shape inference (`ToRealRange`) when the
  weight input is a `Mul(const, const)`, which is the leftover from (B).
- **(D)** JetTagging stores `Quant` zeropoint/bitwidth as INT64. onnx-passes'
  `Quant` function adds them to float tensors without a cast.
- `BipolarQuant` and `Trunc` are not inlined by onnx-passes;
  `origin/feature/bipolar-quant` adds BipolarQuant import/inlining.
- Infrastructure fix: `AddDefaultAttributes_v1` oscillated with onnx-ir 1.1.0's
  `RemoveUnusedNodesPass` (BatchNorm `training_mode`), so `canonicalize` never
  terminated on BatchNorm models. This is fixed on this branch.

### Checkpoint 2 (2026-11-24): gaps A, B, D and BipolarQuant fixed

Pytest: 10 passed, 6 failed. **All 6 failures are SIRA measurement gaps after
phase 1** (see the follow-up section of `handoff_sira_integer_propagation.md`).
Both streamlining phases complete and verify on all 16 in-scope models.

| Model | Linear before | Linear after P1 | Quant before -> after P2 | MultiThreshold | Max err P1/P2 | Remaining blocker |
|---|---:|---:|---:|---:|---:|---|
| CNV_2W2A | 8/1/0 | 9/0/0 | 9 -> 0 | 9 | 1.8e-7 | |
| CNV_1W2A | 8/1/0 | 9/0/0 | 9 -> 0 | 9 | 1.2e-7 | |
| CNV_1W1A, cnv_1w1a_gtsrb | 8/1/0 | - | 9 -> 0 | 9 | ≤ 4.8e-7 | SIRA: GreaterOrEqual |
| TFC_2W2A, TFC_1W2A | 4/0/0 | 4/0/0 | 4 -> 0 | 4 | ≤ 2.4e-7 | |
| TFC_1W1A | 4/0/0 | - | 4 -> 0 | 4 | 1.2e-7 | SIRA: GreaterOrEqual |
| LFC_1W2A | 4/0/0 | 4/0/0 | 4 -> 0 | 4 | 2.4e-7 | |
| LFC_1W1A | 4/0/0 | - | 4 -> 0 | 4 | 2.4e-7 | SIRA: GreaterOrEqual |
| unsw_nb15-mlp-w2a2 | 0/4/0 | - | 4 -> 0 | 4 | 0 | SIRA: GreaterOrEqual |
| kwsmlp_w3a3 | 0/4/0 | 4/0/0 | 4 -> 0 | 4 | 4.8e-7 | |
| cloudsatnet_qat_4bit_exsi | 0/12/0 | 12/0/0 | 12 -> 0 | 12 | 4.8e-7 | |
| 2W2A_rn8_pow2 | 0/10/0 | 8/2/0 | 13 -> 1 (Trunc) | 12 | 0 | |
| 3W3A_rn8_fp | 0/10/0 | 8/2/0 | 12 -> 1 (Trunc) | 11 | 9.5e-7 | |
| mobilenet_4W4A | 0/28/0 | - | 28 -> 1 (Trunc) | 27 | 4.8e-7 | SIRA: CastLike (after Trunc) |
| qkeras_jettagging | 0/3/1 | 3/0/1 | 3 -> 0 | 3 | 0 | |

These scaled-int/non-int ops are expected after phase 1:
- **rn8:** the first Conv reads the scaled-int model input, and the final
  MatMul sits behind the global-pool `Trunc`.
- **JetTagging:** the first MatMul reads the unquantized float input.

Changes:

- **(A)** `RewriteUnsqueezeAsReshape_v1` now gathers from `Concat(Shape(x), [1])`.
  Upstream (iksnagreb/onnx-passes) has no issue or PR for this, and its
  `reshape.py` is unchanged. The fix is on local branch `fix/unsqueeze-scalar`,
  which is based on upstream `main` and ready to upstream.
- **BipolarQuant:**
  - Ported from `origin/feature/bipolar-quant`, which predates the registry
    refactor and could not be merged.
  - Op definitions (QONNX and Brevitas domains) are in `ops/qonnx.py` and
    `ops/brevitas.py`, and inlining is in `inline/quantizer.py`.
  - The new `multithreshold.convert.RewriteStepAsMultiThreshold_v1` turns
    `Where(x >= t, a, b)` into a single-step threshold.
  - Compilation folds the singleton reduction away, so
    `multithreshold.single.InferSingleStepMultiThreshold_v1` recovers the bare
    `Cast(x >= t)` as a `MultiThreshold`. It runs only at the end of phase 2,
    because inside `multithreshold.normalize` it would also match steps of
    unnormalized multi-threshold subgraphs.
- **(B)** New module `reorder/linear.py` (part of `reorder`):
  - per-column weight `Mul` past MatMul;
  - per-output-channel weight `Mul` past Conv, with and without bias;
  - scalar input `Mul` past Conv;
  - `MoveMulPastPad_v1`, the scale half of upstream issue #10.

  Phase 1 also runs `lower.gemm` now.
- **(D)** Integer `Quant` zero points are cast to the input type, in both the op
  function and the inlining rules.
- `eliminate.identity`'s identity-matrix check no longer crashes on values of
  unknown shape.
- Metric: comparisons not absorbed into a `MultiThreshold` now count as
  residual quantizers.

Next gaps:
- SIRA handlers for comparisons, `Where`, `Cast` and `CastLike` (handoff
  follow-up).
- `Trunc` inlining (rn8, MobileNet).

### Checkpoint 3 (2026-11-24): SIRA follow-up merged, Trunc inlined

`feature/sira` 3ca43e5 (handlers for comparisons, `Where`, `CastLike` and
`Cast`) is merged. **Pytest: 16/16 zoo models pass**, together with
`test_sira.py` and `test_sira_zoo.py` (55 tests in total). All repo demos
still verify.

| Model | Linear before | Linear after P1 | Quant before -> after P2 | MultiThreshold | Max err P1/P2 |
|---|---:|---:|---:|---:|---:|
| CNV_1W1A / 1W2A / 2W2A, cnv_1w1a_gtsrb | 8/1/0 | 9/0/0 | 9 -> 0 | 9 | ≤ 4.8e-7 |
| TFC_1W1A / 1W2A / 2W2A | 4/0/0 | 4/0/0 | 4 -> 0 | 4 | ≤ 2.4e-7 |
| LFC_1W1A / 1W2A | 4/0/0 | 4/0/0 | 4 -> 0 | 4 | ≤ 4.8e-7 |
| unsw_nb15-mlp-w2a2 | 0/4/0 | 3/1/0 | 4 -> 0 | 4 | 0 |
| kwsmlp_w3a3 | 0/4/0 | 4/0/0 | 4 -> 0 | 4 | 4.8e-7 |
| cloudsatnet_qat_4bit_exsi | 0/12/0 | 12/0/0 | 12 -> 0 | 12 | 4.8e-7 |
| 2W2A_rn8_pow2 | 0/10/0 | 9/1/0 | 13 -> 1 | 12 | 0 |
| 3W3A_rn8_fp | 0/10/0 | 9/1/0 | 12 -> 1 | 11 | 4.8e-7 |
| mobilenet_4W4A | 0/28/0 | 27/1/0 | 28 -> 1 | 27 | 4.8e-7 |
| qkeras_jettagging | 0/3/1 | 3/0/1 | 3 -> 0 | 3 | 0 |

What remains, all expected:

- **Non-int linear ops after phase 1** read the model input directly (scaled-int
  or float input spec): the first layer of rn8, MobileNet, UNSW and JetTagging.
- **The residual quantizer in rn8 and MobileNet** is the `Round` inside the
  global-pool `Trunc`. Trunc v1 has no clamp, so its integer range is unbounded
  and cannot be expressed as a finite set of thresholds.

Changes:

- **Trunc:**
  - Op functions for QONNX opset versions 1 and 2 (`ops/qonnx.py`) and for
    Brevitas v1 (`ops/brevitas.py`).
  - Inlining (`inline/quantizer.py`) tells the versions apart by input count:
    5 inputs is v1, 6 is v2.
  - v2 clamps to the output bitwidth and rescales by the output scale. v1
    rescales by the input scale, as in the specification.
  - The zoo models only use v1 (`onnx.brevitas`, FLOOR).
  - A spec-conformance test covers 30 cases: v1/v2, both domains, all rounding
    modes, zero points, and signed/narrow.
  - Rounding modes in op functions are compared longest-first, because the ONNX
    reference evaluator refuses to compare strings of different length. As a
    result, CEIL cannot be dispatched by the reference evaluator. The existing
    `Quant` function has the same limitation; inlining is not affected.
- **Function inlining blacklist** now includes `BipolarQuant` and `Trunc`:
  - Generic function inlining in `canonicalize` expanded them before the
    dedicated rules ran.
  - For `Trunc`, this left a string-valued `If` that broke initializer
    deduplication.
  - For `BipolarQuant`, the weight `ConstantStop` was skipped.
- **`MoveChannelMulPastDepthwiseConvInput_v1`** (`reorder/linear.py`): moves a
  per-channel input scale past a depthwise Conv (`group == C_in`, any channel
  multiplier). It covers MobileNet's 12 depthwise Convs (15/13/0 -> 27/1/0).

## Dead and constant channels

Some zoo models have BatchNorm channels whose quantized weight is exactly 0.
2W2A_rn8_pow2 has many: 25 of 64 in `layer1.0/bn1` and 21 in `layer1.0/bn2`.
`MultiThreshold_10` (layer3 `downsample_bn`) has 242 of 256 channels with zero
step weights. For such a channel the BatchNorm output is its bias, so the
quantized activation is a constant independent of the input, mostly 0.

Streamlining keeps these channels. They show up in three forms:

- **Zero step weights in a `MultiThreshold`**, e.g., `[1, -0, -0, …]`. A scale
  following the `MultiThreshold` is fused into its weights
  (`FuseScaleIntoMultiThreshold_v1`), and zero scales give zero weights. The
  thresholds of these channels are leftovers and do not matter.
- **Zero input channels in the next Conv/MatMul weights.** `AbsorbMulIntoComparison`
  handles zero scales by masking the comparison, which ends up as a 0/1 `Mul`
  after the `MultiThreshold`. If a `Pad` sits in between, `MoveMulPastPad_v1`
  can move the mask past the `Pad` before it gets fused, and the mask then
  stops at the Conv. Since 2026-11-26 the sign rules
  (`MoveSignedChannelMulPast{ConvInput,MatMulLhs}_v1`) accept zero channels and
  fold the mask into zero weights. Example: all 25 dead channels of
  `layer1.0/bn1` are zero input channels of `Conv_2`. Note that 2-bit weights
  also produce zero input channels by themselves, so not every zero column is
  a dead channel.
- **A step of weight a at threshold −∞** for a channel with constant nonzero
  output a, other steps at +∞ with weight 0.
  `AbsorbChannelOffsetIntoMultiThreshold_v1` creates this form from a
  zero-weight channel followed by a per-channel offset. Example: channel 35
  of `MultiThreshold_0` (constant 1). This gives the `MultiThreshold`
  per-step weights, e.g., shape (64,1,1,3) instead of (64,1,1,1).

Implications:

- Backends need to accept thresholds at ±∞ and per-step weights, or the
  constant channels need a different encoding (e.g., the most negative
  representable accumulator value instead of −∞).
- The channels could be pruned: drop a dead channel from the `MultiThreshold`
  and the matching input channel of the next layer, and fold a constant
  nonzero output into that layer's bias. Residual `Add`s need care, since both
  branches must agree on the channel set. Not implemented.

## Open follow-ups

### Per-channel sign flips in front of MaxPool (deferred 2026-11-26)

In CNV_1W1A, a per-channel `Mul(±1)` stays between each `MultiThreshold` and
the following `MaxPool`. The sign flips come from batch-norm channels with
negative scale, which invert the threshold comparison. A negative scale turns
the maximum into a minimum, so it cannot move past `MaxPool`, and ONNX has no
MinPool. Everything else (the common magnitude, the offsets) already moves past
`MaxPool` and the next `Conv`.

Proposed fix: push the sign upstream instead of downstream. For a channel with
negative sign and `y = [x ≥ t]` (bipolar output `2y − 1`):

- `−(2y − 1) = 2(1 − y) − 1`, and `1 − y = [x < t] = [−x > −t]`.
- `x` is an integer accumulator, so `[−x > −t] = [−x ≥ ⌊−t⌋ + 1]`, which is
  again a regular `MultiThreshold` on `−x`.
- `−x` comes from negating that output channel of the preceding `Conv`/`MatMul`
  weights, and ±1 weights stay ±1.

After this the scale is a uniform positive factor, which moves past `MaxPool`
(`MoveMulPastMaxPool_v1`) and the next `Conv` like the other cases. Still to
work out: multi-bit thresholds (the steps need mirroring, not just one
comparison flipped), and what SIRA reports for an accumulator whose sign is
flipped.

Update (2026-11-26): the flip itself now exists as
`AbsorbChannelOffsetIntoMultiThreshold_v1` (`multithreshold.fuse`). It handles
multi-bit steps (`w [x ≥ t] = w + |w| [x < t]`) and compares against
`nextafter(−t, +∞)`, which is exact for float accumulators too. It triggers on
`Add(MultiThreshold, offset)` with non-positive weights where the offset
cancels `sum(w)`, as in 2W2A_rn8_pow2. Still open: the CNV_1W1A form, a
`Mul(±1)` after the bipolar `MultiThreshold`, has to be brought into that
shape first, e.g., by fusing the sign into the step weights.

### Redundant rounding in the Trunc tail (deferred 2026-11-26)

2W2A_rn8_pow2 ends with `AveragePool → Reshape → Mul 64 → Round → Mul 4 →
Floor → MatMul`. This is the inlined Brevitas `TruncAvgPool`: `Mul 16` from
the export followed by `Trunc(scale 0.25, in_bw 6, out_bw 8, FLOOR)`.

- `Mul 64` = 16 / 0.25 (scale to the input grid), `Round` snaps to that grid.
- `Mul 4` = 2^(out_bw − in_bw): the exporter's out_bw is larger than in_bw,
  so the "truncation" multiplies instead of dropping bits.
- `Floor` is the truncation rounding.

Both roundings are no-ops: `MultiThreshold_11` outputs multiples of 0.25, the
4x4 average makes them multiples of 1/64, ×64 makes them integers and ×4 keeps
them integers. The tail is ×256 on the pooled sum.

Proposed fix: an `eliminate` rule removing `Round`/`Floor`/`Ceil` whose input
is provably an integer multiple of a step q with q integer. Track q locally
through producers:

- `MultiThreshold`: gcd of the step weights (and offsets).
- `AveragePool`: q / kernel area, if unpadded or `count_include_pad=1`.
- `Mul` by a constant c: q·|c| (gcd over c's elements).
- `Add`/`Sub`: gcd of both operands' steps. `Round`/`Floor`/`Ceil`: 1.
- Shape ops, `Pad` (with a value on the grid), `MaxPool`: unchanged.

The SIRA pass builds a separate analysis graph, so it is not usable as a
cheap check inside a rewrite rule. Once the roundings are gone, the remaining
`Mul`s fold into the MatMul output scale.
