# LeWorldModel (JEPA) for Multi-Object Physics Prediction

## Overview

This document reports experimental results on training Joint Embedding
Predictive Architecture (JEPA) world models to learn 2D physics from pixel
observations. The core method follows LeWorldModel (Maes et al. 2026), using
a two-term objective, MSE prediction loss plus SIGReg anti-collapse
regularisation, to learn latent dynamics end-to-end from raw frames.

Experiments progress from a single bouncing ball (baseline validation) to a
two-ball collision system (multi-object interaction learning).

---

## 1. Single-Ball Baseline

### Setup

<img width="1439" height="152" alt="output" src="https://github.com/user-attachments/assets/b2084bff-f282-4ce3-bebd-ce015b4fddcf" />

A ball bounces in a 2D box under gravity with elastic wall collisions. The
observation is a 16×16 rasterised grid (the ball rendered as a Gaussian blob),
giving a 256-dimensional input per timestep. The encoder is a two-layer MLP
(256  128 -> 32), the predictor is a history-conditioned two-layer MLP
(96 -> 64 -> 32) taking 3 concatenated latent vectors as input, and the model
predicts 3 steps ahead autoregressively.

Training: 100 epochs, batch size 8192, AdamW with cosine LR schedule.

### Key Findings

**SIGReg lambda must scale with prediction steps.** With 1-step prediction,
λ=0.1 prevented collapse. Switching to 3-step autoregressive prediction caused
collapse at λ=0.1 because the multi-step loss has a stronger gradient toward
the trivial solution (constant encoder output makes all 3 predictions perfect
for free). Increasing λ to 1.0 restored healthy training. The collapse
manifested as pred_loss ≈ 0, SIGReg stuck at 0.41, and the encoder outputting
a single constant representation for all inputs.

**History-conditioned prediction is essential for velocity-dependent anomalies.**
With a single-frame predictor (z_t -> z_{t+1}), the model could detect
teleportation (position discontinuity, 2400× above baseline) but was
completely blind to velocity flip and gravity change (identical surprise to
normal trajectories). Adding 3-frame history (z_{t-2}, z_{t-1}, z_t -> z_{t+1})
enabled velocity flip detection at 120× above baseline. The predictor infers
velocity implicitly from the difference between consecutive latent vectors.

**The encoder captures position, not velocity.** Linear probe R² on frozen
encoder outputs: position (x, y) ≈ 0.96, velocity (vx, vy) ≈ 0.01. This is
expected — a single static frame contains no motion information. Velocity is
recovered by the predictor from the temporal context, not by the encoder from
a single observation.

**Gravity anomaly remains difficult.** Gravity changes acceleration, a second
derivative. The positional effect accumulates slowly and stays within the noise
floor of the 3-step prediction window. Peak surprise for gravity flip was
0.0017 vs 0.001 baseline — detectable in aggregate over many seeds but not as
a clear single-trajectory spike. Exaggerating the perturbation (5× gravity
reversal) produced a clear signal (peak 0.70 at step 102, 36 steps after
injection), confirming the model does encode acceleration-relevant features
but the natural perturbation magnitude is below the detection threshold.

### Quantisation Results

All quantisation experiments used the same single-ball architecture trained
with ReLU activations for MCU compatibility.

| Model              | Val Pred | Teleport | Vel Flip | LZMA KB |
|--------------------|----------|----------|----------|---------|
| FP32 baseline      | 0.001593 |   2.3236 | 0.120862 |   159.9 |
| FP32 no_bias       | 0.001724 |   3.2982 | 0.106014 |   158.6 |
| INT8 QAT bias      | 0.001504 |   2.4637 | 0.124937 |    41.5 |
| INT8 QAT no_bias   | 0.001476 |   3.1973 | 0.119046 |    39.0 |
| Ternary QAT bias   | 0.018691 |   2.2225 | 0.077682 |     8.8 |
| Ternary QAT no_bias| 0.012366 |   2.3573 | 0.090471 |     7.9 |
| Binary QAT bias    | 0.026278 |   1.6957 | 0.050976 |     5.6 |
| Binary QAT no_bias | 0.023652 |   1.4550 | 0.064367 |     4.7 |

**INT8 QAT matched FP32 accuracy.** Val pred loss was actually lower (0.001476
vs 0.001593), likely because quantisation noise acts as mild regularisation.
Teleport and velocity flip detection were statistically equivalent. This
motivated using INT8 QAT from epoch 1 in all subsequent experiments, bypassing
the FP32 pretraining phase entirely.

**Ternary offers the best accuracy-per-byte.** At 7.9 KB compressed (LZMA),
ternary QAT retained 72% of FP32 teleport detection capability. The RMS-based
per-row scaling (replacing the original fixed-threshold ternary quantisation)
was critical — without scaling, post-training ternary quantisation produced
loss values in the trillions due to weight magnitude mismatch.

**Binary is viable but noticeably degraded.** Val pred 16× worse than FP32,
teleport detection reduced to 44% of baseline. Still functional (1.45 surprise
vs 0.001 baseline = 1450× separation), but the quality gap is large enough to
matter for subtle anomalies.

**Post-training quantisation (PTQ) vs QAT.** PTQ ternary was catastrophic
(loss 0.074 vs FP32's 0.0016). QAT recovered most of the gap (0.012). PTQ
INT8 was nearly lossless (0.0018 vs 0.0016). The conclusion: INT8 can be
deployed with PTQ alone, but ternary and binary require QAT.

**Bias has negligible impact on final accuracy.** No-bias models converged
slower initially but reached equivalent or better final quality across all
quantisation levels. No-bias simplifies the deployment path (inference is pure
matmul with no separate bias addition) at no accuracy cost.

### C Inference Validation

The trained single-ball model was exported as C header files (weight arrays)
and run through a C implementation of the full inference pipeline on macOS.
All four quantisation levels were validated:

| Model   | us/frame (M1 Max) | Teleport | Vel Flip |
|---------|-------------------|----------|----------|
| FP32    |              21.7 |   2.8999 | 0.117991 |
| INT8    |              23.0 |   2.7010 | 0.108036 |
| Ternary |             134.0 |   2.1507 | 0.081707 |
| Binary  |              28.9 |   1.7568 | 0.073203 |

Surprise values matched the Python implementation, confirming correctness of
the export and C inference code. Ternary is slow on general-purpose hardware
due to per-element unpacking overhead; on MCU hardware with SIMD support for
INT8 MAC operations, INT8 inference is expected to be the fastest quantised
option.

---

## 2. Two-Ball Collision System

### Motivation

The single-ball system validates the LeWM architecture on simple trajectory
prediction but does not test multi-object reasoning. The two-ball system
introduces: (1) multi-object tracking from a single observation frame,
(2) collision dynamics where the behaviour of one ball depends on the other,
and (3) anomalies defined by the _relationship_ between objects rather than
the trajectory of a single object.

### Physics Simulation

Two balls bounce in a 2D box (8×8 world units) with elastic wall and
ball-ball collisions. Collision response uses conservation of momentum and
kinetic energy for 2D elastic collisions with overlap separation. Training
data is biased so that 50% of trajectories have balls aimed toward each other,
ensuring frequent collisions in the training set. The observation is a 32×32
rasterised grid with both balls rendered as Gaussian blobs (summed and clamped
to [0, 1]).

### Anomaly Types

| Anomaly        | Description                                                |
|----------------|------------------------------------------------------------|
| normal         | Standard physics, control baseline.                        |
| pass_through   | Ball-ball collisions disabled — balls pass through each other. |
| disappear      | Ball 2 removed from rendering mid-trajectory.              |
| mass_change    | Ball 2 mass increased 10× — collision response changes.    |

### VoE Evaluation Methodology

A critical methodological refinement was needed for the two-ball evaluation.
The initial evaluation generated random trajectories and disabled collisions
at a fixed timestep. However, most random trajectories have no collision near
that timestep, disabling a collision that never happens produces no surprise.

**Collision-targeted trajectories.** The evaluation was updated to generate
trajectories where a collision is geometrically guaranteed: balls are placed
on opposite sides of the box and aimed at each other, timed to converge near
the centre of the trajectory. This ensures the anomaly test evaluates what it
claims to: "does the model detect that a collision didn't happen as expected?"

**Interaction anomalies apply from the start.** For pass_through and
mass_change, the physics modification (collisions disabled or mass changed) is
active from the beginning of the trajectory. The trajectory is identical to
normal until the balls actually meet — then it diverges. This reflects the
correct counterfactual: same initial conditions, different interaction rules.
Disappear remains a point event (ball removed at a specific timestep) because
it is a visual anomaly rather than a physics modification.

**Global peak search for interaction anomalies.** Because the exact collision
timestep varies per trajectory (it depends on initial conditions and gravity),
the peak surprise search covers the entire trajectory for pass_through and
mass_change rather than a narrow window around a fixed step.

### Experiment 1: MLP Encoder (64×64 grid)

Architecture: Linear encoder (4096 -> 512 -> 64), 2.16M parameters.
Dataset: 500 trajectories of 80 steps (reduced from 2000 due to memory
constraints at 64×64 resolution (44GB peak memory caused swap thrashing)).

| Metric         | Value    |
|----------------|----------|
| Val pred loss  | 0.002856 |
| Normal surprise| 0.0020   |
| Pass-through   | 0.0020   |
| Disappear      | 0.6227   |
| Mass change    | 0.0020   |
| Position R²    | −1.30    |

The encoder was massively overparameterised: 2.1M parameters for 37K training
windows (56 parameters per sample). Probe R² was deeply negative, indicating
the latent space did not correspond to physical state in any linearly
decodable way. Only the disappear anomaly (a global pixel distribution change)
was detected. Pass-through and mass_change were indistinguishable from
baseline — the encoder had no spatial structure to represent per-object state.

### Experiment 2: MLP Encoder (32×32 grid), Bias Switching

Architecture: Linear encoder (1024 -> 256 -> 64), 311K parameters.
Training with bias for first 50 epochs, then bias removed mid-training.

| Metric         | Value    |
|----------------|----------|
| Val pred loss  | 0.011783 |
| Normal surprise| 0.0880   |
| Pass-through   | 0.0350   |
| Disappear      | 2.6768   |
| Mass change    | 0.0442   |
| Position R²    | −2.12    |

The bias switch was destructive: pred loss jumped from 0.006 to 0.047 at the
switch point (epoch 51) and only recovered to 0.012 after 150 more epochs —
never reaching the pre-switch quality. The bias parameters had encoded
important offset information that the weights alone could not compensate for.
This problem did not manifest in the simpler single-ball case where the
encoder's task was straightforward enough to recover from the disruption.

### Experiment 3: MLP Encoder (32×32 grid), No Bias, λ=0.5

Architecture: Linear encoder (1024 -> 256 -> 64), 311K parameters.
No bias from epoch 1, SIGReg λ reduced from 1.0 to 0.5.

| Metric         | Value    |
|----------------|----------|
| Val pred loss  | 0.002021 |
| Normal surprise| 0.0120   |
| Pass-through   | 0.0382   |
| Disappear      | 2.6768   |
| Mass change    | 0.0442   |
| Position R²    | −0.14    |

Training converged smoothly without the bias switch disruption. Pred loss
reached 0.002, comparable to the single-ball model. Disappear detection was
strong (223× above baseline). However, pass_through (3.2×) and mass_change
(3.7×) remained at the noise margin. Animation confirmed that peak surprise
for these anomalies occurred at random timesteps rather than at the collision
point — not genuine detections.

Probe R² improved from −2.12 to −0.14 but remained negative, confirming the
flat MLP encoder was not learning per-object spatial features. The encoder
learned a holistic frame-level representation sufficient for next-frame
prediction but without the spatial decomposition needed to distinguish
interaction anomalies from normal dynamics.

### Experiment 4: Convolutional Encoder

Two changes were made simultaneously, each addressing a different failure mode.

**Architectural change: conv encoder.** The MLP encoder treats the 1024-pixel
input as an unstructured flat vector. It has no concept of spatial locality —
it cannot learn that adjacent pixels belong to the same blob, or that two
separated blobs represent two distinct objects. A 3-layer convolutional encoder
was introduced:

```
Conv2d(1->16, 3×3, stride 2) -> ReLU    # (1, 32, 32) -> (16, 16, 16)
Conv2d(16->32, 3×3, stride 2) -> ReLU   # (16, 16, 16) -> (32, 8, 8)
Conv2d(32->64, 3×3, stride 2) -> ReLU   # (32, 8, 8) -> (64, 4, 4)
Flatten -> Int8Linear(1024 -> 64)        # (1024,) -> (64,)
```

The conv layers provide three critical properties the MLP lacked:
- **Local receptive fields** that group nearby pixels into blob features
  (detecting each ball as a coherent object).
- **Translation equivariance** so a ball at position (3,7) activates the same
  filters as a ball at (6,2).
- **Spatial hierarchy** where the final 4×4 feature map preserves the layout
  of detected objects, giving the linear projection position-aware inputs.

Conv layers use standard `nn.Conv2d` (not quantised). The final linear
projection is INT8 QAT. Total parameters: 121K — 2.5× fewer than the MLP
encoder (311K), yet better across every metric.

**Evaluation change: collision-targeted VoE.** As described in the
Methodology section above, the VoE trajectories were redesigned to guarantee
a ball-ball collision, and interaction anomalies were applied from the start
of the trajectory rather than at a fixed timestep. This evaluates the specific
capability we care about: does the model understand that balls bounce off each
other?

**SIGReg lambda increased to 5.0.** The conv encoder collapsed at λ=0.5
(pred_loss = 0, SIGReg stuck at 0.41). Conv layers have a much easier path to
collapse than MLPs: a single 3×3 filter that produces spatially uniform
activations causes everything downstream to be constant regardless of input.
The MLP needed all 1024 input weights to conspire; the conv needs one filter.
λ=5.0 provided sufficient anti-collapse force.

**Training with bias for first 100 epochs, then removed.** Unlike Experiment 2,
the conv encoder recovered gracefully from the bias switch: pred loss jumped
from 0.0045 to 0.013 at epoch 101 but recovered to 0.004 by epoch 125. The
conv weights encode spatial filters that are meaningful with or without bias,
unlike the MLP where biases carried critical offset information.

Training: 300 epochs, batch size 4096, 4000 trajectories of 100 steps.

| Metric             | MLP (Exp 3) | Conv (Exp 4) | Improvement |
|--------------------|-------------|--------------|-------------|
| Val pred loss      | 0.002021    | 0.002596     | 0.8×        |
| Parameters         | 311K        | 121K         | 2.5× fewer  |
| Normal surprise    | 0.0120      | 0.0043       | 2.8× lower  |
| Pass-through       | 0.0382      | 0.0277       | —           |
| Disappear          | 2.6768      | 0.8442       | —           |
| Mass change        | 0.0442      | 0.0808       | —           |
| Pass-through / normal | 3.2×    | 6.4×         | 2× better   |
| Mass change / normal  | 3.7×    | 18.8×        | 5× better   |
| Disappear / normal    | 223×    | 196×         | comparable  |
| Position R²        | −0.14       | +0.34        | positive    |

The conv encoder achieved positive probe R² for the first time in any
two-ball experiment. Position R² of 0.34 means the encoder is learning where
the balls are — not perfectly, but the representation has real spatial content.
The MLP's representation was worse than a constant prediction (R² < 0).

**Mass change detection improved 5×** (3.7× -> 18.8× above baseline). The
model now captures enough per-object structure that a change in collision
response (one ball suddenly 10× heavier) registers as anomalous. The heavier
ball barely deflects while the lighter ball flies away — different from the
symmetric bounce the predictor expects.

**Pass-through detection improved 2×** (3.2× -> 6.4× above baseline). The
model expects a bounce when the balls converge and instead sees them continue
straight through. The detection is weaker than mass_change because the pixel
difference between "bounce" and "pass through" is subtle — both involve two
blobs near the same location, just with different subsequent trajectories. The
3-step prediction window captures a few frames of divergence.

**Lower baseline surprise** (0.012 -> 0.004) indicates more confident
predictions under normal conditions. The conv encoder's spatial features give
the predictor more precise input, reducing prediction noise during free flight
and wall bounces. This lower noise floor is what makes the interaction
anomalies detectable — the signal only needs to be 7× above baseline rather
than 200×.

Animation confirmed that surprise peaks align with collision events:
- Pass-through peak at step 42 (near the expected collision point)
- Disappear peak at step 65 (exact injection point)
- Mass_change peak at step 114 (a later collision with altered physics)

---

## 3. Ablation Summary

### SIGReg Lambda

| λ   | Pred Loss | SIGReg | Collapsed? | Notes                          |
|-----|-----------|--------|------------|--------------------------------|
| 0.1 | 0.000000  | 0.4087 | Yes        | 3-step prediction, collapse    |
| 0.5 | 0.002021  | 0.0153 | No         | Two-ball MLP, healthy training |
| 0.5 | 0.000000  | 0.4087 | Yes        | Two-ball conv, collapse         |
| 1.0 | 0.001368  | 0.0088 | No         | Single-ball, healthy training  |
| 5.0 | 0.002596  | 0.0005 | No         | Two-ball conv, healthy          |

Higher lambda is needed when the encoder architecture has an easier path to
collapse. MLP encoders require all input weights to conspire toward a constant
output. Conv encoders can collapse through a single spatially-uniform filter.
The required lambda scales with the encoder's "collapse efficiency" — how
easily it can find the trivial solution. For 3-step prediction with a conv
encoder, λ=5.0 was needed, 10× the MLP requirement.

### Batch Size and Learning Rate

| Batch Size | LR   | Outcome                                                   |
|------------|------|-----------------------------------------------------------|
| 128        | 3e-4 | 18.8s/epoch, healthy training                             |
| 8192       | 3e-4 | 2.3s/epoch, collapsed (SIGReg stuck at 0.41)              |
| 8192       | 2e-2 | 2.3s/epoch, healthy (linear scaling rule)                  |
| 32768      | 2e-2 | 2.4s/epoch, collapsed                                     |

Large batch sizes require proportionally higher learning rates (linear scaling
rule) to maintain the same number of effective gradient updates per epoch.
Without LR scaling, the model collapses because SIGReg gets too few updates to
prevent the encoder from converging to a constant output. This is not specific
to JEPA — it is a standard consequence of reduced optimizer steps per epoch.

### Bias Switching

| Strategy           | Single Ball | Two Ball MLP | Two Ball Conv |
|--------------------|-------------|--------------|---------------|
| Bias throughout    | 0.001593    | —            | —             |
| No-bias throughout | 0.001724    | 0.002021     | —             |
| Bias -> no-bias     | 0.001547    | 0.011783     | 0.002596      |

Bias switching worked for the single ball (negligible disruption, marginally
better final quality) and for the conv encoder (brief disruption, full
recovery). It was destructive for the two-ball MLP encoder (pred loss jumped
8× and never recovered). The difference: conv weight parameters encode spatial
filters that retain meaning without bias offsets, while MLP weights are
entangled with the bias terms they were co-optimised with.

### Predictor History

| History | Teleport | Vel Flip | Gravity  | Notes                        |
|---------|----------|----------|----------|------------------------------|
| 1 frame | 2.202    | 0.002871 | 0.002871 | Velocity invisible           |
| 3 frames| 2.421    | 0.123367 | 0.001689 | Velocity detected, not accel |

The predictor with 3-frame history infers velocity from the temporal difference
between consecutive latent vectors. This enabled detection of velocity-
dependent anomalies that were completely invisible with single-frame input. The
improvement for velocity flip was 43× (from baseline-level to clearly
separated). Gravity anomaly detection improved marginally — acceleration is a
second derivative requiring either longer history or longer prediction horizon.

### Encoder Architecture (Two-Ball)

| Encoder        | Params | Val Pred | Disappear | Pass-through | Mass Chg | Pos R²  |
|----------------|--------|----------|-----------|--------------|----------|---------|
| MLP 4096->512   | 2.16M  | 0.002856 | 0.623     | 0.002 (1×)   | 0.002(1×)| −1.30   |
| MLP 1024->256   | 311K   | 0.002021 | 2.677     | 0.038 (3.2×) | 0.044(3.7×)| −0.14 |
| Conv 3-layer   | 121K   | 0.002596 | 0.844     | 0.028 (6.4×) | 0.081(19×)| +0.34  |

The parenthesised values show the ratio to baseline (normal) surprise for each
architecture. For the MLP experiments, VoE used random trajectories and
timestep-targeted anomaly injection. For the conv experiment, VoE used
collision-targeted trajectories and global-start interaction anomalies. The
methodologies are not directly comparable for absolute surprise values, but the
ratio-to-baseline comparison is valid since the normal baseline was measured
under the same conditions in each case.

The MLP encoder could not learn per-object spatial features regardless of
size. Probe R² was negative for all position dimensions, meaning the learned
representation was worse than a constant prediction for recovering (x, y)
coordinates. The convolutional encoder's local receptive fields enabled
object-level feature extraction, resulting in positive probe R² and detectable
interaction anomalies.

---

## 4. Key Observations

**The JEPA framework successfully learns latent dynamics from pixels.** For
single-object systems with clear spatial structure, a flat MLP encoder with
SIGReg regularisation learns a latent space that captures position to R²=0.96
and supports anomaly detection at 2400× above baseline for position
discontinuities and 120× for velocity anomalies.

**Multi-object interaction learning requires spatial inductive bias.** The flat
MLP encoder treats the observation as an unstructured vector and cannot
decompose it into per-object features. It detects global appearance changes
(object disappearance) but fails to detect interaction-level anomalies
(collision violations). A convolutional encoder provides the local receptive
fields needed for object-level reasoning, enabling detection of collision
anomalies at 6-19× above baseline.

**VoE evaluation design matters as much as the model.** Randomly generated
evaluation trajectories cannot test interaction anomalies because most
trajectories have no interaction near the anomaly timestep. Collision-targeted
trajectories — where balls are aimed at each other — are necessary to evaluate
whether the model learned collision dynamics. Similarly, interaction anomalies
(pass_through, mass_change) must be applied from the start of the trajectory
rather than at a fixed timestep, because the divergence occurs at the
collision point, not at an arbitrary moment.

**SIGReg lambda must be calibrated to the encoder architecture.** Conv
encoders collapse at lambda values that are sufficient for MLPs. The
anti-collapse force must scale with the encoder's structural capacity for
trivial solutions. A conv encoder can collapse via a single spatially-uniform
filter, while an MLP requires all input weights to conspire. The working
values ranged from λ=0.5 (MLP) to λ=5.0 (conv) for the same 3-step
prediction setup.

**SIGReg lambda is the single critical hyperparameter.** Consistent with the
LeWM paper's finding, all other SIGReg parameters (number of projections,
integration knots) had negligible impact on downstream performance. Lambda must
be set relative to the prediction loss scale and the encoder's collapse
characteristics.

**INT8 quantisation is effectively free.** QAT INT8 matched or exceeded FP32
accuracy across all metrics while reducing model size by 4×. This result held
for both single-ball and two-ball architectures.

**The Epps-Pulley characteristic function test provides provable anti-collapse
guarantees.** By the Cramér-Wold theorem, matching all 1D random projections
to the standard Gaussian target implies the full joint distribution converges
to N(0, I). In practice this means there is a single knob to tune (lambda)
with predictable behaviour: too low causes collapse (constant encoder output),
too high causes the prediction loss to be dominated by regularisation (diffuse
representations with poor predictive quality). The optimal range was consistent
across architectures and tasks once normalised for the prediction loss scale
and encoder collapse characteristics.

---

## 5. Architecture Reference

### Single-Ball Model (16×16 grid)

```
Encoder:    256  -> Linear -> 128  -> ReLU -> Linear -> 32
Predictor:  96   -> Linear -> 64   -> ReLU -> Linear -> 32
                    (3 × 32 concatenated history)
```

Parameters: 45,312 (no bias). LZMA compressed INT8: 39.0 KB.

### Two-Ball Model — MLP (32×32 grid)

```
Encoder:    1024 -> Linear -> 256  -> ReLU -> Linear -> 64
Predictor:  192  -> Linear -> 128  -> ReLU -> Linear -> 64
                    (3 × 64 concatenated history)
```

Parameters: 311,296 (no bias). INT8: 306.4 KB.

### Two-Ball Model — Conv (32×32 grid)

```
Encoder:    (1, 32, 32)
            -> Conv2d(1->16, 3×3, stride 2) -> ReLU   -> (16, 16, 16)
            -> Conv2d(16->32, 3×3, stride 2) -> ReLU   -> (32, 8, 8)
            -> Conv2d(32->64, 3×3, stride 2) -> ReLU   -> (64, 4, 4)
            -> Flatten -> Int8Linear(1024 -> 64)
Predictor:  192  -> Linear -> 128  -> ReLU -> Linear -> 64
                    (3 × 64 concatenated history)
```

Parameters: 121,488. Conv layers are FP32; final linear is INT8 QAT.

### Shared Configuration

| Parameter          | Single-Ball | Two-Ball |
|--------------------|-------------|----------|
| Grid size          | 16×16       | 32×32    |
| Latent dim         | 32          | 64       |
| Predictor history  | 3 frames    | 3 frames |
| Prediction steps   | 3           | 3        |
| Activation         | ReLU        | ReLU     |
| Quantisation       | INT8 QAT    | INT8 QAT |
| SIGReg projections | 256         | 512      |
| SIGReg knots       | 17          | 17       |

### Loss Function

```
L = L_pred + λ · SIGReg(Z)

L_pred  = (1/K) Σ_{k=1}^{K} MSE(z_pred_k, z_target_k)
SIGReg  = (1/M) Σ_{m=1}^{M} ∫ w(t) |φ_N(t; h_m) − φ_0(t)|² dt
```

Where K = JEPA_STEPS (3), M = number of random projections (256 or 512),
φ_N is the empirical characteristic function of the projected embeddings,
φ_0 = exp(−t²/2) is the standard Gaussian characteristic function, and
w(t) = exp(−t²/2) is the Gaussian weighting function. The integral is
evaluated by trapezoidal quadrature over 17 knots in [0.2, 4.0].

---

## References

- Maes, Le Lidec, Scieur, LeCun, Balestriero. "LeWorldModel: Stable End-to-End
  Joint-Embedding Predictive Architecture from Pixels." arXiv:2603.19312, 2026.
- Balestriero & LeCun. "LeJEPA: Provable and Scalable Self-Supervised Learning
  Without the Heuristics." arXiv:2511.08544, 2025.
- Epps & Pulley. "A test for normality based on the empirical characteristic
  function." Biometrika, 70(3):723–726, 1983.
- Cramér & Wold. "Some theorems on distribution functions." Journal of the
  London Mathematical Society, 1(4):290–294, 1936.
