# Source-grounded 3-D TDV-QSM in PyTorch

This repository implements an energy-based 3-D Total Deep Variation (TDV)
regularizer inside an explicit quantitative susceptibility mapping (QSM)
reconstruction. It does not contain a direct field-to-susceptibility network.

The architectural ideas are grounded in `VLOGroup/tdv`, whose implementation
is for 2-D denoising/eigenfunction work. The 3-D convolutions, QSM physics,
magnitude-derived weight, NRMSE training, CUDA float16 AMP, and medical-volume
handling in this repository are project-specific extensions; they are not
features claimed for the source repository.

## TDV energy and force

For `m = num_features`, the regularizer implements

\[
\rho_\theta(\chi)=\frac1m K_N M_L\cdots M_1K_1\chi,
\qquad
R_\theta(\chi)=\sum_{z,y,x}\rho_\theta(\chi)_{z,y,x}.
\]

`TDVEnergy3D.energy_density`, `energy`, and `force` are distinct APIs. The
production force manually traverses the exact transpose Jacobian of every
micro/macro block and linear operator. `force_autograd_reference` is the slow
test oracle. The energy density is signed; it is not squared or implicitly
masked.

The learned path uses:

- bias-free `3x3x3` analysis with zero-mean, unit-norm initialization and a
  zero-mean/unit-ball projection after optimizer steps;
- persistent three-scale macro state, with exactly five Student-t micro-blocks
  per macro-block;
- \(\phi(a)=\tfrac12\log(1+a^2)\) and
  \(\phi'(a)=a/(1+a^2)\), evaluated in float32;
- edge-inclusive symmetric padding with its exact accumulation adjoint;
- learned scale operators followed by the separable binomial antialias filter
  \([1,4,6,4,1]^{\otimes3}/16^3\);
- independent learned down and up operators, where the up traversal calls the
  exact adjoint of its own scale operator;
- a bias-free one-channel `1x1x1` energy head.

## QSM physics and explicit update

The project extension uses periodic Fourier boundaries:

\[
A=\mathcal F^H D\mathcal F,
\qquad
d(\mathbf k)=\frac13-
\frac{(\mathbf k\cdot\widehat{\mathbf B}_0)^2}{\|\mathbf k\|_2^2},
\qquad d(\mathbf0)=0.
\]

`QSMOperator.forward` and `adjoint` are separate; the adjoint uses `D.conj()`.
The kernel supports batched anisotropic voxel sizes and arbitrary per-sample
field directions. FFT inputs are float32, Fourier tensors complex64, FFTs are
orthonormal, and the forward model performs no masking, cropping, padding,
thresholding, TKD, or inversion.

For stored real nonnegative diagonal `W`, the data force is exactly

\[
g_D=A^H W^2(A\chi-b),
\]

and the explicit branch is

\[
\chi_{s+1}=\chi_s-\frac{T}{S}\nabla R_\theta(\chi_s)
-\frac{\lambda}{S}g_D.
\]

`T` and `lambda` have independent learned bounded or fixed modes. State
masking after each step is configurable. Only the final state is retained by
default; diagnostics can request all states. The manual force can be activation
checkpointed with `use_reentrant=False`.

## Magnitude weighting and evaluation

The COSMOS diagnostic has one immutable preprocessing rule:

```text
input magn: treated as raw, finite, nonnegative, dimensionless values
normalization: none
clipping: none
masking inside W: none
zero magnitude: stored W is zero
stored quantity: W itself, not sqrt(W)
W: sqrt(2) * magn
```

The brain mask is separate from `W`. Training and optional terminal data
consistency use the same stored weight without changing its meaning. NRMSE is
computed per sample and then averaged. `mask_and_reference` applies either an
explicit `already_referenced` convention or per-sample `masked_mean_zero`
referencing before NRMSE.

`QSMSample` defines and validates the full medical-volume contract, including
units, reference convention, immutable processing-version string, anonymized
subject ID, normalized B0, optional affine consistency, and subject-level
split isolation. Subjects must be split before patch extraction. Because the
dipole operator is global, isolated-patch physical updates are unsupported
unless fields were generated globally and a crop/halo policy is documented.

## Mixed precision

CUDA uses mixed precision only for learned convolutions, transposed
convolutions, multiscale features, and their additions. The state, local field,
weights, masks, Student-t calculations, force outputs, physics, reductions,
NRMSE, parameters, and optimizer state remain float32; Fourier tensors remain
complex64. The model is never globally converted to half precision. Training
uses `torch.amp.GradScaler`, unscales before finite-gradient checks/clipping,
and projects the analysis kernel after every optimizer update.

## COSMOS same-volume diagnostic

The included runner is a controlled overfit/smoke diagnostic, not evidence of
cross-subject generalization. It accepts the sibling `chi_cosmos.mat`,
`magn.mat`, and `msk.mat` files (or one equivalent MAT/NPZ file), simulates a
new deterministic complex-noise realization per epoch, and records all
mathematical conventions.

```bash
uv sync --group dev

uv run tdv-qsm-train \
  --data /cosmos_data \
  --output-dir runs/cosmos-snr70 \
  --epochs 100 \
  --snr 70 \
  --features 4 \
  --steps 10 \
  --maximum-time 0.25 \
  --maximum-lambda 1.0 \
  --learning-rate 1e-4
```

The run writes `history.csv`, `history.png`, `reconstruction.png`,
`checkpoint.pt`, and `report.json`. The optional terminal loss is
`NRMSE + beta_dc * ||W(A chi-b)||^2/N`; `beta_dc` is independent of the
dynamics coefficient `lambda` and spatial reliability `W`.

Before optimization begins, the training runner prints the complete model
architecture followed by the number of microblocks, macroblocks, learned
convolutions, total parameters, and trainable parameters. The convolution
count includes every learned `AdjointConv3d`, including those inside scale
operators, and excludes fixed binomial antialias filters.

## Data-only gradient-descent baseline

The baseline evaluator reconstructs susceptibility without constructing or
loading the learned TDV regularizer.  Starting from the same masked weighted
backprojection used by the training diagnostic, it runs the fixed update

\[
\chi_{s+1}=\chi_s-\tau A^H W^2(A\chi_s-b).
\]

The iteration count is fixed before evaluation; ground truth is used only for
the initial and final masked/referenced NRMSE and never for stopping or model
selection. If `--step-size` is omitted, the evaluator uses the conservative
value
`1 / (||W||_inf^2 ||D||_inf^2)` without normalizing or otherwise changing
`W` or `D`.

```bash
uv run tdv-qsm-gradient-baseline \
  --data /cosmos_data \
  --output-dir runs/cosmos-gradient-baseline \
  --steps 100 \
  --snr 70
```

Use `--step-size VALUE` to override the automatic step. The script uses the
same one-realization COSMOS noise simulation, magnitude rule, periodic dipole
operator, mask, and susceptibility-reference convention as the training
diagnostic. It writes `history.csv`, `history.png`, `reconstruction.png`,
`reconstruction.pt`, and `report.json`.

## Learned-regularizer evaluation

The learned-regularizer evaluator accepts a model directory containing
`checkpoint.pt` and `report.json`, restores `TDVEnergy3D`, and runs a fixed
number of explicit reconstruction iterations:

\[
\chi_{s+1}=\chi_s
-\tau_R\nabla R_\theta(\chi_s)
-\tau_D A^H W^2(A\chi_s-b).
\]

By default, the evaluator loads the final per-step model weights directly from
`report.json` at `taus.regularizer` and `taus.data`. It does not reconstruct
them from raw checkpoint parameters. An explicit `taus` argument overrides
the report values and is ordered as `tau_R` followed by `tau_D`. The Python
API is:

```python
evaluate_learned_regularizers(
    model_directory_path,
    dataset_directory_path,
    gradient_parameters,
    total_iterations,
    taus=None,
    snr=None,
)
```

`gradient_parameters` is a mapping that can override `device`,
`voxel_size_zyx`, `b0_direction_zyx`, `snr`, `phase_scale`, `seed`,
`mask_state_each_step`, `use_amp`, and `output_dir`. Architecture and physical
metadata otherwise come from `checkpoint.pt` when available. The explicit
`snr` argument takes precedence over both `gradient_parameters["snr"]` and the
checkpoint value, and must be finite and positive.

The evaluator conditionally resolves a COSMOS directory as follows:

```text
phase.mat present: load phase directly as b; do not run field simulation
phase.mat absent: simulate b from chi.mat or chi_cosmos.mat at the configured SNR
magn.mat present: store the supplied finite nonnegative magn directly as W
magn.mat absent: use mask as W
chi.mat present: compute ground-truth NRMSE after every iteration
chi.mat absent: omit the ground-truth metric
initial.mat present: use the supplied initial state
initial.mat absent: use the masked weighted normal backprojection
```

The raw-`magn` evaluation rule and its exact `mask` fallback are specific to
this evaluator. They do not replace the training runner's documented
`W = sqrt(2) * magn` preprocessing rule.

At iteration `s`, `tol_update` is

\[
\operatorname{NRMSE}(\chi_s,\chi_{s-1})
=\frac{\|\chi_s-\chi_{s-1}\|_2}{\|\chi_{s-1}\|_2},
\]

so the immediately preceding state is treated as the NRMSE reference. When
`chi.mat` is present, masked ground-truth NRMSE is recorded on the same
iteration axis.

```bash
uv run tdv-qsm-evaluate-regularizers \
  runs/cosmos-snr70 \
  /cosmos_data \
  --iterations 100 \
  --snr 70 \
  --gradient-parameters \
  '{"voxel_size_zyx":[1,1,1],"b0_direction_zyx":[0,0,1]}' \
  --output-dir runs/cosmos-regularizer-evaluation
```

`--gradient-parameters` accepts either a JSON object or a path to a JSON file.
`--snr` controls complex-noise simulation when `phase.mat` is absent. When
`phase.mat` exists, its field values are loaded directly and SNR does not alter
them.
Use optional `--taus TAU_REGULARIZER TAU_DATA` to override the report values.
If `--taus` is omitted, `report.json` must contain finite nonnegative
`taus.regularizer` and `taus.data` values. Training writes the final post-update
values into this report so they correspond to the saved `checkpoint.pt`.
The run writes `metrics.csv`, a two-panel `metrics.png` line chart with
`tol_update` and ground-truth NRMSE in separate panels, `chi_pred.mat`, and
`chi_pred.png`. Without `chi.mat`, the ground-truth panel is marked
unavailable. The prediction figure reuses the COSMOS training diagnostic's
three anatomical planes, radiological rotations, grayscale susceptibility
range `[-0.1, 0.1]`, magma absolute-error colormap, layout, labels, and color
bars. Without `chi.mat`, it produces the corresponding prediction-only
three-plane grayscale view.

Run all tests with:

```bash
uv run pytest
```
