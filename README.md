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

Run all tests with:

```bash
uv run pytest
```
