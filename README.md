# TDV-QSM (PyTorch)

This repository implements Total Deep Variation (TDV) for 3-D quantitative
susceptibility mapping in PyTorch.  It is an energy-based reconstruction:
the network produces the scalar energy \(R_\theta(\chi)\), and the learned
force is obtained only as \(\nabla_\chi R_\theta(\chi)\).  It contains no
direct field-to-susceptibility network.

The bias-free TDV CNN and its `1x1x1` head produce a signed response
\(h_\theta(\chi)\).  The implemented local potential is

\[
r_\theta(\chi)=\frac12 h_\theta(\chi)^2,
\qquad
R_\theta(\chi)=\sum_i m_i r_{\theta,i}(\chi)\ge 0,
\]

for the nonnegative integration mask \(m\).  Thus every voxel contribution is
nonnegative and, because the network is bias-free, \(R_\theta(0)=0\).
The energy head is initialized uniformly in
\([-s/\sqrt{q},s/\sqrt{q}]\), with the configurable default `s=0.2` and
`q=features`.  Since the potential is quadratic, this deliberately starts the
regularization energy and force small without initializing them identically
to zero.  Override it with `--regularizer-head-initialization-scale` only as a
documented stability experiment.

The unrolled solver uses explicit Euler / plain gradient descent, not the
semi-implicit discretization used in the original TDV paper:

\[
\chi_{s+1}=\chi_s-\tau\left[A^H W^2(A\chi_s-b)+\nabla_\chi R_\theta(\chi_s)\right],
\qquad \tau=\frac{T_{\max}\sigma(\alpha)}{S}.
\]

## Single-volume COSMOS training

The supplied data layout is supported directly:

```text
/cosmos_data/
  chi_cosmos.mat  # variable: chi_cosmos
  magn.mat        # variable: magn
  msk.mat         # variable: msk
```

When `/cosmos_data` is not mounted, the CLI falls back to the repository's
`cosmos_data/` directory.  A self-contained `.mat` or `.npz` file containing
`chi`/`susceptibility`, `magn`/`magnitude`, and `msk`/`brain_mask` is also
accepted.

```bash
uv sync --group dev

uv run tdv-qsm-train \
  --data /cosmos_data \
  --output-dir runs/cosmos-snr70 \
  --epochs 100 \
  --snr 70 \
  --features 1 \
  --steps 1 \
  --learning-rate 1e-4
```

Equivalently, run `uv run python scripts/train_cosmos_tdv.py ...`.
`--snr` is configurable.  A fresh deterministic random seed (`seed + epoch`)
is used for each epoch, so every epoch receives a unique complex Gaussian
noise realization.

The forward simulation is

\[
b_{\mathrm{clean}}=A\chi,\quad
s=\mathrm{magn}\,e^{-i\,c b_{\mathrm{clean}}},\quad
b=-\arg(s+n)/c,
\]

where `c` is `--phase-scale`; each real and imaginary component of `n` has
standard deviation `mean(magn inside brain mask) / SNR`.  The exact,
documented data weight is

\[
W=\sqrt{2}\,\mathrm{magn}.
\]

The loader uses raw supplied magnitudes: it does not normalize, clip, or
silently multiply the field weight by the brain mask.  The mask is applied to
the reconstruction state and supervised susceptibility comparison.  Run
metadata records this preprocessing rule, the noise rule, SNR, spatial
metadata, and all optimization settings.

Each run produces:

- `history.csv` and `history.png` with loss, NRMSE, data consistency, and the
  nonnegative TDV regularization energy by epoch;
- `reconstruction.png`, showing initial \(X_0\), prediction, truth, and
  prediction-minus-truth residual, each fixed to `[-0.1, 0.1]`;
- `checkpoint.pt` and `report.json`.

The primary loss and reported metric are per-sample NRMSE.  An optional
separate data-consistency loss uses exactly \(W(A\chi-b)\), controlled by
`--data-consistency-weight`; it never changes the definition of `W`.
For a meaningful history comparison, the energy panel plots both terms per
evaluated field/brain voxel: \(\|W(A\chi-b)\|^2/N\) and \(R_\theta(\chi)/N\).
The reconstruction itself still differentiates the full summed
\(R_\theta\); this logging normalization does not change its update.

## Numerical conventions

Tensor layout is `[B, 1, Z, Y, X]`; voxel size and B0 direction are always
`zyx` and are configurable with `--voxel-size VZ VY VX` and
`--b0-direction BZ BY BX`.  The dipole kernel supports anisotropic voxels and
arbitrary nonzero B0 directions, has `d(0)=0`, uses unthresholded unitary FFTs
(`norm="ortho"`), and has periodic physical boundaries.  It has no hidden
padding, crop, mask, TKD, or inverse operation.  CNN convolutions use zero
padding; their multiscale downsampling uses a box low-pass before decimation.

The state, field, masks, weights, energy reduction, physical gradient, loss,
and FFT/IFFT remain float32/complex64.  CUDA TDV convolutions use float16 AMP
autocast; training uses `GradScaler`, gradient clipping, and projects every
analysis filter back to zero mean after each Adam step.  The same TDV
parameters are shared across all explicit steps.

## Tests

```bash
uv run pytest
```

The suite covers the dipole adjoint relation on odd/even and anisotropic grids,
scalar energy gradients and directional finite differences, explicit Euler
with \(W^2\), NRMSE/data-consistency definitions, and a tiny end-to-end
single-volume training run.
