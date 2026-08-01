# TDV-QSM

An energy-based JAX implementation of Total Deep Variation (TDV) for 3D
quantitative susceptibility mapping. It implements the required chain

```text
chi -> R_theta(chi) -> grad_chi R_theta(chi) -> semi-implicit QSM solver -> chi_S
```

It does not contain a direct field-to-susceptibility CNN.

## Quick start

Use `uv` to create and synchronize the project virtual environment:

```bash
uv sync --group dev

uv run python scripts/generate_synthetic_single.py \
  --output data/synthetic/synthetic_0001.npz

uv run python scripts/train_single_dataset.py \
  --dataset data/synthetic/synthetic_0001.npz \
  --output-dir runs/synthetic-overfit \
  --iterations 100 --features 4 --steps 2 --cg-iterations 12

uv run python scripts/validate_single_dataset.py \
  --dataset data/synthetic/synthetic_0001.npz \
  --checkpoint runs/synthetic-overfit/checkpoint.pkl \
  --output-dir runs/synthetic-validation

uv run pytest
```

The training/validation pair is an explicitly labelled *same-sample overfit
diagnostic*. It proves that the higher-order derivatives, TDV parameters, and
stopping time are trainable; it is not a generalisation result.

For the COSMOS overfit diagnostic, use the provided full-volume files (the
script accepts `/cosmos_data` and the workspace-mounted `cosmos_data/`):

```bash
uv run python scripts/train_cosmos_overfit.py \
  --cosmos-data /cosmos_data \
  --output-dir runs/cosmos-snr70-100epochs \
  --iterations 100 --features 1 --steps 1 --cg-iterations 12 --remat-force
```

It computes `b = A(chi_cosmos)`, forms `S = magn * exp(-i * b)`, adds complex
Gaussian noise calibrated to SNR 70, estimates the noisy phase, and trains
against `chi_cosmos`. It saves orthogonal input/output/ground-truth/error slice
figures, `convergence.csv`, `convergence.png`, and masked NRMSE,
`||chi_recon - chi_gt|| / ||chi_gt||`.

The COSMOS runner saves its TDV parameters and Adam state in `checkpoint.pkl`.
If a long full-volume run must be interrupted, continue it without resetting
the fit, for example with `--resume --iterations 6 --total-epochs 100`.
It uses `W = mask * magnitude` by default; pass
`--no-include-magnitude-in-weight` to use `W = mask` instead. The same toggle
is available in the generic single-sample training CLI.

## Evaluate a COSMOS-like input directory

Evaluate a trained checkpoint on any directory containing `msk.mat` and either
`phase_in.mat` or an optional susceptibility reference. `phase_in.mat` must
contain the already prepared real local field `phase_in` in the field units of
the checkpoint. It is used directly as (b). If it is absent, the evaluator
requires `chi_cosmos.mat` / `chi_cosmos` or `chi.mat` / `chi` and simulates
(b=A\chi) with the periodic unitary dipole operator. If `chi` exists in
either case, it is used only as ground truth for reporting NRMSE at every
reconstruction step.

```bash
uv run python scripts/evaluate_qsm_set.py \
  --input-dir /path/to/input_set \
  --checkpoint runs/cosmos-snr70-100epochs/checkpoint.pkl \
  --output-dir runs/input-set-evaluation \
  --iterations 10 --cg-iterations 12
```

`--iterations` (also spelled `--steps`) is the number (S) of TDV-QSM
semi-implicit reconstruction steps. `--cg-iterations` is the fixed number of
conjugate-gradient updates used inside each one of those steps. Both override
the saved checkpoint configuration for this evaluation only. Voxel size and
B0 direction default to the checkpoint metadata; pass `--voxel-size VZ VY VX`
and `--b0-direction BZ BY BX` to override them explicitly.
When `magn.mat` (or `magnitude.mat`) is present, evaluation uses
`W = mask * magnitude` by default; pass `--no-include-magnitude-in-weight` to
use `W = mask`.

The output directory contains `prediction.npy`, `chi_initial.npy` (the weighted
field initial state, \(\chi_0=W\cdot\mathrm{phase\_in}\)),
`iteration_metrics.csv`, `report.json`, and `orthogonal_slices.png`. The CSV
uses `nrmse_to_gt` for ‖\(\chi_s-\chi_{gt}\)‖/‖\(\chi_{gt}\)‖ when a
reference exists. Its `tol_update_nrmse` is the requested convergence update:
‖\(\chi_s-\chi_{s-1}\)‖/‖\(\chi_{s-1}\)‖, so the prior reconstruction
is the ground truth for that row. It is intentionally separate from
`cg_relative_residual`, which reports the accuracy of the inner linear solve.

## Layout

```text
src/qsm_tdv/
  data/          # strict sample contract and deterministic synthetic source
  models/        # scalar TDV energy and zero-mean analysis projection
  physics/       # unitary Fourier dipole model, fixed CG, semi-implicit solver
  training/      # pure-JAX Adam, single-sample training, checkpoint validation
  evaluation/    # reserved challenge and in-vivo evaluation adapters
scripts/         # thin executable entry points
tests/           # numerical and end-to-end regression tests
configs/         # versioned experiment configurations as the project expands
docs/            # data contract and numerical-convention documentation
```

`data/` is deliberately limited to reading one anonymised, metadata-complete
sample today. Subject-level split management, patch extraction, challenge
adapters, and in-vivo importers belong under that same package and must keep
the validation/test subjects disjoint before any patching. Challenge- and
in-vivo-specific evaluation should be added under `src/qsm_tdv/evaluation/`
without changing the core physics or TDV energy modules.

## Numerical formulation

Volumes are always NDHWC: `[B, Z, Y, X, C]`. Voxel size and B0 direction are
always `(z, y, x)`, never inferred from shape. The physical operator is

\[
A(\chi) = \mathcal F^H D\mathcal F\chi,
\]

using `norm="ortho"`, float32 state arrays, complex64 FFTs, an unthresholded
continuous dipole kernel, and periodic Fourier boundaries. CNN convolutions
use zero `SAME` padding; they are separate from the physical boundary model.

The TDV model has a zero-mean learned analysis convolution, followed by `L`
three-scale U-Net-like macro-blocks. Each macro-block has exactly five
bias-free residual micro-blocks and uses the smooth log-Student-t activation.
The final 1×1×1 convolution gives local energy density; summing it gives one
scalar energy per batch item. The regularisation force is always
`jax.grad(sum(R_theta))`; it is never a separately predicted vector field.

For `S` steps and `tau = T/S`, the solver uses fixed-iteration CG to solve

\[
(I + \tau A^H A)\chi_{s+1}
= \chi_s + \tau[A^Hb - \nabla_\chi R_\theta(\chi_s)].
\]

Training and evaluation form one explicit data weight
`W = brain_mask * magnitude` when magnitude is available, and fall back to
`W = brain_mask` otherwise. The optional training/reporting data-consistency
term is `|| W (A chi - b) ||₂`; the semi-implicit data energy is
`0.5 || W (A chi - b) ||²` with its exact adjoint. The weight is never
silently folded into `A`.

After every Adam update the training loop functionally projects each output
filter of the analysis kernel to zero mean. Differentiation remains connected
through the TDV force, all unrolled steps, fixed CG operations, and the
sigmoid-parameterised stopping time.

## Data contract

A sample is `sample.npz` with an adjacent `sample.json` manifest. See
[the data contract](docs/data-contract.md). Arrays are unbatched `[Z,Y,X,1]`:

- required: `local_field`, `susceptibility`, `chi_init` (legacy compatibility;
  canonically emitted as `W * local_field`), `brain_mask`;
- optional: `reference_mask`, `magnitude`, `statistical_weight`.

The manifest makes source, processing version, anonymised subject ID, units,
reference convention, voxel size, and B0 direction explicit. The synthetic
generator also records a deterministic configuration hash and seed.

## Verification

The regression tests cover the Fourier adjoint identity on odd/even,
anisotropic, arbitrary-B0 volumes; DC handling; TDV scalar energy, force,
directional finite difference, and Hessian-vector product; zero-mean
projection; dense-CG agreement and right-hand-side derivatives; the
semi-implicit residual; and a deterministic tiny single-sample overfit run.

See [numerical conventions](docs/numerics.md) for the full implementation
choices and expected extension constraints.
