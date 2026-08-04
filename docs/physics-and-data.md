# TDV-QSM physics and data contract

The field model is `b = A(chi) + eta`, with
`A = F^H D F`.  Input tensors use `[B, 1, Z, Y, X]`; metadata is strictly
`zyx`: `voxel_size_zyx=[vz, vy, vx]` and
`b0_direction_zyx=[bz, by, bx]`.  Susceptibility and field units must agree
with the user-supplied `phase_scale` in the signal simulation.  The source
MATLAB arrays are assumed already in `zyx` order and no unit conversion is
performed by the loader.

`DipoleOperator3D` runs float32/complex64 unitary FFTs and represents periodic
Fourier boundary conditions.  It does not pad, crop, apply a mask, threshold
the dipole (TKD), or materialize a dense convolution matrix.  Its `forward`
and `adjoint` methods are separate interfaces, although a real dipole kernel
makes their numerical action equal.

The TDV CNN produces a signed, bias-free response `h_theta(chi)`, but the
regularizer uses the nonnegative local potential
`0.5 * h_theta(chi)^2`.  Consequently
`R_theta(chi) = sum(mask * 0.5 * h_theta(chi)^2) >= 0` for every finite image
and nonnegative mask, with `R_theta(0)=0`.  The square and spatial reduction
are evaluated in float32 even when the CNN convolutions use CUDA float16 AMP.
The final energy-head weights are initialized uniformly in
`[-s/sqrt(features), s/sqrt(features)]`, with `s=0.2` by default.  This makes
the initial quadratic energy `0.04` of the corresponding unit-scale-head
energy while avoiding both the zero-gradient failure of an exactly zero head
and subnormal float16 head responses.  The scale is stored in training
configuration and checkpoints.

Training history compares like normalizations:
`data_consistency_value = ||W(A chi-b)||^2/N` and
`regularization_energy = R_theta(chi)/N`.  It additionally stores the summed
`regularization_energy_total` for auditing.  Only the displayed/logged
regularizer value is normalized; the explicit reconstruction continues to
use the gradient of the summed scalar energy.

For the COSMOS runner the magnitude preprocessing rule is intentionally
minimal and immutable:

```text
input magn: raw finite nonnegative magn values from magn.mat
W:          sqrt(2) * magn
```

There is no magnitude normalization, clipping, mask multiplication, or
spatial rescaling.  `W` is used both in the unrolled data gradient
`A^H W^2 (A chi - b)` and, if enabled, in the optional data-consistency
residual `W(A chi - b)`.  The brain mask is instead an explicit reconstruction
state and susceptibility-evaluation mask.  `report.json` records this rule
for every training run.

For every epoch, clean field `A(chi_gt)` is encoded in a complex magnitude
signal, independent real/imaginary Gaussian noise is added at the configured
SNR, and the noisy phase field is recovered.  The epoch seed is `seed+epoch`,
so re-running an experiment is reproducible while its epochs retain distinct
noise samples.
