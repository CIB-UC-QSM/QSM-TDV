# QSM physics and medical-data contract

QSM physics and medical-volume handling are project-specific extensions to the
2-D source TDV repository.

## Axes, units, and boundaries

Images and fields use `[B,1,Z,Y,X]`; voxel sizes and field directions use
`[B,3]` in strict `zyx` order. `QSMOperator` normalizes each nonzero B0
direction and supports a separate anisotropic voxel size per batch item.

The physical model is `b = F^H D F chi + eta`. It uses orthonormal float32 /
complex64 FFTs and periodic boundaries. No pad, crop, mask, dipole threshold,
TKD, or inverse is hidden in the operator. `forward` multiplies by `D` and
`adjoint` by `D.conj()`.

Field and susceptibility units are explicit sample metadata. The COSMOS
simulation performs no unit conversion; its configured `phase_scale` must be
consistent with those source units.

The learned TDV convolutions use a different, explicit boundary rule:
edge-inclusive symmetric extension and its exact transpose. Scale changes use
learned convolution, separable binomial antialiasing, and stride two. Their
adjoints recover requested odd and even shapes exactly.

## Stored magnitude weight

For the COSMOS runner, magnitude is treated as dimensionless and the rule is:

```text
W = sqrt(2) * raw_magnitude
normalization = none
clipping = none
masking = none
zero magnitude -> W = 0
stored array = W, not sqrt(W)
```

Thus the data energy and its exact force are

\[
D_W(\chi;b)=\tfrac12\|W(A\chi-b)\|_2^2,
\qquad
\nabla D_W=A^H W^2(A\chi-b).
\]

The two applications of `W` in the force are intentional. The brain mask is a
separate state/evaluation choice. `lambda` controls the data force inside the
unrolled dynamics; `beta_dc` controls an optional terminal loss. Neither
changes the definition of `W`.

## Full sample validation

`QSMSample` contains:

```text
local_field, susceptibility, brain_mask, magnitude_weight, initial [1,Z,Y,X] float32
voxel_size_zyx, b0_direction_zyx                              [3] float32
subject_id, field_unit, susceptibility_unit                   explicit strings
reference_convention, processing_version                     explicit strings
```

Validation rejects nonfinite values, negative weights, empty/negative masks,
nonpositive voxel sizes, nonnormalized B0, mismatched shapes, inconsistent
optional affines, contact/path-like subject IDs, and subjects duplicated across
splits. The frozen sample record prevents accidental reassignment of the
processing version.

The `reference_convention` must be acted on before NRMSE. The provided helper
supports `already_referenced` (mask only) and `masked_mean_zero` (independent
per-sample in-mask mean subtraction followed by masking). The same NRMSE
implementation is used for optimization and reporting.

Subject splitting precedes patch extraction. Physical QSM updates should use
whole volumes because the dipole operator is global. Patch-based work requires
globally generated local fields plus a documented halo/crop boundary policy.
