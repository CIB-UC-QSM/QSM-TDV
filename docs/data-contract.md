# QSM sample contract

One sample is represented by `subject_or_sample.npz` and an adjacent JSON
manifest with the same stem. This intentional one-sample boundary keeps the
current overfit diagnostic separate from future cohort and challenge loaders.

The NPZ uses float32 unbatched `[Z,Y,X,1]` arrays:

- `local_field`: measured local field, \(b\);
- `susceptibility`: reference susceptibility, \(\chi_{ref}\);
- `chi_init`: legacy compatibility array. Newly generated samples store
  `W * local_field`; the loader derives the canonical initial state
  χ_0 = `W * local_field` and ignores any stored value;
- `brain_mask`: non-negative support mask applied to susceptibility before the
  TDV regularizer, \(R_\theta(\text{brain_mask}\odot\chi)\), and used in the
  default observation weight;
- `reference_mask` (optional): non-negative supervised-loss mask;
- `magnitude` (optional): non-negative magnitude map used to form the default
  data weight \(W=\text{brain_mask}\cdot\text{magnitude}\);
- `statistical_weight` (optional): non-negative compatibility map for custom
  physics experiments.

The default data-consistency residual is \(\lVert W(A\chi-b)\rVert_2\). If
the magnitude map is unavailable or `--no-include-magnitude-in-weight` is
selected, \(W=\text{brain_mask}\). The semi-implicit data energy is
\(\frac12\lVert W(A\chi-b)\rVert^2\), and implementation applies \(W^TW\)
explicitly before the dipole adjoint. No multiplier is folded into \(A\).
The reconstruction initial state uses the same resolved weight map:
\(\chi_0=W\cdot\text{phase_in}\) (or \(W\cdot b\) for prepared samples).

The manifest requires these fields:

```json
{
  "anonymized_subject_id": "...",
  "source": "...",
  "processing_version": "...",
  "voxel_size_zyx": [vz, vy, vx],
  "b0_direction_zyx": [bz, by, bx],
  "field_units": "ppm",
  "susceptibility_units": "ppm",
  "susceptibility_reference": "..."
}
```

There must be no identifiable clinical metadata in this manifest or training
logs. New loaders must preserve affine and orientation metadata separately,
transform B0 whenever they reorient image axes, and perform subject-level
splitting before patch extraction.

## COSMOS noisy-signal diagnostic

`qsm_tdv.data.cosmos` reads the supplied `chi_cosmos.mat`, `magn.mat`, and
`msk.mat` files. It requires their named three-dimensional variables and never
deduces physical metadata from their shape. The CLI defaults to explicitly
declared `voxel_size_zyx=(1,1,1)` and `b0_direction_zyx=(0,0,1)`; use its
arguments to provide the acquisition-specific values when known.

It computes `b=A(chi_cosmos)`, `phase=phase_scale*b`, and
`S=magn*exp(-i*phase)`. A circular complex Gaussian realization is globally
scaled to make its realised in-mask L2 SNR exactly 70 by default. The field
given to TDV-QSM is `-angle(S_noisy)/phase_scale`, masked only as an explicit
observation operation. No phase unwrapping is silently applied.
The prepared COSMOS sample records `magnitude` and uses
`W = brain_mask * magnitude` by default.
