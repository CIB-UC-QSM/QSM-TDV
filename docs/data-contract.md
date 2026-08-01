# QSM sample contract

One sample is represented by `subject_or_sample.npz` and an adjacent JSON
manifest with the same stem. This intentional one-sample boundary keeps the
current overfit diagnostic separate from future cohort and challenge loaders.

The NPZ uses float32 unbatched `[Z,Y,X,1]` arrays:

- `local_field`: measured local field, \(b\);
- `susceptibility`: reference susceptibility, \(\chi_{ref}\);
- `chi_init`: legacy compatibility array; newly generated samples store zeros.
  The loader always emits a zero volume for (chi_0), ignoring any nonzero
  value in older files;
- `brain_mask`: non-negative regularizer integration and observation mask;
- `reference_mask` (optional): non-negative supervised-loss mask;
- `statistical_weight` (optional): non-negative data-term multiplier \(W\).

`statistical_weight` means \(W\), not a variance or a precision map. Its data
term is \(\frac12\lVert WM(A\chi-b)\rVert^2\), and implementation applies
\(M^TW^TWM\) explicitly before the dipole adjoint. A future loader that accepts
noise variance must convert it explicitly and document the conversion.

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
