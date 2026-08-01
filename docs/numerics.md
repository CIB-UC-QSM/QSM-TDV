# Numerical conventions and extension rules

The dipole implementation uses native FFT frequency ordering
`fftfreq(Z, vz)`, `fftfreq(Y, vy)`, and `fftfreq(X, vx)`, so no FFT shift is
needed. It evaluates

\[
d(k)=\frac13-\frac{(k\cdot\hat B_0)^2}{\|k\|^2},\qquad d(0)=0.
\]

`B0` is normalized internally after it has been supplied in explicit z-y-x
coordinates. The operator has periodic boundaries, uses orthonormal FFTs, and
returns real float32 arrays after complex64 transforms. It never thresholds
the kernel or substitutes an inversion method for the forward model.

The TDV convolutional network uses zero `SAME` padding. This edge convention
is intentionally documented separately from the periodic forward physics. A
future padded or cropped physical model must implement and test its padding
and cropping adjoints instead of hiding them inside the dipole operator.

Fixed CG makes the unrolled computation static and differentiable. It is the
first implementation route; an implicit-gradient solver may be added only
after equivalence against this path is tested. No full Hessian or Jacobian is
formed: higher-order terms arise only through JAX's reverse-mode derivatives
and JVP-based Hessian-vector products. `cg_iterations` is static: the JAX scan
always executes that many entries. `cg_relative_tolerance` makes later entries
numerical no-ops after the target is reached, preventing finite-precision CG
breakdown without changing the scan length. The final relative residual is
still checked after reconstruction; increase the iteration count if it misses
the documented target.

All model steps share one TDV parameter pytree. `raw_time` represents
\(T=T_{max}\sigma(raw\_time)\), and `tau=T/S` is derived inside reconstruction.
The analysis-kernel projection is applied after—not within—the jitted Adam
update, preserving functional parameter handling.

The COSMOS experiment reports masked NRMSE,
\(\lVert\chi_{recon}-\chi_{gt}\rVert_2/\lVert\chi_{gt}\rVert_2\). The mask
is applied to both norms only to exclude voxels outside the supplied support;
the metric does not redefine the dipole operator. Its generated figure uses a
common susceptibility scale for input, TDV output, and COSMOS ground truth,
and includes sagittal, coronal, and axial views plus absolute error.
