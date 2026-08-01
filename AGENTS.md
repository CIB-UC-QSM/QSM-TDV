# TDV for 3D Quantitative Susceptibility Mapping

## 1. Mission

Implement **Total Deep Variation (TDV)** as a learned scalar regularizer for 3D Quantitative Susceptibility Mapping (QSM).

TDV must not directly predict the susceptibility map. The neural network represents an energy

\[
R_\theta(\chi)\in\mathbb{R},
\]

and the reconstruction uses its gradient

\[
\nabla_\chi R_\theta(\chi)
\]

inside a physics-based variational solver.

The required computational chain is

\[
\chi
\rightarrow
R_\theta(\chi)
\rightarrow
\nabla_\chi R_\theta(\chi)
\rightarrow
\text{semi-implicit QSM reconstruction}
\rightarrow
\chi_S.
\]

Do not replace this structure with a direct CNN mapping \(b\mapsto\chi\).

---

## 2. QSM forward model

Use the unitary discrete Fourier model

\[
b = \mathcal{F}^{H}D\mathcal{F}\chi+\eta,
\]

where:

- \(\chi\): 3D magnetic susceptibility map;
- \(b\): measured local field;
- \(\eta\): measurement and modeling error;
- \(\mathcal{F}\): unitary 3D discrete Fourier transform;
- \(\mathcal{F}^{H}\): inverse/adjoint unitary Fourier transform;
- \(D\): diagonal multiplication by the dipole kernel in k-space.

Define

\[
A = \mathcal{F}^{H}D\mathcal{F}.
\]

For a real, symmetric dipole kernel,

\[
A^{H}=A.
\]

The data fidelity is

\[
D_{\text{data}}(\chi,b)
=
\frac12\|A\chi-b\|_2^2.
\]

Its gradient is

\[
\nabla_\chi D_{\text{data}}
=
A^{H}(A\chi-b).
\]

If masks or statistical weights are used, they must be introduced explicitly and their adjoints must be implemented exactly. Never silently absorb them into \(A\).

---

## 3. Dipole kernel

For spatial frequency \(\mathbf{k}\) and unit field direction \(\widehat{\mathbf{B}}_0\),

\[
d(\mathbf{k})
=
\frac13
-
\frac{(\mathbf{k}\cdot\widehat{\mathbf{B}}_0)^2}
{\|\mathbf{k}\|_2^2},
\qquad
d(\mathbf{0})=0.
\]

Requirements:

- support anisotropic voxel sizes;
- support arbitrary \(\widehat{\mathbf{B}}_0\);
- use one documented axis convention;
- use orthonormal FFT normalization;
- preserve the forward/adjoint identity numerically;
- do not threshold the dipole kernel inside the physical forward operator;
- document periodic, padded, or cropped boundary conditions.

For JAX, use tensors in NDHWC format:

\[
[B,Z,Y,X,C].
\]

Metadata order is always:

- voxel size: \([v_z,v_y,v_x]\);
- field direction: \([B_{0,z},B_{0,y},B_{0,x}]\).

Never infer the \(B_0\) direction from array shape.

Reference code:
```python
def continuous_dipole_kernel(
    shape: tuple[int, int, int],
    voxel_size: tuple[int, int, int] = (1, 1, 1),
    b0_dir: tuple[int, int, int] = (0, 0, 1),
) -> npt.NDArray[np.float64]:
    rx = np.arange(-np.floor(shape[0] / 2), np.ceil(shape[0] / 2))
    ry = np.arange(-np.floor(shape[1] / 2), np.ceil(shape[1] / 2))
    rz = np.arange(-np.floor(shape[2] / 2), np.ceil(shape[2] / 2))

    kx, ky, kz = np.meshgrid(rx, ry, rz, indexing="ij")
    kx /= np.max(np.abs(kx)) * voxel_size[0]
    ky /= np.max(np.abs(ky)) * voxel_size[1]
    kz /= np.max(np.abs(kz)) * voxel_size[2]

    k2 = kx**2 + ky**2 + kz**2
    kernel = ifftshift(
        1 / 3.0
        - ((kx * b0_dir[0] + ky * b0_dir[1] + kz * b0_dir[2]) ** 2)
        / (k2 + np.finfo(np.float64).eps)
    )
    kernel[0, 0, 0] = 0
    return kernel

```

---

## 4. TDV regularizer

The TDV energy is

\[
R_\theta(\chi)
=
\sum_{i=1}^{n} r_\theta(\chi)_i,
\]

with local energy density

\[
r_\theta(\chi)
=
w^{\top}N_\theta(K\chi).
\]

Interpret this voxelwise:

\[
r_{\theta,i}(\chi)
=
w^{\top}N_\theta(K\chi)_i.
\]

Components:

- \(K\): learned analysis convolution;
- \(N_\theta\): learned multiscale convolutional network;
- \(w\): learned channel-combination vector or \(1\times1\times1\) convolution;
- \(R_\theta(\chi)\): one scalar energy per sample.

The regularization force is

\[
g_\theta(\chi)
=
\nabla_\chi R_\theta(\chi).
\]

It must have the same shape as \(\chi\).

The full Hessian must never be materialized. Use automatic differentiation for Hessian-vector products and mixed derivatives.

---

## 5. TDV network structure

A \(\mathrm{TDV}^{L}\) model contains \(L\) consecutive macro-blocks:

\[
N_\theta
=
\operatorname{Ma}^{L}\circ\cdots\circ\operatorname{Ma}^{1}.
\]

Each macro-block is a three-scale U-Net-like module with skip connections and five residual micro-blocks.

Each micro-block has the form

\[
\operatorname{Mi}(u)
=
u+K_2\phi(K_1u),
\]

where \(K_1\) and \(K_2\) are bias-free 3D convolutions.

Use the smooth log-Student-t activation

\[
\phi(a)
=
\frac{1}{2\nu}\log(1+\nu a^2).
\]

Its derivatives are

\[
\phi'(a)=\frac{a}{1+\nu a^2},
\]

\[
\phi''(a)
=
\frac{1-\nu a^2}{(1+\nu a^2)^2}.
\]

Do not replace it with ReLU while claiming to preserve the original smooth TDV formulation.

---

## 6. Zero-mean analysis kernel

Each output filter of \(K\) must satisfy

\[
\sum_j K_{a,j}=0.
\]

For a 3D convolution, sum over all input channels and spatial kernel positions.

After every optimizer update, project

\[
K_a
\leftarrow
K_a-\operatorname{mean}(K_a).
\]

This projection is part of training and must not be omitted.

---

## 7. Variational QSM energy

The reconstruction energy is

\[
E(\chi;\theta,b)
=
\frac12\|A\chi-b\|_2^2
+
R_\theta(\chi).
\]

The continuous gradient flow is

\[
\dot{\widetilde{\chi}}(t)
=
-A^{H}(A\widetilde{\chi}(t)-b)
-
\nabla_\chi R_\theta(\widetilde{\chi}(t)).
\]

Learn a stopping time \(T\in[0,T_{\max}]\). Reparameterize time by

\[
\chi(t)=\widetilde{\chi}(tT),
\qquad t\in[0,1].
\]

Then

\[
\dot{\chi}(t)
=
T\left[
-A^{H}(A\chi(t)-b)
-
\nabla_\chi R_\theta(\chi(t))
\right].
\]

---

## 8. Semi-implicit reconstruction

Fix a number of steps \(S\) and define

\[
\tau=\frac{T}{S}.
\]

Use the semi-implicit update

\[
\chi_{s+1}
=
\chi_s
-\tau A^{H}(A\chi_{s+1}-b)
-\tau\nabla_\chi R_\theta(\chi_s).
\]

Equivalently,

\[
\boxed{
(I+\tau A^{H}A)\chi_{s+1}
=
\chi_s
+
\tau
\left(
A^{H}b
-
\nabla_\chi R_\theta(\chi_s)
\right).
}
\]

Each step must:

1. evaluate \(R_\theta(\chi_s)\);
2. compute \(g_s=\nabla_\chi R_\theta(\chi_s)\);
3. form
   \[
   q_s=\chi_s+\tau(A^{H}b-g_s);
   \]
4. solve
   \[
   (I+\tau A^{H}A)\chi_{s+1}=q_s.
   \]

Use the same \(\theta\) at every step.

The linear system is Hermitian positive definite for \(\tau>0\). Use fixed-iteration conjugate gradient or a validated implicit linear solver.

---

## 9. Training problem

Training samples are triplets

\[
(\chi_{\mathrm{init}}^i,\chi_{\mathrm{ref}}^i,b^i).
\]

The learned variables are:

\[
\theta
\quad\text{and}\quad
T.
\]

The discrete training objective is

\[
\min_{\theta,T}
J_S(T,\theta)
=
\frac1N
\sum_{i=1}^{N}
\ell(\chi_S^i-\chi_{\mathrm{ref}}^i),
\]

subject to the \(S\)-step semi-implicit dynamics.

Training procedure:

1. compute
   \[
   T=T_{\max}\sigma(\alpha),
   \qquad
   \tau=T/S;
   \]
2. initialize
   \[
   \chi_0=\chi_{\mathrm{init}};
   \]
3. unroll \(S\) TDV-QSM steps;
4. compute the terminal supervised loss;
5. optionally add an explicitly weighted data-consistency term;
6. differentiate through:
   - \(\nabla_\chi R_\theta\);
   - all TDV steps;
   - the linear solves;
   - the stopping-time parameter;
7. update parameters with Adam;
8. project \(K\) to zero mean.

TDV is trained with automatic differentiation and Adam. Do not implement MSA or an explicit Hamiltonian maximization unless a new research variant is requested.

---

## 10. Higher-order autodifferentiation

Because every reconstruction step uses

\[
\nabla_\chi R_\theta(\chi),
\]

training requires mixed derivatives

\[
\frac{\partial}{\partial\theta}
\nabla_\chi R_\theta(\chi)
\]

and Hessian-vector products

\[
\nabla_\chi^2R_\theta(\chi)v.
\]

Requirements:

- do not apply `stop_gradient` to the regularization force during training;
- do not replace the energy gradient with an independent network output;
- do not construct full Jacobians or Hessians;
- verify gradients against finite differences on small problems.

In JAX, the intended pattern is:

```python
def total_energy(theta, chi, mask):
    return regularizer.apply({"params": theta}, chi, mask).sum()

regularizer_force = jax.grad(total_energy, argnums=1)
```

The outer training loss must be differentiated with respect to all model parameters.

---

## 11. JAX implementation rules

- Use pure functions for operators and solvers.
- Use `jax.lax.scan` for TDV steps and fixed CG iterations.
- Keep iteration counts static.
- Use NDHWC tensors.
- Use `float32` for states and solvers.
- Use `complex64` for FFT operations.
- Use orthonormal FFT normalization.
- Use `jax.checkpoint`/`jax.remat` when needed for memory.
- Start by differentiating through fixed CG iterations.
- Use implicit differentiation only after validating equivalence.
- Avoid dynamic shapes inside jitted functions.
- Never mutate parameters inside `jit`.

---

## 12. QSM data contract

Each sample must define:

- `local_field`: \(b\), shape `[Z,Y,X,1]`;
- `susceptibility`: reference \(\chi\), same spatial shape;
- `brain_mask`;
- optional `reference_mask`;
- optional statistical weight map;
- `voxel_size_zyx`;
- `b0_direction_zyx`;
- explicit field and susceptibility units;
- explicit susceptibility-reference convention;
- anonymized subject identifier;
- source and processing version.

Rules:

- use subject-level train/validation/test splits;
- split subjects before patch extraction;
- never mix patches from one subject across partitions;
- preserve affine and orientation metadata;
- transform \(B_0\) consistently when reorienting data;
- avoid training the global dipole physics on isolated patches without a documented halo or global-field generation strategy;
- record deterministic seeds and configuration hashes for synthetic data;
- never include identifiable clinical metadata in manifests or logs.

---

## 13. Required numerical tests

### Dipole operator

Verify

\[
\frac{
|\langle A x,y\rangle-\langle x,A^{H}y\rangle|
}{
|\langle A x,y\rangle|
+
|\langle x,A^{H}y\rangle|
+\varepsilon
}
<
10^{-5}
\]

in `float32` on representative shapes.

Also test:

- zero frequency;
- arbitrary \(B_0\);
- anisotropic voxels;
- odd and even dimensions;
- padding/cropping adjoints, if used.

### TDV energy

Verify:

- one scalar energy per batch element;
- \(\nabla_\chi R_\theta\) has the same shape as \(\chi\);
- directional finite differences agree with the autodiff gradient;
- the zero-mean projection holds for every analysis filter.

### Linear solver

Verify:

- CG agrees with a dense solve on tiny problems;
- residuals decrease;
- no NaNs occur in the expected \(\tau\) range;
- gradients with respect to the right-hand side pass finite-difference checks.

### Semi-implicit step

Verify the residual

\[
(I+\tau A^{H}A)\chi_{s+1}
-
\left[
\chi_s+\tau(A^{H}b-\nabla_\chi R_\theta(\chi_s))
\right]
\]

is below the configured tolerance.

### Training

Verify:

- finite gradients reach all trainable parameter groups;
- the stopping time receives a finite gradient;
- a tiny synthetic dataset can be overfit;
- reconstruction improves over the initialization;
- inference is deterministic for fixed parameters and inputs.

---

## 14. Prohibited changes

Do not make the following changes without explicit approval:

1. Replace the scalar energy with a direct \(b\mapsto\chi\) CNN.
2. Use a vector field not derived from an energy as the TDV force.
3. Use different TDV parameters at different steps without documentation.
4. Apply `stop_gradient` to \(\nabla_\chi R_\theta\) during training.
5. Materialize full Hessians.
6. Replace the semi-implicit step with explicit Euler silently.
7. Omit the zero-mean projection of \(K\).
8. Confuse \(T\), \(S\), and \(\tau=T/S\).
9. Threshold the physical dipole kernel inside \(A\).
10. Treat an approximate inverse as the physical forward operator.
11. Change units, orientation, reference, masks, or FFT normalization silently.
12. Claim that TDV was trained with MSA.

---

## 15. Reference execution flow

```text
for each minibatch:
    read chi_init, chi_ref, b, metadata
    construct D from voxel size and B0 direction
    define A(chi) = F^H D F chi

    T = T_max * sigmoid(raw_time)
    tau = T / S
    chi = chi_init

    repeat S times:
        energy = R_theta(chi)
        force = grad_chi energy
        rhs = chi + tau * (A_adjoint(b) - force)
        chi = solve(I + tau * A_adjoint(A), rhs)

    loss = terminal_supervised_loss(chi, chi_ref)
    loss += optional_weight * data_consistency(chi, b)

    differentiate loss with respect to theta and raw_time
    apply Adam update
    project K to zero mean
```

---

## 16. Definition of done

A TDV-QSM implementation is complete only when:

- the forward model is exactly documented;
- \(A\) and \(A^{H}\) pass the adjoint test;
- the network outputs a scalar energy per sample;
- the energy gradient has the same shape as \(\chi\);
- the semi-implicit equation is solved to the required tolerance;
- gradients reach all model parameters and \(T\);
- the analysis kernel remains zero mean;
- a tiny synthetic problem can be overfit;
- units, axes, \(B_0\), voxel size, masks, and susceptibility reference are explicit;
- every deviation from this specification is documented.
