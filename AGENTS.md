# TDV-QSM

## 1. Mission and provenance

Implement a 3D PyTorch adaptation of the Total Deep Variation (TDV) regularizer for Quantitative Susceptibility Mapping (QSM).

The implementation must preserve the central design of `VLOGroup/tdv`:

- TDV is an energy model, not a direct image-to-image predictor.
- The reconstruction uses an explicit `regularizer.force(x)` operation.
- The force is the exact gradient of the summed TDV energy.
- The network uses bias-free analysis, multiscale macro-blocks, smooth Student-t micro-blocks, and a one-channel output head.
- Linear operators have explicit forward and adjoint operations.
- Downsampling is antialiased.
- The first analysis kernel is zero-mean and norm-bounded.
- The explicit variational-network branch uses separate coefficients for the regularizer and data term.

The source repository is a 2D denoising/eigenfunction implementation. The following are project-specific extensions and must be labeled as such:

- 3D convolutions;
- QSM physics;
- magnitude-derived weighting matrix \(W\);
- NRMSE training;
- CUDA AMP with float16;
- medical-volume data handling.

Do not claim that these extensions are present in the source repository.

---

## 2. Required model decomposition

Use separate modules:

```text
TDVEnergy3D
├── analysis: AdjointConv3d
├── macro_blocks: ModuleList[MacroBlock3D]
├── energy_head: AdjointConv3d(kernel_size=1)
├── energy_density(chi)
├── energy(chi)
├── force(chi)
└── force_autograd_reference(chi)  # tests only

QSMOperator
├── forward(chi)
├── adjoint(field)
└── build_dipole_kernel(metadata)

ExplicitTDVQSM3D
├── regularizer: TDVEnergy3D
├── operator: QSMOperator
├── raw_T or fixed_T
├── raw_lambda or fixed_lambda
└── forward(...)
```

The regularizer and physical operator must remain independently testable.

---

## 3. QSM forward model

Use

\[
b=A\chi+\eta,
\qquad
A=\mathcal F^H D\mathcal F,
\]

where:

- \(\chi\): 3D susceptibility;
- \(b\): local field;
- \(\eta\): measurement and model error;
- \(\mathcal F\): orthonormal 3D FFT;
- \(D\): diagonal dipole kernel in k-space.

For spatial frequency \(\mathbf k\) and normalized field direction \(\widehat{\mathbf B}_0\),

\[
d(\mathbf k)
=
\frac13
-
\frac{(\mathbf k\cdot\widehat{\mathbf B}_0)^2}
{\|\mathbf k\|_2^2},
\qquad
d(\mathbf0)=0.
\]

Tensor convention:

```text
image/field:       [B, 1, Z, Y, X]
voxel_size_zyx:    [B, 3]
b0_direction_zyx: [B, 3]
```

Requirements:

- support anisotropic voxel sizes;
- normalize every \(B_0\) direction;
- use `norm="ortho"` in FFT and IFFT;
- implement `forward` and `adjoint` separately;
- use `D.conj()` in the adjoint;
- keep FFT inputs in float32 and Fourier tensors in complex64;
- do not threshold \(D\) inside the forward model;
- explicitly document periodic, padded, or cropped boundary conditions.

```python
FFT_DIMS = (-3, -2, -1)

def fft3(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftn(
        x.float(),
        dim=FFT_DIMS,
        norm="ortho",
    )

def ifft3(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifftn(
        x,
        dim=FFT_DIMS,
        norm="ortho",
    ).real

def forward(chi, dipole_kernel):
    return ifft3(fft3(chi) * dipole_kernel)

def adjoint(field, dipole_kernel):
    return ifft3(
        fft3(field) * dipole_kernel.conj()
    )
```

---

## 4. Magnitude-weighted data term

Let \(W\) be a real, nonnegative diagonal matrix derived from signal magnitude. Store its diagonal as

```text
magnitude_weight: [B, 1, Z, Y, X] float32
```

Define

\[
D_W(\chi;b)
=
\frac12\|W(A\chi-b)\|_2^2.
\]

Its exact gradient is

\[
\nabla_\chi D_W(\chi;b)
=
A^H W^H W(A\chi-b).
\]

For real diagonal \(W\),

\[
\nabla_\chi D_W(\chi;b)
=
A^H W^2(A\chi-b).
\]

PyTorch equivalent:

```python
predicted_field = operator.forward(chi)
field_residual = predicted_field - local_field.float()

weighted_residual = (
    magnitude_weight.float() * field_residual
)

normal_weighted_residual = (
    magnitude_weight.float() * weighted_residual
)

data_force = operator.adjoint(
    normal_weighted_residual
)
```

Never replace this with

```python
operator.adjoint(magnitude_weight * field_residual)
```

because that corresponds to only one application of \(W\), not the gradient of \(\frac12\|W(A\chi-b)\|^2\).

The preprocessing rule that constructs \(W\) from magnitude must specify:

- normalization;
- clipping;
- masking;
- handling of zero magnitude;
- units or dimensionless scaling;
- whether \(W\) or \(\sqrt W\) is stored.

Do not silently modify this rule.

---

## 5. Source-faithful TDV energy

Let \(m\) be `num_features`. Define the TDV transformation

\[
\mathcal T_\theta(\chi)
=
K_N
\circ
M_L
\circ\cdots\circ
M_1
\circ
K_1(\chi),
\]

where:

- \(K_1\): learned \(3\times3\times3\) analysis convolution;
- \(M_\ell\): multiscale macro-block;
- \(K_N\): learned \(1\times1\times1\) one-channel head.

The voxelwise energy density is

\[
\rho_\theta(\chi)
=
\frac{1}{m}\mathcal T_\theta(\chi).
\]

The scalar energy per sample is

\[
R_\theta(\chi)
=
\sum_{z,y,x}
\rho_\theta(\chi)_{z,y,x}.
\]

Canonical shapes:

```text
chi:             [B, 1, Z, Y, X]
analysis:        [B, m, Z, Y, X]
energy_density:  [B, 1, Z, Y, X]
energy:          [B] float32
force:           [B, 1, Z, Y, X] float32
```

Required API:

```python
def energy_density(self, chi: Tensor) -> Tensor:
    ...

def energy(self, chi: Tensor) -> Tensor:
    density = self.energy_density(chi)
    return density.float().flatten(1).sum(dim=1)

def force(self, chi: Tensor) -> Tensor:
    # Source-style differentiable manual gradient.
    ...

def force_autograd_reference(self, chi: Tensor) -> Tensor:
    # Slow correctness oracle used only by tests.
    energy = self.energy(chi)
    force, = torch.autograd.grad(
        energy.sum(),
        chi,
        create_graph=True,
    )
    return force
```

The production `force` and autograd reference must agree within configured tolerance.

---

## 6. Student-t micro-block

The source code uses \(\alpha=1\):

\[
\phi(a)
=
\frac12\log(1+a^2),
\qquad
\phi'(a)
=
\frac{a}{1+a^2}.
\]

A micro-block is

\[
v
=
u+K_2\phi(K_1u),
\]

with bias-free convolutions.

Its Jacobian-transpose action is

\[
J_{\mathrm{Mi}}(u)^T q
=
q
+
K_1^T
\left[
\phi'(K_1u)
\odot
K_2^Tq
\right].
\]

Use native PyTorch operations so training can differentiate through the force:

```python
def student_t_pair(x: torch.Tensor):
    x32 = x.float()
    denominator = 1.0 + x32.square()

    value = 0.5 * torch.log(denominator)
    derivative = x32 / denominator

    return value, derivative
```

The convolutions may run under float16 autocast. Compute the Student-t value and derivative in float32, then cast only where required by the next convolution.

Do not use ReLU.

Do not copy the source pattern of storing `act_prime` as mutable module state. Return local caches or recompute them. Persistent activation caches are unsafe with checkpointing, concurrency, and repeated calls.

---

## 7. Macro-block topology

For `num_scales = 3`, each macro-block contains exactly five micro-blocks:

- scale 0: one before downsampling and one after upsampling;
- scale 1: one before downsampling and one after upsampling;
- scale 2: one at the coarsest scale.

Represent the multiscale state as

```python
[x_scale_0, x_scale_1, x_scale_2]
```

For the first macro-block:

```python
[x, None, None]
```

Later macro-blocks receive and update all existing scale tensors. Do not discard coarse-scale features between macro-blocks.

Forward order:

1. apply the first micro-block at each noncoarsest scale;
2. downsample and add into the next scale;
3. apply the coarsest micro-block;
4. traverse scales in reverse;
5. apply the learned up-adjoint operator;
6. add the skip connection;
7. apply the second micro-block.

The manual force must traverse the exact reverse order with Jacobian-transpose operations.

---

## 8. Linear convolution operators and adjoints

The source implementation does not treat convolution as an opaque layer. Every linear operator has:

```python
forward(x)
adjoint(y, output_shape=None)
```

For a convolution with boundary extension \(P\) and kernel operation \(C\),

\[
K=CP,
\qquad
K^T=P^TC^T.
\]

The 3D implementation must preserve this identity numerically.

### Boundary handling

The source uses symmetric padding and its exact transpose. PyTorch reflection padding is not automatically equivalent to symmetric padding.

Choose one:

1. implement `SymmetricPad3d` and its exact adjoint; or
2. use a different documented boundary rule and validate its exact adjoint; or
3. rely on autograd for the reference gradient and label the boundary rule as a deliberate deviation.

Do not write a guessed crop as the adjoint of padding.

### Downsampling

The source antialiases scaled convolutions using the normalized binomial filter

\[
h=\frac1{16}[1,4,6,4,1].
\]

For 3D, use the separable kernel

\[
H=h\otimes h\otimes h.
\]

The learned scaled-convolution kernel must be antialiased before stride-2 sampling. Its adjoint must use transposed convolution and recover the requested output shape exactly, including odd dimensions.

The learned down operator and learned up operator are independent. The up path is the adjoint of its own learned scaled-convolution operator; it is not required to share weights with the down operator.

Arbitrary trilinear interpolation is not source-faithful.

---

## 9. Analysis-kernel constraints

The first analysis convolution \(K_1\) must be:

- bias-free;
- zero-mean per output filter;
- bounded by unit \(\ell_2\) norm per output filter.

For a weight tensor `[out, in, kz, ky, kx]`:

\[
\sum_{c,z,y,x}K_{o,c,z,y,x}=0,
\]

\[
\|K_o\|_2\le1.
\]

Initialization:

1. sample with standard deviation
   \[
   \sqrt{\frac{1}{C_{\mathrm{in}}k_zk_yk_x}};
   \]
2. subtract the per-filter mean;
3. normalize every nonzero output filter to unit norm.

After every optimizer update:

```python
@torch.no_grad()
def project_analysis_kernel_(weight: torch.Tensor) -> None:
    reduce_dims = (1, 2, 3, 4)

    weight.sub_(
        weight.mean(
            dim=reduce_dims,
            keepdim=True,
        )
    )

    norm = torch.linalg.vector_norm(
        weight,
        ord=2,
        dim=reduce_dims,
        keepdim=True,
    )

    weight.div_(
        norm.clamp_min(1.0)
    )
```

Do not use `.data`.

The source norm bound applies to \(K_1\), not automatically to every convolution.

---

## 10. Explicit variational-network update

The source repository's nonproximal branch uses separate regularizer and data coefficients.

Use:

\[
T\ge0,
\qquad
\lambda\ge0,
\qquad
S\in\mathbb N.
\]

At step \(s\):

\[
g_R^s
=
\nabla_\chi R_\theta(\chi_s),
\]

\[
g_D^s
=
A^H W^H W(A\chi_s-b).
\]

The explicit update is

\[
\boxed{
\chi_{s+1}
=
\chi_s
-
\frac{T}{S}g_R^s
-
\frac{\lambda}{S}g_D^s.
}
\]

This is the QSM equivalent of the source code's explicit branch.

Do not use conjugate gradient, a proximal data step, or a semi-implicit solve in this variant.

Support fixed or learned modes for both \(T\) and \(\lambda\). For learned bounded parameters, prefer:

\[
T=T_{\max}\sigma(\alpha_T),
\]

\[
\lambda=\lambda_{\max}\sigma(\alpha_\lambda).
\]

Compute both once per reconstruction, not once per iteration.

PyTorch skeleton:

```python
T = self.maximum_time * torch.sigmoid(self.raw_T)
lam = self.maximum_lambda * torch.sigmoid(self.raw_lambda)

regularizer_step = T.float() / self.num_steps
data_step = lam.float() / self.num_steps

chi = initial.float()

for _ in range(self.num_steps):
    force_R = self.regularizer.force(chi).float()

    field_residual = (
        self.operator.forward(chi)
        - local_field.float()
    )

    weighted = (
        magnitude_weight.float()
        * field_residual
    )

    force_D = self.operator.adjoint(
        magnitude_weight.float() * weighted
    ).float()

    chi = (
        chi
        - regularizer_step * force_R
        - data_step * force_D
    )

    chi = chi * brain_mask.float()
```

Masking the state after each step is a project choice, not source behavior. Make it configurable and test its effect.

---

## 11. Source-style force versus autograd reference

The source repository computes `R.grad(x)` by manually propagating through transposed linear operators and stored activation derivatives. Preserve that idea for the production path.

However, the 3D implementation must also provide an autograd correctness oracle.

Mandatory invariant:

\[
\frac{
\|g_{\mathrm{manual}}-g_{\mathrm{autograd}}\|_2
}{
\max(\|g_{\mathrm{autograd}}\|_2,\varepsilon)
}
<
\text{configured tolerance}.
\]

Run this test in float64 on small tensors.

During training, `force()` must remain differentiable with respect to every regularizer parameter. Do not detach local derivatives, convolution weights, or intermediate forces.

---

## 12. float16 AMP contract

The source checkpoints are float32. float16 is a project-specific performance extension.

Use mixed precision, not global half precision.

### Allowed under float16 autocast

- 3D learned convolutions;
- transposed convolutions;
- feature additions;
- multiscale feature tensors.

### Must remain float32 or complex64

- \(\chi_s\);
- \(b\);
- \(W\);
- masks;
- dipole kernel;
- FFT and IFFT;
- Student-t value and derivative;
- energy summation;
- TDV force output;
- physical force;
- NRMSE;
- optimizer master parameters and Adam state.

Never call `model.half()`.

Use:

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.float16,
    enabled=use_amp,
):
    feature_output = regularizer_feature_path(...)
```

Use `torch.amp.GradScaler` for CUDA training.

Before clipping gradients:

```python
scaler.unscale_(optimizer)
```

Then validate finite gradients, clip, step, update the scaler, and project \(K_1\).

---

## 13. Activation checkpointing and memory

The source has an `efficient` mode that checkpoints `R.grad`.

For 3D training, support:

```python
force_R = torch.utils.checkpoint.checkpoint(
    self.regularizer.force,
    chi,
    use_reentrant=False,
)
```

Requirements:

- explicitly set `use_reentrant=False`;
- checkpoint only pure deterministic functions;
- do not rely on mutable activation caches;
- compare checkpointed and noncheckpointed outputs and gradients;
- do not retain all states by default.

The source returns every state `x_all`. For 3D QSM, return only \(\chi_S\) by default. Store intermediate states or forces only when diagnostics request them.

---

## 14. NRMSE training and evaluation

Use

\[
\operatorname{NRMSE}
=
\frac{
\|x_{\mathrm{pred}}-x_{\mathrm{true}}\|_2
}{
\|x_{\mathrm{true}}\|_2
}.
\]

Compute it independently per sample, then average across the batch.

```python
def nrmse(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred = x_pred.float().flatten(1)
    true = x_true.float().flatten(1)

    numerator = torch.linalg.vector_norm(
        pred - true,
        ord=2,
        dim=1,
    )
    denominator = torch.linalg.vector_norm(
        true,
        ord=2,
        dim=1,
    ).clamp_min(eps)

    return (numerator / denominator).mean()
```

Apply brain masking and susceptibility referencing before passing tensors to `nrmse`.

Use the same function for:

- training loss;
- validation metric;
- test metric;
- checkpoint selection.

An optional terminal data-consistency loss is

\[
\mathcal L_{\mathrm{dc}}
=
\frac1B
\sum_i
\frac{
\|W^i(A\chi_S^i-b^i)\|_2^2
}{
\max(N_i,1)
}.
\]

The total training loss may be

\[
\mathcal L
=
\mathcal L_{\mathrm{NRMSE}}
+
\beta_{\mathrm{dc}}\mathcal L_{\mathrm{dc}}.
\]

Do not confuse:

- \(W\): spatial reliability from magnitude;
- \(\lambda\): data-force strength inside the unrolled dynamics;
- \(\beta_{\mathrm{dc}}\): optional terminal-loss weight.

---

## 15. Dataset contract

Each sample must contain:

```text
local_field          float32 [1,Z,Y,X]
susceptibility       float32 [1,Z,Y,X]
brain_mask           float32 [1,Z,Y,X]
magnitude_weight     float32 [1,Z,Y,X]
initial              float32 [1,Z,Y,X]
voxel_size_zyx       float32 [3]
b0_direction_zyx    float32 [3]
subject_id           anonymized string
field_unit           explicit string
susceptibility_unit  explicit string
reference_convention explicit string
processing_version   immutable string
```

Validation:

- finite arrays;
- nonnegative \(W\);
- nonempty masks;
- positive voxel sizes;
- normalized \(B_0\);
- shape and affine consistency;
- subject-level split integrity;
- no duplicated subjects across splits;
- documented construction of \(W\);
- no identifiable metadata.

Split subjects before patch extraction. Because the dipole operator is global, do not train physical QSM updates on isolated patches unless local fields were generated globally and the crop/halo policy is documented.

---

## 16. Mandatory tests

### Source-fidelity tests

1. `energy_density` has shape `[B,1,Z,Y,X]`.
2. `energy` equals the float32 spatial sum of `energy_density`.
3. `force` matches `force_autograd_reference` in float64.
4. Reproduce the source directional scale test:
   \[
   \langle x,\nabla R(sx)\rangle
   \approx
   \frac{R((s+\epsilon)x)-R((s-\epsilon)x)}
   {2\epsilon}.
   \]
5. Every micro-block manual Jacobian-transpose matches autograd.
6. Every linear operator passes an adjoint inner-product test.
7. Every down/up-adjoint operator handles odd and even shapes.
8. \(K_1\) is zero-mean and has per-filter norm at most one.

### QSM tests

1. \(d(\mathbf0)=0\).
2. `A` and `A^H` pass:
   \[
   \frac{
   |\langle Ax,y\rangle-\langle x,A^Hy\rangle|
   }{
   |\langle Ax,y\rangle|+
   |\langle x,A^Hy\rangle|+\varepsilon
   }
   <10^{-5}.
   \]
3. Support anisotropic voxels and arbitrary \(B_0\).
4. Weighted data gradient matches finite differences:
   \[
   A^H W^H W(A\chi-b).
   \]
5. An all-ones \(W\) reproduces the unweighted case.
6. Zero entries of \(W\) remove those residual contributions.
7. Negative or nonfinite weights are rejected.

### Explicit dynamics tests

1. One step matches the boxed formula exactly.
2. Parameters are shared across all \(S\) steps.
3. \(T=0\) disables the regularizer force.
4. \(\lambda=0\) disables the data force.
5. With sufficiently small steps, a synthetic energy does not diverge.
6. Checkpointed and noncheckpointed runs match.

### Loss and AMP tests

1. NRMSE matches a hand-computed result.
2. NRMSE is zero for identical tensors.
3. NRMSE is computed per sample before batch averaging.
4. AMP agrees with a float32 reference within tolerance.
5. FFT inputs remain float32.
6. Force outputs remain float32.
7. No NaN or Inf occurs during a 100-step smoke training.
8. Finite gradients reach \(K_1\), all macro-blocks, \(K_N\), \(T\), and \(\lambda\).
9. Tiny-overfit succeeds on 1–4 synthetic phantoms.

---

## 17. Prohibited implementations

Do not:

1. replace TDV with a direct \(b\mapsto\chi\) U-Net;
2. use a force not derived from the TDV energy;
3. omit the manual-force versus autograd-reference test;
4. use ReLU;
5. discard coarse-scale features between macro-blocks;
6. use arbitrary interpolation as the source-faithful up path;
7. guess the adjoint of padding or convolution;
8. omit antialiasing in stride-2 scale changes;
9. omit the \(K_1\) norm bound;
10. apply only one \(W\) when computing the gradient of \(\frac12\|W(A\chi-b)\|^2\);
11. use CG, a proximal update, or a semi-implicit solve in the explicit variant;
12. detach states or forces during training;
13. call `model.half()`;
14. execute FFT in float16;
15. use mutable module-level activation caches;
16. use `.data`;
17. retain all 3D states by default;
18. change magnitude weighting, units, masks, axes, or reference conventions silently;
19. claim that the source repository implements QSM, AMP, NRMSE training, or this 3D extension.

---

## 18. Development workflow

Before editing:

1. read this file;
2. inspect existing operator and regularizer interfaces;
3. identify whether the task changes source-faithful behavior or a QSM extension;
4. add a failing test;
5. implement the smallest coherent change;
6. run focused tests;
7. run the complete suite;
8. report files changed, commands run, numerical tolerances, and unresolved assumptions.

Do not add dependencies or alter mathematical conventions without approval.

---

## 19. Definition of done

The implementation is complete only when:

- the 3D TDV topology matches the source code's macro/micro structure;
- energy density, scalar energy, and force are distinct APIs;
- the production force matches the autograd reference;
- all learned linear operators have validated adjoints;
- \(K_1\) is zero-mean and norm-bounded;
- QSM `A` and `A^H` pass the adjoint test;
- the data force is exactly \(A^H W^H W(A\chi-b)\);
- the explicit update uses separate \(T/S\) and \(\lambda/S\) coefficients;
- NRMSE is used consistently for training and evaluation;
- float16 is limited to AMP-safe learned operations;
- FFT, state updates, force outputs, reductions, and loss remain float32;
- checkpointed and noncheckpointed gradients agree;
- tiny-overfit passes;
- no NaN or Inf occurs;
- every deviation from the source implementation is documented.
