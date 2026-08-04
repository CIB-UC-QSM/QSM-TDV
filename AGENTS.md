# TDV-QSM

## 1. Mission and fidelity boundary

Implement **Total Deep Variation (TDV)** for 3D QSM reconstruction in PyTorch.

The network represents a learned scalar energy

\[
R_\theta(\chi)\in\mathbb{R},
\]

and the reconstruction uses its gradient

\[
\nabla_\chi R_\theta(\chi).
\]

The network must not directly predict susceptibility through a mapping \(b\mapsto\chi\).

The original TDV paper starts from the same continuous gradient flow, but uses a semi-implicit discretization. This project deliberately uses **explicit Euler / plain gradient descent**:

\[
\chi_{s+1}
=
\chi_s-\tau\nabla_\chi E_\theta(\chi_s;b).
\]

Do not describe this implementation as an exact reproduction of the original semi-implicit TDV discretization.

---

## 2. QSM physical model

Use

\[
b=\mathcal F^{H}D\mathcal F\chi+\eta,
\]

where:

- \(\chi\): 3D susceptibility map;
- \(b\): local field;
- \(\eta\): noise and modeling error;
- \(\mathcal F\): unitary 3D FFT;
- \(\mathcal F^{H}\): inverse/adjoint FFT;
- \(D\): diagonal dipole-kernel operator in k-space.

Define

\[
A=\mathcal F^{H}D\mathcal F.
\]

When convolution notation is used below,

\[
X * \chi
\equiv
\mathcal F^{H}D\mathcal F\chi
=
A\chi.
\]

Here, \(X\) denotes the spatial dipole-convolution kernel corresponding to \(D\). In PyTorch, compute \(X*\chi\) through the FFT-based `operator.forward(chi)` implementation rather than by materializing a dense convolution matrix.

Let \(W\) be the magnitude-derived weighting matrix used in the field domain. For QSM, \(W\) is expected to be a real, nonnegative diagonal operator constructed from the measured signal magnitude. In tensor form, store its diagonal as a spatial weight map with shape `[B, 1, Z, Y, X]`.

The weighted data-fidelity residual is

\[
r_{\mathrm{data}}
=
W(A\chi-b).
\]

The data-fidelity term is

\[
D_{\mathrm{data}}(\chi,b;W)
=
\frac12\|W(A\chi-b)\|_2^2.
\]

Its gradient is

\[
\nabla_\chi D_{\mathrm{data}}
=
A^{H}W^{H}W(A\chi-b).
\]

For the expected real diagonal weighting matrix,

\[
W^{H}=W,
\]

so

\[
\nabla_\chi D_{\mathrm{data}}
=
A^{H}W^2(A\chi-b).
\]

Do not replace this gradient with \(A^H W(A\chi-b)\): when the loss is the squared norm of \(W(A\chi-b)\), the chain rule produces \(W^H W\).

For a real dipole kernel, \(A^H=A\), but `forward` and `adjoint` must still be implemented as separate interfaces.

The exact construction of \(W\) from signal magnitude—normalization, clipping, masking, and scaling—must be defined in preprocessing configuration and recorded in dataset metadata. Do not silently invent or change that transformation.

### Dipole kernel

\[
d(\mathbf k)
=
\frac13-
\frac{(\mathbf k\cdot\widehat{\mathbf B}_0)^2}
{\|\mathbf k\|_2^2},
\qquad
d(\mathbf0)=0.
\]

Requirements:

- image tensors use `[B, C, Z, Y, X]`;
- metadata order is always `zyx`;
- `voxel_size_zyx = [vz, vy, vx]`;
- `b0_direction_zyx = [bz, by, bx]`;
- support anisotropic voxels;
- normalize \(\widehat{\mathbf B}_0\);
- use `norm="ortho"` for FFT and IFFT;
- do not apply TKD or dipole thresholding inside the physical operator;
- explicitly document padding, cropping, masks, and boundary conditions.

### PyTorch equivalent

```python
FFT_DIMS = (-3, -2, -1)

def fft3(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftn(x.float(), dim=FFT_DIMS, norm="ortho")

def ifft3(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifftn(x, dim=FFT_DIMS, norm="ortho").real

def dipole_forward(
    chi: torch.Tensor,
    dipole_kernel: torch.Tensor,
) -> torch.Tensor:
    return ifft3(fft3(chi) * dipole_kernel)

def dipole_adjoint(
    value: torch.Tensor,
    dipole_kernel: torch.Tensor,
) -> torch.Tensor:
    return ifft3(fft3(value) * dipole_kernel.conj())
```

FFT operations must always run in `float32/complex64`, never in `float16`.



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

## 3. TDV energy

Define

\[
R_\theta(\chi)
=
\sum_{i=1}^{n}r_{\theta,i}(\chi),
\qquad
r_\theta(\chi)
=
w^\top N_\theta(K\chi).
\]

Components:

- \(K\): learned 3D analysis convolution;
- \(N_\theta\): learned multiscale CNN;
- \(w\): channel-combination layer, implemented as `Conv3d(..., out_channels=1, kernel_size=1)`;
- \(r_\theta\): voxelwise energy density;
- \(R_\theta\): one scalar energy per sample.

Canonical shapes:

```text
chi:             [B, 1, Z, Y, X]
K(chi):          [B, m, Z, Y, X]
N(K(chi)):       [B, q, Z, Y, X]
energy_density:  [B, 1, Z, Y, X]
energy:          [B] float32
grad_R:          [B, 1, Z, Y, X] float32
```

The regularization force must be exactly

\[
g_\theta(\chi)
=
\nabla_\chi R_\theta(\chi).
\]

Do not use an independent vector-valued network output as the force.

### Micro-block

Each micro-block must be

\[
\operatorname{Mi}(u)
=
u+K_2\phi(K_1u),
\]

with bias-free 3D convolutions and

\[
\phi(a)
=
\frac{1}{2\nu}\log(1+\nu a^2).
\]

```python
def log_student_t(
    x: torch.Tensor,
    nu: float = 9.0,
) -> torch.Tensor:
    return torch.log1p(nu * x.square()) / (2.0 * nu)

def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x + self.conv2(
        log_student_t(self.conv1(x), self.nu)
    )
```

### Architecture

A `TDV^L` configuration contains \(L\) macro-blocks. Each macro-block uses:

- three spatial scales;
- five residual micro-blocks;
- skip connections;
- antialiased downsampling;
- upsampling;
- \(1\times1\times1\) fusion convolutions.

Implement a small `TDV^1` model first and validate all gradients before scaling the architecture.

---

## 4. Zero-mean analysis kernel

Each output filter of the first analysis kernel \(K\) must satisfy

\[
\sum_{c,z,y,x}K_{o,c,z,y,x}=0.
\]

After every optimizer step:

```python
@torch.no_grad()
def project_zero_mean_(conv: nn.Conv3d) -> None:
    weight = conv.weight
    weight.sub_(
        weight.mean(
            dim=(1, 2, 3, 4),
            keepdim=True,
        )
    )
```

Do not use `.data`.

---

## 5. Explicit gradient-descent reconstruction

The total energy is

\[
E_\theta(\chi;b,W)
=
\frac12\|W(A\chi-b)\|_2^2
+
R_\theta(\chi).
\]

Its gradient is

\[
\nabla_\chi E_\theta
=
A^{H}W^{H}W(A\chi-b)
+
\nabla_\chi R_\theta(\chi).
\]

For real diagonal \(W\),

\[
\nabla_\chi E_\theta
=
A^{H}W^2(A\chi-b)
+
\nabla_\chi R_\theta(\chi).
\]

Use \(S\) explicit Euler steps:

\[
\boxed{
\chi_{s+1}
=
\chi_s-\tau
\left[
A^{H}W^{H}W(A\chi_s-b)
+
\nabla_\chi R_\theta(\chi_s)
\right].
}
\]

Parameterize the learned stopping time as

\[
T=T_{\max}\sigma(\alpha),
\qquad
\tau=\frac{T}{S}.
\]

PyTorch equivalent:

```python
T = T_max * torch.sigmoid(raw_time)
tau = T / num_steps

predicted_field = operator.forward(chi)
field_residual = predicted_field - local_field.float()

# W is a real diagonal matrix represented by its voxelwise diagonal.
weighted_residual = magnitude_weight.float() * field_residual
normal_weighted_residual = (
    magnitude_weight.float() * weighted_residual
)

data_grad = operator.adjoint(
    normal_weighted_residual
)

energy = regularizer.energy(chi, mask)

grad_R, = torch.autograd.grad(
    energy.sum(),
    chi,
    create_graph=model.training,
    retain_graph=model.training,
)

chi = chi - tau.float() * (data_grad + grad_R)
chi = chi * mask
```

The same parameter set \(\theta\) must be shared across all steps.

Do not use conjugate gradient, matrix inverses, proximal updates, or semi-implicit steps in this variant.

---

## 6. float16 precision contract

“Use float16” means **stable mixed precision with AMP**, not converting the complete system to half precision.

### Run under float16 autocast

- TDV convolutions;
- TDV activations;
- multiscale blocks and fusion layers.

### Keep in float32/complex64

- state \(\chi_s\);
- local field \(b\);
- masks and weights;
- dipole kernel;
- FFT and IFFT;
- physical gradient;
- energy reduction;
- \(\nabla_\chi R_\theta\);
- explicit state update;
- loss;
- master parameters;
- Adam optimizer state.

Never call `model.half()`.

### Mandatory pattern

```python
def energy(
    self,
    chi: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=chi.is_cuda,
    ):
        density = self.energy_density(chi)

    return (
        density.float() * mask.float()
    ).flatten(1).sum(dim=1)
```

Compute the image gradient outside autocast:

```python
energy = regularizer.energy(chi, mask)

grad_R, = torch.autograd.grad(
    energy.sum(),
    chi,
    create_graph=True,
)
```

Training must use `GradScaler`:

```python
scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available(),
)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)

scaler.step(optimizer)
scaler.update()

model.regularizer.project_zero_mean_()
```

---

## 7. Required reconstruction interface

```python
class ExplicitTDVQSM3D(nn.Module):
    regularizer: TDVEnergy3D
    raw_time: nn.Parameter
    num_steps: int
    maximum_time: float

    def forward(
        self,
        local_field: torch.Tensor,
        mask: torch.Tensor,
        dipole_kernel: torch.Tensor,
        magnitude_weight: torch.Tensor,
        initial: torch.Tensor,
        *,
        return_states: bool = False,
    ) -> TDVOutput:
        ...
```

Forward-pass rules:

1. validate `[B,1,Z,Y,X]` shapes;
2. convert all physical inputs, including `magnitude_weight`, to `float32`;
3. validate that `magnitude_weight` is finite, nonnegative, and broadcast-compatible with `local_field`;
4. initialize `chi = initial.float().requires_grad_(True)`;
5. compute \(T\) and \(\tau\) once;
6. execute exactly \(S\) steps;
7. do not detach states between steps during training;
8. during inference, locally enable gradients to evaluate \(\nabla_\chi R_\theta\);
9. return reconstruction, predicted field, and optional intermediate states.

Do not wrap the complete inference pass in `torch.no_grad()` because the gradient with respect to \(\chi\) is still required.

---

## 8. Training contract

Each sample must provide:

```text
local_field       float32 [B,1,Z,Y,X]
susceptibility    float32 [B,1,Z,Y,X]
brain_mask        float32 [B,1,Z,Y,X]
magnitude_weight  float32 [B,1,Z,Y,X]
reference_mask    optional
voxel_size_zyx    float32 [B,3]
b0_direction_zyx float32 [B,3]
initial           float32 [B,1,Z,Y,X]
```

### Primary training loss: NRMSE

Use NRMSE as the primary supervised loss and as the reported evaluation metric:

\[
\operatorname{NRMSE}(x_{\mathrm{pred}},x_{\mathrm{true}})
=
\frac{
\left\|x_{\mathrm{pred}}-x_{\mathrm{true}}\right\|_2
}{
\left\|x_{\mathrm{true}}\right\|_2
}.
\]

Compute it independently for every batch element and then average across the batch. Add a small numerical \(\varepsilon\) only to protect the denominator:

\[
\mathcal L_{\mathrm{NRMSE}}
=
\frac1B
\sum_{i=1}^{B}
\frac{
\left\|x_{\mathrm{pred}}^i-x_{\mathrm{true}}^i\right\|_2
}{
\max\!\left(
\left\|x_{\mathrm{true}}^i\right\|_2,
\varepsilon
\right)
}.
\]

If a brain mask or susceptibility-reference operation is required, apply it first and define the resulting tensors as \(x_{\mathrm{pred}}\) and \(x_{\mathrm{true}}\). The NRMSE formula itself must not be changed.

```python
def nrmse(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred = x_pred.float().flatten(start_dim=1)
    true = x_true.float().flatten(start_dim=1)

    error_norm = torch.linalg.vector_norm(
        pred - true,
        ord=2,
        dim=1,
    )
    true_norm = torch.linalg.vector_norm(
        true,
        ord=2,
        dim=1,
    ).clamp_min(eps)

    return (error_norm / true_norm).mean()
```

Use this same function for:

- the supervised training loss;
- validation NRMSE;
- test-set NRMSE;
- checkpoint selection unless another criterion is explicitly approved.

### Magnitude-weighted data-consistency term

Define \(W\) as a real, nonnegative diagonal matrix derived from the signal magnitude. In PyTorch, represent the diagonal of \(W\) as `magnitude_weight` with shape `[B, 1, Z, Y, X]`.

The data-consistency residual is exactly

\[
r_{\mathrm{dc}}
=
W(A\chi_S-b).
\]

In tensor form:

\[
r_{\mathrm{dc}}
=
W\odot(A\chi_S-b).
\]

The optional data-consistency loss is

\[
\mathcal L_{\mathrm{dc}}
=
\frac1B
\sum_{i=1}^{B}
\frac{
\left\|
W^i(A\chi_S^i-b^i)
\right\|_2^2
}{
\max(N_i,1)
},
\]

where \(N_i\) is the number of evaluated field voxels. If a field-domain mask is required, apply it explicitly and document whether it is already included in \(W\).

```python
def weighted_data_consistency_loss(
    chi_pred: torch.Tensor,
    local_field: torch.Tensor,
    magnitude_weight: torch.Tensor,
    operator,
    *,
    field_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    weight = magnitude_weight.float()

    if not torch.isfinite(weight).all():
        raise ValueError("Magnitude weights must be finite.")
    if torch.any(weight < 0):
        raise ValueError("Magnitude weights must be nonnegative.")

    predicted_field = operator.forward(
        chi_pred.float()
    )
    residual = predicted_field - local_field.float()
    weighted_residual = weight * residual

    if field_mask is not None:
        field_mask = field_mask.float()
        weighted_residual = weighted_residual * field_mask
        voxel_count = (
            field_mask.flatten(1)
            .sum(dim=1)
            .clamp_min(1.0)
        )
    else:
        voxel_count = torch.full(
            (weighted_residual.shape[0],),
            weighted_residual[0].numel(),
            device=weighted_residual.device,
            dtype=torch.float32,
        )

    residual_energy = (
        weighted_residual.square()
        .flatten(1)
        .sum(dim=1)
    )

    return (residual_energy / voxel_count).mean()
```

The total training objective is

\[
\mathcal L
=
\mathcal L_{\mathrm{NRMSE}}
+
\lambda_{\mathrm{dc}}\mathcal L_{\mathrm{dc}},
\]

where \(\lambda_{\mathrm{dc}}\ge0\) is an explicit configuration value. \(W\) defines spatial reliability from magnitude; \(\lambda_{\mathrm{dc}}\) controls the global contribution of data consistency to the training objective. Do not conflate them.

Do not add intermediate losses by default.

Canonical loop:

```python
optimizer.zero_grad(set_to_none=True)

output = model(
    local_field=batch["local_field"],
    mask=batch["brain_mask"],
    dipole_kernel=dipole_kernel,
    magnitude_weight=batch["magnitude_weight"],
    initial=batch["initial"],
)

x_pred = (
    output.susceptibility.float()
    * batch["brain_mask"].float()
)
x_true = (
    batch["susceptibility"].float()
    * batch["brain_mask"].float()
)

loss_nrmse = nrmse(
    x_pred,
    x_true,
)

loss = loss_nrmse

if data_consistency_weight > 0:
    loss_dc = weighted_data_consistency_loss(
        chi_pred=output.susceptibility,
        local_field=batch["local_field"],
        magnitude_weight=batch["magnitude_weight"],
        operator=operator,
        field_mask=batch.get("field_mask"),
    )
    loss = loss + data_consistency_weight * loss_dc

if not torch.isfinite(loss):
    raise FloatingPointError("Non-finite TDV loss")

scaler.scale(loss).backward()
scaler.unscale_(optimizer)

for name, parameter in model.named_parameters():
    if (
        parameter.grad is not None
        and not torch.isfinite(parameter.grad).all()
    ):
        raise FloatingPointError(
            f"Non-finite gradient: {name}"
        )

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    1.0,
)

scaler.step(optimizer)
scaler.update()

model.regularizer.project_zero_mean_()
```

During evaluation:

```python
validation_nrmse = nrmse(
    x_pred,
    x_true,
)
```

Report NRMSE as a dimensionless scalar. Lower is better.

Use Adam with configuration-driven hyperparameters. Do not hard-code a learning rate without a stability test.

---

## 9. Explicit-step stability

Explicit Euler may diverge when \(\tau\) is too large.

For the quadratic physical term, a sufficient condition is

\[
0<\tau<\frac{2}{\|A\|_2^2}.
\]

The learned nonconvex regularizer may require a smaller step.

Rules:

- `maximum_time` must be configurable;
- log \(T\), \(\tau\), loss, `||data_grad||`, `||grad_R||`, and `||chi||`;
- abort on NaN or Inf;
- verify that the physical energy does not explode when \(R=0\);
- do not change \(S\) and \(T\) simultaneously without justification;
- require a tiny-overfit test before scaling the model.

---

## 10. Higher-order automatic differentiation

Training requires

\[
\frac{\partial}{\partial\theta}
\nabla_\chi R_\theta(\chi).
\]

Therefore:

- use `torch.autograd.grad(..., create_graph=True)` during training;
- do not detach `grad_R`;
- do not use `torch.no_grad()` around the energy;
- do not materialize full Hessians;
- do not use in-place operations on `chi`;
- when using activation checkpointing, pass `use_reentrant=False`.

---

## 11. Mandatory tests

### Physics

1. Adjoint test:

\[
\frac{
|\langle Ax,y\rangle-\langle x,A^Hy\rangle|
}{
|\langle Ax,y\rangle|
+
|\langle x,A^Hy\rangle|
+
\varepsilon
}
<10^{-5}.
\]

2. Verify `d(0)=0`.
3. Test arbitrary \(B_0\) directions.
4. Test anisotropic voxels.
5. Test even and odd dimensions.
6. Test adjoint padding/cropping if used.

### TDV energy

1. `energy.shape == (B,)`.
2. `energy.dtype == torch.float32`.
3. `grad_R.shape == chi.shape`.
4. `grad_R.dtype == torch.float32`.
5. Directional finite-difference test in float32.
6. Zero mean for every output filter of \(K\).

### Explicit step

1. Match the direct mathematical formula.
2. With \(R=0\), physical loss decreases for sufficiently small \(\tau\).
3. With \(A=0\), match gradient descent on \(R_\theta\).
4. Confirm shared parameters across all steps.
5. Confirm `chi` remains float32.

### Losses, metrics, AMP, and integration

1. NRMSE matches
   \[
   \|x_{\mathrm{pred}}-x_{\mathrm{true}}\|_2
   /
   \|x_{\mathrm{true}}\|_2
   \]
   on a hand-computed tensor.
2. NRMSE is zero when prediction equals ground truth.
3. NRMSE is computed per sample and then averaged across the batch.
4. The denominator protection is used only when the ground-truth norm is below \(\varepsilon\).
5. The data-consistency residual is exactly
   \[
   W(A\chi-b).
   \]
6. An all-ones \(W\) reproduces the unweighted residual.
7. Zero entries of \(W\) remove the corresponding field-residual contributions.
8. Negative or non-finite entries of \(W\) are rejected.
9. The physical gradient matches
   \[
   A^H W^H W(A\chi-b)
   \]
   on a tiny dense reference problem.
10. AMP output remains close to a float32 reference on a small volume.
11. No NaN or Inf occurs during 100 synthetic training steps.
12. Finite gradients reach:
    - analysis kernel \(K\);
    - macro-blocks;
    - energy head;
    - `raw_time`.
13. Confirm FFT receives float32 inputs.
14. Confirm `model.half()` is never called.
15. Tiny-overfit on 1–4 phantoms.
16. Final reconstruction improves over initialization.
17. Inference is deterministic for fixed inputs and parameters.

---

## 12. Prohibited changes

Do not:

1. implement a direct U-Net \(b\mapsto\chi\);
2. use a force that is not derived from a scalar energy;
3. use different parameters at different steps without documentation;
4. use CG or semi-implicit updates in this variant;
5. detach states between steps during training;
6. run FFT in half precision;
7. call `model.half()`;
8. accumulate energy in half precision;
9. omit `GradScaler`;
10. omit the zero-mean projection;
11. use `.data`;
12. materialize full Hessians;
13. silently change units, axes, masks, or susceptibility reference;
14. replace NRMSE with MSE as the primary supervised loss without explicit approval;
15. report a differently normalized error under the name NRMSE;
16. implement the data-consistency residual as anything other than \(W(A\chi-b)\);
17. treat \(W\) as a scalar when magnitude-dependent spatial weighting is configured;
18. use \(A^H W(A\chi-b)\) as the gradient of \(\frac12\|W(A\chi-b)\|^2\);
19. construct or normalize \(W\) without a documented preprocessing rule;
20. claim that explicit Euler is the original published TDV discretization.

---

## 13. Minimal repository structure

```text
src/tdv_qsm/
  operators/dipole.py
  models/activation.py
  models/blocks.py
  models/energy.py
  models/explicit_tdv.py
  losses.py
  train.py
tests/
  test_dipole_adjoint.py
  test_energy_gradient.py
  test_explicit_step.py
  test_amp.py
  test_tiny_overfit.py
```

Before editing:

1. read this file;
2. inspect existing interfaces;
3. add or update a test first;
4. implement the smallest coherent change;
5. run focused tests;
6. run the complete suite;
7. report modified files, commands, and results.

---

## 14. Definition of done

The implementation is complete only when:

- the physical model and units are documented;
- \(A\) and \(A^H\) pass the adjoint test;
- TDV outputs one scalar energy per sample;
- the force is exactly \(\nabla_\chi R_\theta\);
- the reconstruction uses explicit Euler;
- TDV convolutions run under float16 autocast;
- physics, FFT, state, reductions, and loss remain float32;
- `GradScaler` is enabled on CUDA;
- \(K\) remains zero mean;
- all parameters and \(T\) receive finite gradients;
- NRMSE is used identically for training and evaluation;
- the data-consistency residual is exactly \(W(A\chi-b)\);
- the explicit physical gradient is exactly \(A^H W^H W(A\chi-b)\);
- AMP agrees reasonably with the float32 reference;
- tiny-overfit passes;
- no NaN or Inf occurs;
- every deviation from this specification is explicitly documented.
