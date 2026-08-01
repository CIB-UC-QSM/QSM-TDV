"""Differentiable fixed-CG semi-implicit TDV-QSM reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from qsm_tdv.models.tdv import Params, TDVConfig, tdv_energy, tdv_force
from qsm_tdv.physics.dipole import apply_dipole, dipole_adjoint

Array = jax.Array


@dataclass(frozen=True)
class ReconstructionConfig:
    """Static settings of an unrolled TDV-QSM reconstruction."""

    steps: int = 4
    cg_iterations: int = 12
    max_time: float = 0.25
    cg_epsilon: float = 1e-12
    cg_relative_tolerance: float = 1e-4
    remat_force: bool = False

    def __post_init__(self) -> None:
        if self.steps < 1 or self.cg_iterations < 1:
            raise ValueError("steps and cg_iterations must be positive")
        if self.max_time <= 0 or self.cg_epsilon <= 0 or self.cg_relative_tolerance <= 0:
            raise ValueError("max_time, cg_epsilon, and cg_relative_tolerance must be positive")


class StepDiagnostics(NamedTuple):
    energy: Array
    force_norm: Array
    relative_residual: Array


class ReconstructionDiagnostics(NamedTuple):
    time: Array
    tau: Array
    energies: Array
    force_norms: Array
    relative_residuals: Array


def cg_residuals_within_tolerance(
    diagnostics: ReconstructionDiagnostics | StepDiagnostics,
    reconstruction_config: ReconstructionConfig,
) -> Array:
    """Return whether every fixed-CG solve satisfies its configured residual target.

    CG always executes its configured static scan length.  Once the target is
    met, subsequent scan entries are numerical no-ops rather than a dynamic
    loop exit; this host-side check verifies the resulting residual.
    """

    return jnp.all(diagnostics.relative_residual <= reconstruction_config.cg_relative_tolerance) if isinstance(
        diagnostics, StepDiagnostics
    ) else jnp.all(diagnostics.relative_residuals <= reconstruction_config.cg_relative_tolerance)


def require_cg_residuals_within_tolerance(
    diagnostics: ReconstructionDiagnostics | StepDiagnostics,
    reconstruction_config: ReconstructionConfig,
) -> None:
    """Raise on the host if a completed solve misses its configured tolerance."""

    if not bool(cg_residuals_within_tolerance(diagnostics, reconstruction_config)):
        residuals = diagnostics.relative_residual if isinstance(diagnostics, StepDiagnostics) else diagnostics.relative_residuals
        raise RuntimeError(
            "Fixed CG did not meet cg_relative_tolerance: "
            f"max relative residual={float(jnp.max(residuals)):.3e}, "
            f"tolerance={reconstruction_config.cg_relative_tolerance:.3e}. "
            "Increase cg_iterations or relax the documented tolerance."
        )


def _batch_inner(x: Array, y: Array) -> Array:
    """Real batchwise inner product as ``[B,1,1,1,1]``."""

    return jnp.real(jnp.sum(jnp.conj(x) * y, axis=(1, 2, 3, 4), keepdims=True))


def _explicit_measurement(
    x: Array,
    observation_mask: Array | None,
    statistical_weight: Array | None,
) -> Array:
    """Apply W M, explicitly outside the physical Fourier operator.

    The weighted data term is ``0.5 * ||W M (A chi - b)||²``.  Both arrays are
    real voxelwise multipliers and therefore their adjoints are themselves.
    ``statistical_weight`` is W (not W²).
    """

    output = x
    if observation_mask is not None:
        if observation_mask.shape != x.shape:
            raise ValueError("observation_mask must have the same NDHWC shape as its input")
        output = output * observation_mask.astype(x.dtype)
    if statistical_weight is not None:
        if statistical_weight.shape != x.shape:
            raise ValueError("statistical_weight must have the same NDHWC shape as its input")
        output = output * statistical_weight.astype(x.dtype)
    return output


def _explicit_measurement_adjoint(
    x: Array,
    observation_mask: Array | None,
    statistical_weight: Array | None,
) -> Array:
    """Exact adjoint of :func:`_explicit_measurement` for real maps."""

    return _explicit_measurement(x, observation_mask, statistical_weight)


def data_fidelity(
    chi: Array,
    local_field: Array,
    kernel: Array,
    observation_mask: Array | None = None,
    statistical_weight: Array | None = None,
) -> Array:
    """One explicit weighted data-fidelity scalar per batch element."""

    residual = _explicit_measurement(
        apply_dipole(chi, kernel) - local_field,
        observation_mask,
        statistical_weight,
    )
    return 0.5 * jnp.sum(residual**2, axis=(1, 2, 3, 4))


def data_fidelity_gradient(
    chi: Array,
    local_field: Array,
    kernel: Array,
    observation_mask: Array | None = None,
    statistical_weight: Array | None = None,
) -> Array:
    """Apply ``Aᴴ Mᴴ Wᴴ W M (A chi - b)`` exactly."""

    residual = apply_dipole(chi, kernel) - local_field
    measured = _explicit_measurement(residual, observation_mask, statistical_weight)
    return dipole_adjoint(
        _explicit_measurement_adjoint(measured, observation_mask, statistical_weight), kernel
    )


def data_rhs(
    local_field: Array,
    kernel: Array,
    observation_mask: Array | None = None,
    statistical_weight: Array | None = None,
) -> Array:
    """Compute ``Aᴴ Mᴴ Wᴴ W M b`` for the semi-implicit right-hand side."""

    measured = _explicit_measurement(local_field, observation_mask, statistical_weight)
    return dipole_adjoint(
        _explicit_measurement_adjoint(measured, observation_mask, statistical_weight), kernel
    )


def data_normal(
    chi: Array,
    kernel: Array,
    observation_mask: Array | None = None,
    statistical_weight: Array | None = None,
) -> Array:
    """Compute ``Aᴴ Mᴴ Wᴴ W M A chi`` for an SPD system."""

    measured = _explicit_measurement(apply_dipole(chi, kernel), observation_mask, statistical_weight)
    return dipole_adjoint(
        _explicit_measurement_adjoint(measured, observation_mask, statistical_weight), kernel
    )


def fixed_cg(
    operator: Callable[[Array], Array],
    rhs: Array,
    *,
    iterations: int,
    epsilon: float = 1e-8,
    relative_tolerance: float | None = None,
    initial: Array | None = None,
) -> tuple[Array, Array]:
    """Run a static number of differentiable conjugate-gradient iterations.

    ``operator`` must be Hermitian positive definite.  The loop always has
    ``iterations`` static scan steps.  When an optional
    relative residual target is reached, later scan steps become stable no-ops
    rather than dynamically exiting the loop; this avoids floating-point CG
    breakdown after convergence while retaining static JAX shapes.
    """

    if iterations < 1:
        raise ValueError("iterations must be positive")
    x0 = jnp.zeros_like(rhs) if initial is None else initial
    r0 = rhs - operator(x0)
    p0 = r0
    rr0 = _batch_inner(r0, r0)
    rhs_norm = jnp.sqrt(_batch_inner(rhs, rhs))
    # ``rr`` and ``pᴴAp`` are squared-norm-scale quantities.  ``epsilon`` is
    # specified on an amplitude scale, so use its square for CG-breakdown
    # protection; otherwise low-amplitude QSM fields stop before their stated
    # relative-residual target can be reached.
    breakdown_threshold = epsilon**2
    target_residual_sq = (
        jnp.zeros_like(rr0)
        if relative_tolerance is None
        else (float(relative_tolerance) * rhs_norm) ** 2
    )
    active0 = rr0 > target_residual_sq

    def cg_iteration(carry: tuple[Array, Array, Array, Array, Array], _: None):
        x, residual, direction, rr, active = carry
        applied_direction = operator(direction)
        denominator = _batch_inner(direction, applied_direction)
        valid_denominator = active & (jnp.abs(denominator) > breakdown_threshold)
        safe_denominator = jnp.where(valid_denominator, denominator, jnp.ones_like(denominator))
        alpha = jnp.where(valid_denominator, rr / safe_denominator, jnp.zeros_like(rr))
        x_new = x + alpha * direction
        residual_new = residual - alpha * applied_direction
        rr_new = _batch_inner(residual_new, residual_new)
        valid_rr = active & (rr > breakdown_threshold)
        safe_rr = jnp.where(valid_rr, rr, jnp.ones_like(rr))
        beta = jnp.where(valid_rr, rr_new / safe_rr, jnp.zeros_like(rr_new))
        direction_new = residual_new + beta * direction
        active_new = active & (rr_new > target_residual_sq)
        return (x_new, residual_new, direction_new, rr_new, active_new), jnp.sqrt(rr_new)

    (solution, final_residual, _, _, _), _ = lax.scan(
        cg_iteration, (x0, r0, p0, rr0, active0), xs=None, length=iterations
    )
    relative_residual = jnp.squeeze(
        jnp.sqrt(_batch_inner(final_residual, final_residual)) / (rhs_norm + epsilon), axis=(1, 2, 3, 4)
    )
    return solution, relative_residual


def semi_implicit_step(
    params: Params,
    chi: Array,
    local_field: Array,
    kernel: Array,
    tdv_config: TDVConfig,
    reconstruction_config: ReconstructionConfig,
    tau: Array,
    *,
    regularizer_mask: Array | None = None,
    observation_mask: Array | None = None,
    statistical_weight: Array | None = None,
) -> tuple[Array, StepDiagnostics]:
    """Perform the specified semi-implicit TDV-QSM update exactly.

    This solves ``(I + tau AᴴMᴴWᴴWMA) chi_next = chi + tau(AᴴMᴴWᴴWMb - g)``.
    It reduces to the requested QSM equation when M and W are absent.
    """

    energy = tdv_energy(params, chi, tdv_config, regularizer_mask)
    if reconstruction_config.remat_force:
        # Keep non-array architecture metadata out of remat's traced arguments.
        force = jax.checkpoint(
            lambda image: tdv_force(params, image, tdv_config, regularizer_mask)
        )(chi)
    else:
        force = tdv_force(params, chi, tdv_config, regularizer_mask)
    right_hand_data = data_rhs(local_field, kernel, observation_mask, statistical_weight)
    rhs = chi + tau * (right_hand_data - force)
    operator = lambda image: image + tau * data_normal(
        image, kernel, observation_mask, statistical_weight
    )
    chi_next, relative_residual = fixed_cg(
        operator,
        rhs,
        iterations=reconstruction_config.cg_iterations,
        epsilon=reconstruction_config.cg_epsilon,
        relative_tolerance=reconstruction_config.cg_relative_tolerance,
    )
    force_norm = jnp.sqrt(jnp.sum(force**2, axis=(1, 2, 3, 4)))
    return chi_next, StepDiagnostics(energy, force_norm, relative_residual)


def reconstruct(
    params: Params,
    raw_time: Array,
    chi_init: Array,
    local_field: Array,
    kernel: Array,
    tdv_config: TDVConfig,
    reconstruction_config: ReconstructionConfig,
    *,
    regularizer_mask: Array | None = None,
    observation_mask: Array | None = None,
    statistical_weight: Array | None = None,
) -> tuple[Array, ReconstructionDiagnostics]:
    """Unroll S semi-implicit steps using shared TDV parameters at every step."""

    time = reconstruction_config.max_time * jax.nn.sigmoid(raw_time)
    tau = time / float(reconstruction_config.steps)

    def one_step(current_chi: Array, _: None) -> tuple[Array, StepDiagnostics]:
        return semi_implicit_step(
            params,
            current_chi,
            local_field,
            kernel,
            tdv_config,
            reconstruction_config,
            tau,
            regularizer_mask=regularizer_mask,
            observation_mask=observation_mask,
            statistical_weight=statistical_weight,
        )

    reconstruction, diagnostics = lax.scan(one_step, chi_init, xs=None, length=reconstruction_config.steps)
    return reconstruction, ReconstructionDiagnostics(
        time=time,
        tau=tau,
        energies=diagnostics.energy,
        force_norms=diagnostics.force_norm,
        relative_residuals=diagnostics.relative_residual,
    )


def reconstruct_trajectory(
    params: Params,
    raw_time: Array,
    chi_init: Array,
    local_field: Array,
    kernel: Array,
    tdv_config: TDVConfig,
    reconstruction_config: ReconstructionConfig,
    *,
    regularizer_mask: Array | None = None,
    observation_mask: Array | None = None,
    statistical_weight: Array | None = None,
) -> tuple[Array, ReconstructionDiagnostics]:
    """Unroll TDV-QSM and retain ``chi_0`` through ``chi_S``.

    This diagnostic variant is intended for evaluation reports.  Unlike
    :func:`reconstruct`, it stores every reconstruction state and therefore
    should not be used in the memory-sensitive training objective.
    """

    time = reconstruction_config.max_time * jax.nn.sigmoid(raw_time)
    tau = time / float(reconstruction_config.steps)

    def one_step(
        current_chi: Array, _: None
    ) -> tuple[Array, tuple[Array, StepDiagnostics]]:
        next_chi, diagnostics = semi_implicit_step(
            params,
            current_chi,
            local_field,
            kernel,
            tdv_config,
            reconstruction_config,
            tau,
            regularizer_mask=regularizer_mask,
            observation_mask=observation_mask,
            statistical_weight=statistical_weight,
        )
        return next_chi, (next_chi, diagnostics)

    _, (states, diagnostics) = lax.scan(one_step, chi_init, xs=None, length=reconstruction_config.steps)
    trajectory = jnp.concatenate((chi_init[jnp.newaxis, ...], states), axis=0)
    return trajectory, ReconstructionDiagnostics(
        time=time,
        tau=tau,
        energies=diagnostics.energy,
        force_norms=diagnostics.force_norm,
        relative_residuals=diagnostics.relative_residual,
    )
