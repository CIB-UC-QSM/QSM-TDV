"""Fourier QSM forward model and semi-implicit reconstruction operators."""

from qsm_tdv.physics.dipole import DipoleMetadata, apply_dipole, dipole_adjoint, dipole_kernel
from qsm_tdv.physics.reconstruction import (
    ReconstructionConfig,
    cg_residuals_within_tolerance,
    reconstruct,
    reconstruct_trajectory,
    require_cg_residuals_within_tolerance,
)

__all__ = [
    "DipoleMetadata",
    "ReconstructionConfig",
    "apply_dipole",
    "cg_residuals_within_tolerance",
    "dipole_adjoint",
    "dipole_kernel",
    "reconstruct",
    "reconstruct_trajectory",
    "require_cg_residuals_within_tolerance",
]
