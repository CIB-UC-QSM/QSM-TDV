"""Held-out, challenge, and in-vivo evaluation adapters.

Evaluation modules must consume the same explicit metadata and core
reconstruction API as training; they must not redefine the physical operator.
"""

from qsm_tdv.evaluation.convergence import save_convergence_artifacts
from qsm_tdv.evaluation.input_set import evaluate_checkpoint, load_evaluation_input
from qsm_tdv.evaluation.slices import (
    save_orthogonal_evaluation_slices,
    save_orthogonal_reconstruction_slices,
)

__all__ = [
    "evaluate_checkpoint",
    "load_evaluation_input",
    "save_convergence_artifacts",
    "save_orthogonal_evaluation_slices",
    "save_orthogonal_reconstruction_slices",
]
