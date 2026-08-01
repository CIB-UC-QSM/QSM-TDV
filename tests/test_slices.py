import numpy as np
import pytest

from qsm_tdv.evaluation.slices import (
    COSMOS_SLICE_ROTATIONS_CLOCKWISE,
    _rotate_plane_clockwise,
    _resolved_slice_rotations,
)


def test_slice_rotation_uses_clockwise_90_degree_multiples_and_swaps_axes():
    plane = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    clockwise, horizontal, vertical = _rotate_plane_clockwise(plane, "Z", "Y", 90)
    counter_clockwise, _, _ = _rotate_plane_clockwise(plane, "Z", "Y", -90)

    assert np.array_equal(clockwise, np.asarray([[4, 1], [5, 2], [6, 3]], dtype=np.float32))
    assert np.array_equal(counter_clockwise, np.asarray([[3, 6], [2, 5], [1, 4]], dtype=np.float32))
    assert (horizontal, vertical) == ("Y", "Z")


def test_cosmos_slice_rotations_match_the_requested_orientations():
    rotations = _resolved_slice_rotations(COSMOS_SLICE_ROTATIONS_CLOCKWISE)

    assert rotations == {"sagittal": 180, "coronal": 0, "axial": 90}
    with pytest.raises(ValueError, match="multiples of 90"):
        _resolved_slice_rotations({"sagittal": 45})
