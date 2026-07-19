"""Morphology must match the phantom's analytic volume and flag large structures.

The large-structure test is a safety test, not an accuracy test: a big cystic
structure must be excluded from the follicle count, flagged, and warned about —
and it must never be given a pathology name.
"""

from __future__ import annotations

import numpy as np
import pytest

from ingestion.ultrasound.loader import load_study
from models.ultrasound.encoder import UltrasoundEncoder
from models.ultrasound.morphology_3d import (
    ELLIPSOID_COEFFICIENT,
    count_follicle_voxels_outside_ovary,
    lateral_asymmetry,
    ovary_dimensions_mm,
    ovary_volume_ml_ellipsoid,
    ovary_volume_ml_voxelwise,
)
from tests.fixtures.synthetic_ultrasound import make_phantom

#: Naming a finding is a diagnosis. A large structure may be described by size,
#: never labelled as any of these entities.
FORBIDDEN_DIAGNOSTIC_WORDS = (
    "cyst",
    "corpus luteum",
    "endometrioma",
    "pmos",
    "polycystic",
    "malignan",
    "tumor",
    "tumour",
    "neoplas",
    "suspicious for",
    "consistent with",
)


def _encode(volume, spacing, **kwargs):
    _, metadata = load_study(
        volume,
        patient_id=kwargs.pop("patient_id", "P"),
        study_id=kwargs.pop("study_id", "S"),
        spacing_mm=spacing,
        laterality=kwargs.pop("laterality", "left"),
        route="transvaginal",
        acquisition_mode="volume_3d",
    )
    return UltrasoundEncoder(segmenter_kind="threshold", **kwargs).encode(
        volume, metadata, acquisition_mode="volume_3d"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_volume_matches_the_phantom_truth(seed):
    """Reported ovarian volume must be within 10% of the analytic truth."""
    phantom = make_phantom(seed=seed)
    encoding = _encode(phantom.volume, phantom.spacing)
    measured = encoding.morphology.ovary_volume_ml
    assert measured is not None
    relative_error = abs(measured - phantom.true_ovary_volume_ml) / phantom.true_ovary_volume_ml
    assert relative_error < 0.10, f"volume error {relative_error:.1%} too large"


def test_ellipsoid_formula_agrees_with_voxel_count():
    """Both volume estimates must agree on an ellipsoid, or the cross-check is broken."""
    phantom = make_phantom(seed=0)
    dims = ovary_dimensions_mm(phantom.ovary_mask, phantom.spacing)
    voxelwise = ovary_volume_ml_voxelwise(phantom.ovary_mask, phantom.spacing)
    ellipsoid = ovary_volume_ml_ellipsoid(dims)
    assert abs(ellipsoid - voxelwise) / voxelwise < 0.15
    assert pytest.approx(0.523) == ELLIPSOID_COEFFICIENT


def test_dimensions_recover_the_phantom_axes():
    """Principal-axis dimensions must recover 2x the phantom semi-axes."""
    phantom = make_phantom(seed=0)
    dims = ovary_dimensions_mm(phantom.ovary_mask, phantom.spacing)
    expected = sorted((2 * a for a in phantom.ovary_semi_axes_mm), reverse=True)
    assert np.allclose(dims, expected, rtol=0.10)


def test_large_structure_is_flagged_and_excluded_from_the_count():
    """A large cystic structure is flagged, excluded and warned about — not named."""
    # One big anechoic structure plus normal follicles, all inside the ovary.
    phantom = make_phantom(
        semi_axes_mm=(20.0, 26.0, 22.0),
        shape=(64, 96, 96),
        spacing=(1.0, 0.6, 0.6),
        follicle_diameters_mm=(4.0, 5.0, 6.0, 30.0),
        seed=5,
    )
    encoding = _encode(phantom.volume, phantom.spacing, large_structure_diameter_mm=25.0)
    morphology = encoding.morphology

    assert morphology.large_structure_flag is True
    large = [i for i in morphology.instances if i.is_large_or_uncertain]
    assert large, "the >=25 mm structure must be flagged"

    # Excluded from the count and from the reported diameters.
    assert morphology.follicle_number_per_ovary == len(morphology.instances) - len(large)
    assert all(d < 25.0 for d in morphology.follicle_diameters_mm)

    warning = " ".join(morphology.warnings).lower()
    assert "large" in warning and "excluded" in warning
    assert "not a diagnosis" in warning
    for word in FORBIDDEN_DIAGNOSTIC_WORDS:
        assert word not in warning, f"morphology output must not say '{word}'"


def test_output_is_always_model_generated():
    """Nothing in the pipeline may promote review status."""
    phantom = make_phantom(seed=0)
    encoding = _encode(phantom.volume, phantom.spacing)
    assert encoding.morphology.clinician_review_status == "model_generated"
    assert encoding.morphology.is_clinically_confirmed is False


def test_density_is_count_over_volume():
    """Follicle density must be internally consistent with count and volume."""
    phantom = make_phantom(seed=1)
    morphology = _encode(phantom.volume, phantom.spacing).morphology
    assert morphology.follicle_density_per_ml == pytest.approx(
        morphology.follicle_number_per_ovary / morphology.ovary_volume_ml, rel=1e-6
    )


def test_no_follicle_voxels_outside_the_ovary():
    """The segmenter restricts follicles to the ovary interior by construction."""
    phantom = make_phantom(seed=0)
    encoding = _encode(phantom.volume, phantom.spacing)
    assert encoding.morphology.false_follicle_voxels_outside_ovary == 0
    assert (
        count_follicle_voxels_outside_ovary(
            encoding.segmentation.follicle_mask, encoding.segmentation.ovary_region_mask
        )
        == 0
    )


def test_asymmetry_is_none_when_either_side_abstained():
    """Asymmetry against a withheld measurement would be a fabricated finding."""
    phantom = make_phantom(seed=0)
    left = _encode(phantom.volume, phantom.spacing, laterality="left").morphology
    unmeasured = _encode(phantom.volume, None, laterality="right").morphology

    assert lateral_asymmetry(left, unmeasured)["volume_difference_ml"] is None
    assert lateral_asymmetry(left, None)["volume_asymmetry_index"] is None

    both = lateral_asymmetry(left, left)
    assert both["volume_difference_ml"] == pytest.approx(0.0)


def test_token_carries_clinician_pending_and_no_zero_fill():
    """Missing measurements become missing_fields entries, never zeros."""
    phantom = make_phantom(seed=0)
    encoding = _encode(phantom.volume, None)
    token = encoding.token
    assert "Clinician confirmation pending" in token.warnings
    assert "ovary_volume_ml" in token.missing_fields
    assert token.structured_features["ovary_volume_ml"] is None
