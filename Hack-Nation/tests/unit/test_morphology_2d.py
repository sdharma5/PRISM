"""2D morphology must be right in physical units, or abstain entirely.

The 2D path is the primary one, so its two headline numbers — the per-section
follicle count and the cross-sectional ovary area — are asserted against the
phantom's analytic truth rather than against "it produced a number".

The abstention tests matter as much as the accuracy tests. A follicle diameter in
pixels is not comparable to any published threshold, so when pixel spacing is
unknown the correct output is ``None`` plus a reason, never a pixel-count
masquerading as a measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.ultrasound import per_section_count_mae, unique_track_count_mae
from ingestion.ultrasound.loader import load_ultrasound
from models.ultrasound.encoder import UltrasoundEncoder
from models.ultrasound.morphology_2d import (
    in_plane_spacing_mm,
    ovary_area_mm2,
    ovary_in_plane_dimensions_mm,
)
from models.ultrasound.qc_2d import assess_frame_quality
from tests.fixtures.synthetic_ultrasound import (
    make_cine_phantom,
    make_phantom_2d,
    make_poor_quality_frame,
)

SPACING = (0.35, 0.35, 0.35)

#: Naming a finding is a diagnosis. A large structure may be described by size,
#: never labelled as any of these entities.
FORBIDDEN_DIAGNOSTIC_WORDS = (
    "cyst",
    "corpus luteum",
    "endometrioma",
    "pcos",
    "polycystic",
    "malignan",
    "tumor",
    "tumour",
    "neoplas",
)


def _encode(frame, *, spacing=SPACING, laterality="left"):
    loaded = load_ultrasound(
        frame,
        patient_id="P",
        study_id="S",
        spacing_mm=spacing,
        laterality=laterality,
        route="transvaginal",
        acquisition_mode="single_frame",
    )
    return UltrasoundEncoder(segmenter_kind="threshold").encode(
        loaded.array, loaded.metadata, acquisition_mode="single_frame"
    )


# -- accuracy on the phantom ------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_per_section_count_matches_the_phantom(seed):
    """The per-section count must equal the number of follicles in the plane."""
    phantom = make_phantom_2d(seed=seed)
    morphology = _encode(phantom.frame).morphology
    assert morphology.follicle_number_per_section == phantom.true_follicle_number_per_section, (
        f"got {morphology.follicle_number_per_section}"
    )


def test_ovary_area_matches_the_analytic_ellipse():
    """Reported area must be within 10% of the phantom's analytic ellipse area."""
    phantom = make_phantom_2d(seed=0)
    measured = _encode(phantom.frame).morphology.ovary_area_mm2
    assert measured is not None
    relative_error = abs(measured - phantom.true_ovary_area_mm2) / phantom.true_ovary_area_mm2
    assert relative_error < 0.10, f"area error {relative_error:.1%}"


def test_ovary_area_is_exact_on_the_ground_truth_mask():
    """Area from the true mask is a pure unit conversion and must be exact."""
    phantom = make_phantom_2d(seed=0)
    area = ovary_area_mm2(phantom.ovary_mask, phantom.pixel_spacing_mm)
    expected = phantom.ovary_mask.sum() * phantom.pixel_spacing_mm[0] * phantom.pixel_spacing_mm[1]
    assert area == pytest.approx(expected)
    assert area == pytest.approx(phantom.true_ovary_area_mm2, rel=0.02)


def test_in_plane_dimensions_recover_the_phantom_axes():
    """Principal in-plane axes must recover 2x the phantom semi-axes."""
    phantom = make_phantom_2d(seed=0)
    dims = ovary_in_plane_dimensions_mm(phantom.ovary_mask, phantom.pixel_spacing_mm)
    expected = sorted((2 * a for a in phantom.ovary_semi_axes_mm), reverse=True)
    assert np.allclose(dims, expected, rtol=0.10), (dims, expected)


def test_diameters_are_close_to_the_truth():
    """Reported cross-sectional diameters recover the phantom's follicle sizes."""
    phantom = make_phantom_2d(seed=0)
    predicted = sorted(_encode(phantom.frame).morphology.follicle_diameters_mm)
    assert len(predicted) == len(phantom.true_diameters_mm)
    assert np.allclose(predicted, phantom.true_diameters_mm, atol=1.0), predicted


# -- abstention -------------------------------------------------------------


def test_abstains_when_pixel_spacing_is_unknown():
    """Without spacing there is no millimetre, so there is no measurement."""
    phantom = make_phantom_2d(seed=0)
    encoding = _encode(phantom.frame, spacing=None)
    morphology = encoding.morphology

    assert encoding.quality.pixel_spacing_available is False
    assert morphology.measurement_feasible is False
    assert morphology.follicle_number_per_section is None
    assert morphology.ovary_area_mm2 is None
    assert morphology.follicle_diameters_mm == []
    assert morphology.follicle_count_method == "not_assessed"
    assert any("spacing" in w.lower() for w in morphology.warnings)
    assert "spacing_mm" in encoding.token.missing_fields


def test_in_plane_spacing_accessor_rejects_unusable_spacing():
    """The sanctioned accessor refuses non-finite or non-positive spacing."""
    phantom = make_phantom_2d()
    _, metadata = load_ultrasound(phantom.frame, spacing_mm=None).as_tuple()
    assert in_plane_spacing_mm(metadata) is None

    _, good = load_ultrasound(phantom.frame, spacing_mm=SPACING).as_tuple()
    assert in_plane_spacing_mm(good) == (0.35, 0.35)


def test_area_and_dimensions_return_none_without_spacing():
    phantom = make_phantom_2d()
    assert ovary_area_mm2(phantom.ovary_mask, None) is None
    assert ovary_in_plane_dimensions_mm(phantom.ovary_mask, None) is None


def test_noise_frame_abstains_with_no_measurements():
    """A structureless frame contains no ovary, so nothing may be measured."""
    encoding = _encode(make_poor_quality_frame(seed=3))
    morphology = encoding.morphology

    assert encoding.quality.ovary_visible is False
    assert morphology.measurement_feasible is False
    assert morphology.follicle_number_per_section is None
    assert morphology.ovary_area_mm2 is None
    assert encoding.quality.reasons, "abstention must explain itself"


def test_output_is_always_model_generated():
    """Nothing in the 2D path may promote review status."""
    for frame in (make_phantom_2d().frame, make_poor_quality_frame()):
        morphology = _encode(frame).morphology
        assert morphology.clinician_review_status == "model_generated"
        assert morphology.is_clinically_confirmed is False


def test_no_pathology_is_ever_named():
    """A large structure is described by size, never diagnosed."""
    phantom = make_phantom_2d(
        semi_axes_mm=(24.0, 20.0),
        follicle_diameters_mm=(4.0, 5.0, 28.0),
        shape=(160, 160),
        seed=5,
    )
    morphology = _encode(phantom.frame).morphology
    warning = " ".join(morphology.warnings).lower()
    for word in FORBIDDEN_DIAGNOSTIC_WORDS:
        assert word not in warning, f"2D morphology must not say '{word}'"


# -- structural guarantees --------------------------------------------------


def test_volume_is_structurally_infeasible_in_2d():
    """No 2D frame is ever good enough to unlock a volume."""
    phantom = make_phantom_2d(seed=0)
    _, metadata = load_ultrasound(
        phantom.frame, spacing_mm=SPACING, laterality="left", route="transvaginal"
    ).as_tuple()
    quality = assess_frame_quality(
        phantom.frame,
        metadata,
        ovary_mask=phantom.ovary_mask & ~phantom.follicle_mask,
        follicle_mask=phantom.follicle_mask,
        ovary_confidence=1.0,
    )
    assert quality.ovary_visible is True
    assert quality.measurement_feasible is True
    assert quality.ovarian_volume_feasible is False
    assert any("two dimensions" in r for r in quality.reasons)


def test_the_two_count_metrics_have_disjoint_keys():
    """Per-section and unique-track MAE must never be poolable by accident."""
    phantom_2d = make_phantom_2d()
    phantom_cine = make_cine_phantom()

    section = per_section_count_mae(
        [_encode(phantom_2d.frame).morphology.follicle_number_per_section],
        [phantom_2d.true_follicle_number_per_section],
    )
    unique = unique_track_count_mae([4], [phantom_cine.true_unique_follicle_count])

    assert set(section) & set(unique) == set()
    assert section["per_section_count_mae"] == 0.0


def test_abstained_studies_are_excluded_not_scored_as_zero():
    """Refusing to count is not counting zero."""
    metrics = per_section_count_mae([None, 5], [7, 5])
    assert metrics["per_section_count_mae"] == 0.0
    assert metrics["per_section_n_scored"] == 1.0
    assert metrics["per_section_n_abstained"] == 1.0
