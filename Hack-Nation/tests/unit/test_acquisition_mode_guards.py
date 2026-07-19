"""An acquisition may never claim a measurement it cannot support.

These are the tests that make the rest of the imaging module trustworthy. Every
other number the pipeline produces is only meaningful if the *kind* of number is
right: a per-section follicle count presented as an antral follicle count would
misrepresent the ovary regardless of how accurately it was measured.

Two independent guards are asserted here, and the redundancy is the point:

1. the **schema validator**, which refuses an over-claiming
   :class:`OvarianMorphologyOutput` however it was constructed; and
2. the **pipeline**, which never assembles one in the first place.

A test suite that only checked the pipeline would pass even if the validator were
deleted, and vice versa.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ingestion.ultrasound.loader import detect_acquisition_mode, load_ultrasound
from models.ultrasound.encoder import UltrasoundEncoder, infer_acquisition_mode
from schemas.imaging import OvarianMorphologyOutput
from tests.fixtures.synthetic_ultrasound import make_cine_phantom, make_phantom, make_phantom_2d

SPACING = (0.35, 0.35, 0.35)


def _output(**kwargs):
    base = {"study_id": "S", "patient_id": "P", "quality_score": 0.8}
    return OvarianMorphologyOutput(**{**base, **kwargs})


# -- guard 1: the schema validator -----------------------------------------


def test_single_frame_cannot_claim_an_estimated_per_ovary_count():
    """A cross-section cannot be extrapolated to the whole ovary."""
    with pytest.raises(ValidationError, match="single frame cannot support an estimated"):
        _output(acquisition_mode="single_frame", estimated_follicle_number_per_ovary=12)


def test_single_frame_cannot_claim_a_true_per_ovary_count():
    with pytest.raises(ValidationError, match="single frame cannot support a true per-ovary"):
        _output(acquisition_mode="single_frame", follicle_number_per_ovary=12)


def test_single_frame_cannot_claim_an_ovarian_volume():
    """Two dimensions cannot produce a volume, however good the frame is."""
    with pytest.raises(ValidationError, match="volume cannot be computed from a single"):
        _output(acquisition_mode="single_frame", ovary_volume_ml=9.0)


@pytest.mark.parametrize("mode", ["cine_loop", "multi_frame"])
def test_cine_cannot_claim_a_true_per_ovary_count(mode):
    """A sweep estimates; only a volume counts."""
    with pytest.raises(ValidationError, match="cannot support a true per-ovary count"):
        _output(acquisition_mode=mode, follicle_number_per_ovary=12)


def test_single_frame_may_report_a_per_section_count_and_area():
    """The guard must not be so broad that the legitimate claim is blocked."""
    output = _output(
        acquisition_mode="single_frame",
        follicle_number_per_section=7,
        ovary_area_mm2=540.0,
        follicle_count_method="per_section",
    )
    assert output.reportable_follicle_count == (7, "per_section")


def test_cine_may_report_an_estimated_per_ovary_count():
    output = _output(
        acquisition_mode="cine_loop",
        estimated_follicle_number_per_ovary=11,
        follicle_number_per_section=4,
        follicle_count_method="estimated_per_ovary",
    )
    assert output.reportable_follicle_count == (11, "estimated_per_ovary")


def test_volume_may_report_a_true_per_ovary_count_and_volume():
    output = _output(
        acquisition_mode="volume_3d",
        follicle_number_per_ovary=15,
        ovary_volume_ml=9.4,
        follicle_count_method="per_ovary",
    )
    assert output.reportable_follicle_count == (15, "per_ovary")


def test_declared_method_must_match_a_populated_count():
    """Naming a method whose count is None is an empty claim."""
    with pytest.raises(ValidationError, match="but that count is None"):
        _output(acquisition_mode="single_frame", follicle_count_method="per_section")


# -- guard 2: the pipeline never assembles an over-claim --------------------


def _encode(image, mode, **kwargs):
    loaded = load_ultrasound(
        image,
        patient_id="P",
        study_id="S",
        spacing_mm=SPACING,
        laterality="left",
        route="transvaginal",
        acquisition_mode=mode,
        **kwargs,
    )
    return UltrasoundEncoder(segmenter_kind="threshold").encode(
        loaded.array, loaded.metadata, acquisition_mode=loaded.acquisition_mode
    )


def test_pipeline_single_frame_emits_no_per_ovary_quantity():
    """A real single-frame encode produces a per-section count and nothing more."""
    morphology = _encode(make_phantom_2d().frame, "single_frame").morphology

    assert morphology.acquisition_mode == "single_frame"
    assert morphology.follicle_number_per_section is not None
    assert morphology.estimated_follicle_number_per_ovary is None
    assert morphology.follicle_number_per_ovary is None
    assert morphology.ovary_volume_ml is None
    assert morphology.ovary_area_mm2 is not None
    assert morphology.follicle_count_method == "per_section"


def test_pipeline_single_frame_says_why_the_per_ovary_count_is_missing():
    """The absence must be explained as an acquisition limit, not a model failure."""
    warnings = " ".join(_encode(make_phantom_2d().frame, "single_frame").morphology.warnings)
    assert "per-ovary count unavailable" in warnings
    assert "Clinician confirmation pending" in warnings


def test_pipeline_cine_emits_an_estimate_and_never_a_true_count():
    morphology = _encode(make_cine_phantom().frames, "cine_loop").morphology

    assert morphology.acquisition_mode == "cine_loop"
    assert morphology.estimated_follicle_number_per_ovary is not None
    assert morphology.follicle_number_per_ovary is None
    assert morphology.ovary_volume_ml is None
    assert morphology.follicle_count_method == "estimated_per_ovary"
    assert morphology.tracking_coverage is not None


def test_pipeline_volume_is_the_only_path_to_a_true_count():
    phantom = make_phantom(seed=0)
    loaded = load_ultrasound(
        phantom.volume,
        patient_id="P",
        study_id="S",
        spacing_mm=phantom.spacing,
        laterality="left",
        route="transvaginal",
        acquisition_mode="volume_3d",
    )
    morphology = (
        UltrasoundEncoder(segmenter_kind="threshold")
        .encode(loaded.array, loaded.metadata, acquisition_mode="volume_3d")
        .morphology
    )
    assert morphology.acquisition_mode == "volume_3d"
    assert morphology.follicle_number_per_ovary is not None
    assert morphology.ovary_volume_ml is not None
    assert morphology.follicle_count_method == "per_ovary"


# -- mode detection ---------------------------------------------------------


def test_a_2d_frame_is_detected_as_a_single_frame():
    mode, source, _ = detect_acquisition_mode(make_phantom_2d().frame)
    assert mode == "single_frame"
    assert source == "array_rank"


def test_an_ambiguous_stack_defaults_to_cine_not_volume():
    """The safe direction: a sweep mislabelled as a volume would over-claim."""
    mode, _, warnings = detect_acquisition_mode(make_cine_phantom().frames)
    assert mode == "cine_loop"
    assert any("no true per-ovary" in w for w in warnings)


def test_a_short_stack_is_multi_frame_not_cine():
    """Separately captured stills do not support the continuity tracking assumes."""
    mode, _, _ = detect_acquisition_mode(make_cine_phantom(n_frames=3).frames)
    assert mode == "multi_frame"


def test_slice_geometry_makes_it_a_volume():
    mode, source, _ = detect_acquisition_mode(make_phantom().volume, is_3d_hint=True)
    assert mode == "volume_3d"
    assert source == "source_geometry"


def test_a_declared_mode_always_wins():
    mode, source, _ = detect_acquisition_mode(make_cine_phantom().frames, declared="volume_3d")
    assert (mode, source) == ("volume_3d", "declared")


def test_an_unknown_declared_mode_is_refused():
    with pytest.raises(ValueError, match="Unknown acquisition_mode"):
        detect_acquisition_mode(make_phantom_2d().frame, declared="freehand_guess")


def test_encoder_refuses_to_guess_an_unsupported_mode():
    """``unknown`` must not silently fall through to a measurement path."""
    encoder = UltrasoundEncoder(segmenter_kind="threshold")
    loaded = load_ultrasound(make_phantom_2d().frame, spacing_mm=SPACING)
    with pytest.raises(ValueError, match="Refusing to guess"):
        encoder.encode(loaded.array, loaded.metadata, acquisition_mode="unknown")


def test_encoder_infers_frame_versus_sweep_from_rank():
    loaded_frame = load_ultrasound(make_phantom_2d().frame, spacing_mm=SPACING)
    loaded_cine = load_ultrasound(make_cine_phantom().frames, spacing_mm=SPACING)
    assert infer_acquisition_mode(loaded_frame.array, loaded_frame.metadata) == "single_frame"
    assert infer_acquisition_mode(loaded_cine.array, loaded_cine.metadata) == "cine_loop"
