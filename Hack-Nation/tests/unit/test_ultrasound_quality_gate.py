"""The quality gate must abstain rather than measure a study it cannot measure.

An under-counted follicle count on a poor scan is worse than no count, because it
looks like evidence. These tests assert that abstention returns literal ``None``
measurements plus reasons, not a zero or a silently degraded number.
"""

from __future__ import annotations

import numpy as np

from ingestion.ultrasound.loader import load_study
from models.ultrasound.encoder import CLINICIAN_PENDING_WARNING, UltrasoundEncoder
from models.ultrasound.quality import abstention_reasons, assess_quality
from tests.fixtures.synthetic_ultrasound import make_phantom, make_poor_quality_volume


def _encode(volume, **kwargs):
    _, metadata = load_study(
        volume, patient_id="P", study_id="S", acquisition_mode="volume_3d", **kwargs
    )
    return UltrasoundEncoder(segmenter_kind="threshold").encode(
        volume, metadata, acquisition_mode="volume_3d"
    )


def test_noise_volume_abstains_with_no_measurements():
    """Structureless noise contains no ovary, so nothing may be measured."""
    noise = make_poor_quality_volume(seed=3)
    encoding = _encode(noise, spacing_mm=(1.0, 0.6, 0.6), laterality="left", route="transvaginal")

    assert encoding.quality.measurement_feasible is False
    assert encoding.morphology.measurement_feasible is False
    assert encoding.morphology.ovary_volume_ml is None
    assert encoding.morphology.follicle_number_per_ovary is None
    assert encoding.morphology.follicle_density_per_ml is None
    assert encoding.morphology.follicle_diameters_mm == []
    assert encoding.morphology.instances == []
    assert encoding.quality.reasons, "abstention must explain itself"


def test_unknown_spacing_blocks_physical_measurement():
    """A perfect image with unknown spacing still cannot yield millimetres."""
    phantom = make_phantom(seed=0)
    encoding = _encode(phantom.volume, spacing_mm=None)

    assert encoding.quality.pixel_spacing_available is False
    assert encoding.quality.measurement_feasible is False
    assert encoding.morphology.ovary_volume_ml is None
    assert encoding.morphology.follicle_number_per_ovary is None
    assert "spacing_mm" in encoding.token.missing_fields
    assert any("spacing" in w.lower() for w in encoding.morphology.warnings)


def test_good_study_is_measured():
    """The gate must not be so conservative that valid studies are refused."""
    phantom = make_phantom(seed=0)
    encoding = _encode(
        phantom.volume, spacing_mm=phantom.spacing, laterality="left", route="transvaginal"
    )

    assert encoding.quality.ovary_visible is True
    assert encoding.quality.measurement_feasible is True
    assert encoding.morphology.measurement_feasible is True
    assert encoding.morphology.ovary_volume_ml is not None
    assert encoding.morphology.follicle_number_per_ovary is not None


def test_abstained_output_is_still_model_generated():
    """Abstention never promotes review status; it stays model_generated."""
    noise = make_poor_quality_volume(seed=4)
    encoding = _encode(noise, spacing_mm=(1.0, 0.6, 0.6))
    assert encoding.morphology.clinician_review_status == "model_generated"
    assert CLINICIAN_PENDING_WARNING in encoding.token.warnings


def test_oversized_mask_is_rejected_as_segmentation_failure():
    """A mask covering most of the field of view is not an ovary."""
    volume = np.zeros((20, 24, 24))
    volume[2:18, 2:22, 2:22] = 1.0  # ~70% of the volume
    quality = assess_quality(
        volume,
        load_study(volume, spacing_mm=(1.0, 1.0, 1.0), laterality="left")[1],
        ovary_mask=volume > 0.5,
        follicle_mask=np.zeros_like(volume, dtype=bool),
    )
    assert quality.ovary_visible is False
    assert any("ceiling" in r or "segmentation failure" in r for r in quality.reasons)


def test_abstention_reasons_empty_when_feasible():
    """``abstention_reasons`` is the inverse of ``measurement_feasible``."""
    phantom = make_phantom(seed=1)
    encoding = _encode(
        phantom.volume, spacing_mm=phantom.spacing, laterality="right", route="transvaginal"
    )
    assert abstention_reasons(encoding.quality) == []


def test_partial_ovary_blocks_volume_but_flags_it():
    """An ovary running off the edge of the field cannot yield a valid volume."""
    phantom = make_phantom(seed=0)
    # Crop away half the ovary so it touches the volume border.
    cropped = phantom.volume[:, :32, :]
    encoding = _encode(cropped, spacing_mm=phantom.spacing, laterality="left", route="transvaginal")
    assert encoding.quality.whole_ovary_visible is False
    assert encoding.morphology.ovary_volume_ml is None
    assert any(
        "border" in w.lower() or "truncat" in w.lower() for w in encoding.morphology.warnings
    )


def test_malformed_spacing_is_rejected_rather_than_silently_truncated():
    """A spacing that is not three values must raise, not become a 2-tuple.

    ``preprocess_volume`` used to build ``original_spacing_mm`` with a generic
    ``tuple(float(s) for s in spacing_mm)``, so a 2-entry spacing sailed through
    and was stored as the acquired spacing of a 3D volume. Every downstream
    millimetre figure — follicle diameters, ovarian volume — is computed from
    that field, so the wrong-rank spacing would have produced measurements that
    look real and are not. Failing loudly is the only safe behaviour.
    """
    import pytest

    from ingestion.ultrasound.preprocessing import preprocess_volume

    volume = np.zeros((4, 8, 8), dtype=float)
    with pytest.raises(ValueError, match="three entries"):
        preprocess_volume(volume, spacing_mm=(0.5, 0.5))  # type: ignore[arg-type]

    ok = preprocess_volume(volume, spacing_mm=(0.5, 0.4, 0.4))
    assert ok.measurement_spacing_mm == (0.5, 0.4, 0.4)
