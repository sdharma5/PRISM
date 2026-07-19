"""Smoke test: the ultrasound pipeline runs end to end on a tiny phantom.

Not a scientific result — a check that loading, preprocessing, segmentation, the
quality gate, instance extraction, morphology and token export all still connect
to one another, without torch, pydicom or scikit-image installed.

All three acquisition pathways are exercised, with the 2D ones first because they
are the primary ones: if only the volumetric path were smoke-tested, the routine
clinical input could break without CI noticing.
"""

from __future__ import annotations

import json

import pytest

from ingestion.ultrasound.loader import load_study, load_ultrasound
from ingestion.ultrasound.preprocessing import ForbiddenTransformError, assert_transforms_allowed
from ingestion.ultrasound.validation import validate_study
from models.ultrasound.encoder import CLINICIAN_PENDING_WARNING, UltrasoundEncoder
from models.ultrasound.segmenter_2d import ThresholdSegmenter2D, build_segmenter_2d
from models.ultrasound.segmenter_3d import ThresholdSegmenter, build_segmenter
from tests.fixtures.synthetic_ultrasound import (
    make_cine_phantom,
    make_phantom,
    make_phantom_2d,
)

pytestmark = pytest.mark.smoke


def test_end_to_end_forward_pass(tmp_path):
    """Volume in, ModalityToken out."""
    phantom = make_phantom(shape=(32, 40, 40), semi_axes_mm=(8.0, 10.0, 9.0), seed=0)
    volume, metadata = load_study(
        phantom.volume,
        patient_id="SMOKE001",
        study_id="SMOKE001_L",
        spacing_mm=phantom.spacing,
        laterality="left",
        route="transvaginal",
        source_dataset="synthetic_phantom",
        acquisition_mode="volume_3d",
    )

    report = validate_study(volume, metadata)
    assert report.ok, report.errors

    encoding = UltrasoundEncoder(segmenter_kind="threshold").encode(
        volume, metadata, acquisition_mode="volume_3d", observed_at="2024-01-01"
    )

    token = encoding.token
    assert token.modality == "ovarian_ultrasound"
    assert token.patient_id == "SMOKE001"
    assert token.embedding
    assert CLINICIAN_PENDING_WARNING in token.warnings
    assert encoding.morphology.clinician_review_status == "model_generated"

    path = token.write_json(tmp_path / "ultrasound_token.json")
    reloaded = json.loads(path.read_text())
    assert reloaded["modality"] == "ovarian_ultrasound"


def test_pipeline_runs_without_torch():
    """The torch-free fallback must be selected and functional."""
    segmenter = build_segmenter("threshold")
    assert isinstance(segmenter, ThresholdSegmenter)
    phantom = make_phantom(shape=(24, 32, 32), semi_axes_mm=(6.0, 8.0, 7.0), seed=1)
    probs = segmenter.predict_proba(phantom.volume)
    assert probs.shape == (3, 24, 32, 32)
    assert probs.min() >= 0.0
    assert abs(float(probs.sum(axis=0).mean()) - 1.0) < 1e-6


def test_forbidden_transform_is_rejected():
    """A count-destroying augmentation cannot be enabled from config."""
    assert_transforms_allowed(["intensity_normalize", "crop_or_pad"])
    with pytest.raises(ForbiddenTransformError, match="forbidden"):
        assert_transforms_allowed(["elastic_deformation"])
    with pytest.raises(ForbiddenTransformError):
        assert_transforms_allowed(["mixup"])


def test_unet_is_optional_and_reports_availability():
    """Importing the U-Net must not require torch."""
    from models.ultrasound.segmenter_3d import UNet3D  # noqa: PLC0415

    model = UNet3D()
    assert isinstance(UNet3D.is_available(), bool)
    if not UNet3D.is_available():
        with pytest.raises(ImportError, match="torch"):
            model.build()


def test_model_card_declares_out_of_scope_uses():
    """The card must state what this must not be used for."""
    card = UltrasoundEncoder(segmenter_kind="threshold").export_model_card_metadata()
    assert card.out_of_scope_uses
    joined = " ".join(card.out_of_scope_uses).lower()
    assert "diagnostic" in joined
    assert card.non_diagnostic_statement


def test_single_frame_forward_pass(tmp_path):
    """The primary pathway: one 2D frame in, ModalityToken out."""
    phantom = make_phantom_2d(shape=(96, 96), semi_axes_mm=(11.0, 9.0), seed=0)
    loaded = load_ultrasound(
        phantom.frame,
        patient_id="SMOKE2D",
        study_id="SMOKE2D_L",
        spacing_mm=(0.35, 0.35, 0.35),
        laterality="left",
        route="transvaginal",
        source_dataset="synthetic_phantom",
    )
    assert loaded.acquisition_mode == "single_frame"

    encoding = UltrasoundEncoder(segmenter_kind="threshold").encode(
        loaded.array, loaded.metadata, acquisition_mode=loaded.acquisition_mode
    )
    token = encoding.token
    assert token.modality == "ovarian_ultrasound"
    assert token.structured_features["acquisition_mode"] == "single_frame"
    assert token.structured_features["follicle_count_method"] == "per_section"
    assert CLINICIAN_PENDING_WARNING in token.warnings
    assert encoding.morphology.clinician_review_status == "model_generated"

    reloaded = json.loads(token.write_json(tmp_path / "frame_token.json").read_text())
    assert reloaded["structured_features"]["follicle_number_per_ovary"] is None


def test_cine_loop_forward_pass():
    """The cine pathway: a sweep in, an ESTIMATED per-ovary count out."""
    phantom = make_cine_phantom(n_frames=12, shape=(96, 96), ovary_semi_axes_mm=(11.0, 9.0))
    loaded = load_ultrasound(
        phantom.frames,
        patient_id="SMOKECINE",
        study_id="SMOKECINE_L",
        spacing_mm=(0.35, 0.35, 0.35),
        laterality="left",
        route="transvaginal",
    )
    assert loaded.acquisition_mode == "cine_loop"

    encoding = UltrasoundEncoder(segmenter_kind="threshold").encode(
        loaded.array, loaded.metadata, acquisition_mode=loaded.acquisition_mode
    )
    assert encoding.tracking is not None
    assert encoding.morphology.follicle_count_method == "estimated_per_ovary"
    assert encoding.morphology.follicle_number_per_ovary is None
    assert encoding.token.structured_features["tracking_coverage"] is not None


def test_2d_pipeline_runs_without_torch():
    """The torch-free 2D fallback must be selected and functional."""
    segmenter = build_segmenter_2d("threshold")
    assert isinstance(segmenter, ThresholdSegmenter2D)
    phantom = make_phantom_2d(shape=(64, 64), semi_axes_mm=(8.0, 7.0), seed=1)
    probs = segmenter.predict_proba(phantom.frame)
    assert probs.shape == (3, 64, 64)
    assert probs.min() >= 0.0
    assert abs(float(probs.sum(axis=0).mean()) - 1.0) < 1e-6


def test_unet2d_is_optional_and_reports_availability():
    """Importing the 2D U-Net must not require torch."""
    from models.ultrasound.segmenter_2d import UNet2D  # noqa: PLC0415

    model = UNet2D()
    assert isinstance(UNet2D.is_available(), bool)
    if not UNet2D.is_available():
        with pytest.raises(ImportError, match="torch"):
            model.build()
