#!/usr/bin/env python
"""Tiny runnable demo of the ovarian ultrasound pipeline.

Walks the three acquisition pathways in priority order — single 2D frame, 2D cine
loop, then the optional 3D volume — printing each one's measurements next to the
phantom's known ground truth. The point of the ordering is the point of the
module: what you may measure is decided by how the study was acquired, and 2D is
the routine acquisition.

Then it demonstrates the two abstention paths that matter: a study with no
visible ovary, and a perfectly good study whose physical spacing is unknown.
Writes ``ultrasound_token.json`` next to this file.

Run:  python examples/ultrasound_example/run_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ingestion.ultrasound.loader import load_ultrasound  # noqa: E402
from models.ultrasound.encoder import UltrasoundEncoder  # noqa: E402
from tests.fixtures.synthetic_ultrasound import (  # noqa: E402
    make_cine_phantom,
    make_phantom,
    make_phantom_2d,
    make_poor_quality_frame,
)

HERE = Path(__file__).resolve().parent
SPACING_2D = (0.35, 0.35, 0.35)


def _rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _encode(encoder, image, *, mode, spacing, patient, study, **kwargs):
    loaded = load_ultrasound(
        image,
        patient_id=patient,
        study_id=study,
        spacing_mm=spacing,
        laterality="left",
        route="transvaginal",
        source_dataset="synthetic_phantom",
        acquisition_mode=mode,
        **kwargs,
    )
    return encoder.encode(loaded.array, loaded.metadata, acquisition_mode=loaded.acquisition_mode)


def main() -> int:
    """Run the demo."""
    encoder = UltrasoundEncoder(segmenter_kind="threshold")

    # -- 1. single 2D frame: the most limited real input --------------------
    _rule("1. Single 2D frame (PRIMARY pathway) - per-section count only")
    frame_phantom = make_phantom_2d(seed=0)
    encoding = _encode(
        encoder,
        frame_phantom.frame,
        mode="single_frame",
        spacing=SPACING_2D,
        patient="DEMO001",
        study="DEMO001_FRAME",
    )
    morphology = encoding.morphology
    count, method = morphology.reportable_follicle_count
    print(f"  acquisition mode      : {morphology.acquisition_mode}")
    print(f"  quality score         : {morphology.quality_score:.3f}")
    print(
        f"  follicles per section : {morphology.follicle_number_per_section}"
        f"      [true {frame_phantom.true_follicle_number_per_section}]"
    )
    print(
        f"  ovary area (mm^2)     : {morphology.ovary_area_mm2:.1f}"
        f"   [true {frame_phantom.true_ovary_area_mm2:.1f}]"
    )
    print(f"  reportable count      : {count} by method '{method}'")
    print(f"  per-ovary count       : {morphology.estimated_follicle_number_per_ovary}  <- refused")
    print(f"  ovary volume (ml)     : {morphology.ovary_volume_ml}  <- refused (2D cross-section)")

    token_path = encoding.token.write_json(HERE / "ultrasound_token.json")
    print(f"\n  token written to      : {token_path}")

    # -- 2. cine loop: the best realistic input -----------------------------
    _rule("2. 2D cine loop (PRIMARY pathway) - tracked unique-follicle ESTIMATE")
    cine_phantom = make_cine_phantom(seed=0)
    encoding = _encode(
        encoder,
        cine_phantom.frames,
        mode="cine_loop",
        spacing=SPACING_2D,
        patient="DEMO002",
        study="DEMO002_CINE",
    )
    morphology, tracking = encoding.morphology, encoding.tracking
    print(f"  acquisition mode      : {morphology.acquisition_mode}")
    print(f"  frames analyzed       : {morphology.frames_analyzed}/{cine_phantom.n_frames}")
    print(f"  tracking coverage     : {morphology.tracking_coverage:.2f}")
    print(
        f"  ESTIMATED per ovary   : {morphology.estimated_follicle_number_per_ovary}"
        f"      [true {cine_phantom.true_unique_follicle_count}]"
    )
    print(
        f"  naive per-frame sum   : {cine_phantom.naive_summed_count}"
        "      <- what summing per-section counts would wrongly report"
    )
    print(f"  count method          : {morphology.follicle_count_method}")
    print(f"  true per-ovary count  : {morphology.follicle_number_per_ovary}  <- refused (2D)")
    print("\n  tracks (a follicle on many frames is ONE follicle):")
    for track in tracking.tracks:
        span = f"{track.frame_span[0]}-{track.frame_span[1]}"
        print(
            f"    track {track.track_id}: frames {span:<7} "
            f"({track.n_frames} frames)  max diameter {track.max_diameter_mm:.1f} mm"
        )

    # -- 3. optional 3D volume ---------------------------------------------
    _rule("3. 3D volume (OPTIONAL enhanced mode) - the only TRUE per-ovary count")
    volume_phantom = make_phantom(seed=0)
    encoding = _encode(
        encoder,
        volume_phantom.volume,
        mode="volume_3d",
        spacing=volume_phantom.spacing,
        patient="DEMO003",
        study="DEMO003_VOLUME",
    )
    morphology = encoding.morphology
    print(f"  acquisition mode      : {morphology.acquisition_mode}")
    print(
        f"  TRUE per-ovary count  : {morphology.follicle_number_per_ovary}"
        f"      [true {volume_phantom.true_count}]"
    )
    print(
        f"  ovary volume (ml)     : {morphology.ovary_volume_ml:.2f}"
        f"   [true {volume_phantom.true_ovary_volume_ml:.2f}]"
    )
    print(f"  count method          : {morphology.follicle_count_method}")
    print(f"  follicle voxels outside ovary: {morphology.false_follicle_voxels_outside_ovary}")
    print(f"  review status         : {morphology.clinician_review_status}")

    # -- 4. abstention: no ovary -------------------------------------------
    _rule("4. Abstention: no ovary visible in the frame")
    encoding = _encode(
        encoder,
        make_poor_quality_frame(seed=7),
        mode="single_frame",
        spacing=SPACING_2D,
        patient="DEMO004",
        study="DEMO004_NOISE",
    )
    print(f"  measurement feasible  : {encoding.morphology.measurement_feasible}")
    print(f"  follicles per section : {encoding.morphology.follicle_number_per_section}")
    print(f"  ovary area            : {encoding.morphology.ovary_area_mm2}")
    for reason in encoding.quality.reasons:
        print(f"    - {reason}")

    # -- 5. abstention: unknown spacing ------------------------------------
    _rule("5. Abstention: good frame, but physical spacing is unknown")
    encoding = _encode(
        encoder,
        frame_phantom.frame,
        mode="single_frame",
        spacing=None,
        patient="DEMO005",
        study="DEMO005_NOSPACING",
    )
    print(f"  measurement feasible  : {encoding.morphology.measurement_feasible}")
    print(f"  follicles per section : {encoding.morphology.follicle_number_per_section}")
    print(f"  ovary area            : {encoding.morphology.ovary_area_mm2}")
    print("  a follicle diameter in pixels is not a clinical measurement, so nothing is emitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
