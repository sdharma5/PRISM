"""Follicle instance extraction must recover the known phantom count.

Counting is the clinically consequential output, and the two ways it goes wrong
are merging touching follicles (undercount) and admitting speckle (overcount).
Both have a dedicated test here.
"""

from __future__ import annotations

import numpy as np
import pytest

from ingestion.ultrasound.loader import load_study
from models.ultrasound.encoder import UltrasoundEncoder
from models.ultrasound.follicle_instances import (
    MIN_FOLLICLE_DIAMETER_MM,
    extract_follicle_instances,
    size_distribution,
    track_instances_across_slices,
)
from tests.fixtures.synthetic_ultrasound import make_phantom, make_touching_follicle_phantom


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_recovers_known_follicle_count(seed):
    """The pipeline recovers the phantom's exact follicle count."""
    phantom = make_phantom(seed=seed)
    _, metadata = load_study(
        phantom.volume,
        patient_id="P",
        study_id="S",
        spacing_mm=phantom.spacing,
        laterality="left",
        route="transvaginal",
        acquisition_mode="volume_3d",
    )
    encoding = UltrasoundEncoder(segmenter_kind="threshold").encode(
        phantom.volume, metadata, acquisition_mode="volume_3d"
    )
    assert encoding.morphology.follicle_number_per_ovary == phantom.true_count


def test_touching_follicles_are_separated():
    """Two overlapping follicles must be counted as two, not one.

    This is the undercount failure mode: naive connected-component labelling
    yields a single object here.
    """
    phantom = make_touching_follicle_phantom()

    from scipy import ndimage as ndi

    _, n_naive = ndi.label(phantom.follicle_mask)
    assert n_naive == 1, "fixture must actually produce one merged component"

    result = extract_follicle_instances(
        phantom.follicle_mask,
        spacing_mm=phantom.spacing,
        ovary_mask=phantom.ovary_mask,
        separate_touching_follicles=True,
    )
    assert result.count == 2
    assert result.n_split_by_watershed >= 1


def test_separation_can_be_disabled():
    """Without splitting the same fixture undercounts, proving the split works."""
    phantom = make_touching_follicle_phantom()
    result = extract_follicle_instances(
        phantom.follicle_mask,
        spacing_mm=phantom.spacing,
        ovary_mask=phantom.ovary_mask,
        separate_touching_follicles=False,
    )
    assert result.count == 1


def test_subthreshold_components_are_removed_in_physical_units():
    """A sub-2 mm blob is speckle, not a follicle, and must be dropped."""
    spacing = (0.5, 0.5, 0.5)
    mask = np.zeros((20, 20, 20), dtype=bool)
    ovary = np.ones_like(mask)
    mask[2:4, 2:4, 2:4] = True  # ~1 mm across: below the threshold
    mask[10:18, 10:18, 10:18] = True  # ~4 mm across: above it

    result = extract_follicle_instances(mask, spacing_mm=spacing, ovary_mask=ovary)
    assert result.count == 1
    assert result.n_removed_too_small == 1
    assert result.instances[0].mean_diameter_mm > MIN_FOLLICLE_DIAMETER_MM
    assert any("speckle" in w or "unresolvable" in w for w in result.warnings)


def test_components_outside_ovary_are_removed():
    """A follicle outside the ovary is anatomically impossible."""
    spacing = (0.5, 0.5, 0.5)
    mask = np.zeros((24, 24, 24), dtype=bool)
    ovary = np.zeros_like(mask)
    ovary[:12] = True
    mask[2:10, 2:10, 2:10] = True  # inside
    mask[14:22, 14:22, 14:22] = True  # outside

    result = extract_follicle_instances(mask, spacing_mm=spacing, ovary_mask=ovary)
    assert result.count == 1
    assert result.n_removed_outside_ovary == 1


def test_unknown_spacing_yields_no_physical_size_and_warns():
    """Without spacing there is no millimetre size, and the filter cannot apply."""
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[4:10, 4:10, 4:10] = True
    result = extract_follicle_instances(mask, spacing_mm=None, ovary_mask=np.ones_like(mask))

    assert result.count == 1
    assert result.instances[0].mean_diameter_mm is None
    assert result.instances[0].volume_mm3 is None
    assert any("spacing unknown" in w.lower() for w in result.warnings)


def test_diameters_are_close_to_the_truth():
    """Recovered diameters must match the phantom's within ~1 mm."""
    phantom = make_phantom(seed=0)
    _, metadata = load_study(
        phantom.volume,
        spacing_mm=phantom.spacing,
        laterality="left",
        route="transvaginal",
        patient_id="P",
        study_id="S",
    )
    encoding = UltrasoundEncoder(segmenter_kind="threshold").encode(phantom.volume, metadata)
    predicted = sorted(encoding.morphology.follicle_diameters_mm)
    truth = sorted(phantom.true_diameters_mm)
    assert len(predicted) == len(truth)
    assert np.max(np.abs(np.asarray(predicted) - np.asarray(truth))) < 1.0


def test_instances_are_tracked_across_slices_not_recounted():
    """A follicle spanning many slices is one instance, not one per slice."""
    phantom = make_phantom(seed=0)
    result = extract_follicle_instances(
        phantom.follicle_mask, spacing_mm=phantom.spacing, ovary_mask=phantom.ovary_mask
    )
    tracks = track_instances_across_slices(result.label_volume)
    assert len(tracks) == result.count
    # The largest follicle (9 mm at 1 mm z-spacing) must span multiple slices.
    assert max(len(slices) for slices in tracks.values()) > 3


def test_size_distribution_summarises_the_instances():
    """The size distribution carries more information than the bare count."""
    phantom = make_phantom(seed=0)
    result = extract_follicle_instances(
        phantom.follicle_mask, spacing_mm=phantom.spacing, ovary_mask=phantom.ovary_mask
    )
    stats = size_distribution(result.instances)
    assert stats["n"] == result.count
    assert stats["min_diameter_mm"] <= stats["median_diameter_mm"] <= stats["max_diameter_mm"]
