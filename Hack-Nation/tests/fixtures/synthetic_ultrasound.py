"""Synthetic ovarian ultrasound phantoms with exact ground truth.

Real ovarian ultrasound volumes cannot be committed to this repository, and a
segmentation/counting pipeline that is only ever exercised on data without
ground truth cannot be falsified. These phantoms give us the one thing real
scans do not: an *exactly known* follicle count, an *exactly known* set of
follicle diameters, and an analytically known ovarian volume. That lets the test
suite assert on absolute error rather than on "it ran without crashing".

Design choices and the scientific reason for each:

* The ovary is an **ellipsoid**, because clinical ovarian volume is reported via
  the prolate-ellipsoid formula (D1 x D2 x D3 x 0.523). Using an ellipsoid means
  the voxel-count volume and the ellipsoid-formula cross-check should agree, so
  a disagreement in the test is a real bug and not a shape artefact.
* Follicles are **spheres of specified diameter**, because follicle size
  distribution (and specifically the count of 2-9 mm follicles) is the quantity
  the morphology module reports.
* Noise is **multiplicative speckle plus additive Gaussian**, because ultrasound
  noise is dominantly speckle; additive-only noise would make thresholding
  unrealistically easy.
* Spacing is anisotropic by default, because a pipeline that silently assumes
  isotropic voxels will produce wrong physical measurements on real scans and we
  want that bug to show up in tests.

The 2D phantoms below are the ones that matter most now that 2D is the primary
pathway. :func:`make_cine_phantom` is the only fixture in this repository that can
falsify the *counting* logic end to end: it places a follicle so that it appears
on an exactly known span of consecutive frames, which means the suite can assert
both that the per-section counts are right frame by frame **and** that the unique
follicle appearing on frames 3-7 is counted once rather than five times. Without
that fixture, a tracker that simply summed per-frame counts would pass every test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CinePhantom",
    "FollicleSpec2D",
    "Phantom",
    "Phantom2D",
    "make_cine_phantom",
    "make_phantom",
    "make_phantom_2d",
    "make_poor_quality_frame",
    "make_poor_quality_volume",
    "make_touching_follicle_phantom",
]


@dataclass
class Phantom:
    """One synthetic study with its ground truth."""

    volume: np.ndarray
    ovary_mask: np.ndarray
    follicle_mask: np.ndarray
    spacing: tuple[float, float, float]
    true_count: int
    true_diameters_mm: list[float] = field(default_factory=list)
    ovary_semi_axes_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def true_ovary_volume_ml(self) -> float:
        """Analytic ellipsoid volume in millilitres (1 ml == 1000 mm^3)."""
        a, b, c = self.ovary_semi_axes_mm
        return (4.0 / 3.0) * np.pi * a * b * c / 1000.0

    def as_tuple(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float], int]:
        """Return ``(volume, ovary_mask, follicle_mask, spacing, true_count)``."""
        return self.volume, self.ovary_mask, self.follicle_mask, self.spacing, self.true_count


def _physical_grid(
    shape: tuple[int, int, int], spacing: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxel-centre coordinates in mm, centred on the volume centre."""
    axes = [
        (np.arange(n, dtype=float) - (n - 1) / 2.0) * s for n, s in zip(shape, spacing, strict=True)
    ]
    return np.meshgrid(*axes, indexing="ij")


def make_phantom(
    *,
    shape: tuple[int, int, int] = (48, 64, 64),
    spacing: tuple[float, float, float] = (1.0, 0.6, 0.6),
    semi_axes_mm: tuple[float, float, float] = (11.0, 15.0, 12.0),
    follicle_diameters_mm: tuple[float, ...] = (4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
    noise_sigma: float = 0.03,
    speckle_sigma: float = 0.05,
    seed: int = 0,
) -> Phantom:
    """Build an ellipsoid "ovary" containing spherical "follicles" plus noise.

    Follicles are anechoic (dark) fluid-filled structures inside brighter ovarian
    stroma, which is exactly the contrast real B-mode ultrasound shows, so an
    intensity-threshold fallback segmenter is a scientifically defensible
    baseline on this phantom.

    Args:
        shape: Volume shape in voxels, ``(z, y, x)``.
        spacing: Physical voxel size in mm per axis. Deliberately anisotropic.
        semi_axes_mm: Ovary ellipsoid semi-axes in mm.
        follicle_diameters_mm: Ground-truth follicle diameters in mm.
        noise_sigma: Additive Gaussian noise sd (intensities are 0-1).
        speckle_sigma: Multiplicative speckle sd.
        seed: RNG seed; phantoms are reproducible by construction.

    Returns:
        A :class:`Phantom` carrying volume, masks, spacing and ground truth.
    """
    rng = np.random.default_rng(seed)
    zz, yy, xx = _physical_grid(shape, spacing)

    a, b, c = semi_axes_mm
    ovary = ((zz / a) ** 2 + (yy / b) ** 2 + (xx / c) ** 2) <= 1.0

    follicle = np.zeros(shape, dtype=bool)
    placed: list[float] = []
    placed_spheres: list[tuple[tuple[float, float, float], float]] = []
    for diameter in sorted(follicle_diameters_mm, reverse=True):
        radius = diameter / 2.0
        centre = _find_centre(rng, a, b, c, radius, placed_spheres, min_gap_mm=1.5)
        if centre is None:
            continue
        cz, cy, cx = centre
        sphere = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2
        follicle |= sphere & ovary
        placed_spheres.append((centre, radius))
        placed.append(float(diameter))

    # Anechoic follicle lumen, echogenic stroma, dark background.
    volume = np.full(shape, 0.08, dtype=float)
    volume[ovary] = 0.62
    volume[follicle] = 0.10

    volume = volume * (1.0 + speckle_sigma * rng.standard_normal(shape))
    volume = volume + noise_sigma * rng.standard_normal(shape)
    volume = np.clip(volume, 0.0, 1.0)

    return Phantom(
        volume=volume,
        ovary_mask=ovary,
        follicle_mask=follicle,
        spacing=spacing,
        true_count=len(placed),
        true_diameters_mm=sorted(placed),
        ovary_semi_axes_mm=semi_axes_mm,
    )


def _find_centre(
    rng: np.random.Generator,
    a: float,
    b: float,
    c: float,
    radius: float,
    existing: list[tuple[tuple[float, float, float], float]],
    *,
    min_gap_mm: float,
    attempts: int = 2000,
) -> tuple[float, float, float] | None:
    """Rejection-sample a follicle centre fully inside the ovary and separated.

    Separation is enforced as ``r_i + r_j + min_gap_mm`` so that phantom
    follicles never merge into one connected component — the merged case has its
    own dedicated fixture, :func:`make_touching_follicle_phantom`.
    """
    margin = radius + 1.5
    for _ in range(attempts):
        u = rng.uniform(-1.0, 1.0, size=3)
        cz, cy, cx = u[0] * (a - margin), u[1] * (b - margin), u[2] * (c - margin)
        if (cz / max(a - margin, 1e-6)) ** 2 + (cy / max(b - margin, 1e-6)) ** 2 + (
            cx / max(c - margin, 1e-6)
        ) ** 2 > 1.0:
            continue
        ok = True
        for (ez, ey, ex), other_radius in existing:
            dist = float(np.sqrt((cz - ez) ** 2 + (cy - ey) ** 2 + (cx - ex) ** 2))
            if dist < radius + other_radius + min_gap_mm:
                ok = False
                break
        if ok:
            return float(cz), float(cy), float(cx)
    return None


def make_touching_follicle_phantom(
    *,
    shape: tuple[int, int, int] = (48, 64, 64),
    spacing: tuple[float, float, float] = (0.6, 0.6, 0.6),
    diameter_mm: float = 7.0,
    gap_mm: float = -0.6,
    seed: int = 1,
) -> Phantom:
    """Two follicles placed so their masks touch, to test watershed splitting.

    Touching follicles are the single most common cause of undercounting in
    antral follicle counting, so the separation step needs a dedicated fixture.

    Args:
        shape: Volume shape in voxels.
        spacing: Isotropic voxel spacing in mm.
        diameter_mm: Diameter of both follicles.
        gap_mm: Centre separation offset; negative means the spheres overlap.
        seed: RNG seed.

    Returns:
        A phantom whose ``true_count`` is 2 but whose naive connected-component
        count would be 1.
    """
    rng = np.random.default_rng(seed)
    zz, yy, xx = _physical_grid(shape, spacing)
    a, b, c = 12.0, 16.0, 13.0
    ovary = ((zz / a) ** 2 + (yy / b) ** 2 + (xx / c) ** 2) <= 1.0

    r = diameter_mm / 2.0
    sep = 2 * r + gap_mm
    follicle = np.zeros(shape, dtype=bool)
    for offset in (-sep / 2.0, +sep / 2.0):
        follicle |= ((zz**2 + (yy - offset) ** 2 + xx**2) <= r**2) & ovary

    volume = np.full(shape, 0.08)
    volume[ovary] = 0.62
    volume[follicle] = 0.10
    volume = np.clip(volume * (1.0 + 0.04 * rng.standard_normal(shape)), 0.0, 1.0)

    return Phantom(
        volume=volume,
        ovary_mask=ovary,
        follicle_mask=follicle,
        spacing=spacing,
        true_count=2,
        true_diameters_mm=[diameter_mm, diameter_mm],
        ovary_semi_axes_mm=(a, b, c),
    )


def make_poor_quality_volume(
    *,
    shape: tuple[int, int, int] = (24, 32, 32),
    seed: int = 2,
) -> np.ndarray:
    """Structureless noise: no ovary is visible, so the gate must abstain."""
    rng = np.random.default_rng(seed)
    return np.clip(0.2 + 0.05 * rng.standard_normal(shape), 0.0, 1.0)


# --------------------------------------------------------------------------
# 2D phantoms — the primary pathway
# --------------------------------------------------------------------------

#: Intensities chosen to mimic B-mode: anechoic lumen, echogenic stroma, dark
#: background. The stroma/lumen gap is what every follicle detector keys on.
BACKGROUND_INTENSITY = 0.08
STROMA_INTENSITY = 0.62
LUMEN_INTENSITY = 0.10


@dataclass
class Phantom2D:
    """One synthetic 2D frame with its ground truth."""

    frame: np.ndarray
    ovary_mask: np.ndarray
    follicle_mask: np.ndarray
    #: In-plane ``(row_mm, col_mm)``.
    pixel_spacing_mm: tuple[float, float]
    #: Follicles visible in THIS cross-section. Not a per-ovary count.
    true_follicle_number_per_section: int
    true_diameters_mm: list[float] = field(default_factory=list)
    ovary_semi_axes_mm: tuple[float, float] = (0.0, 0.0)

    @property
    def true_ovary_area_mm2(self) -> float:
        """Analytic ellipse area in mm^2."""
        a, b = self.ovary_semi_axes_mm
        return float(np.pi * a * b)


def _physical_grid_2d(
    shape: tuple[int, int], spacing: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre coordinates in mm, centred on the frame centre."""
    axes = [
        (np.arange(n, dtype=float) - (n - 1) / 2.0) * s for n, s in zip(shape, spacing, strict=True)
    ]
    return np.meshgrid(*axes, indexing="ij")


def _speckle(
    base: np.ndarray, rng: np.random.Generator, noise_sigma: float, speckle_sigma: float
) -> np.ndarray:
    """Multiplicative speckle plus additive Gaussian, as real B-mode shows."""
    out = base * (1.0 + speckle_sigma * rng.standard_normal(base.shape))
    out = out + noise_sigma * rng.standard_normal(base.shape)
    return np.clip(out, 0.0, 1.0)


def make_phantom_2d(
    *,
    shape: tuple[int, int] = (128, 128),
    pixel_spacing_mm: tuple[float, float] = (0.35, 0.35),
    semi_axes_mm: tuple[float, float] = (15.0, 12.0),
    follicle_diameters_mm: tuple[float, ...] = (4.0, 5.0, 6.0, 7.0),
    follicle_centres_mm: tuple[tuple[float, float], ...] | None = None,
    noise_sigma: float = 0.02,
    speckle_sigma: float = 0.03,
    seed: int = 0,
) -> Phantom2D:
    """Build one 2D "ovary" cross-section containing circular "follicles".

    This is the analogue of a single frozen transvaginal frame, and its ground
    truth is a **per-section** count: the number of follicles visible in this one
    plane. It is deliberately not called a follicle count, because a per-section
    count is a different quantity from a per-ovary one.

    Follicle centres are laid out deterministically on a ring inside the ovary
    when not supplied, rather than rejection-sampled. A fixed layout means the
    per-section count is reproducible across seeds, so a test failure is a
    regression rather than an unlucky draw.

    Args:
        shape: Frame shape in pixels, ``(rows, cols)``.
        pixel_spacing_mm: In-plane spacing in mm per axis.
        semi_axes_mm: Ovary ellipse semi-axes in mm.
        follicle_diameters_mm: Ground-truth follicle diameters in mm.
        follicle_centres_mm: Explicit centres; a ring layout is used when omitted.
        noise_sigma: Additive Gaussian noise sd (intensities are 0-1).
        speckle_sigma: Multiplicative speckle sd.
        seed: RNG seed; phantoms are reproducible by construction.

    Returns:
        A :class:`Phantom2D` carrying frame, masks, spacing and ground truth.
    """
    rng = np.random.default_rng(seed)
    rows, cols = _physical_grid_2d(shape, pixel_spacing_mm)

    a, b = semi_axes_mm
    ovary = ((rows / a) ** 2 + (cols / b) ** 2) <= 1.0

    diameters = list(follicle_diameters_mm)
    if follicle_centres_mm is None:
        # A ring at 55% of the ovary radius keeps every follicle well inside the
        # stroma and separated by more than the sum of any two radii, so
        # touching-follicle merging (which has its own dedicated fixture) is not
        # silently under test here.
        n = max(len(diameters), 1)
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        centres = [(0.55 * a * np.sin(t), 0.55 * b * np.cos(t)) for t in angles]
    else:
        centres = [(float(r), float(c)) for r, c in follicle_centres_mm]

    follicle = np.zeros(shape, dtype=bool)
    placed: list[float] = []
    for diameter, (cr, cc) in zip(diameters, centres, strict=False):
        radius = diameter / 2.0
        disc = ((rows - cr) ** 2 + (cols - cc) ** 2) <= radius**2
        disc &= ovary
        if not disc.any():
            continue
        follicle |= disc
        placed.append(float(diameter))

    base = np.full(shape, BACKGROUND_INTENSITY, dtype=float)
    base[ovary] = STROMA_INTENSITY
    base[follicle] = LUMEN_INTENSITY

    return Phantom2D(
        frame=_speckle(base, rng, noise_sigma, speckle_sigma),
        ovary_mask=ovary,
        follicle_mask=follicle,
        pixel_spacing_mm=pixel_spacing_mm,
        true_follicle_number_per_section=len(placed),
        true_diameters_mm=sorted(placed),
        ovary_semi_axes_mm=semi_axes_mm,
    )


def make_poor_quality_frame(*, shape: tuple[int, int] = (96, 96), seed: int = 2) -> np.ndarray:
    """Structureless 2D noise: no ovary is visible, so the gate must abstain."""
    rng = np.random.default_rng(seed)
    return np.clip(0.2 + 0.05 * rng.standard_normal(shape), 0.0, 1.0)


# --------------------------------------------------------------------------
# Synthetic cine loop — ground truth for BOTH counting quantities
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FollicleSpec2D:
    """A follicle that appears on an exactly known span of consecutive frames.

    Specifying the *span* rather than a 3D position is what makes this fixture
    useful: the test can say "this follicle is on frames 3 through 7" and then
    assert that it contributes exactly 1 to the unique count and exactly 1 to the
    per-section count of each of those five frames.
    """

    diameter_mm: float
    first_frame: int
    last_frame: int
    #: Position within the ovary, in mm from the ovary centre.
    centre_row_mm: float
    centre_col_mm: float

    @property
    def frames(self) -> list[int]:
        """Frames this follicle is visible on — the ground-truth span."""
        return list(range(self.first_frame, self.last_frame + 1))

    @property
    def n_frames(self) -> int:
        return self.last_frame - self.first_frame + 1


@dataclass
class CinePhantom:
    """A swept cine loop with per-frame and per-ovary ground truth."""

    frames: np.ndarray
    ovary_masks: np.ndarray
    follicle_masks: np.ndarray
    pixel_spacing_mm: tuple[float, float]
    specs: list[FollicleSpec2D] = field(default_factory=list)
    #: Frames deliberately corrupted, which the quality gate should reject.
    unusable_frames: list[int] = field(default_factory=list)
    #: Rigid probe drift applied, in mm per frame.
    drift_mm_per_frame: tuple[float, float] = (0.0, 0.0)

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def true_unique_follicle_count(self) -> int:
        """The number of distinct follicles in the loop — the tracking target."""
        return len(self.specs)

    @property
    def true_diameters_mm(self) -> list[float]:
        return sorted(float(s.diameter_mm) for s in self.specs)

    @property
    def true_per_section_counts(self) -> list[int]:
        """Ground-truth follicle count in each frame's cross-section."""
        return [
            sum(1 for s in self.specs if s.first_frame <= k <= s.last_frame)
            for k in range(self.n_frames)
        ]

    @property
    def true_frames_by_follicle(self) -> dict[int, list[int]]:
        """``{follicle index: [frame indices]}`` — the tracking ground truth."""
        return {i: s.frames for i, s in enumerate(self.specs)}

    def true_observations_mm(self) -> dict[int, dict[int, tuple[float, float]]]:
        """``{follicle index: {frame: (row_mm, col_mm)}}`` in image coordinates.

        Positions are expressed from the image origin, matching what the tracker
        measures, rather than from the ovary centre used to construct the phantom.
        This is the ground truth that
        :func:`evaluation.ultrasound.match_tracks_to_truth` matches against, so
        fragmentation and merging can be measured rather than inferred from frame
        spans — which is impossible when several follicles share frames.
        """
        rows, cols = self.frames.shape[1], self.frames.shape[2]
        row_mm, col_mm = self.pixel_spacing_mm
        origin_row = (rows - 1) / 2.0 * row_mm
        origin_col = (cols - 1) / 2.0 * col_mm
        drift_row, drift_col = self.drift_mm_per_frame
        return {
            i: {
                k: (
                    origin_row + spec.centre_row_mm + drift_row * k,
                    origin_col + spec.centre_col_mm + drift_col * k,
                )
                for k in spec.frames
                if k not in self.unusable_frames and k < self.n_frames
            }
            for i, spec in enumerate(self.specs)
        }

    @property
    def naive_summed_count(self) -> int:
        """What a pipeline that summed per-frame counts would wrongly report.

        Kept as an explicit property so a test can assert the tracker's estimate
        is nothing like it. On the default loop this is several times the truth.
        """
        return int(sum(self.true_per_section_counts))


#: How much the follicle cross-section shrinks towards the ends of its span.
#: A true sphere would taper to zero, which would push the end frames below the
#: 2 mm minimum-diameter filter and silently shorten every span. 0.8 keeps the
#: end cross-sections at ~45% of the peak diameter: tapered, but still resolvable.
_SPAN_TAPER = 0.8


def make_cine_phantom(
    *,
    n_frames: int = 16,
    shape: tuple[int, int] = (128, 128),
    pixel_spacing_mm: tuple[float, float] = (0.35, 0.35),
    ovary_semi_axes_mm: tuple[float, float] = (15.0, 12.0),
    specs: tuple[FollicleSpec2D, ...] | None = None,
    drift_mm_per_frame: tuple[float, float] = (0.0, 0.0),
    unusable_frames: tuple[int, ...] = (),
    noise_sigma: float = 0.02,
    speckle_sigma: float = 0.03,
    seed: int = 0,
) -> CinePhantom:
    """Build a swept 2D cine loop with exactly known follicle frame spans.

    The probe sweeps through the ovary; each frame is one cross-section. A
    follicle's cross-sectional diameter tapers towards the ends of its span, as a
    sphere's would when the plane of section approaches its pole, so the tracker
    is exercised on changing sizes rather than on identical repeated discs.

    Args:
        n_frames: Number of frames in the loop.
        shape: Frame shape in pixels.
        pixel_spacing_mm: In-plane spacing in mm per axis.
        ovary_semi_axes_mm: Ovary ellipse semi-axes in mm, at mid-sweep.
        specs: Follicle specifications. A default set spanning different frame
            ranges (including one on frames 3-7) is used when omitted.
        drift_mm_per_frame: Rigid probe drift applied cumulatively, in mm per
            frame. Exercises the tracker's centroid gate; large values will
            legitimately fragment tracks.
        unusable_frames: Frame indices to replace with structureless noise, so
            the quality gate rejects them and tracking coverage drops below 1.0.
        noise_sigma: Additive Gaussian noise sd.
        speckle_sigma: Multiplicative speckle sd.
        seed: RNG seed.

    Returns:
        A :class:`CinePhantom`.
    """
    rng = np.random.default_rng(seed)
    if specs is None:
        specs = (
            # The canonical case: one follicle on frames 3 through 7. It must be
            # counted ONCE, not five times.
            FollicleSpec2D(6.0, 3, 7, -5.0, -4.0),
            FollicleSpec2D(7.0, 1, 6, 5.0, 4.0),
            FollicleSpec2D(5.0, 8, 13, -4.0, 5.0),
            FollicleSpec2D(8.0, 6, 12, 4.0, -5.0),
        )
    spec_list = list(specs)

    rows_base, cols_base = _physical_grid_2d(shape, pixel_spacing_mm)
    a, b = ovary_semi_axes_mm

    frames = np.zeros((n_frames, *shape), dtype=float)
    ovary_masks = np.zeros((n_frames, *shape), dtype=bool)
    follicle_masks = np.zeros((n_frames, *shape), dtype=bool)

    for k in range(n_frames):
        drift_r = drift_mm_per_frame[0] * k
        drift_c = drift_mm_per_frame[1] * k
        rows = rows_base - drift_r
        cols = cols_base - drift_c

        # The ovary cross-section is widest mid-sweep and narrows at the ends,
        # because the sweep enters and leaves the organ.
        u = (2.0 * k / max(n_frames - 1, 1)) - 1.0
        scale = float(np.sqrt(max(1.0 - 0.35 * u * u, 1e-3)))
        ovary = ((rows / (a * scale)) ** 2 + (cols / (b * scale)) ** 2) <= 1.0

        follicle = np.zeros(shape, dtype=bool)
        for spec in spec_list:
            if not (spec.first_frame <= k <= spec.last_frame):
                continue
            centre = (spec.first_frame + spec.last_frame) / 2.0
            half = max((spec.n_frames - 1) / 2.0, 0.5)
            t = (k - centre) / (half + 0.5)
            radius = (spec.diameter_mm / 2.0) * float(np.sqrt(max(1.0 - _SPAN_TAPER * t * t, 1e-6)))
            disc = (
                (rows - spec.centre_row_mm) ** 2 + (cols - spec.centre_col_mm) ** 2
            ) <= radius**2
            follicle |= disc & ovary

        base = np.full(shape, BACKGROUND_INTENSITY, dtype=float)
        base[ovary] = STROMA_INTENSITY
        base[follicle] = LUMEN_INTENSITY

        if k in unusable_frames:
            # Structureless noise: the ovary is not detectable, so the quality
            # gate must reject this frame and tracking coverage must drop.
            frames[k] = np.clip(0.2 + 0.05 * rng.standard_normal(shape), 0.0, 1.0)
            continue

        frames[k] = _speckle(base, rng, noise_sigma, speckle_sigma)
        ovary_masks[k] = ovary
        follicle_masks[k] = follicle

    return CinePhantom(
        frames=frames,
        ovary_masks=ovary_masks,
        follicle_masks=follicle_masks,
        pixel_spacing_mm=pixel_spacing_mm,
        specs=spec_list,
        unusable_frames=list(unusable_frames),
        drift_mm_per_frame=drift_mm_per_frame,
    )
