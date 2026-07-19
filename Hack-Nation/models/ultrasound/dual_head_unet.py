"""Shared-encoder, two-head U-Net for ovary and follicle segmentation.

**Why two heads rather than a three-class softmax.** A follicle is a structure
*inside* the ovary, so the two labels are nested, not mutually exclusive. A
three-class softmax forces a voxel to choose, which is the wrong inductive bias
for nested anatomy and makes the ovary boundary compete with follicle detection
for the same probability mass. It is also unstable here: follicle voxels are a
small minority of an already-minority ovary region, and the audit shows whole
slices with no follicle at all (2,742 of 10,372 label slices are entirely black).
Two independent sigmoid heads over a shared encoder let a voxel be *both* ovary
and follicle, which is what it anatomically is.

**Why it is small.** Twelve training volumes. The capacity that a 30M-parameter
network would spend is spent memorising, and the held-out set is two volumes --
far too small to detect that it happened. Base width 16 and depth 3 gives ~1.5M
parameters in 3D, which is already generous for this cohort.

The same class serves 2D and 3D through ``dims``, so the deployment-oriented 2D
slice model and the scientific 3D volume model share one implementation and
cannot silently diverge.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

__all__ = ["DualHeadUNet", "SegmentationHeads"]


class SegmentationHeads(nn.Module):
    """Container making the two logit maps explicit rather than a tuple index."""

    def __init__(self, ovary: torch.Tensor, follicle: torch.Tensor) -> None:
        super().__init__()
        self.ovary = ovary
        self.follicle = follicle


def _conv_block(dims: int, cin: int, cout: int) -> nn.Sequential:
    """Two convolutions with instance norm.

    InstanceNorm rather than BatchNorm: batches here are one or two volumes, and
    BatchNorm statistics over a batch that small are noise. InstanceNorm is also
    the standard choice for medical segmentation for exactly this reason.
    """
    conv = nn.Conv3d if dims == 3 else nn.Conv2d
    norm = nn.InstanceNorm3d if dims == 3 else nn.InstanceNorm2d
    return nn.Sequential(
        conv(cin, cout, 3, padding=1),
        norm(cout, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
        conv(cout, cout, 3, padding=1),
        norm(cout, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )


class DualHeadUNet(nn.Module):
    """U-Net with one encoder and separate ovary / follicle decoder heads."""

    def __init__(
        self,
        *,
        dims: int = 3,
        in_channels: int = 1,
        base_channels: int = 16,
        depth: int = 3,
    ) -> None:
        """
        Args:
            dims: 2 for the slice model, 3 for the volume model.
            in_channels: Input channels (1 for B-mode greyscale).
            base_channels: Width of the first level.
            depth: Number of down/up levels.

        Raises:
            ValueError: If ``dims`` is not 2 or 3.
        """
        super().__init__()
        if dims not in (2, 3):
            raise ValueError(f"dims must be 2 or 3, got {dims}.")

        self.dims = dims
        self.base_channels = base_channels
        self.depth = depth

        pool = nn.MaxPool3d if dims == 3 else nn.MaxPool2d
        up = nn.ConvTranspose3d if dims == 3 else nn.ConvTranspose2d
        conv = nn.Conv3d if dims == 3 else nn.Conv2d

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        channels = in_channels
        widths: list[int] = []
        for level in range(depth):
            width = base_channels * (2**level)
            self.encoders.append(_conv_block(dims, channels, width))
            self.pools.append(pool(2))
            channels = width
            widths.append(width)

        self.bottleneck = _conv_block(dims, channels, base_channels * (2**depth))

        # Two decoders. They do NOT share weights: the ovary boundary and the
        # follicle interiors are different shapes at different scales, and a
        # shared decoder would force one set of filters to serve both.
        self.ovary_ups, self.ovary_decoders = self._build_decoder(up, dims, widths, base_channels)
        self.follicle_ups, self.follicle_decoders = self._build_decoder(
            up, dims, widths, base_channels
        )

        self.ovary_head = conv(base_channels, 1, 1)
        self.follicle_head = conv(base_channels, 1, 1)

    def _build_decoder(
        self, up: Any, dims: int, widths: list[int], base_channels: int
    ) -> tuple[nn.ModuleList, nn.ModuleList]:
        ups = nn.ModuleList()
        decoders = nn.ModuleList()
        channels = base_channels * (2 ** len(widths))
        for width in reversed(widths):
            ups.append(up(channels, width, 2, stride=2))
            decoders.append(_conv_block(dims, width * 2, width))
            channels = width
        return ups, decoders

    def _decode(
        self,
        x: torch.Tensor,
        skips: list[torch.Tensor],
        ups: nn.ModuleList,
        decoders: nn.ModuleList,
    ) -> torch.Tensor:
        for up, decoder, skip in zip(ups, decoders, reversed(skips), strict=True):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                # Odd input dimensions make the transposed conv undershoot by one
                # voxel. Interpolating to the skip's size keeps arbitrary volume
                # shapes valid without demanding padded inputs.
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = decoder(torch.cat([skip, x], dim=1))
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return ``{"ovary": logits, "follicle": logits}``, each ``(N, 1, ...)``."""
        skips: list[torch.Tensor] = []
        for encoder, pool in zip(self.encoders, self.pools, strict=True):
            x = encoder(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)

        ovary = self._decode(x, skips, self.ovary_ups, self.ovary_decoders)
        follicle = self._decode(x, skips, self.follicle_ups, self.follicle_decoders)
        return {
            "ovary": self.ovary_head(ovary),
            "follicle": self.follicle_head(follicle),
        }

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
