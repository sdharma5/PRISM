"""The ``indeterminate`` class: an explicit place to put probability mass.

Scientific WHY
--------------
A K-way softmax is forced to spend all of its mass on the K discovered groups,
even for a participant who resembles none of them, or whose evidence is too thin
to place. Reserving an explicit ``indeterminate`` outcome means "we do not know"
is a *reportable answer* rather than an artifact of normalization — which is the
whole point of an abstention-capable research tool.
"""

from __future__ import annotations

from collections.abc import Mapping

from schemas.phenotype import INDETERMINATE

__all__ = ["add_indeterminate_mass", "dominant_with_indeterminate", "is_indeterminate"]


def add_indeterminate_mass(
    probabilities: Mapping[str, float],
    mass: float,
    key: str = INDETERMINATE,
) -> dict[str, float]:
    """Reserve ``mass`` for ``indeterminate`` and rescale the rest to fill 1 - mass.

    The existing profile probabilities keep their *relative* ordering and ratios;
    only their total shrinks. ``mass`` is clamped to [0, 1]. An all-zero or empty
    input collapses to full indeterminate mass, which is the correct answer when
    there is nothing to be confident about.
    """
    mass = float(min(max(mass, 0.0), 1.0))
    existing = {k: float(v) for k, v in probabilities.items() if k != key}
    total = sum(v for v in existing.values() if v > 0)

    if total <= 0 or mass >= 1.0:
        return {**{k: 0.0 for k in existing}, key: 1.0}

    scale = (1.0 - mass) / total
    out = {k: max(v, 0.0) * scale for k, v in existing.items()}
    out[key] = mass + float(probabilities.get(key, 0.0)) * 0.0
    return out


def is_indeterminate(probabilities: Mapping[str, float], key: str = INDETERMINATE) -> bool:
    """True when ``indeterminate`` holds at least as much mass as any profile."""
    if not probabilities:
        return True
    indeterminate = float(probabilities.get(key, 0.0))
    others = [float(v) for k, v in probabilities.items() if k != key]
    return not others or indeterminate >= max(others)


def dominant_with_indeterminate(
    probabilities: Mapping[str, float],
    key: str = INDETERMINATE,
) -> tuple[str, float]:
    """Return the argmax label and its probability, ``indeterminate`` included.

    Ties resolve to ``indeterminate`` on purpose: an exact tie between two
    profiles is precisely the case where naming one of them would overstate what
    the model knows.
    """
    if not probabilities:
        return key, 1.0
    best_key, best_value = key, float(probabilities.get(key, 0.0))
    for candidate, value in probabilities.items():
        if candidate == key:
            continue
        if float(value) > best_value:
            best_key, best_value = candidate, float(value)
    return best_key, best_value
