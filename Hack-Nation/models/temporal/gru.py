"""Sequence encoders over participant-days: a GRU with a numpy-only fallback.

The temporal model reads the previous 14-30 participant-days and produces a
*current-state* embedding. The lookback is bounded because hormonal state has a
cycle-length memory: days more than one cycle back carry little information about
today's state, and a longer window mostly adds missing values.

Feature construction (:func:`build_feature_matrix`) is the scientifically
load-bearing part. Every time-varying channel enters as a **triple**:

* ``value`` — the observed number, or a *carried-forward* last observation,
* ``is_observed`` — 1 when today's number is real, 0 when it is carried,
* ``time_since_last_observed`` — how stale the carried value is.

Missing is never silently zero. A zero LH is a real measurement; an unobserved
LH is not, and a model that cannot tell them apart will learn the testing
*schedule* instead of the physiology — which matters here because the
missingness is non-ignorable (people test around expected ovulation).

Cycle day enters as ``sin``/``cos`` of the phase angle so that day 28 and day 1
are adjacent, which they physically are. Modality identifiers are appended as a
fixed indicator block so the model knows which channel group a value came from.

GRU-D-style exponential decay is available: a carried-forward value decays toward
the channel's empirical mean at rate ``exp(-gamma * delta_t)``, which encodes the
prior that a week-old measurement is nearly uninformative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from schemas.temporal import ParticipantDay

DEFAULT_LOOKBACK_DAYS = 21
MIN_LOOKBACK_DAYS = 14
MAX_LOOKBACK_DAYS = 30

#: Channel-group indicators appended to every timestep.
MODALITY_GROUPS: tuple[str, ...] = ("hormone", "wearable", "cgm", "symptom")


@dataclass
class FeatureSpec:
    """Describes the layout of the built feature matrix, for auditability."""

    channels: tuple[str, ...]
    channel_groups: dict[str, str] = field(default_factory=dict)
    use_decay: bool = True
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    channel_means: dict[str, float] = field(default_factory=dict)
    channel_scales: dict[str, float] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        """3 per channel (value, is_observed, staleness) + 2 cycle + group block."""
        return 3 * len(self.channels) + 2 + len(MODALITY_GROUPS)

    def feature_names(self) -> list[str]:
        """Human-readable column names, in matrix order."""
        names: list[str] = []
        for channel in self.channels:
            names += [f"{channel}__value", f"{channel}__is_observed", f"{channel}__staleness"]
        names += ["cycle_day_sin", "cycle_day_cos"]
        names += [f"group__{g}" for g in MODALITY_GROUPS]
        return names


def infer_channels(days: list[ParticipantDay]) -> tuple[str, ...]:
    """Union of every channel key seen, in sorted order for determinism."""
    channels: set[str] = set()
    for day in days:
        channels |= set(day.values.keys()) | set(day.is_observed.keys())
    return tuple(sorted(channels))


def fit_feature_spec(
    days: list[ParticipantDay],
    *,
    channels: tuple[str, ...] | None = None,
    channel_groups: dict[str, str] | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    use_decay: bool = True,
) -> FeatureSpec:
    """Fit per-channel standardisation statistics on OBSERVED values only.

    Statistics come from observed values alone. Including carried-forward values
    would let the imputation scheme leak into the normalisation constants.
    """
    resolved = channels if channels is not None else infer_channels(days)
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for channel in resolved:
        observed = [
            float(raw)
            for day in days
            if day.is_observed.get(channel) and (raw := day.values.get(channel)) is not None
        ]
        if observed:
            array = np.asarray(observed, dtype=float)
            means[channel] = float(array.mean())
            scales[channel] = float(array.std()) if array.std() > 1e-8 else 1.0
        else:
            means[channel], scales[channel] = 0.0, 1.0
    return FeatureSpec(
        channels=tuple(resolved),
        channel_groups=dict(channel_groups or {}),
        use_decay=use_decay,
        lookback_days=int(np.clip(lookback_days, MIN_LOOKBACK_DAYS, MAX_LOOKBACK_DAYS)),
        channel_means=means,
        channel_scales=scales,
    )


def _group_vector(spec: FeatureSpec) -> np.ndarray:
    """Indicator of which channel groups are present in this feature set."""
    present = {spec.channel_groups.get(c, "hormone") for c in spec.channels}
    return np.array([1.0 if g in present else 0.0 for g in MODALITY_GROUPS], dtype=float)


def build_feature_matrix(
    window: list[ParticipantDay],
    spec: FeatureSpec,
    *,
    decay_gamma: float = 0.35,
) -> np.ndarray:
    """Build a ``(T, F)`` matrix from a window of participant-days.

    Args:
        window: Consecutive days, oldest first.
        spec: Fitted feature spec.
        decay_gamma: GRU-D decay rate per day of staleness.

    Returns:
        Float matrix of shape ``(len(window), spec.n_features)``.

    Raises:
        ValueError: If a day carries a value without the matching observation
            flag, which would let a missing value masquerade as observed.
    """
    group_block = _group_vector(spec)
    rows: list[np.ndarray] = []
    last_value: dict[str, float | None] = dict.fromkeys(spec.channels)

    for day in window:
        row: list[float] = []
        for channel in spec.channels:
            mean = spec.channel_means.get(channel, 0.0)
            scale = spec.channel_scales.get(channel, 1.0)
            is_observed = bool(day.is_observed.get(channel, False))
            raw = day.values.get(channel)

            if is_observed and raw is None:
                raise ValueError(
                    f"{day.participant_id} day {day.study_day}: channel '{channel}' is flagged "
                    "observed but carries no value. Refusing to guess."
                )

            default_gap = 0.0 if is_observed else 1.0
            staleness = float(day.time_since_last_observed.get(channel, default_gap))

            if is_observed:
                last_value[channel] = float(raw)  # type: ignore[arg-type]
                normalized = (float(raw) - mean) / scale  # type: ignore[arg-type]
            elif last_value[channel] is not None:
                carried = (float(last_value[channel]) - mean) / scale  # type: ignore[arg-type]
                # GRU-D decay: an old observation shrinks toward the channel mean
                # (0 after standardisation) rather than being trusted forever.
                normalized = (
                    carried * float(np.exp(-decay_gamma * max(staleness, 0.0)))
                    if spec.use_decay
                    else carried
                )
            else:
                # Never observed yet: the standardised mean, flagged unobserved.
                normalized = 0.0

            row += [float(normalized), float(is_observed), float(np.log1p(max(staleness, 0.0)))]

        cycle_day = day.cycle_day
        if cycle_day is None:
            row += [0.0, 0.0]
        else:
            angle = 2.0 * np.pi * (float(cycle_day) % 28.0) / 28.0
            row += [float(np.sin(angle)), float(np.cos(angle))]

        rows.append(np.concatenate([np.asarray(row, dtype=float), group_block]))

    return np.vstack(rows) if rows else np.zeros((0, spec.n_features), dtype=float)


def make_windows(
    days: list[ParticipantDay],
    spec: FeatureSpec,
    *,
    stride: int = 1,
    decay_gamma: float = 0.35,
) -> tuple[np.ndarray, list[ParticipantDay]]:
    """Slide a lookback window over one participant's days.

    Windows never cross participants (this function takes one participant's days
    at a time), because a window spanning two people would fabricate a person who
    does not exist.

    Returns:
        ``(X, target_days)`` where ``X`` is ``(N, T, F)`` and ``target_days[i]``
        is the day the i-th window predicts (the last day of the window).
    """
    ordered = sorted(days, key=lambda d: d.study_day)
    lookback = spec.lookback_days
    windows: list[np.ndarray] = []
    targets: list[ParticipantDay] = []
    for end in range(lookback, len(ordered) + 1, stride):
        window = ordered[end - lookback : end]
        windows.append(build_feature_matrix(window, spec, decay_gamma=decay_gamma))
        targets.append(window[-1])
    if not windows:
        return np.zeros((0, lookback, spec.n_features), dtype=float), []
    return np.stack(windows), targets


def build_dataset(
    days: list[ParticipantDay],
    spec: FeatureSpec,
    *,
    decay_gamma: float = 0.35,
) -> tuple[np.ndarray, list[ParticipantDay], list[str]]:
    """Build windows for a whole cohort, returning the participant id per window."""
    by_participant: dict[str, list[ParticipantDay]] = {}
    for day in days:
        by_participant.setdefault(day.participant_id, []).append(day)

    all_x: list[np.ndarray] = []
    all_targets: list[ParticipantDay] = []
    groups: list[str] = []
    for participant_id in sorted(by_participant):
        x, targets = make_windows(by_participant[participant_id], spec, decay_gamma=decay_gamma)
        if x.shape[0] == 0:
            continue
        all_x.append(x)
        all_targets.extend(targets)
        groups.extend([participant_id] * x.shape[0])
    if not all_x:
        return (
            np.zeros((0, spec.lookback_days, spec.n_features), dtype=float),
            [],
            [],
        )
    return np.concatenate(all_x, axis=0), all_targets, groups


# --------------------------------------------------------------------------
# numpy GRU
# --------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


class NumpyGRU:
    """A GRU forward pass in pure numpy, used when torch is unavailable.

    Weights are fixed random projections by default. That sounds unhelpful, but a
    randomly initialised recurrent encoder followed by a *trained* linear head is
    an echo-state network, which is a legitimate and well-behaved model for short
    sequences and small cohorts — precisely this regime. It keeps the whole
    temporal pipeline trainable and testable without torch, and the trained part
    (the heads) is the part whose calibration matters.
    """

    def __init__(self, input_size: int, hidden_size: int = 32, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(max(hidden_size, 1))
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_z = rng.normal(0, scale, (input_size, hidden_size))
        self.U_z = rng.normal(0, scale, (hidden_size, hidden_size))
        self.b_z = np.zeros(hidden_size)
        self.W_r = rng.normal(0, scale, (input_size, hidden_size))
        self.U_r = rng.normal(0, scale, (hidden_size, hidden_size))
        self.b_r = np.zeros(hidden_size)
        self.W_h = rng.normal(0, scale, (input_size, hidden_size))
        self.U_h = rng.normal(0, scale, (hidden_size, hidden_size))
        self.b_h = np.zeros(hidden_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run the GRU over ``(N, T, F)`` and return the final ``(N, H)`` state."""
        x = np.atleast_3d(np.asarray(x, dtype=float))
        n, t, _ = x.shape
        h = np.zeros((n, self.hidden_size))
        for step in range(t):
            xt = x[:, step, :]
            z = _sigmoid(xt @ self.W_z + h @ self.U_z + self.b_z)
            r = _sigmoid(xt @ self.W_r + h @ self.U_r + self.b_r)
            h_tilde = np.tanh(xt @ self.W_h + (r * h) @ self.U_h + self.b_h)
            h = (1.0 - z) * h + z * h_tilde
        return h

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class TorchGRU:
    """Thin lazy wrapper around ``torch.nn.GRU`` with the same interface."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        *,
        num_layers: int = 1,
        seed: int | None = 0,
    ) -> None:
        """
        Args:
            seed: Seeds the weight initialisation. Defaults to 0 rather than None
                because this encoder's weights are NOT trained -- they are fixed
                random projections, exactly like :class:`NumpyGRU`. An unseeded
                random projection makes every run produce different features and
                therefore different metrics, for reasons unrelated to the data.
                ``NumpyGRU`` has always taken a seed; this one did not, so
                installing torch silently switched ``backend="auto"`` from a
                reproducible encoder to a non-reproducible one. Pass None only if
                you deliberately want fresh weights per instance.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seed = seed
        self._module: Any | None = None

    @staticmethod
    def is_available() -> bool:
        try:
            import torch  # noqa: F401, PLC0415
        except ImportError:
            return False
        return True

    def build(self) -> Any:
        """Instantiate and cache the ``torch.nn.GRU``."""
        if self._module is not None:
            return self._module
        try:
            from torch import nn  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "TorchGRU requires the optional 'torch' extra. Use NumpyGRU instead."
            ) from exc
        if self.seed is not None:
            import torch  # noqa: PLC0415

            # Scoped to construction: a global manual_seed here would silently
            # reseed the caller's RNG for everything that follows.
            generator_state = torch.random.get_rng_state()
            torch.manual_seed(self.seed)
            try:
                self._module = nn.GRU(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=True,
                )
            finally:
                torch.random.set_rng_state(generator_state)
            return self._module

        self._module = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        return self._module

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return the final hidden state as ``(N, H)`` numpy."""
        import torch  # noqa: PLC0415

        module = self.build()
        module.eval()
        with torch.no_grad():
            _, hidden = module(torch.as_tensor(np.asarray(x), dtype=torch.float32))
        return hidden[-1].numpy()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


def build_sequence_encoder(
    input_size: int,
    *,
    hidden_size: int = 32,
    backend: str = "auto",
    seed: int = 0,
) -> NumpyGRU | TorchGRU:
    """Return a GRU encoder, preferring torch when it is installed."""
    if backend == "numpy":
        return NumpyGRU(input_size, hidden_size, seed=seed)
    if backend == "torch":
        return TorchGRU(input_size, hidden_size)
    if backend == "auto":
        return (
            TorchGRU(input_size, hidden_size)
            if TorchGRU.is_available()
            else NumpyGRU(input_size, hidden_size, seed=seed)
        )
    raise ValueError(f"Unknown backend '{backend}'.")
