#!/usr/bin/env python
"""Consolidate the mcPHASES per-stream CSVs into model-ready participant-days.

    python scripts/consolidate_mcphases.py --data-root $PRISM_DATA_ROOT/mcphases/raw

Why this exists
---------------
The official mcPHASES distribution ships ~25 CSVs, one per stream, each with its
own column names and a ``day_in_study`` time base rather than absolute
timestamps. Nothing in the repository consumed that shape:

* ``McPhasesAdapter`` expects ONE long-format table with ``participant_id`` and
  ``timestamp`` columns, so no mcPHASES file is directly ingestible; and
* its output keys are ``{canonical_code}_{statistic}`` (``cgm_mean_glucose_mean``
  and so on), whereas ``TemporalStateModel`` reads the bare channel names
  ``lh, e3g, pdg, resting_heart_rate, wrist_temperature, hrv_rmssd,
  mean_glucose`` that ``tests/fixtures/synthetic_cycles.py`` produces.

Those two vocabularies never met, because the only cohort Step 9 had ever run on
was the synthetic one, which bypasses the adapter entirely. This script targets
the vocabulary the *model* actually reads, and writes ``participant_days.jsonl``
directly.

Source selection is deliberate
------------------------------
``heart_rate.csv`` (63M rows) is instantaneous heart rate, not resting heart
rate, so ``resting_heart_rate.csv`` is used instead — it is both the correct
quantity and four orders of magnitude smaller.

``wrist_temperature.csv`` carries ``temperature_diff_from_baseline`` (values
around +/-0.02), a BASELINE-RELATIVE DELTA. The model's ``wrist_temperature``
channel is an absolute temperature near 33.5 C. Feeding the delta into that
channel would put a different physical quantity on the same input, so the
absolute nightly skin temperature from ``computed_temperature.csv``
(``type == "SKIN"``, mean 33.7 C) is used instead.

Unit conversion
---------------
Every value is converted through ``registry.loader.convert_to_canonical``, so
the factors live in ``registry/units.yaml`` where they are unit-tested once,
and an unknown unit pairing raises instead of passing through untouched.

Two traps this dataset sets, both caught by the range audit below:

* **Glucose is mixed-unit across participants.** Most report mmol/L (median ~6)
  but two report mg/dL (median ~110). Declaring one unit for the file multiplies
  those two by 18 into impossible ~2000 mg/dL readings, so the unit is detected
  per participant.
* **Resting heart rate uses 0.0 as a missing marker** (with ``error`` also 0.0,
  where real readings carry error >= 6.79). Ingested as measurements those 1,380
  zeros pull the cohort mean from ~70 down to 63.75.

After conversion every value is checked against the registry's ``valid_range``
and the run is refused if more than ``MAX_OUT_OF_RANGE_FRACTION`` of a channel
falls outside it. That check is what turned both traps above from silent
corruption into a stop.

Study intervals
---------------
``day_in_study`` is a single continuous counter per person, but the study ran in
two blocks (days 1-90 in 2022, days 838-1004 in 2024). Days are emitted densely
*within* each block and not at all between them: materialising the 747-day gap
would invent three-quarters of the series, and a 14-30 day model window must
never straddle it. ``participant_id`` remains the person, so a grouped split
still keeps both of one person's blocks on the same side.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingestion.base import file_checksum  # noqa: E402
from ingestion.mcphases.alignment import derive_time_since_last_observed  # noqa: E402
from ingestion.mcphases.validation import assert_use_permitted  # noqa: E402
from registry.loader import (  # noqa: E402
    convert_to_canonical,
    in_valid_range,
    load_variable_registry,
)
from schemas.temporal import ParticipantDay  # noqa: E402
from scripts._cli import add_standard_arguments, make_parser  # noqa: E402

DATASET_ID = "mcphases"

#: mcPHASES cycle phase -> the schema's CyclePhase literal.
PHASE_MAP: dict[str, str] = {
    "Menstrual": "menstrual",
    "Follicular": "follicular",
    "Fertility": "peri_ovulatory",
    "Luteal": "luteal",
}

#: mcPHASES self-report column -> the model's symptom name.
SYMPTOM_MAP: dict[str, str] = {
    "cramps": "cramps",
    "bloating": "bloating",
    "moodswing": "mood_low",
    "sorebreasts": "breast_tenderness",
}

#: Self-report intensities counted as the symptom being present. The scale is
#: ordinal text, and the model wants a boolean, so the cut has to be stated
#: somewhere; it is stated here rather than buried in a comprehension.
SYMPTOM_PRESENT_AT_OR_ABOVE: tuple[str, ...] = ("Moderate", "High", "Very High")

#: Model channel -> (canonical registry code, the unit mcPHASES reports it in).
#:
#: Declaring the SOURCE unit here and converting through
#: :func:`registry.loader.convert_to_canonical` is what keeps conversion
#: honest. Factors live in registry/units.yaml, are unit-tested centrally, and
#: raise on an unknown pairing rather than passing the value through unchanged.
#: An earlier revision hardcoded 18.016 in this file, silently duplicating the
#: registry's own `cgm_mean_glucose` entry — the kind of copy that survives
#: right up until the registry is corrected and this file is not.
CHANNEL_SPECS: dict[str, tuple[str, str]] = {
    "lh": ("luteinizing_hormone", "mIU/mL"),
    "e3g": ("e3g", "ng/mL"),
    "pdg": ("pdg", "ug/mL"),
    "resting_heart_rate": ("resting_heart_rate", "bpm"),
    "wrist_temperature": ("skin_temperature", "degC"),
    "hrv_rmssd": ("hrv_rmssd", "ms"),
    "mean_glucose": ("cgm_mean_glucose", "mmol/L"),
}

#: Every channel the temporal model reads, in its own vocabulary.
CHANNELS: tuple[str, ...] = tuple(CHANNEL_SPECS)

#: Fraction of a channel's values that may fall outside the registry's declared
#: valid range before consolidation is refused.
#:
#: This is the backstop for a wrong unit. Real physiological outliers are rare;
#: a channel converted with the wrong factor (or not converted at all) lands
#: almost entirely outside its range. Unconverted mmol/L glucose reads as ~5
#: against a 20-600 mg/dL range, so it fails at essentially 100% rather than
#: flowing into training as a plausible-looking number.
MAX_OUT_OF_RANGE_FRACTION = 0.10


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser, following the shared CLI contract."""
    parser = make_parser(description=__doc__)
    # No --config: this script reads no analyst choices from one, and _cli.py's
    # rule is that a flag which parses but does nothing is worse than none.
    add_standard_arguments(parser, config=False, seed=False, experiment_id=False)
    return parser


def _read(root: Path, name: str, columns: list[str]) -> pd.DataFrame:
    """Read the needed columns of one stream file, or return an empty frame."""
    path = root / name
    if not path.exists():
        print(f"  WARNING: {name} absent; its channels stay unobserved.", file=sys.stderr)
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path, usecols=lambda c: c in set(columns))


def _daily_mean(frame: pd.DataFrame, day_col: str, value_col: str) -> pd.DataFrame:
    """Collapse a high-frequency stream to one mean per participant-day."""
    if frame.empty:
        return pd.DataFrame(columns=["id", "day", "value"])
    out = frame[["id", day_col, value_col]].copy()
    out.columns = ["id", "day", "value"]
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])
    return out.groupby(["id", "day"], as_index=False)["value"].mean()


def collect_channels(
    root: Path,
) -> tuple[dict[tuple[Any, int], dict[str, float]], pd.DataFrame, dict[str, int], dict[Any, str]]:
    """Read every source stream into a {(participant, day): {channel: value}} map."""
    per_day: dict[tuple[Any, int], dict[str, float]] = {}
    sentinel_drops: dict[str, int] = {}

    def absorb(
        frame: pd.DataFrame, channel: str, unit_by_participant: dict[Any, str] | None = None
    ) -> None:
        """Convert one stream into canonical units and index it by day."""
        code, declared_unit = CHANNEL_SPECS[channel]
        for row in frame.itertuples(index=False):
            unit = (unit_by_participant or {}).get(row.id, declared_unit)
            # convert_to_canonical raises on an unknown unit pairing rather than
            # returning the value untouched, so a mapping mistake here is loud.
            result = convert_to_canonical(code, float(row.value), unit)
            per_day.setdefault((row.id, int(row.day)), {})[channel] = result.value

    hormones = _read(
        root,
        "hormones_and_selfreport.csv",
        ["id", "study_interval", "day_in_study", "phase", "lh", "estrogen", "pdg", *SYMPTOM_MAP],
    )
    for source_col, channel in (("lh", "lh"), ("estrogen", "e3g"), ("pdg", "pdg")):
        if source_col in hormones.columns:
            absorb(_daily_mean(hormones, "day_in_study", source_col), channel)

    # 1,380 of 13,737 resting-heart-rate rows carry value == 0.0 with error == 0.0,
    # while every genuine reading has error >= 6.79. Zero bpm is not a measurement,
    # it is this file's missing marker. Dropping it here — rather than letting the
    # range check discard it later — keeps the range check meaningful: a blanket
    # "drop anything out of range" rule would also swallow the wrong-unit errors
    # that check exists to catch.
    rhr = _read(root, "resting_heart_rate.csv", ["id", "day_in_study", "value", "error"])
    if not rhr.empty:
        sentinel = (pd.to_numeric(rhr["value"], errors="coerce") == 0.0) & (
            pd.to_numeric(rhr["error"], errors="coerce") == 0.0
        )
        sentinel_drops["resting_heart_rate"] = int(sentinel.sum())
        rhr = rhr[~sentinel]
    absorb(_daily_mean(rhr, "day_in_study", "value"), "resting_heart_rate")

    temp = _read(
        root,
        "computed_temperature.csv",
        ["id", "sleep_start_day_in_study", "type", "nightly_temperature"],
    )
    if not temp.empty and "type" in temp.columns:
        temp = temp[temp["type"] == "SKIN"]
    absorb(
        _daily_mean(temp, "sleep_start_day_in_study", "nightly_temperature"), "wrist_temperature"
    )

    hrv = _read(root, "heart_rate_variability_details.csv", ["id", "day_in_study", "rmssd"])
    absorb(_daily_mean(hrv, "day_in_study", "rmssd"), "hrv_rmssd")

    glucose = _read(root, "glucose.csv", ["id", "day_in_study", "glucose_value"])
    daily_glucose = _daily_mean(glucose, "day_in_study", "glucose_value")
    glucose_units = detect_glucose_units(daily_glucose)
    absorb(daily_glucose, "mean_glucose", glucose_units)
    mixed = {str(k): v for k, v in glucose_units.items() if v != CHANNEL_SPECS["mean_glucose"][1]}
    if mixed:
        print(
            f"  NOTE: {len(mixed)} participant(s) report glucose in mg/dL, not mmol/L: "
            f"{', '.join(sorted(mixed))}",
            file=sys.stderr,
        )

    return per_day, hormones, sentinel_drops, glucose_units


#: A participant-median glucose above this is mg/dL, not mmol/L. Sustained
#: 25 mmol/L is ~450 mg/dL and not survivable as a median; observed mg/dL
#: medians sit near 110. The two populations are three-fold apart with nothing
#: in between, so the cut is not delicate.
GLUCOSE_MG_DL_MEDIAN_THRESHOLD = 25.0


def detect_glucose_units(daily_glucose: pd.DataFrame) -> dict[Any, str]:
    """Decide, per participant, whether glucose is reported in mmol/L or mg/dL.

    mcPHASES is MIXED-UNIT: most participants report mmol/L (median ~6) but a
    minority report mg/dL (median ~110). Declaring one unit for the file and
    converting everything multiplies the mg/dL participants by 18 and yields
    impossible values around 2000 mg/dL — which is what the first version of
    this script did. Per-participant detection is not a nicety here.
    """
    if daily_glucose.empty:
        return {}
    medians = daily_glucose.groupby("id")["value"].median()
    return {
        participant: ("mg/dL" if median > GLUCOSE_MG_DL_MEDIAN_THRESHOLD else "mmol/L")
        for participant, median in medians.items()
    }


def _unit_summary(units_by_participant: dict[Any, str]) -> str:
    """Describe a per-participant unit assignment for the manifest."""
    if not units_by_participant:
        return "unknown"
    counts: dict[str, int] = {}
    for unit in units_by_participant.values():
        counts[unit] = counts.get(unit, 0) + 1
    return ", ".join(f"{unit} x{n}" for unit, n in sorted(counts.items()))


def range_audit(per_day: dict[tuple[Any, int], dict[str, float]]) -> dict[str, dict[str, Any]]:
    """Check every converted value against the registry's declared valid range.

    Returns one entry per channel carrying the counts and the offending sample.
    Channels the registry gives no range are reported as ``checked=0`` rather
    than silently passing, so an unranged channel is visible instead of assumed
    correct.
    """
    audit: dict[str, dict[str, Any]] = {}
    for channel, (code, source_unit) in CHANNEL_SPECS.items():
        values = [cell[channel] for cell in per_day.values() if channel in cell]
        spec = load_variable_registry().variables.get(code)
        has_range = spec is not None and spec.valid_range is not None
        outside = [v for v in values if not in_valid_range(code, v)] if has_range else []
        audit[channel] = {
            "canonical_code": code,
            "source_unit": source_unit,
            "n_values": len(values),
            "checked": len(values) if has_range else 0,
            "n_outside_valid_range": len(outside),
            "fraction_outside": round(len(outside) / len(values), 4) if values else 0.0,
            "example_outside": round(outside[0], 4) if outside else None,
        }
    return audit


def assert_units_are_sane(audit: dict[str, dict[str, Any]]) -> None:
    """Refuse to write a cohort whose values contradict their declared units."""
    offenders = {
        channel: entry
        for channel, entry in audit.items()
        if entry["checked"] and entry["fraction_outside"] > MAX_OUT_OF_RANGE_FRACTION
    }
    # A channel the registry gives no valid_range cannot be checked at all. Say
    # so: "no range declared" and "range checked and passed" must not look alike
    # in the output, or an unverifiable channel reads as a verified one.
    unranged = [
        f"{channel} ({entry['canonical_code']})"
        for channel, entry in audit.items()
        if entry["n_values"] and not entry["checked"]
    ]
    if unranged:
        print(
            f"  WARNING: no valid_range in registry/variables.yaml for {', '.join(unranged)}; "
            "their units are NOT verified. Add a range to close this gap.",
            file=sys.stderr,
        )

    if not offenders:
        return
    detail = "\n".join(
        f"  {channel}: {entry['fraction_outside']:.1%} of {entry['checked']} values outside the "
        f"valid range for '{entry['canonical_code']}' (e.g. {entry['example_outside']}), "
        f"declared source unit '{entry['source_unit']}'"
        for channel, entry in offenders.items()
    )
    raise SystemExit(
        "ERROR: consolidated values contradict their declared units.\n"
        f"{detail}\n"
        "\n"
        "This almost always means the source unit declared in CHANNEL_SPECS is wrong, or "
        "registry/units.yaml has no conversion for that pairing and the value passed through "
        "unchanged. Fix the declaration or the registry — do not raise the threshold."
    )


def _symptoms_and_phase(hormones: pd.DataFrame) -> dict[tuple[Any, int], dict[str, Any]]:
    """Index self-reported symptoms and cycle phase by (participant, day)."""
    annotations: dict[tuple[Any, int], dict[str, Any]] = {}
    if hormones.empty:
        return annotations
    present = set(SYMPTOM_PRESENT_AT_OR_ABOVE)
    for row in hormones.itertuples(index=False):
        key = (row.id, int(row.day_in_study))
        symptoms = {
            name: (str(getattr(row, col, "")) in present) for col, name in SYMPTOM_MAP.items()
        }
        annotations[key] = {
            "symptoms": symptoms,
            "phase": PHASE_MAP.get(str(getattr(row, "phase", "")), "unknown"),
        }
    return annotations


def consolidate(root: Path) -> tuple[list[ParticipantDay], dict[str, dict[str, Any]]]:
    """Build participant-days and the unit audit that had to pass to produce them."""
    per_day, hormones, sentinel_drops, glucose_units = collect_channels(root)
    if not per_day:
        raise SystemExit(f"ERROR: no usable stream files under {root}")

    # Audit before building: a unit error must stop the run, not be discovered
    # in a manifest after the cohort has already been written.
    audit = range_audit(per_day)
    for channel, dropped in sentinel_drops.items():
        audit[channel]["n_missing_sentinels_dropped"] = dropped
    audit["mean_glucose"]["source_unit"] = _unit_summary(glucose_units)
    assert_units_are_sane(audit)

    return _days_from(per_day, hormones), audit


def build_participant_days(root: Path) -> list[ParticipantDay]:
    """Build dense-within-block participant-days for every mcPHASES participant."""
    return consolidate(root)[0]


def _days_from(
    per_day: dict[tuple[Any, int], dict[str, float]], hormones: pd.DataFrame
) -> list[ParticipantDay]:
    """Assemble participant-days from the per-day channel map."""
    annotations = _symptoms_and_phase(hormones)

    by_participant: dict[Any, list[int]] = {}
    for participant, day in per_day:
        by_participant.setdefault(participant, []).append(day)

    days: list[ParticipantDay] = []
    for participant in sorted(by_participant, key=str):
        observed_days = sorted(set(by_participant[participant]))

        # Split the person's days into contiguous blocks, allowing short
        # within-block gaps but never bridging the multi-year study break.
        blocks: list[list[int]] = [[observed_days[0]]]
        for day in observed_days[1:]:
            if day - blocks[-1][-1] > 60:
                blocks.append([day])
            else:
                blocks[-1].append(day)

        for block in blocks:
            span = list(range(block[0], block[-1] + 1))
            observations = {
                channel: [channel in per_day.get((participant, d), {}) for d in span]
                for channel in CHANNELS
            }
            gaps = {
                channel: derive_time_since_last_observed(flags)
                for channel, flags in observations.items()
            }
            for offset, day in enumerate(span):
                cell = per_day.get((participant, day), {})
                note = annotations.get((participant, day), {})
                days.append(
                    ParticipantDay(
                        participant_id=f"{DATASET_ID}:{participant}",
                        study_day=day,
                        cycle_phase=note.get("phase", "unknown"),
                        values={c: cell.get(c) for c in CHANNELS},
                        is_observed={c: observations[c][offset] for c in CHANNELS},
                        time_since_last_observed={c: gaps[c][offset] for c in CHANNELS},
                        daily_symptoms=note.get("symptoms", {}),
                        source_dataset=DATASET_ID,
                    )
                )
    return days


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    assert_use_permitted("temporal_state_model", DATASET_ID)

    if args.data_root is None:
        print(
            "ERROR: --data-root is required and must point at the mcPHASES raw directory, "
            "e.g. --data-root $PRISM_DATA_ROOT/mcphases/raw",
            file=sys.stderr,
        )
        return 1
    root = Path(args.data_root).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: mcPHASES root does not exist: {root}", file=sys.stderr)
        return 1

    days, unit_audit = consolidate(root)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else root
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "participant_days.jsonl"
    with target.open("w") as fh:
        for day in days:
            fh.write(day.model_dump_json() + "\n")

    sources = [
        "hormones_and_selfreport.csv",
        "resting_heart_rate.csv",
        "computed_temperature.csv",
        "heart_rate_variability_details.csv",
        "glucose.csv",
    ]
    coverage = {
        channel: round(sum(d.is_observed.get(channel, False) for d in days) / len(days), 4)
        for channel in CHANNELS
    }
    manifest = {
        "dataset_id": DATASET_ID,
        "source_root": str(root),
        "n_participant_days": len(days),
        "n_participants": len({d.participant_id for d in days}),
        "channels": list(CHANNELS),
        "channel_observed_fraction": coverage,
        # Conversions come from registry/units.yaml; recording the audit rather
        # than a list of factors means the manifest reflects what was actually
        # applied and range-checked, not what this file believes should be.
        "unit_audit": unit_audit,
        "symptom_present_at_or_above": list(SYMPTOM_PRESENT_AT_OR_ABOVE),
        "source_checksums": {
            name: file_checksum(root / name) for name in sources if (root / name).exists()
        },
        "excluded_sources": {
            "heart_rate.csv": "Instantaneous heart rate, not resting; 63M rows.",
            "wrist_temperature.csv": (
                "Carries a baseline-relative delta, not the absolute skin temperature "
                "the wrist_temperature channel represents."
            ),
        },
    }
    (output_dir / "consolidation_manifest.json").write_text(json.dumps(manifest, indent=2))

    if not args.quiet:
        print(f"Wrote {target}")
        print(f"  {len(days)} participant-days across {manifest['n_participants']} participants")
        for channel, fraction in coverage.items():
            print(f"    {channel:22} observed on {fraction:6.1%} of days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
