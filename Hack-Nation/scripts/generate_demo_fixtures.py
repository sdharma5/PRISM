"""Produce the frontend's demo fixtures by running real inference.

Mock mode must show what the models actually do. Hand-authored fixtures drift
from real behaviour and end up demonstrating a system that does not exist -- a
dominant profile the stability gate would withhold, an axis met on evidence the
rules would reject. So the fixtures are generated: each demo patient in
``apps/api/demo_patients.py`` is run through the same orchestrator, adapter and
mapper that serve ``POST /api/v1/patients/infer``, and the response is written
out verbatim.

Usage::

    python scripts/generate_demo_fixtures.py
    python scripts/generate_demo_fixtures.py --check   # CI: fail if stale

Regenerate whenever the models, the mapper, or the demo inputs change.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apps.api.demo_patients import DEMO_PATIENTS, DemoPatient
from apps.api.registry import ModelRegistry
from apps.api.schemas.requests import PatientInferenceRequest
from inference.presentation.website_mapper import to_website_response

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT = _ROOT.parent / "UI/prism-app/src/lib/demo"

#: Fixed so regenerating without a model change produces an identical file.
#: A moving timestamp would make every regeneration look like a content change.
_FIXED_TIMESTAMP = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)


def _temporal_days(patient: DemoPatient) -> list[dict[str, Any]]:
    """Deterministic longitudinal days for a demo patient.

    Deliberately arithmetic rather than random: a fixture that changes every
    time it is generated cannot be reviewed in a diff, and "do not use random
    mock values" applies to the temporal branch too.
    """
    if patient.temporal_days <= 0:
        return []

    days: list[dict[str, Any]] = []
    irregular = patient.temporal_profile == "irregular"
    period = 41 if irregular else 28

    for i in range(patient.temporal_days):
        phase = i % period
        # A crude but stable cycle shape: LH rises near mid-cycle, PdG after it.
        lh = 4.0 + 9.0 * (1.0 if abs(phase - period // 2) <= 1 else 0.0) + (phase % 3) * 0.4
        e3g = 30.0 + 1.6 * phase
        pdg = 2.0 + (4.5 if phase > period // 2 else 0.5) + (phase % 4) * 0.2
        days.append(
            {
                "participant_id": patient.patient_id,
                "study_day": i,
                "cycle_day": phase + 1,
                "values": {
                    "urinary_lh": round(lh, 3),
                    "e3g": round(e3g, 3),
                    "pdg": round(pdg, 3),
                },
                "is_observed": {"urinary_lh": True, "e3g": True, "pdg": True},
                "time_since_last_observed": {"urinary_lh": 0.0, "e3g": 0.0, "pdg": 0.0},
            }
        )
    return days


def build_fixture(registry: ModelRegistry, patient: DemoPatient) -> dict[str, Any]:
    """Run one demo patient through the real pipeline."""
    request = PatientInferenceRequest(
        patient_id=patient.patient_id,
        clinical_features=patient.clinical_features,
        temporal_observations=_temporal_days(patient),  # type: ignore[arg-type]
    )
    report = registry.orchestrator.run(request.to_bundle())
    response = to_website_response(
        report,
        generated_at=_FIXED_TIMESTAMP,
        report_id=f"rpt_demo_{patient.key}",
    )
    return {
        "key": patient.key,
        "display_name": patient.display_name,
        "summary": patient.summary,
        "exercises": list(patient.exercises),
        "notes": list(patient.notes),
        "response": response.model_dump(mode="json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    registry = ModelRegistry.load()
    args.output.mkdir(parents=True, exist_ok=True)

    stale: list[str] = []
    index: list[dict[str, Any]] = []

    for patient in DEMO_PATIENTS:
        fixture = build_fixture(registry, patient)
        rendered = json.dumps(fixture, indent=2, sort_keys=True) + "\n"
        path = args.output / f"{patient.key}.json"

        if args.check:
            if not path.exists() or path.read_text() != rendered:
                stale.append(str(path))
        else:
            path.write_text(rendered)

        assessment = fixture["response"]["pcos_assessment"]
        index.append(
            {
                "key": patient.key,
                "display_name": patient.display_name,
                "summary": patient.summary,
                "evidence_level": assessment["evidence_level"],
                "score_available": assessment["available"],
            }
        )
        print(
            f"{patient.key:8} evidence={assessment['evidence_level']:14} "
            f"score={assessment['calibrated_model_score']}"
        )

    index_rendered = json.dumps(index, indent=2, sort_keys=True) + "\n"
    index_path = args.output / "index.json"
    if args.check:
        if not index_path.exists() or index_path.read_text() != index_rendered:
            stale.append(str(index_path))
        if stale:
            print("Stale demo fixtures:\n  " + "\n  ".join(stale), file=sys.stderr)
            return 1
        print("Demo fixtures are up to date.")
        return 0

    index_path.write_text(index_rendered)
    print(f"Wrote {len(DEMO_PATIENTS)} fixtures to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
