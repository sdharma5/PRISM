"""Run the document pipeline over the synthetic report corpus and write metrics.

Like the speech evaluation, this runs entirely offline: the committed reports are
plain text with page markers, so no PDF library is needed to reproduce these
numbers.

Usage:
    python scripts/evaluate_document_pipeline.py \
        --config configs/experiments/exp_document_extraction.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.documents import evaluate_documents  # noqa: E402
from ingestion.documents.lab_extractor import LabExtractor, to_events  # noqa: E402
from ingestion.documents.parser import TextFixtureParser  # noqa: E402
from ingestion.documents.report_extractor import (  # noqa: E402
    ReportExtractor,
    findings_to_events,
)
from ingestion.documents.validation import drop_ungrounded, validate_lab_results  # noqa: E402
from models.documents.evidence_encoder import encode_document_events  # noqa: E402
from schemas.event import HormonalHealthEvent  # noqa: E402
from scripts._cli import add_standard_arguments, make_parser, resolve_output_dir  # noqa: E402


def _predicted_record(result: Any) -> dict[str, Any]:
    """Flatten an ExtractedLabResult into the evaluator's comparison shape."""
    reference = result.reference_range
    return {
        "document_id": result.document_id,
        "canonical_code": result.canonical_code,
        "source_test_name": result.source_test_name,
        "value_source": result.value_source,
        "unit_source": result.unit_source,
        "value_canonical": result.value_canonical,
        "unit_canonical": result.unit_canonical,
        "conversion_applied": result.conversion_applied,
        "reference_low": reference.low if reference else None,
        "reference_high": reference.high if reference else None,
        "collected_date": result.collected_date,
        "reported_date": result.reported_date,
        "page": result.page_number,
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Execute the document pipeline and return the serializable outcome."""
    fixtures_dir = REPO_ROOT / config["fixtures_dir"]
    with (fixtures_dir / config["ground_truth"]).open() as fh:
        truth = yaml.safe_load(fh)

    parser = TextFixtureParser()
    lab_extractor = LabExtractor()
    report_extractor = ReportExtractor()

    predicted: list[dict[str, Any]] = []
    finding_predictions: list[dict[str, Any]] = []
    events: list[HormonalHealthEvent] = []
    n_unsupported = 0
    n_unmapped = 0
    n_validation_errors = 0
    grounding_checked = 0
    grounding_ok = 0

    for spec in truth["documents"]:
        document = parser.parse(fixtures_dir / spec["path"], document_id=spec["document_id"])
        patient_id = f"{truth['patient_id_prefix']}-{spec['document_id']}"

        if spec["kind"] in {"lab", "summary"}:
            lab_result = lab_extractor.extract(document, patient_id=patient_id)
            n_unmapped += len(lab_result.unmapped)
            n_unsupported += len(lab_result.unsupported)

            grounded, ungrounded = drop_ungrounded(document, lab_result.results)
            n_unsupported += len(ungrounded)
            grounding_checked += len(lab_result.results)
            grounding_ok += len(grounded)

            report = validate_lab_results(document, grounded)
            n_validation_errors += report.n_errors

            predicted.extend(_predicted_record(r) for r in grounded)
            events.extend(to_events(grounded, source_dataset=config.get("dataset_id")))

        if spec["kind"] in {"ultrasound", "summary"}:
            findings = report_extractor.extract(document, patient_id=patient_id)
            modality = "ultrasound_report" if spec["kind"] == "ultrasound" else "clinical_document"
            for finding in findings.findings:
                finding_predictions.append(
                    {
                        "document_id": finding.document_id,
                        "canonical_code": finding.canonical_code,
                        "value": finding.value,
                        "side": finding.side,
                        "page": finding.page_number,
                    }
                )
            events.extend(
                findings_to_events(
                    findings.findings,
                    source_dataset=config.get("dataset_id"),
                    modality=modality,
                )
            )

    gold = truth["lab_values"]
    report = evaluate_documents(
        gold,
        predicted,
        corpus_id=truth["corpus_id"],
        n_documents=len(truth["documents"]),
        n_unsupported=n_unsupported,
        n_unmapped=n_unmapped,
    )

    finding_score = _score_findings(truth["report_findings"], finding_predictions)
    report.notes = [
        "Synthetic de-identified reports. No real patient data. Supports no real-world claim.",
        f"report-finding F1 (ultrasound/summary): {finding_score['f1']} "
        f"over {finding_score['support']} gold findings.",
        f"page-grounding checked on {grounding_checked} value(s); {grounding_ok} grounded.",
        f"{n_validation_errors} validation error(s) raised.",
        "Layout variety (table / line / two-page) is scripted, so these numbers are a "
        "regression guard on parsing and unit handling, not a real-world estimate.",
    ]

    token = encode_document_events(
        events,
        patient_id=config.get("token_patient_id", "SYNTH-DOC-lab_0001"),
        source_dataset=config.get("dataset_id"),
        n_unsupported=n_unsupported,
        allow_unconfirmed=bool(config.get("allow_unconfirmed", True)),
    )
    return {
        "report": report,
        "token": token,
        "finding_score": finding_score,
        "n_events": len(events),
    }


def _score_findings(gold: list[dict[str, Any]], predicted: list[dict[str, Any]]) -> dict[str, Any]:
    """Simple set-based F1 for ultrasound findings, keyed by (doc, code, side)."""

    def key(item: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(item["document_id"]),
            str(item["canonical_code"]),
            str(item.get("side", "unspecified")),
        )

    remaining = list(predicted)
    tp = 0
    for gold_item in gold:
        match = next(
            (
                p
                for p in remaining
                if key(p) == key(gold_item) and str(p["value"]) == str(gold_item["value"])
            ),
            None,
        )
        if match is not None:
            remaining.remove(match)
            tp += 1
    fp, fn = len(remaining), len(gold) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": len(gold),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Exposed so the CLI contract test can inspect it."""
    parser = make_parser(description=__doc__)
    add_standard_arguments(
        parser,
        config_default=REPO_ROOT / "configs/experiments/exp_document_extraction.yaml",
        data_root=False,
        seed=False,
        experiment_id=False,
        quiet=False,
    )
    # File-level overrides. Not deprecated: they name individual files rather
    # than a directory, and they win over --output-dir when both are given.
    parser.add_argument("--metrics-out", type=Path, default=None, help="Metrics JSON path.")
    parser.add_argument("--token-out", type=Path, default=None, help="Lab token JSON path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    with args.config.open() as fh:
        config = yaml.safe_load(fh)

    outcome = run(config)
    output = config.get("output", {}) or {}
    if args.output_dir is not None:
        base = resolve_output_dir(config, args.output_dir, experiment_id="document_extraction")
        default_metrics = base / Path(output["metrics_path"]).name
        default_token = base / Path(output["token_path"]).name
    else:
        default_metrics = REPO_ROOT / output["metrics_path"]
        default_token = REPO_ROOT / output["token_path"]
    metrics_path = args.metrics_out or default_metrics
    token_path = args.token_out or default_token

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(outcome["report"].model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    outcome["token"].write_json(token_path)

    report = outcome["report"]
    print(
        json.dumps(
            {
                "corpus_id": report.corpus_id,
                "n_documents": report.n_documents,
                "n_gold_values": report.n_gold_values,
                "n_extracted_values": report.n_extracted_values,
                "test_name_f1": report.test_name.f1,
                "source_value_f1": report.source_value.f1,
                "canonical_value_f1": report.canonical_value.f1,
                "unit_f1": report.unit.f1,
                "collected_date_f1": report.collected_date.f1,
                "reference_range_f1": report.reference_range.f1,
                "page_grounding_accuracy": report.page_grounding_accuracy,
                "unsupported_value_rate": report.unsupported_value_rate,
                "unit_conversion_rate": report.unit_conversion_rate,
                "report_finding_f1": outcome["finding_score"]["f1"],
                "n_events": outcome["n_events"],
                "metrics_path": str(metrics_path.relative_to(REPO_ROOT)),
                "token_path": str(token_path.relative_to(REPO_ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
