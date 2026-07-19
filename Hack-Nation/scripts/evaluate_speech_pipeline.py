"""Run the speech pipeline over the scripted corpus and write metrics.

The whole pipeline runs offline: scripted transcription, rule-based extraction,
validation, simulated confirmation, encoding. No network, no API key, no model
download. That is a deliberate property — a metric you cannot reproduce on a
laptop with no credentials is a metric nobody re-checks.

Usage:
    python scripts/evaluate_speech_pipeline.py --config configs/data/speech_eval.yaml
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

from evaluation.speech import evaluate_corpus  # noqa: E402
from ingestion.speech.confirmation import ConfirmationSession, summarize  # noqa: E402
from ingestion.speech.extraction import RuleBasedExtractor  # noqa: E402
from ingestion.speech.transcription import ScriptedTranscriptionAdapter  # noqa: E402
from ingestion.speech.validation import drop_unsupported, validate_extractions  # noqa: E402
from models.speech.symptom_encoder import encode_speech_events  # noqa: E402
from schemas.event import HormonalHealthEvent  # noqa: E402
from scripts._cli import add_standard_arguments, make_parser, resolve_output_dir  # noqa: E402


def _gold_records(utterance: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize gold annotations into the evaluator's comparison shape."""
    records: list[dict[str, Any]] = []
    for event in utterance.get("events") or []:
        records.append(
            {
                "canonical_code": event["code"],
                "attribution": event.get("attribution", "patient"),
                "negated": bool(event.get("negated", False)),
                "historical": bool(event.get("historical", False)),
                "uncertain": bool(event.get("uncertain", False)),
                "speaker_role": event.get("speaker_role", utterance.get("speaker_role", "patient")),
                "value": event.get("value", True),
                "medication_action": event.get("medication_action"),
                "utterance_id": utterance["id"],
            }
        )
    return records


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    """Follow an experiment config's ``data_config`` reference, if it has one.

    Two kinds of YAML are valid here and the documented command uses the first:

    * an experiment config (``configs/experiments/exp_speech_extraction.yaml``),
      which names its corpus indirectly via ``data_config:``;
    * a data config (``configs/data/speech_eval.yaml``), which holds the corpus
      settings directly and remains this script's default.

    Previously only the second worked, while README.md and TRAINING.md both
    documented the first — running it raised ``KeyError: 'corpus_path'``.
    Data-config keys form the base; the experiment config overrides them, so an
    experiment can redirect its own outputs without editing shared data settings.
    """
    reference = config.get("data_config")
    if not reference:
        return config

    path = Path(reference)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open() as fh:
        base = yaml.safe_load(fh) or {}

    merged = {**base, **config}
    return merged


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Execute the pipeline and return the serializable report."""
    config = resolve_config(config)
    corpus_path = REPO_ROOT / config["corpus_path"]
    with corpus_path.open() as fh:
        corpus = yaml.safe_load(fh)

    patient_id = config.get("patient_id", "SYNTH-SPEECH-001")
    transcriber = ScriptedTranscriptionAdapter(
        words_per_second=float(config["transcription"].get("words_per_second", 2.5))
    )
    extractor = RuleBasedExtractor()

    gold_all: list[dict[str, Any]] = []
    pred_all: list[dict[str, Any]] = []
    transcript_pairs: list[tuple[str, str]] = []
    by_category: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    confirmed_events: list[HormonalHealthEvent] = []
    unmapped_total = 0
    unsupported_total = 0
    suppressed_questions = 0
    n_reviewed = 0
    n_corrected = 0

    for utterance in corpus["utterances"]:
        speaker_role = utterance.get("speaker_role", "patient")
        transcript = transcriber.transcribe(
            [{"speaker_role": speaker_role, "text": utterance["text"]}],
            recording_id=utterance["id"],
            language=corpus.get("language", "en"),
        )
        transcript_pairs.append((utterance["text"], transcript.text))

        result = extractor.extract(transcript, patient_id=patient_id)
        unmapped_total += len(result.unmapped)
        suppressed_questions += result.suppressed_questions

        supported, unsupported = drop_unsupported(transcript, result.events)
        unsupported_total += len(unsupported)
        report = validate_extractions(transcript, supported)
        rejected_ids = set(report.unsupported_extraction_ids)
        supported = [e for e in supported if e.extraction_id not in rejected_ids]

        gold = _gold_records(utterance)
        predicted = [
            {
                "canonical_code": e.canonical_code,
                "attribution": e.attribution,
                "negated": e.negated,
                "historical": e.historical,
                "uncertain": e.uncertain,
                "speaker_role": e.speaker_role,
                "value": e.value,
                "medication_action": e.medication_action,
                "utterance_id": utterance["id"],
            }
            for e in supported
        ]
        gold_all.extend(gold)
        pred_all.extend(predicted)

        category = utterance.get("category", "unspecified")
        cat_gold, cat_pred = by_category.setdefault(category, ([], []))
        cat_gold.extend(gold)
        cat_pred.extend(predicted)

        session = ConfirmationSession.from_result(
            result,
            source_dataset=config.get("dataset_id"),
            audio_quality=float(config["audio"]["assumed_quality_score"]),
        )
        # Only the extractions that survived validation are offered for review;
        # an ungrounded extraction must never reach a human as if it were real.
        session.items = {
            k: v for k, v in session.items.items() if any(s.extraction_id == k for s in supported)
        }
        if config["confirmation"].get("auto_confirm", True):
            session.confirm_all(config["confirmation"]["reviewer_id"])
        summary = summarize(session)
        n_reviewed += summary.n_confirmed + summary.n_rejected
        n_corrected += summary.n_corrected
        confirmed_events.extend(session.to_events())

    report = evaluate_corpus(
        gold_all,
        pred_all,
        corpus_id=corpus["corpus_id"],
        n_utterances=len(corpus["utterances"]),
        transcript_pairs=transcript_pairs,
        n_unsupported=unsupported_total,
        n_reviewed=n_reviewed,
        n_corrected=n_corrected,
        categories=by_category,
    )
    report.notes = [
        "Synthetic scripted corpus. No real speech. Supports no real-world performance claim.",
        f"{unmapped_total} mention(s) recognized but with no canonical patient variable.",
        f"{suppressed_questions} question(s) suppressed (a question asserts nothing).",
        "WER is 0.0 by construction with the scripted adapter; it is non-trivial only "
        "when a real ASR engine is substituted.",
        "Scores near 1.0 are expected and are NOT evidence of a good extractor: the "
        "scripted utterances were written within the lexicon's surface-form coverage. "
        "This run is a regression guard on negation, temporality and attribution logic, "
        "not a performance estimate.",
    ]

    token = encode_speech_events(
        confirmed_events,
        patient_id=config.get("patient_id", "SYNTH-SPEECH-001"),
        audio_quality=float(config["audio"]["assumed_quality_score"]),
        source_dataset=config.get("dataset_id"),
        n_unsupported=unsupported_total,
    )
    return {"report": report, "token": token, "n_confirmed_events": len(confirmed_events)}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Exposed so the CLI contract test can inspect it."""
    parser = make_parser(description=__doc__)
    add_standard_arguments(
        parser,
        config_default=REPO_ROOT / "configs/data/speech_eval.yaml",
        data_root=False,
        seed=False,
        experiment_id=False,
        quiet=False,
    )
    # File-level overrides. Not deprecated: they name individual files rather
    # than a directory, and they win over --output-dir when both are given.
    parser.add_argument("--metrics-out", type=Path, default=None, help="Metrics JSON path.")
    parser.add_argument("--token-out", type=Path, default=None, help="Symptom token JSON path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    with args.config.open() as fh:
        config = yaml.safe_load(fh)

    outcome = run(config)
    output = config.get("output", {}) or {}
    if args.output_dir is not None:
        base = resolve_output_dir(config, args.output_dir, experiment_id="speech_extraction")
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
                "n_utterances": report.n_utterances,
                "extraction_f1": report.symptom_extraction.f1,
                "negation_f1": report.negation.f1,
                "temporality_f1": report.temporality.f1,
                "medication_f1": report.medication_events.f1,
                "speaker_attribution_accuracy": report.speaker_attribution_accuracy,
                "unsupported_event_rate": report.unsupported_event_rate,
                "word_error_rate": report.word_error_rate,
                "n_confirmed_events": outcome["n_confirmed_events"],
                "metrics_path": str(metrics_path.relative_to(REPO_ROOT)),
                "token_path": str(token_path.relative_to(REPO_ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
