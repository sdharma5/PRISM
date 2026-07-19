"""Validate the scripted speech corpus and materialize its derived JSONL form.

WHY A BUILD STEP AT ALL: the YAML corpus is the human-authored source of truth,
but the evaluator wants a flat, one-record-per-utterance stream, and CI wants an
early failure when an annotation references a canonical code that no longer
exists in the registry. Both come from here, so the YAML never drifts silently.

Usage:
    python scripts/build_speech_eval_set.py --config configs/data/speech_eval.yaml
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

from registry.loader import load_variable_registry  # noqa: E402
from scripts._cli import (  # noqa: E402
    add_deprecated_alias,
    add_standard_arguments,
    make_parser,
    resolve_output_dir,
)

REQUIRED_CATEGORIES = {
    "present",
    "negated",
    "historical",
    "uncertain",
    "family_history",
    "medication_start",
    "medication_stop",
    "cycle_timing",
    "fertility_goal",
    "clinician_question",
    "speaker_confusion",
    "approximate_date",
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh)


def load_corpus(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh)


def validate_corpus(corpus: dict[str, Any], *, min_utterances: int) -> list[str]:
    """Return a list of problems. An empty list means the corpus is usable."""
    problems: list[str] = []
    registry = load_variable_registry().variables
    utterances = corpus.get("utterances", [])

    if not corpus.get("synthetic"):
        problems.append("corpus must declare synthetic: true")
    if len(utterances) < min_utterances:
        problems.append(f"only {len(utterances)} utterances; at least {min_utterances} required")

    seen_ids: set[str] = set()
    categories: set[str] = set()
    for utterance in utterances:
        uid = utterance.get("id", "<missing id>")
        if uid in seen_ids:
            problems.append(f"{uid}: duplicate utterance id")
        seen_ids.add(uid)
        categories.add(str(utterance.get("category", "")))
        if not str(utterance.get("text", "")).strip():
            problems.append(f"{uid}: empty text")
        for event in utterance.get("events") or []:
            code = event.get("code")
            if code not in registry:
                problems.append(f"{uid}: '{code}' is not in registry/variables.yaml")
            if str(code).startswith("family_history_") and event.get("attribution") != (
                "family_member"
            ):
                problems.append(f"{uid}: family-history code without family_member attribution")

    missing = REQUIRED_CATEGORIES - categories
    if missing:
        problems.append(f"corpus is missing required categories: {sorted(missing)}")
    return problems


def write_jsonl(corpus: dict[str, Any], destination: Path) -> int:
    """Flatten to JSONL: one utterance with its gold annotations per line."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as fh:
        for utterance in corpus["utterances"]:
            record = {
                "id": utterance["id"],
                "category": utterance.get("category", "unspecified"),
                "speaker_role": utterance.get("speaker_role", "patient"),
                "text": utterance["text"],
                "events": utterance.get("events") or [],
                "synthetic": True,
            }
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    return len(corpus["utterances"])


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
    # '--out' was this script's original name for the JSONL destination.
    add_deprecated_alias(parser, "--out", dest="output_dir", replacement="--output-dir", type=Path)
    parser.add_argument(
        "--min-utterances",
        type=int,
        default=50,
        help="Reject the corpus if it holds fewer utterances than this.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = load_config(args.config)
    corpus_path = REPO_ROOT / config["corpus_path"]
    corpus = load_corpus(corpus_path)

    problems = validate_corpus(corpus, min_utterances=args.min_utterances)
    if problems:
        for problem in problems:
            print(f"ERROR {problem}")
        return 1

    configured = REPO_ROOT / config["jsonl_path"]
    if args.output_dir is None:
        destination = configured
    elif args.output_dir.suffix == ".jsonl":
        # '--out' historically named the file itself; keep that meaning.
        destination = args.output_dir
    else:
        destination = (
            resolve_output_dir(config, args.output_dir, experiment_id="speech_eval")
            / configured.name
        )
    count = write_jsonl(corpus, destination)
    summary = {
        "corpus_id": corpus["corpus_id"],
        "version": corpus["version"],
        "n_utterances": count,
        "n_gold_events": sum(len(u.get("events") or []) for u in corpus["utterances"]),
        "categories": sorted({u.get("category", "") for u in corpus["utterances"]}),
        "jsonl_path": str(destination.relative_to(REPO_ROOT)),
        "synthetic": True,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
