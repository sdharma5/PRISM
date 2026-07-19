#!/usr/bin/env python3
"""Regenerate the auto-generated sections of the documentation.

Only content between AUTO-GENERATED markers is ever replaced:

    <!-- AUTO-GENERATED: DATASET-REGISTRY START -->
    ...
    <!-- AUTO-GENERATED: DATASET-REGISTRY END -->

Run with ``--check`` in CI to fail when generated documentation is stale.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from registry.loader import (  # noqa: E402
    load_dataset_registry,
    load_phenotype_domains,
    load_variable_registry,
)

MARKER = re.compile(
    r"(<!-- AUTO-GENERATED: (?P<name>[A-Z0-9-]+) START -->)"
    r"(?P<body>.*?)"
    r"(<!-- AUTO-GENERATED: (?P=name) END -->)",
    re.DOTALL,
)


def render_dataset_registry() -> str:
    rows = [
        "| Dataset | Access | Longitudinal | Allowed uses | Prohibited claims |",
        "|:--|:--|:--|:--|:--|",
    ]
    for dataset_id, spec in sorted(load_dataset_registry().datasets.items()):
        allowed = "<br>".join(f"`{u}`" for u in spec.allowed_uses)
        prohibited = "<br>".join(f"`{c}`" for c in spec.prohibited_claims) or "—"
        longitudinal = "Yes" if spec.longitudinal else "No"
        rows.append(
            f"| **{dataset_id}**<br><sub>{spec.name}</sub> | {spec.access} | "
            f"{longitudinal} | {allowed} | {prohibited} |"
        )
    return "\n".join(rows)


def render_variable_registry() -> str:
    variables = load_variable_registry().variables
    by_domain: dict[str, list[str]] = {}
    for code, spec in variables.items():
        for domain in spec.domain or ["uncategorized"]:
            by_domain.setdefault(domain, []).append(code)

    lines = [f"Total canonical variables: **{len(variables)}**", ""]
    lines.append("| Domain | Count | Variables |")
    lines.append("|:--|--:|:--|")
    for domain, codes in sorted(by_domain.items()):
        listed = ", ".join(f"`{c}`" for c in sorted(codes))
        lines.append(f"| {domain} | {len(codes)} | {listed} |")
    return "\n".join(lines)


def render_phenotype_domains() -> str:
    domains = load_phenotype_domains()["domains"]
    lines = ["| Domain | Features | Min coverage to report | Qualifier |", "|:--|--:|--:|:--|"]
    for name, domain in domains.items():
        qualifier = domain.get("symptom_only_qualifier", "—")
        lines.append(
            f"| **{name}** — {domain['label']} | {len(domain['features'])} | "
            f"{domain.get('min_coverage_to_report', 0)} | {qualifier} |"
        )
    return "\n".join(lines)


def render_implementation_status() -> str:
    """Status table driven by which modules actually exist on disk."""
    components = [
        (
            "1",
            "Schemas and registries",
            ["schemas/event.py", "registry/datasets.yaml"],
            "Contract tests",
        ),
        (
            "2",
            "Ingestion + event store",
            ["event_store/store.py"],
            "Unit tests, synthetic fixtures",
        ),
        (
            "3",
            "Static baselines",
            ["models/tabular/logistic.py"],
            "Cross-validated on the public cohort",
        ),
        (
            "4",
            "Phenotype domains",
            ["features/phenotype_domains.py"],
            "Reconstruction vs mean-imputation baseline",
        ),
        (
            "5",
            "Subtype + stability",
            ["models/stability/abstention.py"],
            "Stability metrics only; no external validation",
        ),
        (
            "6",
            "Speech pipeline",
            ["ingestion/speech/extraction.py"],
            "Synthetic scripted corpus only",
        ),
        (
            "7",
            "Document pipeline",
            ["ingestion/documents/lab_extractor.py"],
            "Synthetic report corpus only",
        ),
        (
            "8",
            "Ultrasound pipeline",
            ["models/ultrasound/morphology_2d.py"],
            "Segmentation/counting metrics on labelled data",
        ),
        (
            "9",
            "Dynamic hormonal state",
            ["models/temporal/gru.py"],
            "Grouped participant-level held-out evaluation",
        ),
    ]
    lines = ["| Step | Component | Status | Independently validated? |", "|:--|:--|:--|:--|"]
    for step, name, paths, validation in components:
        exists = all((REPO_ROOT / p).exists() for p in paths)
        status = "Implemented" if exists else "**Not implemented**"
        lines.append(f"| {step} | {name} | {status} | {validation} |")
    lines.append(
        "| — | Cross-modal fusion | **Not implemented** | "
        "Requires genuinely matched multimodal patients |"
    )
    return "\n".join(lines)


def render_experiments() -> str:
    """Table of experiment artifact directories that exist."""
    exp_root = REPO_ROOT / "artifacts" / "experiments"
    if not exp_root.exists():
        return "_No experiment artifacts present. Run a training script to populate this table._"

    rows = ["| Experiment | Model | Dataset version | Primary metrics |", "|:--|:--|:--|:--|"]
    found = False
    for metrics_path in sorted(exp_root.glob("*/metrics.json")):
        try:
            data = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        found = True
        agg = data.get("aggregate_metrics", {})
        summary = ", ".join(f"{k}={v:.3f}" for k, v in list(agg.items())[:3]) or "—"
        rows.append(
            f"| `{data.get('experiment_id', metrics_path.parent.name)}` | "
            f"{data.get('model', '—')} | {data.get('dataset_version', '—')} | {summary} |"
        )
    if not found:
        return "_No experiment artifacts present. Run a training script to populate this table._"
    return "\n".join(rows)


GENERATORS = {
    "DATASET-REGISTRY": render_dataset_registry,
    "VARIABLE-REGISTRY": render_variable_registry,
    "PHENOTYPE-DOMAINS": render_phenotype_domains,
    "IMPLEMENTATION-STATUS": render_implementation_status,
    "EXPERIMENTS": render_experiments,
}

TARGETS = [
    "README.md",
    "DATASET_REGISTRY.md",
    "docs/index.md",
    "docs/datasets/index.md",
    "docs/contracts.md",
    "docs/experiments/index.md",
    "docs/concepts/phenotype_profiles.md",
]


def update_file(path: Path, check: bool) -> bool:
    """Return True when the file is (or would be) changed."""
    if not path.exists():
        return False
    original = path.read_text()

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        generator = GENERATORS.get(name)
        if generator is None:
            raise SystemExit(f"{path}: unknown AUTO-GENERATED section '{name}'")
        return f"{match.group(1)}\n{generator()}\n{match.group(4)}"

    updated = MARKER.sub(replace, original)
    if updated == original:
        return False
    if not check:
        path.write_text(updated)
    return True


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Exposed so the CLI contract test can inspect it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated documentation is stale instead of rewriting it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    changed = [t for t in TARGETS if update_file(REPO_ROOT / t, args.check)]

    if args.check and changed:
        print("\033[31mGenerated documentation is stale:\033[0m")
        for target in changed:
            print(f"  - {target}")
        print("\nRun: python scripts/update_repo_docs.py")
        return 1

    if changed:
        print(f"Updated {len(changed)} file(s):")
        for target in changed:
            print(f"  - {target}")
    else:
        print("Generated documentation is up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
