#!/usr/bin/env python3
"""Generate release notes from conventional-commit prefixes.

Commits since the previous tag are grouped by prefix. A release that changes a
schema or a dataset registry entry gets an explicit callout — those are
scientific changes and a reader should not have to infer them from a diff.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

SECTIONS: dict[str, str] = {
    "feat": "Added",
    "fix": "Fixed",
    "model": "Models",
    "data": "Data and ingestion",
    "eval": "Evaluation",
    "docs": "Documentation",
    "refactor": "Refactoring",
    "test": "Tests",
    "chore": "Tooling",
}

#: Paths whose change means a scientific claim may have moved.
CLAIM_SENSITIVE = ("schemas/", "registry/", "MODEL_CARD.md", "docs/decisions/")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def previous_tag() -> str | None:
    tags = _git("tag", "--sort=-creatordate").splitlines()
    return tags[1] if len(tags) > 1 else None


def collect_commits(since: str | None) -> list[tuple[str, str, str]]:
    """Return (prefix, subject, sha) for each commit since ``since``."""
    rev_range = f"{since}..HEAD" if since else "HEAD"
    log = _git("log", rev_range, "--pretty=format:%h%x00%s")
    commits: list[tuple[str, str, str]] = []
    for line in log.splitlines():
        if not line:
            continue
        sha, _, subject = line.partition("\0")
        prefix, sep, rest = subject.partition(":")
        prefix = prefix.split("(")[0].strip().lower()
        if sep and prefix in SECTIONS:
            commits.append((prefix, rest.strip(), sha))
        else:
            commits.append(("chore", subject.strip(), sha))
    return commits


def claim_sensitive_changes(since: str | None) -> list[str]:
    rev_range = f"{since}..HEAD" if since else "HEAD"
    files = _git("diff", "--name-only", rev_range).splitlines()
    return sorted({f for f in files if f.startswith(CLAIM_SENSITIVE)})


def render(commits: list[tuple[str, str, str]], sensitive: list[str], since: str | None) -> str:
    lines = ["# Release notes", ""]
    if since:
        lines.append(f"Changes since `{since}`.")
        lines.append("")

    if sensitive:
        lines += [
            "## ⚠️ Contract and claim review required",
            "",
            "This release touches files that define data contracts or scientific claims:",
            "",
        ]
        lines += [f"- `{path}`" for path in sensitive]
        lines += [
            "",
            "Confirm the schema version ledger, the CHANGELOG, and MODEL_CARD.md agree "
            "with these changes before publishing.",
            "",
        ]

    for prefix, heading in SECTIONS.items():
        entries = [(s, sha) for p, s, sha in commits if p == prefix]
        if not entries:
            continue
        lines += [f"## {heading}", ""]
        lines += [f"- {subject} ({sha})" for subject, sha in entries]
        lines.append("")

    lines += [
        "---",
        "",
        "PRISM is a research artifact. It does not diagnose any condition, does not "
        "provide medical advice, and is not validated for clinical use.",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Exposed so the CLI contract test can inspect it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("RELEASE_NOTES.md"))
    parser.add_argument("--since", default=None, help="Tag to diff from (default: previous tag).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    since = args.since or previous_tag()
    commits = collect_commits(since)
    notes = render(commits, claim_sensitive_changes(since), since)
    args.output.write_text(notes)
    print(f"Wrote {args.output} ({len(commits)} commits since {since or 'the beginning'}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
