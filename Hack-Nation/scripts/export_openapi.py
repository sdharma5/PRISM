"""Emit the API's OpenAPI document to docs/openapi.yaml.

The frontend's TypeScript types are generated from this file rather than
hand-written, so it is a build input, not documentation. Regenerate it whenever
apps/api/schemas changes and commit the result: a stale spec silently produces
client types that disagree with the server.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from apps.api.main import create_app

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "docs/openapi.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # create_app() does not run the lifespan, so no model is loaded here: the
    # schema is a property of the code, not of which checkpoints happen to be
    # on disk. That keeps this script runnable in CI without artifacts.
    document = create_app().openapi()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(document, sort_keys=False, width=100))
    print(f"Wrote {args.output} ({len(document.get('paths', {}))} paths)")


if __name__ == "__main__":
    main()
