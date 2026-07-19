"""Generate the frontend's TypeScript types from the API's OpenAPI schema.

The frontend must never hand-write the response types. A hand-written copy
drifts silently: the server renames a field, the client keeps compiling against
the old name, and the mismatch surfaces as a blank panel in the UI rather than
as a build error.

This emits Python-side, rather than via ``openapi-typescript``, because no Node
runtime is available in this environment -- so a JS-based generator could not be
run or verified here. The output is plain TypeScript with no runtime dependency,
and is committed so the frontend builds without a generation step.

Usage::

    python scripts/generate_frontend_types.py
    python scripts/generate_frontend_types.py --check   # CI: fail if stale

Regenerate whenever ``apps/api/schemas/`` changes, after re-running
``scripts/export_openapi.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SPEC = _ROOT / "docs/openapi.yaml"
_DEFAULT_OUTPUT = _ROOT.parent / "UI/prism-app/src/types/api.ts"

_HEADER = """// GENERATED FILE -- DO NOT EDIT BY HAND.
//
// Source: docs/openapi.yaml (from apps/api/schemas/).
// Regenerate: python scripts/export_openapi.py
//             python scripts/generate_frontend_types.py
//
// These mirror the server's response contract exactly. Editing them here does
// not change the server; it only hides a mismatch until runtime.

"""

#: OpenAPI primitive -> TypeScript primitive.
_PRIMITIVES: dict[str, str] = {
    "string": "string",
    "integer": "number",
    "number": "number",
    "boolean": "boolean",
    "null": "null",
}


def _ref_name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def _type_of(schema: dict[str, Any] | None) -> str:
    """Render one OpenAPI schema node as a TypeScript type expression."""
    if not schema:
        return "unknown"

    if "$ref" in schema:
        return _ref_name(schema["$ref"])

    # `const` is how a pinned literal (is_diagnosis: false) reaches the schema.
    if "const" in schema:
        return _literal(schema["const"])

    if "enum" in schema:
        return " | ".join(_literal(v) for v in schema["enum"]) or "never"

    for key in ("anyOf", "oneOf"):
        if key in schema:
            parts = [_type_of(s) for s in schema[key]]
            # Deduplicate while preserving order; `T | null` is extremely common
            # here because every optional field is Optional[...] in Pydantic.
            seen: list[str] = []
            for part in parts:
                if part not in seen:
                    seen.append(part)
            return " | ".join(seen)

    if "allOf" in schema:
        parts = [_type_of(s) for s in schema["allOf"]]
        return " & ".join(parts) if parts else "unknown"

    declared = schema.get("type")

    if declared == "array":
        return f"{_type_of(schema.get('items'))}[]"

    if declared == "object" or "properties" in schema:
        if "properties" in schema:
            return _inline_object(schema)
        extra = schema.get("additionalProperties")
        if isinstance(extra, dict):
            return f"Record<string, {_type_of(extra)}>"
        return "Record<string, unknown>"

    if isinstance(declared, list):
        return " | ".join(_PRIMITIVES.get(t, "unknown") for t in declared)

    return _PRIMITIVES.get(str(declared), "unknown")


def _inline_object(schema: dict[str, Any]) -> str:
    required = set(schema.get("required", []))
    fields = [
        f"{name}{'' if name in required else '?'}: {_type_of(body)}"
        for name, body in schema.get("properties", {}).items()
    ]
    return "{ " + "; ".join(fields) + " }"


def _literal(value: Any) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _render_interface(name: str, schema: dict[str, Any]) -> str:
    """One named schema as an exported interface or type alias."""
    doc = schema.get("description")
    lines: list[str] = []
    if doc:
        # Keep only the summary line: the full Python docstrings are long, and a
        # wall of prose above every field hurts more than it helps in an editor.
        summary = str(doc).strip().split("\n\n")[0].replace("\n", " ").strip()
        lines.append(f"/** {summary} */")

    if "enum" in schema or "const" in schema:
        lines.append(f"export type {name} = {_type_of(schema)}")
        return "\n".join(lines)

    if "properties" not in schema:
        lines.append(f"export type {name} = {_type_of(schema)}")
        return "\n".join(lines)

    required = set(schema.get("required", []))
    lines.append(f"export interface {name} {{")
    for field, body in schema["properties"].items():
        field_doc = body.get("description")
        if field_doc:
            summary = str(field_doc).strip().replace("\n", " ")
            lines.append(f"  /** {summary} */")
        optional = "" if field in required else "?"
        lines.append(f"  {field}{optional}: {_type_of(body)}")
    lines.append("}")
    return "\n".join(lines)


def generate(spec: dict[str, Any]) -> str:
    schemas: dict[str, Any] = spec.get("components", {}).get("schemas", {})
    blocks = [_render_interface(name, body) for name, body in sorted(schemas.items())]

    paths = spec.get("paths", {})
    routes = "\n".join(f"  | '{path}'" for path in sorted(paths))
    blocks.append(f"/** Every path this API serves. */\nexport type ApiPath =\n{routes}")

    return _HEADER + "\n\n".join(blocks) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=_DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the committed output is stale, without writing.",
    )
    args = parser.parse_args()

    if not args.spec.exists():
        print(f"No spec at {args.spec}; run scripts/export_openapi.py first.", file=sys.stderr)
        return 2

    rendered = generate(yaml.safe_load(args.spec.read_text()))

    if args.check:
        current = args.output.read_text() if args.output.exists() else ""
        if current != rendered:
            print(f"{args.output} is stale; regenerate it.", file=sys.stderr)
            return 1
        print(f"{args.output} is up to date.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(f"Wrote {args.output} ({rendered.count('export ')} exported types)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
