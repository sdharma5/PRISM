"""The CLI contract: documented commands must be commands that actually exist.

This file exists because the documentation and the code disagreed for a long
time without anything noticing. README.md, TRAINING.md and slurm/train_temporal.sbatch
all told users to run ``scripts/train_temporal.py --config ...`` while the script
only accepted ``--experiment``; the output-directory flag had four different
names across seven scripts; and ``train_temporal.py`` imported a function
(``load_participant_days``) that did not exist anywhere in the tree — a crash on
the real-data path that no test reached, because the synthetic path never
executed the import.

Every check here is cheap and none of them run a training job. They are the
checks that would have caught all of the above:

1. ``--help`` exits 0 for every entry point. This alone catches an import error
   like the missing ``load_participant_days``, because argparse cannot print
   help for a module that will not import.
2. Any script taking a config takes it as ``--config``, and any script writing
   artifacts takes ``--output-dir``. One name each, everywhere.
3. Every command written in the documentation and in the Slurm templates names a
   script that exists and passes flags that its parser accepts. This is the
   check that fails when documentation drifts.
4. Every config path mentioned anywhere resolves to a file on disk.

Checks 3 and 4 parse the commands; they never execute them.
"""

from __future__ import annotations

import argparse
import importlib
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

#: Sources that document how to invoke this project.
DOC_FILES: list[Path] = sorted(
    [
        *(p for p in (REPO_ROOT / "README.md", REPO_ROOT / "TRAINING.md") if p.exists()),
        *sorted((REPO_ROOT / "docs").rglob("*.md")),
        *sorted((REPO_ROOT / "slurm").glob("*.sbatch")),
    ]
)


def _script_paths() -> list[Path]:
    """Every runnable entry point in scripts/ (private helpers excluded)."""
    return sorted(
        p
        for p in SCRIPTS_DIR.glob("*.py")
        if not p.name.startswith("_") and p.name != "__init__.py"
    )


def _has_main(path: Path) -> bool:
    return re.search(r"^def main\(", path.read_text(), re.MULTILINE) is not None


SCRIPTS = [p for p in _script_paths() if _has_main(p)]
SCRIPT_NAMES = [p.name for p in SCRIPTS]


def _load_parser(script: Path) -> argparse.ArgumentParser | None:
    """Import a script and return its parser, or None if it exposes none."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    module = importlib.import_module(f"scripts.{script.stem}")
    builder = getattr(module, "build_parser", None)
    return builder() if callable(builder) else None


def _config_is_required(parser: argparse.ArgumentParser) -> bool:
    """True when the parser will refuse to parse without ``--config``."""
    action = parser._option_string_actions.get("--config")  # noqa: SLF001
    return bool(action is not None and action.required)


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    """Every flag the parser accepts, including hidden deprecated aliases."""
    return set(parser._option_string_actions)  # noqa: SLF001 - no public accessor exists


# ---------------------------------------------------------------------------
# Command extraction
# ---------------------------------------------------------------------------

#: `python scripts/foo.py <args>` up to the end of the (continued) line.
COMMAND_RE = re.compile(r"python3?\s+(scripts/[\w.-]+\.py)((?:\s+[^\n]*)?)")
#: Any configs/... yaml path mentioned in prose, a command, or another config.
CONFIG_RE = re.compile(r"configs/[\w./-]+\.ya?ml")


def _join_continuations(text: str) -> str:
    """Fold shell line-continuations so a multi-line command reads as one."""
    return re.sub(r"\\\s*\n\s*", " ", text)


def _extract_commands(path: Path) -> list[tuple[str, str, int]]:
    """Return (script, argument string, line number) for each command in a file."""
    raw = path.read_text()
    joined = _join_continuations(raw)
    out: list[tuple[str, str, int]] = []
    for match in COMMAND_RE.finditer(joined):
        line = joined[: match.start()].count("\n") + 1
        out.append((match.group(1), match.group(2), line))
    return out


def _flags(argument_string: str) -> list[str]:
    """The long/short flags used in an argument string, ignoring their values."""
    try:
        tokens = shlex.split(argument_string, comments=True)
    except ValueError:
        # Unbalanced quoting from a shell snippet; fall back to whitespace.
        tokens = argument_string.split()
    return [t.split("=", 1)[0] for t in tokens if t.startswith("-") and t != "-"]


ALL_COMMANDS = [
    (doc, script, args, line) for doc in DOC_FILES for script, args, line in _extract_commands(doc)
]


# ---------------------------------------------------------------------------
# 1. Every entry point can print its help
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", SCRIPTS, ids=SCRIPT_NAMES)
def test_help_exits_zero(script: Path) -> None:
    """``--help`` must exit 0 — which requires the module to import cleanly.

    This is the check that catches a script importing a name that does not
    exist. It is deliberately a subprocess: an in-process import could be
    masked by a module another test already imported successfully.
    """
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"{script.name} --help exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.stdout.strip(), f"{script.name} --help printed nothing."


# ---------------------------------------------------------------------------
# 2. One name per concept
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", SCRIPTS, ids=SCRIPT_NAMES)
def test_canonical_flag_names(script: Path) -> None:
    """A config is always ``--config``; a destination is always ``--output-dir``."""
    parser = _load_parser(script)
    if parser is None:
        pytest.skip(f"{script.name} exposes no build_parser()")
    options = _option_strings(parser)

    forbidden = {
        "--experiment": "--config",
        "--output-root": "--output-dir",
        "--artifact-dir": "--output-dir",
    }
    for bad, good in forbidden.items():
        if bad in options:
            # Keeping the old spelling is fine, but only as an alias alongside
            # the canonical one — never as the only way to pass the value.
            assert good in options, (
                f"{script.name} accepts {bad} but not {good}. "
                f"{bad} may exist only as a deprecated alias for {good}."
            )

    if "--config" in options:
        assert "--output-dir" in options, (
            f"{script.name} takes --config but has no --output-dir. "
            "Every config-driven script must let the caller redirect its artifacts."
        )


@pytest.mark.parametrize("script", SCRIPTS, ids=SCRIPT_NAMES)
def test_deprecated_aliases_still_work(script: Path) -> None:
    """A renamed flag must keep working, not break someone's saved invocation."""
    parser = _load_parser(script)
    if parser is None:
        pytest.skip(f"{script.name} exposes no build_parser()")
    options = _option_strings(parser)

    for old, new in (
        ("--experiment", "--config"),
        ("--output-root", "--output-dir"),
        ("--artifact-dir", "--output-dir"),
    ):
        if old not in options:
            continue
        # Supply --config when the parser requires it, so this test exercises
        # the alias rather than argparse's missing-required-argument path.
        required = ["--config", "some/config.yaml"] if _config_is_required(parser) else []
        namespace = parser.parse_args([*required, old, "some/value"])
        target = new.lstrip("-").replace("-", "_")
        assert str(getattr(namespace, target)) == "some/value", (
            f"{script.name}: {old} did not populate {new}."
        )


# ---------------------------------------------------------------------------
# 3. Documented commands must be real commands
# ---------------------------------------------------------------------------


def test_documentation_contains_commands() -> None:
    """Guard the guard: if extraction silently matches nothing, say so."""
    assert ALL_COMMANDS, (
        "No commands were extracted from the documentation. Either the docs lost "
        "their examples or COMMAND_RE stopped matching — both make the checks "
        "below vacuous."
    )


@pytest.mark.parametrize(
    ("doc", "script", "args", "line"),
    ALL_COMMANDS,
    ids=[f"{d.name}:{ln}:{s.split('/')[-1]}" for d, s, _, ln in ALL_COMMANDS],
)
def test_documented_command_script_exists(doc: Path, script: str, args: str, line: int) -> None:
    """Every documented command must name a script that exists."""
    assert (REPO_ROOT / script).exists(), (
        f"{doc.relative_to(REPO_ROOT)}:{line} documents '{script}', which does not exist."
    )


@pytest.mark.parametrize(
    ("doc", "script", "args", "line"),
    ALL_COMMANDS,
    ids=[f"{d.name}:{ln}:{s.split('/')[-1]}" for d, s, _, ln in ALL_COMMANDS],
)
def test_documented_command_flags_are_accepted(
    doc: Path, script: str, args: str, line: int
) -> None:
    """Every flag in a documented command must be one its parser accepts.

    The command is never executed — only its flags are checked against the
    parser. That keeps this test fast and side-effect free while still failing
    the moment a documented flag stops existing.
    """
    path = REPO_ROOT / script
    if not path.exists():
        pytest.skip("covered by test_documented_command_script_exists")

    parser = _load_parser(path)
    if parser is None:
        pytest.skip(f"{path.name} exposes no build_parser()")
    accepted = _option_strings(parser)

    used = _flags(args)
    unknown = [f for f in used if f not in accepted]
    assert not unknown, (
        f"{doc.relative_to(REPO_ROOT)}:{line} runs '{script}' with {unknown}, "
        f"which its parser does not accept.\n"
        f"Accepted flags: {sorted(accepted)}"
    )


# ---------------------------------------------------------------------------
# 4. Referenced configs must exist
# ---------------------------------------------------------------------------


def _config_references() -> list[tuple[Path, str, int]]:
    """Every configs/*.yaml path referenced by a doc, template or other config."""
    sources = [*DOC_FILES, *sorted((REPO_ROOT / "configs").rglob("*.yaml"))]
    out: list[tuple[Path, str, int]] = []
    for source in sources:
        text = source.read_text()
        for match in CONFIG_RE.finditer(text):
            line = text[: match.start()].count("\n") + 1
            out.append((source, match.group(0), line))
    return out


CONFIG_REFERENCES = _config_references()


@pytest.mark.parametrize(
    ("source", "config", "line"),
    CONFIG_REFERENCES,
    ids=[f"{s.name}:{ln}:{c.split('/')[-1]}" for s, c, ln in CONFIG_REFERENCES],
)
def test_referenced_config_exists(source: Path, config: str, line: int) -> None:
    """A config named in the docs or in another config must be on disk."""
    assert (REPO_ROOT / config).exists(), (
        f"{source.relative_to(REPO_ROOT)}:{line} references '{config}', which does not exist."
    )


# ---------------------------------------------------------------------------
# 5. Shared resolution behaves as documented
# ---------------------------------------------------------------------------


def test_precedence_cli_beats_env_beats_config(tmp_path, monkeypatch) -> None:
    """explicit flag > environment variable > config value > built-in default."""
    from scripts._cli import ARTIFACT_ROOT_ENV, resolve_output_dir
    from scripts._experiment_io import resolve_data_root

    config = {"data": {"root": "from/config"}, "output": {"dir": "out/from/config"}}

    monkeypatch.delenv("PRISM_DATA_ROOT", raising=False)
    assert resolve_data_root(config).as_posix().endswith("from/config")

    monkeypatch.setenv("PRISM_DATA_ROOT", str(tmp_path / "from_env"))
    assert resolve_data_root(config) == tmp_path / "from_env"

    # The flag outranks the environment.
    assert resolve_data_root(config, tmp_path / "from_flag") == tmp_path / "from_flag"

    monkeypatch.delenv("PRISM_DATA_ROOT", raising=False)
    # create=False throughout: resolving a path must not litter the repository
    # with directories just because a test asked where they would go.
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(tmp_path / "artifacts_env"))
    assert resolve_output_dir(config, create=False) == tmp_path / "artifacts_env"
    assert resolve_output_dir(config, tmp_path / "flag", create=False) == tmp_path / "flag"

    monkeypatch.delenv(ARTIFACT_ROOT_ENV, raising=False)
    assert resolve_output_dir(config, create=False).as_posix().endswith("out/from/config")

    # Nothing configured at all: fall back to the experiment-scoped default.
    fallback = resolve_output_dir({"experiment_id": "exp_x"}, create=False)
    assert fallback.as_posix().endswith("artifacts/experiments/exp_x")


def test_empty_env_var_is_treated_as_unset(monkeypatch) -> None:
    """An exported-but-empty variable must not resolve paths against '/'.

    Slurm templates export variables to "" to mean "not configured". Honouring
    that literally would silently relocate every artifact to the filesystem root.
    """
    from scripts._cli import env_path

    monkeypatch.setenv("PRISM_DATA_ROOT", "")
    assert env_path("PRISM_DATA_ROOT") is None
    monkeypatch.setenv("PRISM_DATA_ROOT", "   ")
    assert env_path("PRISM_DATA_ROOT") is None


def test_legacy_config_keys_still_resolve(monkeypatch) -> None:
    """The pre-unification key spellings keep working."""
    from scripts._experiment_io import resolve_data_path, resolve_data_root

    # $PRISM_DATA_ROOT outranks every config key, so a developer who has
    # actually configured a data root would otherwise see this test fail — it
    # asserts config-key precedence, which only applies when the env is unset.
    monkeypatch.delenv("PRISM_DATA_ROOT", raising=False)

    assert resolve_data_root({"root": "legacy/root"}).as_posix().endswith("legacy/root")
    assert (
        resolve_data_path({"dataset": {"path": "legacy/file.csv"}})
        .as_posix()
        .endswith("legacy/file.csv")
    )
    # Canonical wins when both are present.
    both = {"data": {"root": "canonical"}, "root": "legacy"}
    assert resolve_data_root(both).as_posix().endswith("canonical")
