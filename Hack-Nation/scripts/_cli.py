"""Shared argument parsing for every ``scripts/*.py`` entry point.

This module is the single source of truth for the PRISM command-line surface.
Before it existed, each script invented its own names for the same three ideas
(``--experiment`` vs ``--config``, ``--output-root`` vs ``--artifact-dir`` vs
``--output-dir``), the documented commands were never executed, and the
documentation drifted away from the code. ``tests/integration/test_cli_contract.py``
now fails when that happens again.

Canonical flags
---------------
``--config PATH``
    The ONLY name for an experiment or data config.
``--data-root PATH``
    Root directory holding externally obtained datasets.
``--output-dir PATH``
    The ONLY name for a destination directory.
``--seed INT``, ``--experiment-id STR``, ``--quiet``
    Common run controls.

Precedence
----------
Resolution is identical in every script and every helper here::

    explicit CLI flag  >  environment variable  >  config file value  >  built-in default

The environment variables are ``PRISM_DATA_ROOT`` (for ``--data-root``) and
``PRISM_ARTIFACT_ROOT`` (for ``--output-dir``). They are read directly from
``os.environ``; nothing in this repository auto-loads a ``.env`` file. See
``.env.example`` for how to source one.

Deprecated aliases
------------------
Every flag that was renamed keeps its old spelling as a hidden alias that still
works and prints a deprecation warning to stderr, so no existing invocation
breaks silently:

===================  ==================  ============================================
Old flag             Canonical flag      Scripts
===================  ==================  ============================================
``--experiment``     ``--config``        train_temporal, train_ultrasound
``--output-root``    ``--output-dir``    train_static_baselines, train_tabular_autoencoder
``--artifact-dir``   ``--output-dir``    discover_phenotypes, run_stability_analysis
``--out``            ``--output-dir``    build_speech_eval_set
===================  ==================  ============================================

``--metrics-out`` and ``--token-out`` are NOT deprecated: they name individual
files rather than a directory, and they take precedence over ``--output-dir``
when both are given.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Environment variable consulted when ``--data-root`` is not given.
DATA_ROOT_ENV = "PRISM_DATA_ROOT"
#: Environment variable consulted when ``--output-dir`` is not given.
ARTIFACT_ROOT_ENV = "PRISM_ARTIFACT_ROOT"

__all__ = [
    "ARTIFACT_ROOT_ENV",
    "DATA_ROOT_ENV",
    "REPO_ROOT",
    "add_data_root_argument",
    "add_deprecated_alias",
    "add_output_argument",
    "add_standard_arguments",
    "env_path",
    "make_parser",
    "resolve_output_dir",
    "resolve_output_root",
    "resolve_path",
    "resolve_seed",
]


def env_path(name: str) -> Path | None:
    """Return ``$name`` as a Path, or None when unset or empty.

    An empty string is treated as unset. Slurm templates routinely export a
    variable to "" to mean "not configured", and honouring that literally would
    resolve every relative path against the filesystem root.
    """
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    """Absolutise ``value``, resolving relative paths against ``base``.

    ``base`` defaults to the repository root so that a config committed to the
    repo means the same thing regardless of the caller's working directory.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base or REPO_ROOT) / path


class _DeprecatedAliasAction(argparse.Action):
    """Accept a renamed flag, warn, and store into the canonical destination."""

    def __init__(
        self,
        option_strings: list[str],
        dest: str,
        replacement: str = "",
        **kwargs: Any,
    ) -> None:
        self.replacement = replacement
        super().__init__(option_strings, dest, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        print(
            f"WARNING: {option_string} is deprecated and will be removed in a future "
            f"release; use {self.replacement} instead.",
            file=sys.stderr,
        )
        setattr(namespace, self.dest, values)


def add_deprecated_alias(
    parser: argparse.ArgumentParser,
    old_flag: str,
    *,
    dest: str,
    replacement: str,
    type: Any = None,  # noqa: A002 - mirrors argparse's own parameter name
) -> None:
    """Register ``old_flag`` as a hidden, still-working alias for ``replacement``.

    The alias is suppressed from ``--help`` so the documented surface stays
    small, but it keeps working so that scripts, notebooks and cluster job files
    written against the old names do not break without warning.
    """
    parser.add_argument(
        old_flag,
        dest=dest,
        action=_DeprecatedAliasAction,
        replacement=replacement,
        type=type,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )


def make_parser(description: str | None = None) -> argparse.ArgumentParser:
    """Create a parser with PRISM's standard formatting."""
    return argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def add_data_root_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--data-root``."""
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Root directory holding externally obtained datasets. "
            f"Falls back to ${DATA_ROOT_ENV}, then to the config value."
        ),
    )


def add_output_argument(parser: argparse.ArgumentParser, *, help_suffix: str = "") -> None:
    """Add ``--output-dir``."""
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            f"Destination directory for artifacts.{help_suffix} "
            f"Falls back to ${ARTIFACT_ROOT_ENV}, then the config value, then "
            "artifacts/experiments/<experiment_id>."
        ),
    )


def add_standard_arguments(
    parser: argparse.ArgumentParser,
    *,
    config: bool = True,
    config_required: bool = True,
    config_default: Path | None = None,
    data_root: bool = True,
    output: bool = True,
    output_help_suffix: str = "",
    seed: bool = True,
    experiment_id: bool = True,
    quiet: bool = True,
) -> argparse.ArgumentParser:
    """Attach the canonical PRISM flags to ``parser``.

    Every flag is opt-out so a script only advertises what it actually honours.
    A script that ignores ``--seed`` must not accept it: a flag that parses and
    does nothing is worse than one that errors, because it produces a run that
    looks controlled and is not.
    """
    if config:
        parser.add_argument(
            "--config",
            type=Path,
            required=config_required and config_default is None,
            default=config_default,
            help="Path to the YAML config.",
        )
    if data_root:
        add_data_root_argument(parser)
    if output:
        add_output_argument(parser, help_suffix=output_help_suffix)
    if seed:
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Override the config's seed.",
        )
    if experiment_id:
        parser.add_argument(
            "--experiment-id",
            type=str,
            default=None,
            help="Override the config's experiment_id.",
        )
    if quiet:
        parser.add_argument(
            "--quiet",
            action="store_true",
            help="Suppress console log echo.",
        )
    return parser


def resolve_output_dir(
    config: dict[str, Any] | None = None,
    override: str | Path | None = None,
    *,
    experiment_id: str | None = None,
    config_keys: tuple[str, ...] = ("output_dir",),
    create: bool = True,
) -> Path:
    """Resolve the artifact destination under the documented precedence.

    Order: ``override`` (the CLI flag) > ``$PRISM_ARTIFACT_ROOT`` > the config
    value > ``artifacts/experiments/<experiment_id>``.

    ``config_keys`` names the dotted config keys to consult, in order, for the
    legacy per-script spellings (``output.artifact_dir``, ``output.root``,
    ``output_dir``). The canonical key is ``output.dir``.
    """
    config = config or {}
    chosen: str | Path | None = None

    if override is not None:
        chosen = override
    else:
        chosen = env_path(ARTIFACT_ROOT_ENV)
        if chosen is None:
            for dotted in ("output.dir", *config_keys):
                value: Any = config
                for part in dotted.split("."):
                    value = value.get(part) if isinstance(value, dict) else None
                    if value is None:
                        break
                if value:
                    chosen = str(value)
                    break

    if chosen is None:
        run_id = experiment_id or config.get("experiment_id") or "unnamed"
        chosen = f"artifacts/experiments/{run_id}"

    path = resolve_path(chosen)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_output_root(
    config: dict[str, Any] | None = None,
    override: str | Path | None = None,
    *,
    default: str = "artifacts/experiments",
) -> Path:
    """Resolve the *root* under which a timestamped run directory is created.

    Used by the scripts whose ``--output-dir`` names a parent rather than the
    final destination (``train_static_baselines``, ``train_tabular_autoencoder``);
    those scripts append ``<experiment_id>[_<timestamp>]`` themselves. Same
    precedence as :func:`resolve_output_dir`, but nothing is created here.
    """
    config = config or {}
    if override is not None:
        return resolve_path(override)
    from_env = env_path(ARTIFACT_ROOT_ENV)
    if from_env is not None:
        return from_env
    configured = (config.get("output") or {}).get("root")
    return resolve_path(str(configured) if configured else default)


def resolve_seed(
    config: dict[str, Any] | None = None,
    override: int | None = None,
    *,
    default: int = 0,
) -> int:
    """Resolve the run seed: CLI flag > config ``seed`` > ``default``."""
    if override is not None:
        return int(override)
    value = (config or {}).get("seed")
    return int(value) if value is not None else default
