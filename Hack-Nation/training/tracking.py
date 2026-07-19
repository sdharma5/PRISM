"""Experiment tracking: the artifact contract every run must satisfy.

A metric is only a result if you can say which code, which config and which
environment produced it. This module owns that directory layout so no script has
to remember it.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import yaml

from training.logging import DEFAULT_LOG_NAME, JsonlLogger

#: Filenames the artifact contract requires in every experiment directory.
CONFIG_FILENAME = "config.resolved.yaml"
ENVIRONMENT_FILENAME = "environment.json"
GIT_COMMIT_FILENAME = "git_commit.txt"
SPLIT_MANIFEST_FILENAME = "split_manifest.json"
FEATURE_MANIFEST_FILENAME = "feature_manifest.json"
METRICS_FILENAME = "metrics.json"
README_FILENAME = "README.md"

#: Packages whose exact version can change results; recorded on every run.
TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "pydantic",
    "pyyaml",
    "xgboost",
    "torch",
)


def git_commit(repo_root: Path | None = None) -> str:
    """Current commit hash, suffixed ``-dirty`` when the tree has changes."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return commit
    return f"{commit}-dirty" if status else commit


def package_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    """Installed versions of the tracked packages; ``not_installed`` when absent."""
    versions: dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def environment_snapshot() -> dict[str, Any]:
    """Python, platform and package versions — enough to explain a reproduction gap."""
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": package_versions(),
    }


class ExperimentTracker:
    """Owns one experiment directory and writes the required artifacts into it."""

    def __init__(
        self,
        experiment_dir: str | Path,
        *,
        experiment_id: str,
        resolved_config: Mapping[str, Any] | None = None,
        echo: bool = True,
    ) -> None:
        self.dir = Path(experiment_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.resolved_config: dict[str, Any] = dict(resolved_config or {})
        self.git_commit: str = git_commit()
        self.logger = JsonlLogger(self.dir / DEFAULT_LOG_NAME, echo=echo, run_id=experiment_id)

    # -- Required artifacts ------------------------------------------------

    def write_config(self, resolved_config: Mapping[str, Any] | None = None) -> Path:
        """Write ``config.resolved.yaml`` — the config after every default was applied."""
        if resolved_config is not None:
            self.resolved_config = dict(resolved_config)
        path = self.dir / CONFIG_FILENAME
        path.write_text(yaml.safe_dump(_yamlable(self.resolved_config), sort_keys=False))
        return path

    def write_environment(self) -> Path:
        path = self.dir / ENVIRONMENT_FILENAME
        path.write_text(json.dumps(environment_snapshot(), indent=2) + "\n")
        return path

    def write_git_commit(self) -> Path:
        path = self.dir / GIT_COMMIT_FILENAME
        path.write_text(self.git_commit + "\n")
        return path

    def start(self) -> ExperimentTracker:
        """Write the provenance trio up front, so a crashed run is still traceable."""
        self.write_config()
        self.write_environment()
        self.write_git_commit()
        self.logger.log(
            "run_start",
            experiment_id=self.experiment_id,
            git_commit=self.git_commit,
            experiment_dir=str(self.dir),
        )
        return self

    def finish(self, **fields: Any) -> None:
        self.logger.log("run_end", experiment_id=self.experiment_id, **fields)

    # -- Convenience -------------------------------------------------------

    def path(self, filename: str) -> Path:
        return self.dir / filename

    def log(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.logger.log(event, **fields)

    def write_json(self, filename: str, payload: Any) -> Path:
        path = self.dir / filename
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        return path

    def verify_artifacts(self, *, require_readme: bool = True) -> list[str]:
        """Return the names of required artifacts that are missing."""
        required = [
            CONFIG_FILENAME,
            ENVIRONMENT_FILENAME,
            GIT_COMMIT_FILENAME,
            METRICS_FILENAME,
        ]
        if require_readme:
            required.append(README_FILENAME)
        return [name for name in required if not (self.dir / name).exists()]


def resolve_experiment_dir(
    root: str | Path,
    experiment_id: str,
    *,
    timestamped: bool = True,
) -> Path:
    """Build the experiment directory path, optionally timestamped to avoid clobbering."""
    root = Path(root)
    if not timestamped:
        return root / experiment_id
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return root / f"{experiment_id}_{stamp}"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, following an optional ``defaults: [...]`` chain."""
    path = Path(path)
    with path.open() as fh:
        config = yaml.safe_load(fh) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path}: config must be a YAML mapping.")

    defaults = config.pop("defaults", []) or []
    merged: dict[str, Any] = {}
    for entry in defaults:
        parent = (path.parent / str(entry)).resolve()
        merged = deep_merge(merged, load_config(parent))
    return deep_merge(merged, config)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (override wins on conflicts)."""
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _yamlable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _yamlable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_yamlable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


__all__ = [
    "CONFIG_FILENAME",
    "ENVIRONMENT_FILENAME",
    "FEATURE_MANIFEST_FILENAME",
    "GIT_COMMIT_FILENAME",
    "METRICS_FILENAME",
    "README_FILENAME",
    "SPLIT_MANIFEST_FILENAME",
    "ExperimentTracker",
    "deep_merge",
    "environment_snapshot",
    "git_commit",
    "load_config",
    "package_versions",
    "resolve_experiment_dir",
]
