"""End-to-end smoke run on a tiny synthetic cohort.

This asserts the *plumbing* — that every artifact the contract requires exists
and is internally consistent. It deliberately makes no scientific claim about the
numbers; a smoke run on 80 synthetic patients cannot support one.
"""

from __future__ import annotations

import json

import pytest
import yaml

from features.feature_manifest import FeatureManifest
from schemas.model_output import ExperimentResult
from scripts.train_static_baselines import main as train_static_baselines
from scripts.train_tabular_autoencoder import main as train_tabular_autoencoder
from training.splits import load_split_manifest

pytestmark = pytest.mark.smoke


TINY_BASELINES_CONFIG: dict = {
    "seed": 3,
    "experiment_id": "smoke_static_baselines",
    "data": {
        "path": None,
        "dataset_id": "pmos_tabular_public",
        "id_column": "patient_id",
        "label_column": "pmos_binary",
        "synthetic": {"n": 80, "seed": 1, "missing_rate": 0.2},
    },
    "features": {"add_missingness_indicators": True, "min_observed_fraction": 0.1},
    "preprocessing": {"numeric_impute_strategy": "median", "scale": True},
    "split": {"strategy": "repeated_stratified_kfold", "n_splits": 3, "seeds": [0]},
    "evaluation": {"threshold": 0.5, "calibration_bins": 5},
    "models": [
        {
            "name": "baseline_majority_class",
            "class": "models.tabular.baselines.MajorityClassBaseline",
            "params": {},
        },
        {
            "name": "static_logistic",
            "class": "models.tabular.logistic.LogisticStaticModel",
            "params": {"C": 1.0},
        },
    ],
    "output": {"root": None, "timestamped": False, "predictions_format": "csv"},
}


def _write_config(tmp_path, config: dict) -> str:
    config = json.loads(json.dumps(config))
    config["output"]["root"] = str(tmp_path / "experiments")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return str(path)


@pytest.fixture(scope="module")
def baselines_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("smoke_baselines")
    config_path = _write_config(tmp_path, TINY_BASELINES_CONFIG)
    return train_static_baselines(["--config", config_path, "--no-timestamp", "--quiet"])


def test_every_required_artifact_exists(baselines_run):
    for filename in (
        "config.resolved.yaml",
        "environment.json",
        "git_commit.txt",
        "split_manifest.json",
        "feature_manifest.json",
        "metrics.json",
        "predictions.csv",
        "README.md",
        "training_log.jsonl",
    ):
        assert (baselines_run / filename).exists(), filename


def test_metrics_json_parses_as_the_experiment_contract(baselines_run):
    result = ExperimentResult.model_validate(
        json.loads((baselines_run / "metrics.json").read_text())
    )
    assert result.model == "static_logistic"
    assert result.target == "pmos_binary"
    assert len(result.fold_metrics) == 3
    assert 0.0 <= result.aggregate_metrics["auroc"] <= 1.0
    assert result.limitations


def test_feature_manifest_records_columns_and_transforms(baselines_run):
    manifest = FeatureManifest.read_json(baselines_run / "feature_manifest.json")
    assert manifest.n_features == len(manifest.feature_columns)
    assert manifest.indicator_columns, "missingness indicators should be recorded"
    assert manifest.column_statistics
    assert any(t["class"] == "ColumnTransformer" for t in manifest.transforms)


def test_split_manifest_is_disjoint_on_reload(baselines_run):
    manifest = load_split_manifest(baselines_run / "split_manifest.json")
    manifest.assert_disjoint()
    assert len(manifest.folds) == 3


def test_predictions_cover_every_patient_once_per_model(baselines_run):
    import pandas as pd

    predictions = pd.read_csv(baselines_run / "predictions.csv")
    assert set(predictions.columns) >= {"model", "fold", "patient_id", "y_true", "y_prob"}
    for _, chunk in predictions.groupby("model"):
        assert chunk["patient_id"].nunique() == len(chunk) == 80
    assert predictions["y_prob"].between(0, 1).all()


def test_readme_carries_the_non_clinical_framing(baselines_run):
    text = (baselines_run / "README.md").read_text()
    assert "not a diagnosis" in text.lower()
    assert "Limitations" in text
    assert "Model comparison" in text


def test_resolved_config_is_reloadable(baselines_run):
    config = yaml.safe_load((baselines_run / "config.resolved.yaml").read_text())
    assert config["experiment_id"] == "smoke_static_baselines"
    assert config["split"]["n_splits"] == 3


def test_model_card_declares_non_diagnostic_use(baselines_run):
    card = json.loads((baselines_run / "model_card.json").read_text())
    assert "does not diagnose" in card["non_diagnostic_statement"]
    assert card["out_of_scope_uses"]
    assert card["limitations"]


def test_training_log_records_the_fold_loop(baselines_run):
    events = [
        json.loads(line)
        for line in (baselines_run / "training_log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    kinds = {event["event"] for event in events}
    assert {"run_start", "fold_start", "fold_end", "model_end", "run_end"} <= kinds


def test_autoencoder_script_runs_end_to_end(tmp_path):
    config = {
        "seed": 1,
        "experiment_id": "smoke_autoencoder",
        "data": {
            "path": None,
            "dataset_id": "pmos_tabular_public",
            "id_column": "patient_id",
            "label_column": "pmos_binary",
            "synthetic": {"n": 80, "seed": 2, "missing_rate": 0.2},
        },
        "features": {"min_observed_fraction": 0.1},
        "split": {"strategy": "repeated_stratified_kfold", "n_splits": 2, "seeds": [0]},
        "model": {"params": {"latent_dim": 6, "hidden_dim": 16, "epochs": 25}},
        "phenotype": {"export_tokens": True, "max_tokens_written": 5, "include_embedding": True},
        "output": {"root": str(tmp_path / "experiments"), "timestamped": False},
    }
    path = tmp_path / "ae.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))

    run_dir = train_tabular_autoencoder(["--config", str(path), "--no-timestamp", "--quiet"])

    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "embeddings.csv").exists()
    assert (run_dir / "token_summary.json").exists()
    assert len(list((run_dir / "tokens").glob("*.json"))) == 5

    result = ExperimentResult.model_validate(json.loads((run_dir / "metrics.json").read_text()))
    assert result.target == "masked_value_reconstruction"
    assert result.aggregate_metrics["masked_reconstruction_mse"] > 0
