"""Export a patient's static-clinical evidence as a single ``ModalityToken``.

This is the boundary between "features" and "what the rest of PRISM consumes". It
combines two complementary views:

* transparent, auditable domain composites (``features/phenotype_domains``), and
* a learned latent embedding from the masked autoencoder.

Both are qualified by coverage. A token built from three observed symptom flags
must not look, downstream, like a token built from a full hormonal panel — so
``quality_score`` carries the coverage and ``missing_fields`` names what was
absent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from features.missingness import coverage_by_row
from features.phenotype_domains import PhenotypeDomainScorer
from features.static_features import build_static_features, value_columns_of
from models.tabular.masked_autoencoder import MaskedTabularAutoencoder
from schemas.modality_token import ModalityToken
from schemas.phenotype import DomainScore

#: Coverage below which the token is flagged as too sparse to interpret.
LOW_COVERAGE_THRESHOLD = 0.3


class StaticClinicalTokenizer:
    """Fit on a training cohort, then emit one ``ModalityToken`` per patient."""

    version = "0.1.0"
    modality = "static_clinical"

    def __init__(
        self,
        *,
        autoencoder: MaskedTabularAutoencoder | None = None,
        domain_scorer: PhenotypeDomainScorer | None = None,
        source_dataset: str | None = None,
        include_embedding: bool = True,
        id_column: str = "patient_id",
    ) -> None:
        self.domain_scorer = domain_scorer or PhenotypeDomainScorer()
        self.autoencoder = autoencoder
        self.source_dataset = source_dataset
        self.include_embedding = include_embedding
        self.id_column = id_column
        self.embedding_columns_: list[str] = []
        self.is_fitted: bool = False

    # -- Fitting -----------------------------------------------------------

    def fit(self, df: pd.DataFrame, **autoencoder_kwargs: Any) -> StaticClinicalTokenizer:
        """Fit reference statistics and (optionally) the embedding, on training rows only."""
        matrix = build_static_features(df, add_missingness_indicators=False)
        enriched = self._enriched_frame(df)
        self.domain_scorer.fit(enriched)

        if self.include_embedding:
            self.embedding_columns_ = value_columns_of(matrix.X)
            if self.autoencoder is None:
                self.autoencoder = MaskedTabularAutoencoder(**autoencoder_kwargs)
            if not self.autoencoder.is_fitted:
                self.autoencoder.fit(matrix.X[self.embedding_columns_])

        self.is_fitted = True
        return self

    def _enriched_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived codes (lh_fsh_ratio, homa_ir, ...) the domain registry expects."""
        from features.static_features import derive_features

        enriched, _ = derive_features(df)
        return enriched

    # -- Tokenizing --------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> list[ModalityToken]:
        """Emit one token per row of ``df``."""
        if not self.is_fitted:
            raise RuntimeError("StaticClinicalTokenizer.fit() must be called before transform().")

        enriched = self._enriched_frame(df)
        scored = self.domain_scorer.score_frame(enriched)
        expected_codes = self.domain_scorer.required_codes
        coverage = coverage_by_row(enriched, expected_codes)

        embeddings: np.ndarray | None = None
        if self.include_embedding and self.autoencoder is not None and self.embedding_columns_:
            matrix = build_static_features(df, add_missingness_indicators=False)
            available = [c for c in self.embedding_columns_ if c in matrix.X.columns]
            frame = matrix.X.reindex(columns=self.embedding_columns_)
            if available:
                embeddings = self.autoencoder.embed(frame)

        ids = df[self.id_column].astype(str).tolist()
        tokens: list[ModalityToken] = []

        for i, patient_id in enumerate(ids):
            domain_scores = {name: scores[i] for name, scores in scored.items()}
            tokens.append(
                self._build_token(
                    patient_id=patient_id,
                    domain_scores=domain_scores,
                    coverage=float(coverage.iloc[i]),
                    embedding=None if embeddings is None else embeddings[i],
                    expected_codes=expected_codes,
                    row=enriched.iloc[i],
                )
            )
        return tokens

    def _build_token(
        self,
        *,
        patient_id: str,
        domain_scores: dict[str, DomainScore],
        coverage: float,
        embedding: np.ndarray | None,
        expected_codes: list[str],
        row: pd.Series,
    ) -> ModalityToken:
        structured: dict[str, float | int | str | bool | None] = {}
        warnings: list[str] = []
        reportable = 0

        for name, score in domain_scores.items():
            structured[f"domain_{name}_score"] = score.score
            structured[f"domain_{name}_coverage"] = round(score.coverage, 4)
            if score.evidence_qualifier:
                structured[f"domain_{name}_evidence_qualifier"] = score.evidence_qualifier
            if score.is_reportable:
                reportable += 1
            warnings.extend(score.warnings)

        missing_fields = sorted(
            code for code in expected_codes if code not in row.index or pd.isna(row.get(code))
        )

        # Confidence blends how much data exists with how many domains cleared
        # their reporting threshold — a token with data but no reportable domain
        # is not something downstream code should lean on.
        domain_fraction = reportable / max(len(domain_scores), 1)
        confidence = float(np.clip(0.5 * coverage + 0.5 * domain_fraction, 0.0, 1.0))

        if coverage < LOW_COVERAGE_THRESHOLD:
            warnings.append(
                f"Static-clinical coverage is {coverage:.2f}; this profile is sparse and should "
                "not be interpreted as a characterization of the patient."
            )
        if embedding is None and self.include_embedding:
            warnings.append("No latent embedding was produced for this patient.")

        return ModalityToken(
            patient_id=patient_id,
            modality="static_clinical",
            embedding=[] if embedding is None else [float(v) for v in embedding],
            structured_features=structured,
            quality_score=float(np.clip(coverage, 0.0, 1.0)),
            confidence_score=confidence,
            model_version=self.version,
            source_dataset=self.source_dataset,
            missing_fields=missing_fields,
            warnings=sorted(set(warnings)),
        )

    def fit_transform(self, df: pd.DataFrame, **kwargs: Any) -> list[ModalityToken]:
        """Convenience for single-cohort exploratory use, never for held-out evaluation."""
        return self.fit(df, **kwargs).transform(df)

    def manifest(self) -> dict[str, Any]:
        """Reproducibility record for the feature manifest."""
        return {
            "tokenizer_version": self.version,
            "modality": self.modality,
            "embedding_columns": list(self.embedding_columns_),
            "embedding_model": None if self.autoencoder is None else self.autoencoder.name,
            "latent_dim": None if self.autoencoder is None else self.autoencoder.latent_dim,
            "domain_scoring": self.domain_scorer.manifest(),
        }


__all__ = ["LOW_COVERAGE_THRESHOLD", "StaticClinicalTokenizer"]
