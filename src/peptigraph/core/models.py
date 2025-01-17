"""Module containing molecular property prediction models."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from skfp.fingerprints import (
    AtomPairFingerprint,
    ECFPFingerprint,
    ERGFingerprint,
    FunctionalGroupsFingerprint,
    LayeredFingerprint,
    MACCSFingerprint,
    MordredFingerprint,
    PharmacophoreFingerprint,
    TopologicalTorsionFingerprint,
)
from skfp.metrics import extract_pos_proba
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler

from utils import get_logger

logger = get_logger(__name__)


class MolecularPropertyPredictor:
    """Predictor for molecular properties using fingerprint types and models.

    Supports various molecular fingerprint types and machine learning models
    for predicting properties of chemical compounds.
    """

    _FINGERPRINT_TRANSFORMERS: ClassVar[dict[str, type]] = {
        "ecfp": ECFPFingerprint,
        "maccs": MACCSFingerprint,
        "tt": TopologicalTorsionFingerprint,
        "pharmacophore": PharmacophoreFingerprint,
        "ap": AtomPairFingerprint,
        "layered": LayeredFingerprint,
        "mordred": MordredFingerprint,
        "erg": ERGFingerprint,
        "functional": FunctionalGroupsFingerprint,
    }

    _MODEL_CLASSES: ClassVar[dict[str, callable]] = {
        "rf": lambda: RandomForestClassifier(n_estimators=400, max_depth=20)
    }

    def __init__(
        self: MolecularPropertyPredictor,
        model_type: str = "rf",
        fingerprint_types: list[str] | None = None,
        *,
        use_pipeline: bool = False,
    ) -> None:
        """Initialize the MolecularPropertyPredictor.

        Args:
            model_type: Type of model to use for prediction.
            fingerprint_types: List of fingerprint types to use.
            use_pipeline: Whether to use scikit-learn pipeline.
        """
        self.model_type = self._validate_model_type(model_type)
        self.fingerprint_types = fingerprint_types or ["ecfp"]
        self.use_pipeline = use_pipeline
        self.model = self._setup_model()
        logger.info(
            "Initialized MolecularPropertyPredictor",
            extra={
                "model_type": model_type,
                "fingerprint_types": self.fingerprint_types,
                "use_pipeline": self.use_pipeline,
            },
        )

    def _validate_model_type(self: MolecularPropertyPredictor, model_type: str) -> str:
        """Validate the model type.

        Args:
            model_type: Type of model to validate.

        Returns:
            Validated model type.

        Raises:
            ValueError: If model type is not supported.
        """
        if model_type not in self._MODEL_CLASSES:
            msg = f"Unsupported model type: {model_type}"
            logger.error(msg)
            raise ValueError(msg)
        return model_type

    def _get_fingerprint_transformer(
        self: MolecularPropertyPredictor, fp_type: str
    ) -> object:
        """Get fingerprint transformer instance.

        Args:
            fp_type: Type of fingerprint transformer.

        Returns:
            Fingerprint transformer instance.

        Raises:
            ValueError: If fingerprint type is not supported.
        """
        if fp_type not in self._FINGERPRINT_TRANSFORMERS:
            msg = f"Unsupported fingerprint type: {fp_type}"
            logger.error(msg)
            raise ValueError(msg)
        logger.debug("Selected fingerprint transformer: %s", fp_type)
        return self._FINGERPRINT_TRANSFORMERS[fp_type]()

    def _create_feature_union(self: MolecularPropertyPredictor) -> FeatureUnion:
        """Create feature union from fingerprint transformers.

        Returns:
            Feature union of fingerprint transformers.
        """
        transformers = [
            (fp, self._get_fingerprint_transformer(fp)) for fp in self.fingerprint_types
        ]
        logger.debug("Created feature union for fingerprint transformers.")
        return FeatureUnion(transformers)

    def _setup_model(
        self: MolecularPropertyPredictor,
    ) -> RandomForestClassifier | Pipeline:
        """Set up the model with or without pipeline.

        Returns:
            Configured model instance.
        """
        if not self.use_pipeline:
            logger.debug("Setting up model without pipeline.")
            self.fp_transformer = (
                self._create_feature_union()
                if len(self.fingerprint_types) > 1
                else self._get_fingerprint_transformer(self.fingerprint_types[0])
            )
            return self._MODEL_CLASSES[self.model_type]()

        logger.debug("Setting up model with pipeline.")
        return Pipeline(
            [
                ("fps_union", self._create_feature_union()),
                ("scaler", MinMaxScaler()),
                ("classifier", self._MODEL_CLASSES[self.model_type]()),
            ]
        )

    def fit(
        self: MolecularPropertyPredictor,
        data: np.ndarray,
        target: np.ndarray,
    ) -> None:
        """Fit the model to the training data.

        Args:
            data: Training data.
            target: Target values.
        """
        logger.info("Fitting model...")
        if not self.use_pipeline:
            logger.debug("Transforming input data using fingerprint transformer.")
            data = self.fp_transformer.transform(data)
        self.model.fit(data, target)
        logger.info("Model fitting complete.")

    def predict_proba(self: MolecularPropertyPredictor, data: np.ndarray) -> np.ndarray:
        """Predict class probabilities for input data.

        Args:
            data: Input data for prediction.

        Returns:
            Array of predicted probabilities.
        """
        logger.info("Predicting probabilities...")
        if not self.use_pipeline:
            logger.debug("Transforming input data using fingerprint transformer.")
            data = self.fp_transformer.transform(data)
        y_pred = self.model.predict_proba(data)
        logger.info("Probability prediction complete.")
        binary_threshold = 2
        return (
            extract_pos_proba(y_pred)
            if y_pred[0].shape[0] > binary_threshold
            else y_pred[:, 1]
        )
