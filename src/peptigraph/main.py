"""Main module of PeptiGraph."""

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from core.models import MolecularPropertyPredictor
from skfp.datasets.lrgb import load_peptides_func
from skfp.datasets.moleculenet import load_bace, load_sider

from peptigraph.helpers import (
    evaluate_predictions,
    load_and_preprocess_data,
    log_evaluation_results,
)
from utils.logger import get_logger

src_path = str(Path(__file__).resolve().parent.parent.parent / "src")
sys.path.insert(0, src_path)

logger = get_logger(__name__)


def run_experiment(
    dataset_loader: Callable[[], tuple[list[str], np.ndarray]],
    model_params: dict[str, str | list[str] | bool | float],
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Run a single molecular property prediction experiment.

    Args:
        dataset_loader: Function to load dataset, returns (smiles_list, labels)
        model_params: Dictionary of model parameters
        metrics: List of metrics to evaluate

    Returns:
        dict: Evaluation results containing metric scores
    """
    if metrics is None:
        metrics = ["auroc", "auprc"]

    logger.info("=" * 50)
    logger.info("Starting Molecular Property Prediction Experiment")
    logger.info("=" * 50)
    logger.info("Evaluating metrics: %s", ", ".join(metrics))

    # Extract data splitting parameters
    data_params = {
        "test_size": model_params.pop("test_size", 0.2),
        "use_valid": model_params.pop("use_valid", False),
    }

    smiles_list, y = dataset_loader()
    splits = load_and_preprocess_data(smiles_list, y, **data_params)

    logger.info("\nModel Configuration:")
    logger.info("Model type: %s", model_params.get("model_type", "rf"))
    logger.info(
        "Fingerprint types: %s",
        ", ".join(model_params.get("fingerprint_types", ["ecfp"])),
    )
    logger.info("Using pipeline: %s", model_params.get("use_pipeline", False))
    logger.info("\nTraining model...")

    model = MolecularPropertyPredictor(**model_params)
    model.fit(splits["train"][0], splits["train"][1])

    logger.info("\nEvaluating model...")
    y_pred = model.predict_proba(splits["test"][0])
    results = evaluate_predictions(splits["test"][1], y_pred, metrics)

    log_evaluation_results(results)
    return results


def run_all_experiments() -> dict[str, dict[str, float]]:
    """Run experiments on multiple datasets with different configurations.

    Returns:
        dict: Results for all experiments, mapping dataset names to their metric
            scores
    """
    base_params = {
        "model_type": "rf",
        "fingerprint_types": ["ecfp"],
        "test_size": 0.2,
        "use_valid": False,
        "use_pipeline": False,
    }

    peptides_params = {
        **base_params,
        "use_valid": True,
        "use_pipeline": True,
        "fingerprint_types": [
            "ecfp",
            "maccs",
            "tt",
            "pharmacophore",
            "ap",
            "layered",
            "mordred",
            "erg",
            "functional",
        ],
    }

    experiments = [
        ("BACE", load_bace, base_params.copy()),
        ("SIDER", load_sider, base_params.copy()),
        ("Peptides", load_peptides_func, peptides_params),
    ]

    all_results = {}
    logger.info("\nRunning Multiple Dataset Experiments")
    logger.info("=" * 50)

    for name, loader, params in experiments:
        logger.info("\nRunning %s dataset experiment...", name)
        all_results[name] = run_experiment(loader, params, metrics=["auroc", "auprc"])

    logger.info("\nSummary of All Results:")
    logger.info("=" * 50)
    for name, results in all_results.items():
        logger.info("\n%s Dataset Results:", name)
        log_evaluation_results(results)

    return all_results


if __name__ == "__main__":
    logger.info("Starting the PeptiGraph application...")
    run_all_experiments()
