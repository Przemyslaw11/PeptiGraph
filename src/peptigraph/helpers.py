"""Utility functions for preprocessing and evaluating molecular data."""

import numpy as np
from skfp.metrics import multioutput_auprc_score, multioutput_auroc_score
from skfp.model_selection import (
    scaffold_train_test_split,
    scaffold_train_valid_test_split,
)
from skfp.preprocessing import MolFromSmilesTransformer, MolStandardizer

from utils.logger import get_logger

logger = get_logger(__name__)


def load_and_preprocess_data(
    smiles_list: list[str],
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    use_valid: bool = False,
) -> dict:
    """Load and preprocess molecular data.

    Args:
        smiles_list: List of SMILES strings
        y: Target values
        test_size: Size of test split
        use_valid: Whether to include validation split

    Returns:
        Dictionary containing data splits
    """
    _log_input_summary(smiles_list, y)

    logger.info("Converting SMILES to molecules...")
    mols = MolFromSmilesTransformer().transform(smiles_list)
    mols_array = np.array(mols)

    splits = _split_data(mols, mols_array, y, test_size=test_size, use_valid=use_valid)

    logger.info("Standardizing molecules...")
    standardizer = MolStandardizer()
    splits = {
        split: (standardizer.transform(mols_split), y_split)
        for split, (mols_split, y_split) in splits.items()
    }

    _log_split_sizes(splits)
    return splits


def _log_input_summary(smiles_list: list[str], y: np.ndarray) -> None:
    """Log summary of input data.

    Args:
        smiles_list: List of SMILES strings
        y: Target values
    """
    logger.info("\nPreprocessing Data:")
    logger.info("Total number of molecules: %d", len(smiles_list))
    logger.info("Example SMILES: %s", smiles_list[0])
    logger.info("Example label: %s", y[0])
    if len(y[0].shape) > 0:
        logger.info("Number of tasks: %d", y[0].shape[0])


def _split_data(
    mols: list,
    mols_array: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float,
    use_valid: bool,
) -> dict:
    """Split data into train/valid/test sets.

    Args:
        mols: List of molecular objects
        mols_array: Array of molecular data
        y: Target values
        test_size: Size of test split
        use_valid: Whether to include validation split

    Returns:
        Dictionary containing data splits
    """
    if use_valid:
        logger.info("Performing train/valid/test split using scaffold splitting...")
        train_idx, valid_idx, test_idx = scaffold_train_valid_test_split(
            mols, train_size=0.8, valid_size=0.1, test_size=0.1, return_indices=True
        )
        return {
            "train": (mols_array[train_idx], y[train_idx]),
            "valid": (mols_array[valid_idx], y[valid_idx]),
            "test": (mols_array[test_idx], y[test_idx]),
        }

    logger.info("Performing train/test split using scaffold splitting...")
    train_idx, test_idx = scaffold_train_test_split(
        mols, test_size=test_size, return_indices=True
    )
    return {
        "train": (mols_array[train_idx], y[train_idx]),
        "test": (mols_array[test_idx], y[test_idx]),
    }


def _log_split_sizes(splits: dict) -> None:
    """Log sizes of data splits.

    Args:
        splits: Dictionary containing data splits
    """
    logger.info("\nDataset split sizes:")
    for split, (mols, _) in splits.items():
        logger.info("%s set size: %d", split.capitalize(), len(mols))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metrics: list[str] | None = None,
) -> dict:
    """Evaluate model predictions.

    Args:
        y_true: True target values
        y_pred_proba: Predicted probabilities
        metrics: List of metrics to evaluate

    Returns:
        Dictionary containing evaluation results
    """
    if metrics is None:
        metrics = ["auroc", "auprc"]

    metric_functions = {
        "auroc": multioutput_auroc_score,
        "auprc": multioutput_auprc_score,
    }

    results = {}
    for metric in metrics:
        if metric in metric_functions:
            results[metric] = metric_functions[metric](y_true, y_pred_proba)

    return results


def log_evaluation_results(results: dict, task_names: list[str] | None = None) -> None:
    """Log evaluation results.

    Args:
        results: Dictionary containing evaluation results
        task_names: Optional list of task names
    """
    logger.info("\nEvaluation Results:")
    logger.info("-" * 50)

    for metric in ["auroc", "auprc"]:
        if metric in results:
            logger.info("\n%s Scores:", metric.upper())
            scores = results[metric]

            if isinstance(scores, (list, np.ndarray)):
                for i, score in enumerate(scores):
                    task_label = f"Task {i}" if task_names is None else task_names[i]
                    logger.info("%s: %.3f", task_label, score)
            else:
                logger.info("Score: %.3f", scores)
