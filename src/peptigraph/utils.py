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
) -> tuple:
    """Loads and preprocesses the data by performing splitting and logging details.

    Args:
        smiles_list (list[str]): List of SMILES strings.
        y (np.ndarray): Labels corresponding to SMILES.
        test_size (float, optional): Fraction for test split. Defaults to 0.2.
        use_valid (bool, optional): Whether to use validation split. Defaults to False.

    Returns:
        tuple: Train, validation, and test splits.
    """
    _log_input_summary(smiles_list, y)

    transformer = MolFromSmilesTransformer()
    standardizer = MolStandardizer()

    mols = transformer.transform(smiles_list)
    mols = standardizer.transform(mols)
    mols_array = np.array([mol.to_numpy() for mol in mols])

    return _split_data(mols, mols_array, y, test_size=test_size, use_valid=use_valid)


def _log_input_summary(smiles_list: list[str], y: np.ndarray) -> None:
    """Logs summary of the input data.

    Args:
        smiles_list (list[str]): List of SMILES strings.
        y (np.ndarray): Labels corresponding to SMILES.
    """
    logger.info("Preprocessing Data:")
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
    """Splits data into train, validation, and test sets.

    Args:
        mols (list): List of molecular objects.
        mols_array (np.ndarray): Numpy array of molecular data.
        y (np.ndarray): Labels corresponding to SMILES.
        test_size (float): Fraction for test split.
        use_valid (bool): Whether to use validation split.

    Returns:
        dict: Dictionary containing the splits.
    """
    if use_valid:
        logger.info("Performing train/valid/test split using scaffold splitting...")
        return scaffold_train_valid_test_split(mols, mols_array, y, test_size=test_size)

    logger.info("Performing train/test split using scaffold splitting...")
    return scaffold_train_test_split(mols, mols_array, y, test_size=test_size)


def _log_split_sizes(splits: dict) -> None:
    """Logs the sizes of each split.

    Args:
        splits (dict): Dictionary containing the splits.
    """
    logger.info("Dataset split sizes:")
    for split, (mols, _) in splits.items():
        logger.info("%s set size: %d", split.capitalize(), len(mols))


def evaluate_predictions(
    y_true: np.ndarray, y_pred_proba: np.ndarray, metrics: list[str] | None = None
) -> dict:
    """Evaluates predictions using specified metrics.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred_proba (np.ndarray): Predicted probabilities.
        metrics (list[str], optional): List of metrics. Defaults to ["auroc", "auprc"].

    Returns:
        dict: Evaluation results.
    """
    if metrics is None:
        metrics = ["auroc", "auprc"]

    results = {}
    for metric in metrics:
        if metric == "auroc":
            results[metric] = multioutput_auroc_score(y_true, y_pred_proba)
        elif metric == "auprc":
            results[metric] = multioutput_auprc_score(y_true, y_pred_proba)
    return results


def log_evaluation_results(results: dict, task_names: list[str] | None = None) -> None:
    """Logs the evaluation results.

    Args:
        results (dict): Evaluation results.
        task_names (list[str], optional): List of task names. Defaults to None.
    """
    logger.info("\nEvaluation Results:")
    logger.info("-" * 50)

    for metric in ["auroc", "auprc"]:
        if metric in results:
            logger.info("\n%s Scores:", metric.upper())
            scores = results[metric]

            if isinstance(scores, list | np.ndarray):
                for i, score in enumerate(scores):
                    task_label = f"Task {i}" if task_names is None else task_names[i]
                    logger.info("%s: %.3f", task_label, score)
            else:
                logger.info("Score: %.3f", scores)
