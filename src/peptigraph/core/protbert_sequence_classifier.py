"""Module for ProtBERT-based sequence classification."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from transformers import PreTrainedModel, PreTrainedTokenizer

from utils import get_logger

logger = get_logger(__name__)


class ProtBertSequenceClassifier:
    """Sequence classifier using ProtBERT embeddings and Random Forest."""

    def __init__(
        self: ProtBertSequenceClassifier,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        batch_size: int = 32,
        n_estimators: int = 400,
        max_depth: int = 20,
        random_state: int = 42,
    ) -> None:
        """Initialize the ProtBERT sequence classifier.

        Args:
            tokenizer: Pre-trained tokenizer for sequence encoding
            model: Pre-trained ProtBERT model
            batch_size: Batch size for embedding generation
            n_estimators: Number of trees in random forest
            max_depth: Maximum depth of trees
            random_state: Random state for reproducibility
        """
        self.tokenizer = tokenizer
        self.model = model
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )

        logger.info(
            "Initialized ProtBertSequenceClassifier",
            extra={
                "device": self.device.type,
                "batch_size": batch_size,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
            },
        )

    def get_embeddings(
        self: ProtBertSequenceClassifier, sequences: list[str]
    ) -> np.ndarray:
        """Generate embeddings for input sequences using ProtBERT.

        Args:
            sequences: List of protein sequences

        Returns:
            Array of sequence embeddings
        """
        logger.info("Generating embeddings for %d sequences", len(sequences))
        all_embeddings = []

        for i in range(0, len(sequences), self.batch_size):
            batch_sequences = sequences[i : i + self.batch_size]
            batch_sequences = [" ".join(seq) for seq in batch_sequences]

            encoded = self.tokenizer.batch_encode_plus(
                batch_sequences,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return np.vstack(all_embeddings)

    def train_and_evaluate(
        self: ProtBertSequenceClassifier,
        x_train_embeddings: np.ndarray,
        x_test_embeddings: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        np.ndarray,
        np.ndarray,
        float,
    ]:
        """Train classifier and evaluate performance.

        Args:
            x_train_embeddings: Training set embeddings
            x_test_embeddings: Test set embeddings
            y_train: Training set labels
            y_test: Test set labels

        Returns:
            Tuple containing predictions and evaluation metrics
        """
        logger.info("Training and evaluating classifier")
        self.classifier.fit(x_train_embeddings, y_train)

        y_pred = self.classifier.predict(x_test_embeddings)
        y_pred_proba = self.classifier.predict_proba(x_test_embeddings)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auroc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auprc = auc(recall, precision)

        logger.info(
            "Model evaluation complete",
            extra={"auroc": auroc, "auprc": auprc},
        )

        return y_pred, y_pred_proba, fpr, tpr, auroc, precision, recall, auprc

    def plot_metrics(
        self: ProtBertSequenceClassifier,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auroc: float,
        precision: np.ndarray,
        recall: np.ndarray,
        auprc: float,
        dataset_name: str,
    ) -> None:
        """Plot ROC and Precision-Recall curves.

        Args:
            fpr: False positive rates
            tpr: True positive rates
            auroc: Area under ROC curve
            precision: Precision values
            recall: Recall values
            auprc: Area under PR curve
            dataset_name: Name of dataset for plot titles
        """
        logger.info("Plotting performance metrics for %s", dataset_name)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
        ax1.plot([0, 1], [0, 1], "k--")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title(f"ROC Curve - {dataset_name}")
        ax1.legend()
        ax1.grid(visible=True)

        ax2.plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title(f"Precision-Recall Curve - {dataset_name}")
        ax2.legend()
        ax2.grid(visible=True)

        plt.tight_layout()
        plt.show()

    def analyze_dataset(
        self: ProtBertSequenceClassifier,
        sequences: np.ndarray,
        targets: np.ndarray,
        dataset_name: str,
    ) -> tuple[float, float]:
        """Analyze a dataset using the ProtBERT classifier.

        Args:
            sequences: Array of protein sequences
            targets: Array of target labels
            dataset_name: Name of the dataset

        Returns:
            Tuple of (AUROC, AUPRC) scores
        """
        logger.info("Analyzing dataset: %s", dataset_name)

        x_train, x_test, y_train, y_test = train_test_split(
            sequences, targets, test_size=0.2, random_state=42, stratify=targets
        )

        x_train_embeddings = self.get_embeddings(x_train)
        x_test_embeddings = self.get_embeddings(x_test)

        y_pred, y_pred_proba, fpr, tpr, auroc, precision, recall, auprc = (
            self.train_and_evaluate(
                x_train_embeddings, x_test_embeddings, y_train, y_test
            )
        )

        logger.info(
            "\nModel Performance Metrics:",
            extra={
                "dataset": dataset_name,
                "auroc": f"{auroc:.3f}",
                "auprc": f"{auprc:.3f}",
            },
        )

        logger.info("\nDetailed Classification Report:")
        logger.info(classification_report(y_test, y_pred))

        self.plot_metrics(fpr, tpr, auroc, precision, recall, auprc, dataset_name)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return auroc, auprc
