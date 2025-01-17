"""Module for processing and managing molecular datasets."""

from pathlib import Path
import pandas as pd
import numpy as np

from skfp.datasets.moleculenet import load_bace, load_sider
from skfp.datasets.lrgb import load_peptides_func
from utils.logger import get_logger


logger = get_logger(__name__)


class DatasetProcessor:
    """Class for processing and managing molecular datasets."""

    def __init__(self, data_dir: str = "../data") -> None:
        """Initialize DatasetProcessor.

        Args:
            data_dir: Path to data directory (default: "../data")
        """
        self.data_dir = Path(data_dir)
        self.datasets = [
            (load_bace, "bace"),
            (load_sider, "sider"),
            (load_peptides_func, "peptides"),
        ]
        self._ensure_data_directory()

    def _ensure_data_directory(self) -> None:
        """Create data directory if it doesn't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _create_dataframe(
        self, smiles: list, labels: list, task_index: int = 0
    ) -> pd.DataFrame:
        """Create DataFrame from SMILES and labels.

        Args:
            smiles: List of SMILES strings
            labels: List of labels
            task_index: Index of task for multi-task datasets (default: 0)

        Returns:
            DataFrame with SMILES and target columns
        """
        labels_array = np.array(labels)
        if len(labels_array.shape) > 1:
            labels_array = labels_array[:, task_index]

        return pd.DataFrame({"smiles": np.array(smiles), "target": labels_array})

    def _validate_dataframe(self, df: pd.DataFrame) -> dict:
        """Return validation statistics for the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary containing validation statistics
        """
        return {
            "total_samples": len(df),
            "unique_smiles": df["smiles"].nunique(),
            "unique_targets": df["target"].nunique(),
            "has_missing": df.isna.any().any(),
            "label_distribution": df["target"].value_counts(),
        }

    def print_dataset_info(self, name: str, df: pd.DataFrame, stats: dict) -> None:
        """Print dataset information and validation statistics.

        Args:
            name: Dataset name
            df: Dataset DataFrame
            stats: Validation statistics
        """
        logger.info("\n%s Dataset:", name.upper())
        logger.info("SMILES shape: %s", df["smiles"].shape)
        logger.info("Total samples: %d", stats["total_samples"])
        logger.info("Number of unique SMILES: %d", stats["unique_smiles"])
        logger.info("Number of unique targets: %d", stats["unique_targets"])
        logger.info("Any missing values: %s", stats["has_missing"])
        logger.info("Label distribution:\n%s", stats["label_distribution"])

    def save_dataset(self, name: str, df: pd.DataFrame) -> str:
        """Save DataFrame to CSV.

        Args:
            name: Dataset name
            df: Dataset DataFrame

        Returns:
            Path to saved file

        Raises:
            Exception: If saving fails
        """
        try:
            filename = self.data_dir / f"{name}_original.csv"
            df.to_csv(filename, index=False)
            return str(filename)
        except Exception as e:
            logger.info("Error saving dataset %s: %s", name, e)
            raise

    def process_dataset(self, loader: callable, name: str) -> str:
        """Load, process, and save a dataset.

        Args:
            loader: Function to load dataset
            name: Dataset name

        Returns:
            Path to saved file
        """
        smiles, labels = loader()
        dataset_df = self._create_dataframe(smiles, labels)
        stats = self._validate_dataframe(dataset_df)
        filename = self.save_dataset(name, dataset_df)
        self._logger.info_dataset_info(name, dataset_df, stats)
        return filename

    def process_all_datasets(self) -> list[str]:
        """Process all configured datasets.

        Returns:
            List of paths to saved files
        """
        files = []

        for loader, name in self.datasets:
            logger.info("\nProcessing %s dataset...", name.upper())
            filename = self.process_dataset(loader, name)
            files.append(filename)

        logger.info("\nAll datasets have been saved successfully!")
        logger.info("Files created:")
        for file in files:
            logger.info("- %s", file)

        return files


if __name__ == "__main__":
    processor = DatasetProcessor()
    processor.process_all_datasets()
