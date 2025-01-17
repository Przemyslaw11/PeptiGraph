"""Module for creating and processing various peptide datasets."""

from pathlib import Path
from typing import dict, tuple
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class DatasetCreator:
    """Class for creating and processing different peptide datasets."""

    def __init__(self: "DatasetCreator", data_dir: str = "data") -> None:
        """Initialize DatasetCreator.

        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialized DatasetCreator with data directory: %s", self.data_dir)

    def read_fasta_file(self: "DatasetCreator", file_path: str) -> list[str]:
        """Read sequences from a FASTA file."""
        sequences: list[str] = []
        current_sequence = ""

        with Path(file_path).open() as fasta_file:
            for current_line in fasta_file:
                stripped_line = current_line.strip()
                if stripped_line.startswith(">"):
                    if current_sequence:
                        sequences.append(current_sequence)
                    current_sequence = ""
                else:
                    current_sequence += stripped_line
            if current_sequence:
                sequences.append(current_sequence)

        logger.debug("Read %d sequences from %s", len(sequences), file_path)
        return sequences

    def create_bioactive_peptides(self: "DatasetCreator") -> pd.DataFrame:
        """Process bioactive peptides dataset."""
        logger.info("Processing Bioactive Peptides dataset...")
        bioactive_df = pd.read_csv(
            self.data_dir / "bioactive_peptides" / "bioactive_peptides.csv"
        )
        bioactive_df = bioactive_df[["Sequence", "Class"]]
        bioactive_df["target"] = (bioactive_df["Class"] == "A").astype(int)
        logger.info("Processed %d bioactive peptide sequences", len(bioactive_df))
        return bioactive_df[["Sequence", "target"]].rename(
            columns={"Sequence": "sequence"}
        )

    def create_grampa(self: "DatasetCreator") -> tuple[pd.DataFrame, dict[str, int]]:
        """Process GRAMPA dataset."""
        logger.info("Processing GRAMPA dataset...")
        grampa_df = pd.read_csv(self.data_dir / "grampa" / "grampa.csv")
        grampa_df = grampa_df.drop_duplicates(subset=["sequence"], keep="first")
        bacteria_mapping = {
            bacteria: idx
            for idx, bacteria in enumerate(grampa_df["is_modified"].unique())
        }
        grampa_df["target"] = grampa_df["is_modified"].map(bacteria_mapping)
        logger.info("Processed %d GRAMPA sequences", len(grampa_df))
        logger.debug("GRAMPA bacteria mapping: %s", bacteria_mapping)
        return grampa_df[["sequence", "target"]], bacteria_mapping

    def create_hemopi(self: "DatasetCreator", version: int) -> pd.DataFrame:
        """Process HemoPI dataset.

        Args:
            version: HemoPI version (1-3)
        """
        logger.info("Processing HemoPI-%d dataset...", version)
        dataset_path = self.data_dir / f"hemoPI-{version}"
        pos_sequences = self.read_fasta_file(dataset_path / "positive.txt")
        neg_sequences = self.read_fasta_file(dataset_path / "negative.txt")

        sequences = pos_sequences + neg_sequences
        targets = [1] * len(pos_sequences) + [0] * len(neg_sequences)

        logger.info(
            "Processed HemoPI-%d: %d positive and %d negative sequences",
            version,
            len(pos_sequences),
            len(neg_sequences),
        )
        return pd.DataFrame({"sequence": sequences, "target": targets})

    def create_versa(self: "DatasetCreator") -> tuple[pd.DataFrame, dict[str, int]]:
        """Process VERSA dataset."""
        logger.info("Processing VERSA dataset...")
        files = ["amp.fasta", "namp_faba.fasta", "namp_viri.fasta"]
        target_mapping = {"amp": 0, "namp_faba": 1, "namp_viri": 2}

        sequences = []
        targets = []

        for file in files:
            file_path = self.data_dir / "versa" / file
            target_name = file.split(".")[0]

            file_sequences = self.read_fasta_file(str(file_path))
            sequences.extend(file_sequences)
            targets.extend([target_mapping[target_name]] * len(file_sequences))
            logger.debug("Processed %d sequences from %s", len(file_sequences), file)

        logger.info("Processed total of %d VERSA sequences", len(sequences))
        logger.debug("VERSA target mapping: %s", target_mapping)
        return pd.DataFrame({"sequence": sequences, "target": targets}), target_mapping

    def create_all_datasets(
        self: "DatasetCreator",
    ) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, int]]]:
        """Create all available datasets."""
        logger.info("Starting creation of all datasets...")
        datasets: dict[str, pd.DataFrame] = {}
        mappings: dict[str, dict[str, int]] = {}

        try:
            datasets["bioactive_peptides"] = self.create_bioactive_peptides()
            datasets["grampa"], mappings["grampa"] = self.create_grampa()

            for i in range(1, 4):
                try:
                    datasets[f"hemopi_{i}"] = self.create_hemopi(i)
                except (OSError, ValueError):
                    logger.exception("Error processing HemoPI-%d", i)

            datasets["versa"], mappings["versa"] = self.create_versa()

            for name, dataset_df in datasets.items():
                output_path = self.processed_dir / f"{name}.csv"
                dataset_df.to_csv(output_path, index=False)
                logger.info("Saved %s dataset to %s", name, output_path)

        except (OSError, ValueError):
            logger.exception("Error during dataset creation")
            raise

        logger.info("Successfully completed creation of all datasets")
        return datasets, mappings


if __name__ == "__main__":
    creator = DatasetCreator()
    datasets, mappings = creator.create_all_datasets()

    for dataset_name, mapping in mappings.items():
        logger.info("%s mapping:", dataset_name.upper())
        for key, value in mapping.items():
            logger.info("%s: %d", key, value)
