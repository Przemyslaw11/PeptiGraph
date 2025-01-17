"""Module for converting peptide sequences to SMILES notation and processing sequence data."""

from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem

from utils.logger import get_logger

logger = get_logger(__name__)


class SequenceProcessor:
    """Process peptide sequences and convert them to SMILES notation."""

    def __init__(
        self: "SequenceProcessor",
        input_dir: str = "../data/processed",
        output_dir: str = "../data/smiles",
    ) -> None:
        """Initialize the SequenceProcessor.

        Args:
            input_dir: Directory containing input CSV files
            output_dir: Directory where processed files will be saved
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

    def sequence_to_smiles(self: "SequenceProcessor", sequence: str) -> str | None:
        """Convert a peptide sequence to SMILES notation.

        Args:
            sequence: Peptide sequence string

        Returns:
            SMILES string if conversion successful, None otherwise
        """
        if not sequence or not isinstance(sequence, str):
            logger.warning("Invalid sequence input: %s", sequence)
            return None

        try:
            mol = Chem.MolFromSequence(sequence)
            if mol is None:
                logger.warning("Failed to create molecule from sequence: %s", sequence)
                return None
            return Chem.MolToSmiles(mol)
        except ValueError:
            logger.exception("Error converting sequence %s", sequence)
            return None

    def process_file(
        self: "SequenceProcessor", input_path: str, output_path: str
    ) -> pd.DataFrame | None:
        """Process a single CSV file, converting sequences to SMILES.

        Args:
            input_path: Path to input CSV file
            output_path: Path where processed file will be saved

        Returns:
            Processed DataFrame if successful, None otherwise
        """
        try:
            logger.info("Reading file: %s", input_path)
            sequences_df = pd.read_csv(input_path)

            if "sequence" not in sequences_df.columns:
                logger.error("Required column 'sequence' not found in %s", input_path)
                return None

            logger.info("Converting %d sequences to SMILES", len(sequences_df))
            sequences_df["smiles"] = sequences_df["sequence"].apply(
                self.sequence_to_smiles
            )

            # Count successful conversions
            success_count = sequences_df["smiles"].notna().sum()
            logger.info(
                "Successfully converted %d/%d sequences",
                success_count,
                len(sequences_df),
            )

            # Keep only necessary columns
            result_df = sequences_df[["sequence", "smiles", "target"]]

            # Save processed data
            result_df.to_csv(output_path, index=False)
            logger.info("Saved processed data to %s", output_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            logger.exception("Error processing file %s", input_path)
            return None
        else:
            return result_df

    def process_all_files(self: "SequenceProcessor") -> dict[str, Any]:
        """Process all CSV files in the input directory.

        Returns:
            Dictionary containing processing statistics
        """
        output_path = Path(self.output_dir)
        if not output_path.exists():
            output_path.mkdir(parents=True)
            logger.info("Created output directory: %s", self.output_dir)

        stats = {
            "files_processed": 0,
            "total_sequences": 0,
            "successful_conversions": 0,
            "failed_files": [],
        }

        logger.info("Starting SMILES conversion process...")

        for filename in Path(self.input_dir).iterdir():
            if not filename.name.endswith(".csv"):
                continue

            input_path = Path(self.input_dir) / filename.name
            output_path = Path(self.output_dir) / filename.name

            result_df = self.process_file(str(input_path), str(output_path))

            if result_df is not None:
                stats["files_processed"] += 1
                stats["total_sequences"] += len(result_df)
                stats["successful_conversions"] += result_df["smiles"].notna().sum()
            else:
                stats["failed_files"].append(filename.name)

        logger.info("Conversion process completed")
        logger.info("Processed %d files", stats["files_processed"])
        logger.info(
            "Converted %d/%d sequences",
            stats["successful_conversions"],
            stats["total_sequences"],
        )

        if stats["failed_files"]:
            logger.warning(
                "Failed to process %d files: %s",
                len(stats["failed_files"]),
                stats["failed_files"],
            )

        return stats


def main() -> dict[str, Any]:
    """Execute the sequence processing pipeline."""
    processor = SequenceProcessor()
    return processor.process_all_files()


if __name__ == "__main__":
    main()
