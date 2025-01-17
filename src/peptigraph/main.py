"""Main module of PeptiGraph."""

import sys
from pathlib import Path

from utils.logger import get_logger

src_path = str(Path(__file__).resolve().parent.parent.parent / "src")
sys.path.insert(0, src_path)


logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Starting the PeptiGraph application...")
