"""Configuration settings for PeptiGraph application."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOG_DIR = Path(os.getenv('LOG_DIR', 'logs'))
