"""Make the adjacent source-layout package importable during root discovery."""

from pathlib import Path
import sys


TRAINING_ROOT = str(Path(__file__).resolve().parents[1])
if TRAINING_ROOT not in sys.path:
    sys.path.insert(0, TRAINING_ROOT)
