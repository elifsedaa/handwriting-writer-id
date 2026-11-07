"""Gün 14: all_samples.csv'ye split_closed (sayfa bazında) ve split_open (yazar bazında) ekler."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splits import main

if __name__ == "__main__":
    main()
