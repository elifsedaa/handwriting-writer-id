"""Gün 17: all_samples.csv'deki (split_closed) sayfalardan patch veri seti üretir + grafikleri çizer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.patch_dataset import main

if __name__ == "__main__":
    main()
