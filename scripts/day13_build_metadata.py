"""Gün 13: all_samples.csv üretir. Firemaker dosya adı düzeni biliniyorsa --firemaker-regex ver."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.build_metadata import main

if __name__ == "__main__":
    main()
