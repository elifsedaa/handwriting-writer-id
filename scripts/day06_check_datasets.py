"""Gün 6: Veri seti durumunu raporlar. Çalıştırma: python scripts/day06_check_datasets.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.check_datasets import main

if __name__ == "__main__":
    main()
