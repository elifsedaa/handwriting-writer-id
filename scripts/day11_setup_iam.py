"""Gün 11: IAM arşivlerini aç ve doğrula (data/raw/iam/_downloads -> data/raw/iam/<ad>/)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.iam_setup import main

if __name__ == "__main__":
    main()
