"""Gün 18: İlk 18 gün özet raporu -> docs/ilk_18_gun_ozeti.md (sayılar gerçek dosyalardan okunur)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.report import main

if __name__ == "__main__":
    main()
