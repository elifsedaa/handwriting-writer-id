"""Gün 15: Ön işleme hattını bir görüntüde dene; önce/sonra grafiği çıkar.

Kullanım: python scripts/day15_preprocess_demo.py [görüntü_yolu]
Yol verilmezse all_samples.csv'deki ilk görüntü kullanılır.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data.preprocess import load_gray, preprocess_line
from src.utils import ROOT, fig_path

if len(sys.argv) > 1:
    path = Path(sys.argv[1])
else:
    csv = ROOT / "data" / "metadata" / "all_samples.csv"
    if not csv.exists():
        sys.exit("Yol ver ya da önce all_samples.csv üret (day13_build_metadata.py).")
    meta = pd.read_csv(csv)
    if meta.empty:
        sys.exit("all_samples.csv boş; görüntü yolunu elle ver.")
    path = ROOT / meta.iloc[0]["path"]

raw = load_gray(path)
out = preprocess_line(path, height=64, width=512)
print(f"Ham: {raw.shape}, Ön işlenmiş: {out.shape}, değer aralığı [{out.min():.2f}, {out.max():.2f}]")

fig, axes = plt.subplots(2, 1, figsize=(8, 5))
axes[0].imshow(raw, cmap="gray"); axes[0].set_title(f"Ham {raw.shape}")
axes[1].imshow(out, cmap="gray"); axes[1].set_title(f"Ön işlenmiş {out.shape}")
for a in axes: a.axis("off")
p = fig_path("day15_preprocess.png"); fig.tight_layout(); fig.savefig(p, dpi=150)
print("Grafik:", p)
