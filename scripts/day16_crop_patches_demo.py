"""Gün 16: Kırpma ve patch çıkarma denemesi (NumPy). Bir sayfada adımları görselleştirir.

Kullanım: python scripts/day16_crop_patches_demo.py [görüntü_yolu]
Yol verilmezse all_samples.csv'deki ilk görüntü kullanılır.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import Config
from src.data.crop import crop_to_ink, extract_patches_np, filter_blank_np, pad_min_size
from src.data.preprocess import load_gray
from src.utils import ROOT, fig_path

cfg = Config()
if len(sys.argv) > 1:
    path = Path(sys.argv[1])
else:
    csv = ROOT / "data" / "metadata" / "all_samples.csv"
    meta = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    if meta.empty:
        sys.exit("Yol ver ya da önce all_samples.csv üret (day13_build_metadata.py).")
    path = ROOT / meta.iloc[0]["path"]

page = load_gray(path)
cropped = crop_to_ink(page, margin=16)
P = cfg.patch_size
padded = pad_min_size(cropped, P)
patches = extract_patches_np(padded, P)
kept, ratios = filter_blank_np(patches, min_ink=0.02)
print(f"Sayfa {page.shape} -> kırpılmış {cropped.shape} -> patch {len(patches)} adet -> boş elendikten sonra {len(kept)}")
print(f"Elenen boş patch oranı: {1 - len(kept) / max(len(patches), 1):.1%}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(page, cmap="gray"); axes[0].set_title(f"Ham {page.shape}")
axes[1].imshow(cropped, cmap="gray"); axes[1].set_title(f"Kırpılmış {cropped.shape}")
for a in axes: a.axis("off")
fig.tight_layout(); p1 = fig_path("day16_crop.png"); fig.savefig(p1, dpi=130); plt.close(fig)

n = min(8, len(kept))
if n:
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.4), squeeze=False)
    for a, k in zip(axes[0], range(n)):
        a.imshow(kept[k], cmap="gray", vmin=0, vmax=1); a.axis("off"); a.set_title(f"ink {ratios[k]:.2f}", fontsize=8)
    fig.tight_layout(); p2 = fig_path("day16_patches.png"); fig.savefig(p2, dpi=130); plt.close(fig)
    print("Grafikler:", p1, p2)
