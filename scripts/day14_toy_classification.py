"""Gün 14: Sınıflandırma oyuncak verisi (iki daire): girdi/çıktı şekilleri ve grafik.

TEST EDİLMEDİ (matplotlib/sklearn ile çalışır, torch gerekmez ama bu ortamda çalıştırılmadı). sklearn gerekir: pip install scikit-learn
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

from src.utils import fig_path

X, y = make_circles(1000, noise=0.03, random_state=42)
print("X:", X.shape, "y:", y.shape, "| ilk 3 örnek:", X[:3].round(3).tolist(), y[:3].tolist())
print("Girdi özellik sayısı: 2  |  Çıktı: tek değer (0 veya 1)  |  sınıf başına örnek:",
      {int(k): int((y == k).sum()) for k in set(y)})

plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, s=8)
plt.title("İki daire verisi")
out = fig_path("day14_circles.png")
plt.tight_layout()
plt.savefig(out, dpi=150)
print("Grafik:", out)
