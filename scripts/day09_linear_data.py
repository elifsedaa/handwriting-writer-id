"""Gün 9: Doğrusal regresyon oyuncak verisi, eğitim/test bölmesi ve görselleştirme."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

from src.data.synthetic import make_linear_data
from src.utils import fig_path, set_seed
from src.viz import plot_predictions

set_seed(42)
WEIGHT, BIAS = 0.7, 0.3
X_train, y_train, X_test, y_test = make_linear_data(WEIGHT, BIAS, 0.0, 1.0, 0.02, 0.8)

print(f"Gerçek formül: y = {WEIGHT} * X + {BIAS}")
print("X_train:", tuple(X_train.shape), " y_train:", tuple(y_train.shape))
print("X_test :", tuple(X_test.shape), " y_test :", tuple(y_test.shape))
print("İlk 5 X:", X_train[:5].squeeze().tolist())
print("İlk 5 y:", [round(v, 3) for v in y_train[:5].squeeze().tolist()])

out = fig_path("day09_linear_data.png")
plot_predictions(X_train, y_train, X_test, y_test, title="Doğrusal regresyon verisi", path=out)
print("Grafik kaydedildi:", out)
print("\nNot: Proje verisinde de eğitim/test ayrımı sayfa/yazar bazında yapılacak (rastgele patch bölmesi değil).")
