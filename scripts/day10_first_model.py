"""Gün 10: İlk PyTorch modeli (nn.Module), parametreler, state_dict ve inference_mode."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import torch

from src.data.synthetic import make_linear_data
from src.models.linear_regression import LinearRegressionModel
from src.utils import fig_path, set_seed
from src.viz import plot_predictions

set_seed(42)
X_train, y_train, X_test, y_test = make_linear_data()

model = LinearRegressionModel()
print(model)

print("\n=== Parametreler (rastgele başladı) ===")
for name, p in model.named_parameters():
    print(f"  {name:8s} değer={p.data.tolist()}  requires_grad={p.requires_grad}")
print("state_dict:", dict(model.state_dict()))
print("Gerçek değerler: weights=0.7, bias=0.3")

print("\n=== Tahmin: inference_mode ile ve olmadan ===")
y_grad = model(X_test)
print("Olmadan  -> requires_grad:", y_grad.requires_grad, " grad_fn:", y_grad.grad_fn is not None)
with torch.inference_mode():
    y_preds = model(X_test)
print("inference_mode -> requires_grad:", y_preds.requires_grad)

mae = torch.mean(torch.abs(y_preds - y_test)).item()
print(f"Eğitimsiz modelin test hatası (MAE): {mae:.4f}")

out = fig_path("day10_random_model_preds.png")
plot_predictions(X_train, y_train, X_test, y_test, predictions=y_preds,
                 title="Eğitilmemiş model tahminleri", path=out)
print("Grafik kaydedildi:", out)
