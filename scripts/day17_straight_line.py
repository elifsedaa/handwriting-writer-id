"""Gün 17: Düz çizgi verisi: modelimiz hiç öğreniyor mu? (kursun hata ayıklama numarası)

TEST EDİLMEDİ (torch). Fikir: sınıflandırma verisinde öğrenemeyen model, çok basit bir regresyon verisinde
öğreniyorsa sorun kodda değil, modelin kapasitesinde/doğrusal oluşunda olabilir.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import torch
from torch import nn

from src.utils import fig_path, get_device, set_seed
from src.viz import plot_predictions

set_seed(42)
device = get_device()
X = torch.arange(0, 1, 0.01).unsqueeze(1)
y = 0.7 * X + 0.3
split = int(0.8 * len(X))
X_train, y_train, X_test, y_test = (t.to(device) for t in (X[:split], y[:split], X[split:], y[split:]))
print("Şekiller:", tuple(X_train.shape), tuple(X_test.shape))

model = nn.Sequential(nn.Linear(1, 10), nn.Linear(10, 10), nn.Linear(10, 1)).to(device)
loss_fn = nn.L1Loss()
opt = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    model.train()
    loss = loss_fn(model(X_train), y_train)
    opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 100 == 0:
        model.eval()
        with torch.inference_mode():
            tl = loss_fn(model(X_test), y_test)
        print(f"epoch {epoch:4d} | train loss {loss:.4f} | test loss {tl:.4f}")

model.eval()
with torch.inference_mode():
    preds = model(X_test)
out = fig_path("day17_straight_line.png")
plot_predictions(X_train.cpu(), y_train.cpu(), X_test.cpu(), y_test.cpu(), predictions=preds.cpu(),
                 title="Düz çizgi verisinde tahminler", path=out)
print("Grafik:", out)
