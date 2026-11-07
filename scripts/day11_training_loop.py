"""Gün 11: Kursun eğitim döngüsü (doğrusal regresyon). Kayıp, optimizer, 5 adım.

TEST EDİLMEDİ (yazıldığı ortamda torch yoktu) -- ilk çalıştırmada çıktıyı kontrol et.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from src.data.synthetic import make_linear_data
from src.models.linear_regression import LinearRegressionModel
from src.utils import fig_path, set_seed

set_seed(42)
X_train, y_train, X_test, y_test = make_linear_data()
model = LinearRegressionModel()
loss_fn = nn.L1Loss()                                   # MAE
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 200

epoch_log, train_losses, test_losses = [], [], []
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)                # 1) ileri geçiş
    loss = loss_fn(y_pred, y_train)        # 2) kayıp
    optimizer.zero_grad()                  # 3) gradyanı sıfırla
    loss.backward()                        # 4) geri yayılım
    optimizer.step()                       # 5) parametre güncelle

    model.eval()
    with torch.inference_mode():
        test_loss = loss_fn(model(X_test), y_test)
    if epoch % 10 == 0:
        epoch_log.append(epoch)
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        print(f"Epoch {epoch:3d} | train loss {loss.item():.4f} | test loss {test_loss.item():.4f}")

print("\nÖğrenilen:", dict(model.state_dict()), "| Gerçek: weights=0.7, bias=0.3")

plt.figure(figsize=(6, 4))
plt.plot(epoch_log, train_losses, label="train")
plt.plot(epoch_log, test_losses, label="test")
plt.xlabel("epoch"); plt.ylabel("L1 kayıp"); plt.title("Kayıp eğrileri"); plt.legend()
out = fig_path("day11_loss_curves.png")
plt.tight_layout(); plt.savefig(out, dpi=150)
print("Grafik:", out)
