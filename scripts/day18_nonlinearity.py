"""Gün 18: Doğrusal olmayanlık (ReLU) ile daire verisi + aktivasyonları saf PyTorch ile yazma.

TEST EDİLMEDİ (torch). BEKLENTİ: ReLU'lu modelin karar sınırı eğrisel olur; sonuç sayısını kendi çıktına göre yaz.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn

from src.models.activations import relu, sigmoid
from src.models.toy_classifier import CircleModelV2
from src.utils import fig_path, get_device, set_seed
from src.viz import plot_decision_boundary

# --- 1) Aktivasyonlar: saf PyTorch ile yazılanlar yerleşik olanlarla aynı mı? ---
x = torch.arange(-10, 10, 1.0)
print("relu    saf == yerleşik:", torch.allclose(relu(x), torch.relu(x)))
print("sigmoid saf == yerleşik:", torch.allclose(sigmoid(x), torch.sigmoid(x)))
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].plot(x, relu(x)); axes[0].set_title("ReLU")
axes[1].plot(x, sigmoid(x)); axes[1].set_title("Sigmoid")
p = fig_path("day18_activations.png"); fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
print("Grafik:", p)

# --- 2) ReLU'lu model daire verisinde ---
set_seed(42)
device = get_device()
X, y = make_circles(1000, noise=0.03, random_state=42)
X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = (t.to(device) for t in (X_train, X_test, y_train, y_test))

model = CircleModelV2().to(device)
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)


def accuracy(pred, true):
    return (pred == true).sum().item() / len(true) * 100


for epoch in range(1001):
    model.train()
    logits = model(X_train).squeeze()
    loss = loss_fn(logits, y_train)
    opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 100 == 0:
        model.eval()
        with torch.inference_mode():
            tl = model(X_test).squeeze()
            tloss = loss_fn(tl, y_test)
            tacc = accuracy(torch.round(torch.sigmoid(tl)), y_test)
        print(f"epoch {epoch:4d} | loss {loss:.4f} | test loss {tloss:.4f} | test acc {tacc:5.1f}%")

out = fig_path("day18_boundary_relu.png")
plot_decision_boundary(model, X_test, y_test, path=out, title="ReLU'lu model (V2)", device=device)
print("Grafik:", out)
