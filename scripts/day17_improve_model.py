"""Gün 17: Modeli iyileştirme seçenekleri: daha fazla katman / gizli birim / epoch. V0 ile V1 karşılaştırması.

TEST EDİLMEDİ (torch). BEKLENTİ: V1'de de aktivasyon olmadığı için karar sınırı doğrusal kalır;
doğruluğun belirgin artmaması beklenir. Gerçek sonucu deftere kendi çıktına göre yaz.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn

from src.models.toy_classifier import CircleModelV0, CircleModelV1
from src.utils import fig_path, get_device, set_seed
from src.viz import plot_decision_boundary

device = get_device()
X, y = make_circles(1000, noise=0.03, random_state=42)
X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = (t.to(device) for t in (X_train, X_test, y_train, y_test))


def accuracy(pred, true):
    return (pred == true).sum().item() / len(true) * 100


def train(model, epochs=1000, lr=0.1):
    set_seed(42)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
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
            print(f"  epoch {epoch:4d} | loss {loss:.4f} | test loss {tloss:.4f} | test acc {tacc:5.1f}%")


for name, cls in [("V0 (2 katman, 10 gizli)", CircleModelV0), ("V1 (3 katman, 10 gizli)", CircleModelV1)]:
    print(f"\n=== {name} ===")
    set_seed(42)
    model = cls().to(device)
    train(model)
    out = fig_path(f"day17_boundary_{cls.__name__}.png")
    plot_decision_boundary(model, X_test, y_test, path=out, title=name, device=device)
    print("Grafik:", out)
