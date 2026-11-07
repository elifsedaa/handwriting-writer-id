"""Gün 16: Kayıp, optimizer, logits -> olasılık -> etiket ve sınıflandırma eğitim/test döngüsü.

TEST EDİLMEDİ (torch). Doğrusal modelin daire verisinde ~%50 civarında kalması BEKLENİR (aktivasyon yok).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn

from src.models.toy_classifier import CircleModelV0
from src.utils import get_device, set_seed

set_seed(42)
device = get_device()
X, y = make_circles(1000, noise=0.03, random_state=42)
X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = (t.to(device) for t in (X_train, X_test, y_train, y_test))

model = CircleModelV0().to(device)

with torch.inference_mode():
    logits = model(X_test[:5]).squeeze()
    probs = torch.sigmoid(logits)
    labels = torch.round(probs)
print("logits  :", logits.cpu().numpy().round(3))
print("olasılık:", probs.cpu().numpy().round(3))
print("etiket  :", labels.cpu().numpy())

loss_fn = nn.BCEWithLogitsLoss()          # sigmoid + ikili çapraz entropi tek parça
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def accuracy(pred, true):
    return (pred == true).sum().item() / len(true) * 100


for epoch in range(101):
    model.train()
    logits = model(X_train).squeeze()
    loss = loss_fn(logits, y_train)
    acc = accuracy(torch.round(torch.sigmoid(logits)), y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        model.eval()
        with torch.inference_mode():
            tl = model(X_test).squeeze()
            tloss = loss_fn(tl, y_test)
            tacc = accuracy(torch.round(torch.sigmoid(tl)), y_test)
        print(f"Epoch {epoch:3d} | loss {loss:.4f} acc {acc:5.1f}% | test loss {tloss:.4f} acc {tacc:5.1f}%")
