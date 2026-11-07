"""Kurstaki doğrusal regresyon oyuncak verisi (Gün 9-10)."""
import torch


def make_linear_data(weight=0.7, bias=0.3, start=0.0, end=1.0, step=0.02, train_frac=0.8):
    """y = weight * X + bias verisini üretir ve eğitim/test olarak böler."""
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias
    split = int(train_frac * len(X))
    return X[:split], y[:split], X[split:], y[split:]
