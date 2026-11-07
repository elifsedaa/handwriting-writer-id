"""Gün 14-18: Kurstaki sınıflandırma oyuncak modelleri (iki daire verisi)."""
import torch
from torch import nn


class CircleModelV0(nn.Module):
    """Doğrusal katmanlarla küçük ağ (doğrusal olmayan aktivasyon YOK)."""

    def __init__(self, hidden: int = 10):
        super().__init__()
        self.layer_1 = nn.Linear(2, hidden)
        self.layer_2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_2(self.layer_1(x))


def make_sequential(hidden: int = 10, nonlinear: bool = False) -> nn.Sequential:
    """nn.Sequential ile aynı fikir; nonlinear=True ise araya ReLU konur (Gün 18'e hazırlık)."""
    if nonlinear:
        return nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                             nn.Linear(hidden, 1))
    return nn.Sequential(nn.Linear(2, hidden), nn.Linear(hidden, 1))


class CircleModelV1(nn.Module):
    """Gün 17: Daha fazla katman ve gizli birim (hâlâ doğrusal; aktivasyon yok)."""

    def __init__(self, hidden: int = 10):
        super().__init__()
        self.layer_1 = nn.Linear(2, hidden)
        self.layer_2 = nn.Linear(hidden, hidden)
        self.layer_3 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_3(self.layer_2(self.layer_1(x)))


class CircleModelV2(nn.Module):
    """Gün 18: Katmanlar arasına ReLU (doğrusal olmayan) eklenmiş model."""

    def __init__(self, hidden: int = 10):
        super().__init__()
        self.layer_1 = nn.Linear(2, hidden)
        self.layer_2 = nn.Linear(hidden, hidden)
        self.layer_3 = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
