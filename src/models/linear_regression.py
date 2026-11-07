"""Kurstaki ilk PyTorch modeli: elle tanımlı doğrusal regresyon (Gün 10)."""
import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    """y = weights * x + bias. Parametreler rastgele başlar, eğitimle öğrenilir."""

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
