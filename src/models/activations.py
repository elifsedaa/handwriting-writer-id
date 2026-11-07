"""Gün 18: Doğrusal olmayan aktivasyonların saf PyTorch ile yeniden yazımı (kurs alıştırması)."""
import torch


def relu(x: torch.Tensor) -> torch.Tensor:
    """ReLU: negatifleri 0'a çeker, pozitifleri aynen bırakır."""
    return torch.maximum(torch.tensor(0.0, dtype=x.dtype, device=x.device), x)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid: değerleri (0, 1) aralığına sıkıştırır."""
    return 1 / (1 + torch.exp(-x))
