"""Sayfa tensöründen patch çıkarma ve boş patch'leri eleme (Gün 7).

Tensör şekilleri:  sayfa [C, H, W]  ->  patch'ler [N, C, P, P]
"""
import torch


def extract_patches(page: torch.Tensor, patch: int = 128, stride: int = None) -> torch.Tensor:
    """Sayfayı kaydırmalı pencerelerle patch'lere böler (artan kenarlar atılır)."""
    stride = stride or patch
    c = page.shape[0]
    p = page.unfold(1, patch, stride).unfold(2, patch, stride)   # [C, nh, nw, P, P]
    p = p.permute(1, 2, 0, 3, 4)                                 # [nh, nw, C, P, P]
    return p.reshape(-1, c, patch, patch)                        # [N, C, P, P]


def ink_ratio(patches: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Her patch'te koyu (yazı) piksellerin oranı. Beyaz zemin = 1.0 varsayılır."""
    return (patches < threshold).float().mean(dim=(1, 2, 3))


def filter_blank(patches: torch.Tensor, min_ink: float = 0.02) -> torch.Tensor:
    """Yeterince yazı içermeyen patch'leri eler (boolean maske ile indeksleme)."""
    keep = ink_ratio(patches) >= min_ink
    return patches[keep]
