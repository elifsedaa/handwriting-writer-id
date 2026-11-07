"""Gün 16: Sayfa kırpma ve NumPy ile patch çıkarma (torch gerektirmez).

Sayfa dizisi: float32 [H, W], [0, 1], beyaz zemin = 1.0 (preprocess.load_gray çıktısı).
Torch'lu karşılığı `patches.py` içinde; bu modül diske patch üretirken torch yüklemeden çalışmak için var.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def trim_border(arr: np.ndarray, frac: float = 0.0) -> np.ndarray:
    """Her kenardan sayfa boyutunun `frac` oranını keser (tarama kenar gölgeleri / köşe işaretleri için)."""
    if frac <= 0:
        return arr
    h, w = arr.shape
    dy, dx = int(h * frac), int(w * frac)
    return arr[dy:h - dy, dx:w - dx]


def ink_bbox(arr: np.ndarray, threshold: float = 0.5, min_row_ink: float = 0.002):
    """Yazı içeren bölgenin (üst, alt, sol, sağ) sınırlarını döndürür; yazı yoksa None.

    Satır/sütunlar, koyu piksel oranı `min_row_ink` üstündeyse yazı sayılır (tek tek gürültü pikselleri etkilemesin).
    """
    mask = arr < threshold
    rows = mask.mean(axis=1) > min_row_ink
    cols = mask.mean(axis=0) > min_row_ink
    if not rows.any() or not cols.any():
        return None
    r, c = np.where(rows)[0], np.where(cols)[0]
    return int(r[0]), int(r[-1]) + 1, int(c[0]), int(c[-1]) + 1


def crop_to_ink(arr: np.ndarray, margin: int = 16, threshold: float = 0.5) -> np.ndarray:
    """Beyaz kenar boşluklarını atıp yazı bölgesini (+ `margin` piksel) bırakır. Yazı yoksa dizi aynen döner."""
    box = ink_bbox(arr, threshold)
    if box is None:
        return arr
    t, b, l, r = box
    h, w = arr.shape
    return arr[max(0, t - margin):min(h, b + margin), max(0, l - margin):min(w, r + margin)]


def pad_min_size(arr: np.ndarray, size: int, value: float = 1.0) -> np.ndarray:
    """Her iki kenarı da en az `size` olacak şekilde beyazla (sağ/alt) doldurur."""
    h, w = arr.shape
    if h >= size and w >= size:
        return arr
    out = np.full((max(h, size), max(w, size)), value, dtype=arr.dtype)
    out[:h, :w] = arr
    return out


def extract_patches_np(arr: np.ndarray, patch: int = 128, stride: int | None = None) -> np.ndarray:
    """[H, W] -> [N, P, P]. Pencere kaydırma; artan kenarlar atılır (sayfa önce `pad_min_size` ile doldurulmalı)."""
    stride = stride or patch
    win = sliding_window_view(arr, (patch, patch))[::stride, ::stride]   # [nh, nw, P, P]
    return win.reshape(-1, patch, patch).copy()


def ink_ratio_np(patches: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Her patch'te koyu piksel oranı -> [N]."""
    return (patches < threshold).mean(axis=(1, 2))


def filter_blank_np(patches: np.ndarray, min_ink: float = 0.02):
    """Yeterince yazı içermeyen patch'leri eler. (kalan patch'ler, kalanların ink_ratio değerleri) döndürür."""
    ratios = ink_ratio_np(patches)
    keep = ratios >= min_ink
    return patches[keep], ratios[keep]


def stretch_contrast(arr: np.ndarray, q: float = 0.1) -> np.ndarray:
    """Sayfan?n kendi kontrast?na g?re ?l?ekler: en koyu (q. y?zdelik) -> 0, zemin (medyan) -> 1.

    Soluk taramalarda yaz? 0.5 e?i?inin ?st?nde kalabiliyor; bu ad?m sabit e?iklerin her sayfada ?al??mas?n? sa?lar.
    """
    lo, bg = np.percentile(arr, q), np.percentile(arr, 50)
    if bg - lo < 0.05:
        return arr
    return np.clip((arr - lo) / (bg - lo), 0, 1).astype(np.float32)
