"""Gün 15: Ön işleme fonksiyonları (gri tonlama, yeniden boyutlandırma, normalizasyon).

Hepsi NumPy/PIL ile yazıldı (torch gerektirmez). `to_tensor` torch'u fonksiyon içinde import eder.
Dönüş biçimi: float32 dizi [H, W], değerler [0, 1], beyaz zemin = 1.0 (patches.py ile uyumlu).
"""
from __future__ import annotations
import numpy as np
from PIL import Image


def to_gray(im: Image.Image) -> Image.Image:
    """RGB/RGBA/P görüntüyü 8-bit gri tonlamaya çevirir (alfa varsa beyaza yaslar)."""
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("L")


def resize_keep_aspect(im: Image.Image, height: int | None = None, max_side: int | None = None) -> Image.Image:
    """En-boy oranını koruyarak yeniden boyutlandırır (height VEYA max_side ver)."""
    w, h = im.size
    if height is not None:
        scale = height / h
    elif max_side is not None:
        scale = max_side / max(w, h)
    else:
        raise ValueError("height veya max_side verilmeli")
    new = (max(1, round(w * scale)), max(1, round(h * scale)))
    return im.resize(new, Image.BILINEAR)


def pad_to(arr: np.ndarray, height: int, width: int, value: float = 1.0) -> np.ndarray:
    """Dizi [H, W]'yi (height, width) boyuna beyaza doldurarak (ortalayarak) getirir; büyükse kırpar."""
    h, w = arr.shape
    out = np.full((height, width), value, dtype=arr.dtype)
    ch, cw = min(h, height), min(w, width)
    sy, sx = (h - ch) // 2, (w - cw) // 2          # kaynaktan alınacak başlangıç
    dy, dx = (height - ch) // 2, (width - cw) // 2  # hedefe yazılacak başlangıç
    out[dy:dy + ch, dx:dx + cw] = arr[sy:sy + ch, sx:sx + cw]
    return out


def normalize(arr: np.ndarray, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    """(x - mean) / std. Varsayılan: [0,1] -> [-1,1]. Eğitim verisinden hesaplanan değerler de verilebilir."""
    return ((arr - mean) / std).astype(np.float32)


def load_gray(path, height: int | None = None, max_side: int | None = None) -> np.ndarray:
    """Dosyayı okuyup gri tonlamaya çevirir, istenirse yeniden boyutlandırır -> float32 [H, W] in [0, 1]."""
    with Image.open(path) as im:
        im = to_gray(im)
        if height or max_side:
            im = resize_keep_aspect(im, height=height, max_side=max_side)
        return np.asarray(im, dtype=np.float32) / 255.0


def preprocess_line(path, height: int = 64, width: int = 512) -> np.ndarray:
    """Satır görüntüsü için standart hat: gri -> yüksekliğe ölçekle -> genişliğe doldur/kırp -> [0,1]."""
    arr = load_gray(path, height=height)
    return pad_to(arr, height, width, value=1.0)


def to_tensor(arr: np.ndarray, normalize_to_pm1: bool = False):
    """[H, W] -> torch.Tensor [1, H, W]."""
    import torch

    if normalize_to_pm1:
        arr = normalize(arr)
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0)


def compute_mean_std(arrays) -> tuple[float, float]:
    """Verilen dizi listesi (yalnızca EĞİTİM bölmesi!) için tek kanal ortalama ve std."""
    s = n = s2 = 0.0
    for a in arrays:
        s += float(a.sum())
        s2 += float((a.astype(np.float64) ** 2).sum())
        n += a.size
    mean = s / n
    return mean, float(np.sqrt(max(s2 / n - mean ** 2, 0.0)))
