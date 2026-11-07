"""Gün 6: Matmul shape hataları, min/max/mean/sum ve konumsal min/max."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.utils import set_seed

set_seed(42)

print("=== Shape hatası ve çözümü ===")
A = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)       # (3, 2)
B = torch.tensor([[7, 10], [8, 11], [9, 12]], dtype=torch.float32)    # (3, 2)
print("A:", tuple(A.shape), " B:", tuple(B.shape))
try:
    torch.matmul(A, B)
except RuntimeError as e:
    print("HATA (beklenen):", str(e).splitlines()[0])
print("B.T şekli:", tuple(B.T.shape))
out = torch.matmul(A, B.T)                  # (3,2) @ (2,3) -> (3,3)
print("A @ B.T ->", tuple(out.shape))
print(out)
print("A.T @ B ->", tuple(torch.matmul(A.T, B).shape), "(iç boyut yine eşleşmeli)")

print("\n=== Toplulaştırma (aggregation) ===")
x = torch.arange(0, 100, 10)
print("x:", x, x.dtype)
print("min:", x.min().item(), " max:", x.max().item(), " sum:", x.sum().item())
try:
    x.mean()
except RuntimeError as e:
    print("HATA (beklenen, mean float ister):", str(e).splitlines()[0])
print("float32'ye çevirip mean:", x.type(torch.float32).mean().item())

print("\n=== Konumsal min / max ===")
print("argmin:", x.argmin().item(), " argmax:", x.argmax().item())
print("x[argmax] == max ?", x[x.argmax()].item() == x.max().item())

print("\n=== Proje bağlantısı: batch üzerinde istatistik ===")
batch = torch.rand(8, 1, 64, 64)            # 8 sahte görüntü
print("Görüntü başına ortalama parlaklık:", batch.mean(dim=(1, 2, 3)))
print("Görüntü başına en koyu piksel   :", batch.amax(dim=(1, 2, 3)).shape)
scores = torch.rand(8, 20)                  # 8 örnek x 20 yazar skoru
print("Tahmin edilen yazar indeksleri  :", scores.argmax(dim=1).tolist())
