"""Gün 5: Tensör öznitelikleri, işlemler ve matris çarpımı kuralları."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn

from src.utils import get_device, set_seed

set_seed(42)
device = get_device()

print("=== Öznitelikler ===")
x = torch.rand(3, 4)
print("shape :", x.shape)
print("ndim  :", x.ndim)
print("dtype :", x.dtype)
print("device:", x.device)
print("numel :", x.numel())

print("\n=== Eleman bazlı işlemler ===")
print("x * 2 şekli :", tuple((x * 2).shape))
print("x + 10 ilk satır:", (x + 10)[0])

print("\n=== Matris çarpımı: iç boyutlar eşleşmeli ===")
a = torch.rand(3, 4)
b = torch.rand(4, 5)
print("(3,4) @ (4,5) ->", tuple((a @ b).shape))
print("matmul == @ mi?", torch.allclose(a @ b, torch.matmul(a, b)))

bad = torch.rand(3, 5)
print("\n(3,4) @ (3,5) deneniyor...")
try:
    a @ bad
except RuntimeError as e:
    print("HATA (beklenen):", str(e).splitlines()[0])
print("Çözüm: (3,4) @ (4,5) ->", tuple((a @ b).shape))

print("\n=== nn.Linear aslında matris çarpımıdır ===")
layer = nn.Linear(4, 5, bias=False)
y1 = layer(a)                  # katman çıktısı
y2 = a @ layer.weight.T        # elle matris çarpımı
print("Katman çıktısı == elle çarpım mı?", torch.allclose(y1, y2))
print("weight şekli:", tuple(layer.weight.shape), "(çıkış, giriş)")

print("\n=== Proje bağlantısı: düzleştirilmiş patch -> 20 yazar ===")
PATCH = 128
N_WRITERS = 20
BATCH = 8
fc = nn.Linear(PATCH * PATCH, N_WRITERS).to(device)
batch = torch.rand(BATCH, PATCH * PATCH, device=device)
out = fc(batch)
print(f"Giriş : [{BATCH}, {PATCH * PATCH}]")
print(f"Çıkış : {list(out.shape)}  (her örnek için {N_WRITERS} yazar skoru)")
print("Parametre sayısı:", sum(p.numel() for p in fc.parameters()),
      f"= {PATCH * PATCH}*{N_WRITERS} + {N_WRITERS}")
