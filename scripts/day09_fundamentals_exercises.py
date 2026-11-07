"""Gün 9: PyTorch Fundamentals alıştırmaları (kurs alıştırmalarına benzer çözümler)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.utils import get_device

device = get_device()
print("Cihaz:", device)

print("\n[1] (7, 7) şeklinde rastgele tensör")
a = torch.rand(7, 7)
print(tuple(a.shape))

print("\n[2] (7,7) ile (1,7) tensörün matris çarpımı (transpose gerekir)")
b = torch.rand(1, 7)
r = a @ b.T
print("sonuç şekli:", tuple(r.shape))

print("\n[3] Seed 0 ile iki kez aynı sonuç")
torch.manual_seed(0); r1 = torch.rand(7, 7) @ torch.rand(1, 7).T
torch.manual_seed(0); r2 = torch.rand(7, 7) @ torch.rand(1, 7).T
print("eşit mi:", torch.equal(r1, r2))

print("\n[4] GPU seed ve cihaz üzerinde (2, 3) tensörler")
if device == "cuda":
    torch.cuda.manual_seed(1234)
torch.manual_seed(1234)
x = torch.rand(2, 3, device=device)
y = torch.rand(2, 3, device=device)
print("x.device:", x.device)

print("\n[5] (2,3) @ (2,3) hata verir; y.T ile çöz")
out = x @ y.T
print(out, tuple(out.shape))

print("\n[6] Çıktının max / min / argmax / argmin")
print("max:", out.max().item(), " min:", out.min().item())
print("argmax:", out.argmax().item(), " argmin:", out.argmin().item())

print("\n[7] (1,1,1,10) tensör -> squeeze (seed 7)")
torch.manual_seed(7)
s = torch.rand(1, 1, 1, 10)
s2 = s.squeeze()
print("önce:", tuple(s.shape), "-> sonra:", tuple(s2.shape))
print("s:", s)
print("s2:", s2)
