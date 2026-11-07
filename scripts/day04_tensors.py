"""Gün 4: Tensör oluşturma yöntemleri ve veri tipleri.

Son bölümde bir "el yazısı sayfası"nın tensör olarak nasıl temsil edildiğini gösterir.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.utils import fig_path, set_seed

set_seed(42)

print("=== Tensör oluşturma ===")
rastgele = torch.rand(3, 224, 224)        # 0-1 arası rastgele, görüntü benzeri
sifirlar = torch.zeros(2, 3)
birler = torch.ones(2, 3)
aralik = torch.arange(0, 10)
benzer = torch.zeros_like(rastgele)       # rastgele ile aynı şekil ve tip
print("rastgele :", tuple(rastgele.shape))
print("sifirlar :\n", sifirlar)
print("birler   :\n", birler)
print("aralik   :", aralik)
print("benzer   :", tuple(benzer.shape), benzer.dtype)

print("\n=== Veri tipleri ===")
for dt in (torch.float32, torch.float16, torch.int64, torch.uint8):
    t = torch.tensor([1, 2, 3], dtype=dt)
    print(f"{str(dt):15s} eleman başına {t.element_size()} bayt")

a = torch.rand(2, 3)                                  # float32
b = torch.ones(3, 2, dtype=torch.float64)             # float64
print("\nfloat32 @ float64 matris çarpımı deneniyor...")
try:
    a @ b
except RuntimeError as e:
    print("HATA (beklenen):", str(e).splitlines()[0])
print("Çözüm: .to(torch.float32) ->", tuple((a @ b.to(torch.float32)).shape))

print("\n=== Proje bağlantısı: bir el yazısı görüntüsü bir tensördür ===")
# Gerçek görüntüler genelde uint8 (0-255) gelir; modele float32 (0-1) vermemiz gerekir.
img_uint8 = torch.randint(0, 256, (1, 224, 224), dtype=torch.uint8)
img = img_uint8.to(torch.float32) / 255.0
print("uint8  :", img_uint8.dtype, img_uint8.min().item(), "-", img_uint8.max().item())
print("float32:", img.dtype, round(img.min().item(), 3), "-", round(img.max().item(), 3))
print("Şekil  : [kanal, yükseklik, genişlik] =", list(img.shape))

# Sahte bir "yazı satırı" çizip kaydedelim (beyaz zemin, siyah çizgiler)
sayfa = torch.ones(1, 224, 224)
for satir in (50, 90, 130, 170):
    sayfa[:, satir:satir + 3, 20:200] = 0.0
plt.figure(figsize=(3, 3))
plt.imshow(sayfa.squeeze(0), cmap="gray", vmin=0, vmax=1)
plt.title("Sahte sayfa: [1, 224, 224]")
plt.axis("off")
out = fig_path("day04_fake_page.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print("Görsel kaydedildi:", out)
