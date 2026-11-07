"""Gün 7: reshape/view/stack, squeeze/unsqueeze/permute, indexing ve patch çıkarma."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data.patches import extract_patches, filter_blank, ink_ratio
from src.utils import fig_path, set_seed

set_seed(42)

print("=== reshape ve view ===")
x = torch.arange(1.0, 10.0)
print("x:", x, tuple(x.shape))
print("reshape(1,9):", tuple(x.reshape(1, 9).shape), "  reshape(3,3):", tuple(x.reshape(3, 3).shape))
z = x.view(1, 9)                 # view, x ile aynı belleği paylaşır
z[:, 0] = 99
print("z[0,0] = 99 yapınca x[0]:", x[0].item(), "(bellek paylaşımı)")

print("\n=== stack ===")
x = torch.arange(1.0, 10.0)
print("dim=0:", tuple(torch.stack([x, x, x, x], dim=0).shape))
print("dim=1:", tuple(torch.stack([x, x, x, x], dim=1).shape))

print("\n=== squeeze / unsqueeze ===")
xr = x.reshape(1, 9)
print("önce:", tuple(xr.shape), " squeeze:", tuple(xr.squeeze().shape))
print("unsqueeze(0):", tuple(x.unsqueeze(0).shape), " unsqueeze(1):", tuple(x.unsqueeze(1).shape))

print("\n=== permute: HWC -> CHW ===")
img_hwc = torch.rand(224, 224, 3)            # PIL/numpy düzeni: yükseklik, genişlik, kanal
img_chw = img_hwc.permute(2, 0, 1)           # PyTorch düzeni: kanal, yükseklik, genişlik
print("HWC:", tuple(img_hwc.shape), "-> CHW:", tuple(img_chw.shape))
img_hwc[0, 0, 0] = 5.0
print("permute da bellek paylaşır, img_chw[0,0,0] =", img_chw[0, 0, 0].item())
try:
    img_chw.view(-1)
except RuntimeError as e:
    print("HATA (beklenen, bitişik değil):", str(e).splitlines()[0])
print("Çözüm: reshape ->", tuple(img_chw.reshape(-1).shape))

print("\n=== İndeksleme ===")
t = torch.arange(1, 10).reshape(1, 3, 3)
print(t)
print("t[0]:", t[0].tolist())
print("t[0][0]:", t[0][0].tolist(), "  t[0,0,0]:", t[0, 0, 0].item())
print("t[:, :, 1]:", t[:, :, 1].tolist())
print("t[:, 1, 1]:", t[:, 1, 1].tolist(), "  t[0, 0, :]:", t[0, 0, :].tolist())

print("\n=== Proje bağlantısı: sahte sayfadan patch çıkarma ===")
page = torch.ones(1, 512, 384)               # [C, H, W], beyaz zemin
for r in (50, 90, 170, 210, 420):            # yazı satırı gibi koyu çizgiler
    page[:, r:r + 4, 20:200] = 0.0
patches = extract_patches(page, patch=128)
print("Sayfa:", tuple(page.shape), "-> patch'ler:", tuple(patches.shape))
ratios = ink_ratio(patches)
kept = filter_blank(patches, min_ink=0.02)
print("Patch başına yazı oranı:", [round(v, 3) for v in ratios.tolist()])
print(f"Eşik 0.02 ile kalan patch: {kept.shape[0]} / {patches.shape[0]}")

n_cols = page.shape[2] // 128
n_rows = patches.shape[0] // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 7))
for i, ax in enumerate(axes.flat):
    ax.imshow(patches[i, 0], cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{ratios[i]:.3f}", fontsize=8, color="g" if ratios[i] >= 0.02 else "r")
    ax.axis("off")
fig.suptitle("Patch'ler ve yazı oranı (yeşil: kalır, kırmızı: elenir)")
out = fig_path("day07_patches.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print("Görsel kaydedildi:", out)
