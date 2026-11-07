"""Gün 2: Sinir ağının anatomisi.

Giriş -> gizli -> çıkış katmanlarını, parametre sayısını ve softmax ile
"yazar olasılıklarını" küçük bir örnekle gösterir. Ağ şemasını PNG olarak kaydeder.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # ekran açmadan dosyaya çizer
import matplotlib.pyplot as plt
import torch
from torch import nn

from src.utils import fig_path, set_seed

set_seed(42)

# 4 özellik -> 8 gizli nöron -> 3 yazar (oyuncak örnek)
N_IN, N_HIDDEN, N_OUT = 4, 8, 3
model = nn.Sequential(
    nn.Linear(N_IN, N_HIDDEN),
    nn.ReLU(),
    nn.Linear(N_HIDDEN, N_OUT),
)
print(model)

print("\nKatman parametreleri:")
for name, p in model.named_parameters():
    print(f"  {name:12s} şekil={str(tuple(p.shape)):10s} adet={p.numel()}")
print("Toplam parametre:", sum(p.numel() for p in model.parameters()))

# 5 örnek, her birinin 4 özelliği var
x = torch.rand(5, N_IN)
logits = model(x)                       # ham skorlar
probs = torch.softmax(logits, dim=1)    # her satır toplamı 1
print("\nGiriş şekli :", tuple(x.shape))
print("Logit şekli :", tuple(logits.shape))
print("Olasılıklar (her satır bir örnek, her sütun bir yazar):")
print(probs.detach().round(decimals=3))
print("Tahmin edilen yazar indeksleri:", probs.argmax(dim=1).tolist())


def draw_network(sizes, path):
    """Katman boyutlarına göre basit bir ağ şeması çizer."""
    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["Giriş", "Gizli", "Çıkış"]
    coords = []
    for i, n in enumerate(sizes):
        ys = [j - (n - 1) / 2 for j in range(n)]
        coords.append([(i, y) for y in ys])
    for a, b in zip(coords[:-1], coords[1:]):
        for (x1, y1) in a:
            for (x2, y2) in b:
                ax.plot([x1, x2], [y1, y2], color="gray", alpha=0.25, lw=0.7)
    for i, layer in enumerate(coords):
        ax.scatter([c[0] for c in layer], [c[1] for c in layer], s=250, zorder=3)
        ax.text(i, max(c[1] for c in layer) + 1.0, f"{names[i]}\n({sizes[i]})", ha="center")
    ax.axis("off")
    ax.set_title("Sinir ağı anatomisi: 4 -> 8 -> 3")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


out = fig_path("day02_nn_anatomy.png")
draw_network([N_IN, N_HIDDEN, N_OUT], out)
print("\nŞema kaydedildi:", out)
