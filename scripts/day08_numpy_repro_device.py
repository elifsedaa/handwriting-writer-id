"""Gün 8: NumPy-PyTorch dönüşümü, tekrarlanabilirlik, device-agnostic kod, deney klasörü."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.config import Config
from src.experiment import make_run_dir
from src.utils import get_device, set_seed

print("=== NumPy -> PyTorch ===")
arr = np.arange(1.0, 8.0)
t = torch.from_numpy(arr)                      # bellek paylaşılır
print("NumPy dtype:", arr.dtype, " -> tensör dtype:", t.dtype, "(float64!)")
arr[0] = 100.0
print("arr[0]=100 yapınca t[0]:", t[0].item(), "(paylaşım)")
arr = arr + 1                                  # yeni bir dizi oluşur, t etkilenmez
print("arr = arr + 1 sonrası t[0]:", t[0].item(), "(değişmedi)")
t32 = t.float()
print("Modele vermeden önce .float() ->", t32.dtype)

print("\n=== PyTorch -> NumPy ===")
tt = torch.ones(5)
nn_arr = tt.numpy()
tt.add_(1)
print("tensör +1 yapınca numpy dizisi de değişir:", nn_arr)

print("\n=== Tekrarlanabilirlik ===")
torch.manual_seed(42); a = torch.rand(2, 2)
torch.manual_seed(42); b = torch.rand(2, 2)
c = torch.rand(2, 2)                           # seed tüketildi, farklı sayılar
print("Aynı seed, aynı sayılar mı? (a, b):", torch.equal(a, b))
print("Seed'i yeniden vermeden (a, c)    :", torch.equal(a, c))
set_seed(42, deterministic=True)
print("set_seed(42, deterministic=True) çağrıldı")

print("\n=== Device-agnostic kod ===")
device = get_device()
print("Cihaz:", device)
x = torch.rand(3, 3).to(device)
print("x.device:", x.device)
if device == "cuda":
    try:
        x.numpy()
    except TypeError as e:
        print("HATA (beklenen):", str(e).splitlines()[0])
    print("Çözüm: x.cpu().numpy() ->", x.cpu().numpy().shape)
else:
    print("GPU yok; CPU tensörleri doğrudan .numpy() ile dönüşür.")

print("\n=== Deney klasörü ===")
cfg = Config(run_name="gun8_debug")
run_dir = make_run_dir(cfg)
print("Oluşturuldu:", run_dir)
for f in sorted(run_dir.iterdir()):
    print("  -", f.name)
print((run_dir / "env.json").read_text(encoding="utf-8"))
