"""Gün 3: Ortam kontrolü ve CPU/GPU hız karşılaştırması."""
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.utils import get_device, set_seed

print("=== Ortam bilgisi ===")
print("Python        :", sys.version.split()[0])
print("İşletim sis.  :", platform.platform())
print("PyTorch       :", torch.__version__)
try:
    import torchvision
    print("Torchvision   :", torchvision.__version__)
except ImportError:
    print("Torchvision   : kurulu değil")

print("CUDA mevcut mu:", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU           :", props.name)
    print("GPU belleği   : %.1f GB" % (props.total_memory / 1024**3))
    print("CUDA sürümü   :", torch.version.cuda)

device = get_device()
print("Seçilen cihaz :", device)


def bench(dev: str, n: int = 2048, repeats: int = 5) -> float:
    """n x n iki matrisin çarpımının ortalama süresini (sn) ölçer."""
    a = torch.rand(n, n, device=dev)
    b = torch.rand(n, n, device=dev)
    _ = a @ b  # ısınma turu
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _i in range(repeats):
        c = a @ b
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


set_seed(42)
print("\n=== 2048x2048 matris çarpımı ===")
t_cpu = bench("cpu")
print(f"CPU : {t_cpu * 1000:8.2f} ms")
if torch.cuda.is_available():
    t_gpu = bench("cuda")
    print(f"GPU : {t_gpu * 1000:8.2f} ms  (yaklaşık {t_cpu / t_gpu:.1f}x hızlı)")
else:
    print("GPU bulunamadı, sadece CPU ölçüldü.")

print("\n=== Tensörün cihaza taşınması ===")
x = torch.rand(3, 3)
print("Önce  :", x.device)
x = x.to(device)
print("Sonra :", x.device)
