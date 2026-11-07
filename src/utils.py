"""Projenin ortak yardımcı fonksiyonları."""
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "experiments" / "figures"


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Tüm rastgelelik kaynaklarını sabitler.

    deterministic=True ise cuDNN'i deterministik moda alır (biraz yavaşlayabilir).
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """GPU varsa 'cuda', yoksa 'cpu' döndürür (device-agnostic kod)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def fig_path(name: str) -> Path:
    """Grafiklerin kaydedileceği yolu döndürür (klasör yoksa oluşturur)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR / name


def get_env_info() -> dict:
    """Deney kaydı için ortam bilgisi (Python, işletim sistemi, torch, GPU)."""
    import platform
    import sys

    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception as e:  # torch kurulu değil / yüklenemedi
        info["torch"] = f"yüklenemedi: {type(e).__name__}"
    return info
