"""Gün 15: Veriyi tensöre çevir + böl, küçük sinir ağı (nn.Module), nn.Sequential karşılığı.

TEST EDİLMEDİ (torch). Eğitim döngüsü ve kayıp Gün 16'dadır (day16_classification_loop.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

from src.models.toy_classifier import CircleModelV0, make_sequential
from src.utils import get_device, set_seed

set_seed(42)
device = get_device()
print("Cihaz:", device)

X, y = make_circles(1000, noise=0.03, random_state=42)
X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()    # float64 -> float32
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Bölme:", tuple(X_train.shape), tuple(X_test.shape), tuple(y_train.shape), tuple(y_test.shape))

model = CircleModelV0().to(device)
seq = make_sequential().to(device)
print("\nnn.Module     :", model)
print("nn.Sequential :", seq)
print("state_dict anahtarları:", list(model.state_dict().keys()))

with torch.inference_mode():
    out = model(X_test.to(device))
print("\nÇıktı şekli:", tuple(out.shape), "(ham çıktı, henüz olasılık değil)")
print("İlk 5 ham çıktı:", out[:5].squeeze().cpu().numpy().round(3))
print("İlk 5 gerçek etiket:", y_test[:5].numpy())
