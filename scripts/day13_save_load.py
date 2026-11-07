"""Gün 13: Modeli kaydetme/yükleme (state_dict) ve eğitilmiş modelle tekrar tahmin.

TEST EDİLMEDİ (yazıldığı ortamda torch yoktu).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn

from src.config import Config
from src.data.synthetic import make_linear_data
from src.experiment import make_run_dir
from src.models.linear_regression import LinearRegressionModel
from src.utils import set_seed

cfg = Config(run_name="day13_save_load", epochs=200, lr=0.01)
set_seed(cfg.seed)
run_dir = make_run_dir(cfg)

X_train, y_train, X_test, y_test = make_linear_data()
model = LinearRegressionModel()
loss_fn = nn.L1Loss()
opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)
for _ in range(cfg.epochs):
    model.train()
    loss = loss_fn(model(X_train), y_train)
    opt.zero_grad(); loss.backward(); opt.step()

path = run_dir / "model.pth"
torch.save(model.state_dict(), path)                    # yalnızca state_dict kaydedilir (önerilen yol)
print("Kaydedildi:", path)

loaded = LinearRegressionModel()                        # önce aynı mimariyi kur
loaded.load_state_dict(torch.load(path, weights_only=True))
loaded.eval()
with torch.inference_mode():
    a, b = model.eval()(X_test), loaded(X_test)
print("Kaydet/yükle sonrası tahminler aynı mı:", torch.equal(a, b))
