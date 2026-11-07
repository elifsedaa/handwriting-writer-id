"""Hafta 4 torch'suz kısımların testi (sentetik veriyle). Çalıştır: python tests/test_week4.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image

import src.utils as U
from src.data import crop, patch_dataset as pdset, report, splits
from src.data import build_metadata as bm


def make_page(path: Path, h=400, w=600, seed=0):
    """Kenarlarında beyaz boşluk olan, ortasında 'yazı' çizgileri bulunan sahte sayfa."""
    rng = np.random.default_rng(seed)
    a = np.full((h, w), 255, np.uint8)
    for y in range(120, 300, 30):
        x0 = 100
        while x0 < 500:
            a[y:y + 8, x0:x0 + rng.integers(10, 30)] = 0
            x0 += rng.integers(35, 50)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)


def main():
    # --- Gün 16: kırpma ve patch ---
    a = np.ones((400, 600), np.float32)
    a[150:200, 120:480] = 0.0
    box = crop.ink_bbox(a)
    assert box == (150, 200, 120, 480), box
    c = crop.crop_to_ink(a, margin=10)
    assert c.shape == (70, 380), c.shape
    assert crop.crop_to_ink(np.ones((50, 50), np.float32)).shape == (50, 50)      # yazı yok -> aynen
    assert crop.trim_border(a, 0.1).shape == (320, 480)
    pm = crop.pad_min_size(np.ones((20, 300), np.float32), 64)
    assert pm.shape == (64, 300)
    pt = crop.extract_patches_np(np.arange(100 * 100, dtype=np.float32).reshape(100, 100), 32)
    assert pt.shape == (9, 32, 32)                                  # 3x3, artan kenar atıldı
    assert pt[1][0, 0] == 32 and pt[3][0, 0] == 32 * 100            # satır-öncelikli sıra
    pt2 = crop.extract_patches_np(np.zeros((64, 64), np.float32), 32, stride=16)
    assert pt2.shape == (9, 32, 32)
    blank = np.ones((5, 32, 32), np.float32); blank[2] = 0.0
    kept, ratios = crop.filter_blank_np(blank, 0.02)
    assert len(kept) == 1 and ratios[0] == 1.0
    print("Gün 16 tamam")

    # --- Gün 17: patch veri seti ---
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        U.FIG_DIR = root / "figs"
        rows = []
        for w in range(1, 6):
            for p in range(1, 5):
                rel = f"data/raw/firemaker/W{w:03d}/W{w:03d}_p{p:02d}.png"
                make_page(root / rel, seed=w * 10 + p)
                rows.append(("firemaker", f"firemaker_W{w:03d}", f"firemaker_W{w:03d}_p{p:02d}", rel, ""))
        # kısa bir "satır" görüntüsü (yüksekliği patch'ten küçük -> doldurulmalı)
        save = root / "data/raw/iam/line.png"
        save.parent.mkdir(parents=True, exist_ok=True)
        la = np.full((40, 200), 255, np.uint8); la[15:22, 10:190] = 0
        Image.fromarray(la).save(save)
        meta = pd.DataFrame(rows, columns=bm.COLUMNS)
        meta = splits.add_splits(meta)
        assert splits.check_leakage(meta) == []

        pt, ink = pdset.page_to_patches(root / rows[0][3], patch=32, scale=1.0, min_ink=0.02)
        assert pt.ndim == 3 and pt.shape[1:] == (32, 32) and len(pt) > 0 and (ink >= 0.02).all()
        lp, _ = pdset.page_to_patches(save, patch=64, scale=1.0, min_ink=0.0)
        assert lp.shape[1:] == (64, 64) and len(lp) >= 1                 # doldurma çalışıyor
        half, _ = pdset.page_to_patches(root / rows[0][3], patch=32, scale=0.5, min_ink=0.0)
        full, _ = pdset.page_to_patches(root / rows[0][3], patch=32, scale=1.0, min_ink=0.0)
        assert len(half) < len(full)                                     # ölçek küçülünce patch azalır

        idx = pdset.build_patch_dataset(meta, root=root, patch=32, scale=1.0, max_per_page=10, verbose=False)
        assert list(idx.columns) == pdset.INDEX_COLS and len(idx) > 0
        assert (idx.groupby(["writer_id", "page_id"])["split"].nunique() == 1).all()     # sayfa tek bölmede
        assert (idx.groupby(["writer_id", "page_id"]).size() <= 10).all()               # max_per_page
        assert all((root / p).exists() for p in idx["patch_path"])
        assert set(idx["split"]) <= {"train", "val", "test"}
        idx2 = pdset.build_patch_dataset(meta, root=root, out_dir=root / "o2", patch=32, scale=1.0,
                                         max_per_page=10, verbose=False)
        assert idx["ink_ratio"].round(6).tolist() == idx2["ink_ratio"].round(6).tolist()  # seed'li, tekrarlanabilir
        # patch seviyesinde sızıntı yok: bir yazarın open-set bölmesi açıkken closed split patch'te sızmıyor
        for g in (pdset.plot_patch_grid(idx, root=root), pdset.plot_patch_summary(idx)):
            assert g.exists() and g.stat().st_size > 1000
        assert pdset.plot_patch_grid(idx.iloc[0:0], root=root) is None
        # bozuk dosya atlanıyor
        bad = meta.copy(); bad.loc[0, "path"] = "data/raw/firemaker/yok.png"
        idx3 = pdset.build_patch_dataset(bad, root=root, out_dir=root / "o3", patch=32, scale=1.0,
                                         max_per_page=10, verbose=False)
        assert len(idx3) < len(idx) + 1
        print("Gün 17 tamam:", len(idx), "patch;", idx.groupby("split").size().to_dict())

        # --- Gün 18: rapor ---
        txt = report.build_report(root)               # CSV yok -> [ ] dolu rapor
        assert "[ ]" in txt and "İlk 18 gün" in txt
        (root / "data/metadata").mkdir(parents=True, exist_ok=True)
        meta.to_csv(root / "data/metadata/all_samples.csv", index=False)
        (root / "data/processed/patches").mkdir(parents=True, exist_ok=True)
        idx.to_csv(root / "data/processed/patches/patches_index.csv", index=False)
        txt2 = report.build_report(root)
        assert f"**{len(idx)}**" in txt2 and "firemaker" in txt2
        assert "| firemaker | 20 | 5 | 20 | 20 |" in txt2, [l for l in txt2.splitlines() if l.startswith("| firemaker")]
        print("Gün 18 tamam")
    print("\nTÜM TESTLER GEÇTİ")


if __name__ == "__main__":
    main()
