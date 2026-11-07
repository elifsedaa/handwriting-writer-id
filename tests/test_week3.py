"""Hafta 3 torch'suz kısımların testi (sentetik veriyle). Çalıştır: python tests/test_week3.py

IAM/CVL dosya adı düzenleri VARSAYIM; gerçek veriyle ayrıca doğrulanmalıdır.
"""
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image

import src.utils as U
from src.data import build_metadata as bm
from src.data import eda, iam_setup, preprocess, splits


FM_RE = r"^W(\d{3})_p(\d{2})$"   # sahte Firemaker dosya adı düzeni (yalnızca test)


def save_img(path: Path, w=300, h=80, rgba=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((h, w), 255, np.uint8)
    arr[20:30, 10:w - 10] = 0
    im = Image.fromarray(arr)
    im.convert("RGBA" if rgba else "RGB").save(path)


def build_fake_root(root: Path):
    iam = root / "data/raw/iam"
    # forms.txt (varsayılan biçim) + lines görüntüleri, önce _downloads içinde tgz olarak
    stage = root / "stage"
    (stage / "ascii").mkdir(parents=True)
    forms = ["a01-000u 000 2 prt 5", "a01-003 001 3 prt 4", "b02-010 001 1 prt 2", "c03-001a 002 2 prt 6",
             "# yorum satırı"]
    (stage / "ascii/forms.txt").write_text("\n".join(forms), encoding="utf-8")
    for fid in ["a01-000u", "a01-003", "b02-010", "c03-001a"]:
        for k in range(2):
            save_img(stage / f"lines/{fid[:3]}/{fid}/{fid}-{k:02d}.png")
    dl = iam / "_downloads"; dl.mkdir(parents=True)
    for name in ["ascii", "lines"]:
        with tarfile.open(dl / f"{name}.tgz", "w:gz") as t:
            t.add(stage / name, arcname=name)
    # CVL: 6 yazar x 4 metin ; firemaker (sahte düzen): 8 yazar x 4 sayfa
    for w in range(1, 7):
        for t in range(1, 5):
            save_img(root / f"data/raw/cvl/{w:04d}-{t}.png")
    for w in range(1, 9):
        for p in range(1, 5):
            save_img(root / f"data/raw/firemaker/W{w:03d}/W{w:03d}_p{p:02d}.png")
    save_img(root / "data/raw/firemaker/kotu_ad.png")   # eşleşmeyen ad: atlanmalı


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        build_fake_root(root)
        U.FIG_DIR = root / "figs"

        # Gün 11: açma + doğrulama
        done = iam_setup.extract_archives(root / "data/raw/iam/_downloads", root / "data/raw/iam")
        assert sorted(done) == ["ascii", "lines"], done
        again = iam_setup.extract_archives(root / "data/raw/iam/_downloads", root / "data/raw/iam")
        assert again == [], "ikinci çalıştırma tekrar açmamalı"
        r = iam_setup.verify(root / "data/raw/iam")
        assert r["forms_txt"] and r["n_images"]["lines"] == 8, r
        m = iam_setup.read_forms_txt(root / "data/raw/iam" / r["forms_txt"])
        assert m == {"a01-000u": "000", "a01-003": "001", "b02-010": "001", "c03-001a": "002"}, m
        print("Gün 11 tamam")

        # Gün 13: metadata
        df = bm.build(root, firemaker_regex=FM_RE)
        assert list(df.columns) == bm.COLUMNS
        cnt = df.groupby("dataset")["writer_id"].nunique().to_dict()
        assert cnt == {"cvl": 6, "iam": 3, "firemaker": 8}, cnt
        assert (df["dataset"] == "iam").sum() == 8 and (df["dataset"] == "firemaker").sum() == 32
        assert not df["path"].str.contains("kotu_ad").any() and not df["path"].str.contains("\\\\").any()
        print("Gün 13 tamam:", cnt)

        # Gün 14: bölmeler
        s = splits.add_splits(df, seed=42)
        assert splits.check_leakage(s) == []
        own = s[s.dataset == "firemaker"]
        # 4 sayfalı yazar: 2 train, 1 val, 1 test
        per = own.groupby("writer_id")["split_closed"].apply(lambda x: sorted(x.tolist()))
        assert all(v == ["test", "train", "train", "val"] for v in per), per.iloc[0]
        # IAM: her yazarın 1-2 formu var (<3 sayfa) -> excluded
        assert (s[s.dataset == "iam"]["split_closed"] == "excluded").all()
        # open-set: her veri setinde train/val/test yazarı var, yazar tek bölmede
        assert set(own["split_open"]) == {"train", "val", "test"}
        # tekrarlanabilirlik
        s2 = splits.add_splits(df, seed=42)
        assert s.equals(s2)
        assert not s.equals(splits.add_splits(df, seed=7))
        # kasıtlı sızıntılar yakalanıyor mu (a: aynı sayfa iki bölmede, b: aynı yazar iki bölmede)
        row = s[(s.dataset == "firemaker") & (s.split_closed == "train")].iloc[[0]].copy()
        row["split_closed"] = "test"
        assert splits.check_leakage(pd.concat([s, row], ignore_index=True)), "sayfa sızıntısı yakalanmadı"
        bad = s.copy(); i = bad.index[bad.dataset == "cvl"][0]
        bad.loc[i, "split_open"] = "test" if bad.loc[i, "split_open"] != "test" else "train"
        assert splits.check_leakage(bad), "yazar sızıntısı yakalanmadı"
        print("Gün 14 tamam")
        print(splits.summary(s))

        # Gün 12: EDA
        sizes = eda.image_sizes(s, root)
        assert len(sizes) == len(s)
        files = eda.plot_all(s, sizes)
        assert all(f.exists() and f.stat().st_size > 1000 for f in files), files
        print("Gün 12 tamam:", [f.name for f in files])

        # Gün 15: ön işleme
        p = root / "data/raw/firemaker/W001/W001_p01.png"
        out = preprocess.preprocess_line(p, height=64, width=512)
        assert out.shape == (64, 512) and out.dtype == np.float32 and 0 <= out.min() and out.max() <= 1
        assert out[0, 0] == 1.0                                      # doldurma beyaz
        assert preprocess.normalize(out).min() >= -1 and preprocess.normalize(out).max() <= 1
        rgba = root / "rgba.png"; save_img(rgba, rgba=True)
        assert preprocess.to_gray(Image.open(rgba)).mode == "L"
        mean, std = preprocess.compute_mean_std([out, out])
        assert abs(mean - out.mean()) < 1e-4 and abs(std - out.std()) < 1e-3
        assert preprocess.pad_to(np.zeros((100, 700), np.float32), 64, 512).shape == (64, 512)
        print("Gün 15 tamam")
    print("\nTÜM TESTLER GEÇTİ")


if __name__ == "__main__":
    main()
