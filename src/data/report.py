"""Gün 18: İlk 18 gün özet raporu. Gerçek dosyalardan sayıları okuyup docs/ilk_18_gun_ozeti.md üretir.

Bulunamayan her bilgi için `[ ]` bırakılır; hiçbir sayı uydurulmaz.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from src.config import Config
from src.utils import ROOT

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _count_images(folder: Path) -> int | None:
    if not folder.exists():
        return None
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def build_report(root: Path = ROOT) -> str:
    cfg = Config()
    L = [f"# İlk 18 gün özeti (otomatik üretildi {datetime.now():%Y-%m-%d %H:%M})", "",
         "Bilinmeyen değerler `[ ]` olarak bırakıldı; elle doldurulacak.", "", "## 1. Veri durumu", "",
         "| Veri seti | Ham görüntü | Yazar | Sayfa | Örnek (metadata) |", "|---|---|---|---|---|"]
    csv = root / "data" / "metadata" / "all_samples.csv"
    meta = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    for name in ["iam", "cvl", "firemaker"]:
        raw = _count_images(root / "data" / "raw" / name)
        if len(meta) and (meta["dataset"] == name).any():
            g = meta[meta["dataset"] == name]
            L.append(f"| {name} | {raw if raw is not None else '[ ]'} | {g['writer_id'].nunique()} | "
                     f"{g['page_id'].nunique()} | {len(g)} |")
        else:
            L.append(f"| {name} | {raw if raw is not None else '[ ]'} | [ ] | [ ] | [ ] |")

    L += ["", "IAM başvuru/onay durumu: [ ]", "",
          "## 2. Bölme", ""]
    if len(meta) and "split_closed" in meta.columns:
        L.append("closed-set (sayfa bazında) örnek sayıları:\n")
        L.append("```\n" + meta.groupby(["dataset", "split_closed"]).size().unstack(fill_value=0).to_string() + "\n```")
        L.append("\nopen-set (yazar bazında) yazar sayıları:\n")
        L.append("```\n" + meta.drop_duplicates("writer_id").groupby(["dataset", "split_open"]).size()
                 .unstack(fill_value=0).to_string() + "\n```")
    else:
        L.append("Bölme henüz üretilmedi (`day14_make_splits.py`). [ ]")

    L += ["", "## 3. Ön işleme raporu", "",
          f"- Patch boyutu: {cfg.patch_size}x{cfg.patch_size} (config.py), ölçek 0.5 (≈150 dpi), boş patch eşiği: koyu piksel oranı < 0.02",
          "- Sıra: gri tonlama -> kenar kırpma -> yazı bölgesine kırpma -> ölçekleme -> patch -> boş patch eleme",
          "- Satır görüntüsü hattı (preprocess_line): 64x512, beyaz doldurma, [0,1]",
          "- Yatay çevirme gibi el yazısını bozan artırma kullanılmıyor",
          "- Kural: önce sayfa bölünür, sonra patch çıkarılır", ""]
    pidx = root / "data" / "processed" / "patches" / "patches_index.csv"
    if pidx.exists():
        p = pd.read_csv(pidx)
        L.append(f"Üretilen patch: **{len(p)}** | yazar: {p['writer_id'].nunique()} | "
                 f"sayfa: {p['page_id'].nunique()} | ortalama koyu piksel oranı: {p['ink_ratio'].mean():.3f}")
        L.append("\n```\n" + p.groupby(["dataset", "split"]).size().unstack(fill_value=0).to_string() + "\n```")
    else:
        L.append("Patch veri seti henüz üretilmedi (`day17_make_patches.py`). [ ]")

    L += ["", "## 4. Çalıştırılan kod", "",
          "- Torch gerektirmeyen kısımlar: `python tests/test_week3.py`, `python tests/test_week4.py`",
          "- Torch'lu betiklerin çalıştırma durumu: [ ]", "", "## 5. Açık sorunlar / sonraki adımlar", "", "- [ ]", ""]
    return "\n".join(L)


def main() -> None:
    text = build_report()
    out = ROOT / "docs" / "ilk_18_gun_ozeti.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print("Yazıldı:", out)


if __name__ == "__main__":
    main()
