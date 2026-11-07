"""Gün 13: Tüm veri setlerinden tek metadata CSV üretir: data/metadata/all_samples.csv

Sütunlar: dataset, writer_id, page_id, path, split   (split burada boş; Gün 14'te doldurulur)
- writer_id veri seti önekiyle benzersizleştirilir: iam_000, cvl_0052, firemaker_017 ...
- path proje köküne göre, '/' ayraçlı göreli yoldur (kimlik bilgisi içermez).
- IAM: satır görüntüleri (data/raw/iam/**/lines/**) varsa onlar, yoksa form görüntüleri.
  Form görüntüleri basılı metin içerdiği için satır görüntüleri tercih edilir.
- CVL: dosya adı `<4 hane yazar>-<metin no>...` varsayılır.
- Firemaker: dosya adı düzeni doğrulanmadı; --firemaker-regex ile verilmelidir
  (iki grup: yazar, sayfa).
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from src.data.iam_setup import IMG_EXTS, read_forms_txt
from src.utils import ROOT

COLUMNS = ["dataset", "writer_id", "page_id", "path", "split"]
IAM_FORM_RE = re.compile(r"^([a-z]\d{2}-\d{3}[a-z]?)", re.IGNORECASE)   # a01-000u-00.png -> a01-000u
CVL_RE = re.compile(r"^(\d{4})-(\d+)")


def _rel(p: Path, root: Path) -> str:
    return p.relative_to(root).as_posix()


def _images(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def collect_iam(root: Path) -> pd.DataFrame:
    iam = root / "data" / "raw" / "iam"
    forms_txts = list(iam.rglob("forms.txt")) if iam.exists() else []
    if not forms_txts:
        print("  IAM: forms.txt yok, atlandı.")
        return pd.DataFrame(columns=COLUMNS)
    form2writer = read_forms_txt(forms_txts[0])
    imgs = [p for p in _images(iam) if "lines" in {x.lower() for x in p.relative_to(iam).parts[:-1]}]
    if not imgs:
        print("  IAM: lines görüntüsü yok, form görüntüleri kullanılıyor.")
        imgs = [p for p in _images(iam) if "lines" not in {x.lower() for x in p.relative_to(iam).parts[:-1]}]
    rows, unmatched = [], 0
    for p in imgs:
        m = IAM_FORM_RE.match(p.stem)
        writer = form2writer.get(m.group(1)) if m else None
        if writer is None:
            unmatched += 1
            continue
        rows.append(("iam", f"iam_{writer}", m.group(1), _rel(p, root), ""))
    if unmatched:
        print(f"  IAM: {unmatched} görüntü forms.txt ile eşleşmedi (atlandı).")
    return pd.DataFrame(rows, columns=COLUMNS)


def collect_cvl(root: Path) -> pd.DataFrame:
    rows = []
    for p in _images(root / "data" / "raw" / "cvl"):
        m = CVL_RE.match(p.stem)
        if m:
            rows.append(("cvl", f"cvl_{m.group(1)}", f"cvl_{m.group(1)}-{m.group(2)}", _rel(p, root), ""))
    return pd.DataFrame(rows, columns=COLUMNS)


def collect_firemaker(root: Path, pattern: str | None) -> pd.DataFrame:
    imgs = _images(root / "data" / "raw" / "firemaker")
    if not imgs:
        return pd.DataFrame(columns=COLUMNS)
    if not pattern:
        print(f"  Firemaker: {len(imgs)} görüntü var ama --firemaker-regex verilmedi, atlandı.")
        print("    Örnek dosya adı:", imgs[0].name)
        return pd.DataFrame(columns=COLUMNS)
    rx, rows = re.compile(pattern), []
    for p in imgs:
        m = rx.search(p.stem)
        if m:
            rows.append(("firemaker", f"firemaker_{m.group(1)}", f"firemaker_{m.group(1)}-{m.group(2)}",
                         _rel(p, root), ""))
    return pd.DataFrame(rows, columns=COLUMNS)


def build(root: Path = ROOT, firemaker_regex: str | None = None) -> pd.DataFrame:
    parts = [collect_iam(root), collect_cvl(root), collect_firemaker(root, firemaker_regex)]
    parts = [p for p in parts if len(p)]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
    return df.sort_values(["dataset", "writer_id", "page_id", "path"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--firemaker-regex", default=None,
                    help="dosya adı (uzantısız) için regex; grup1=yazar, grup2=sayfa")
    args = ap.parse_args()
    print("=== metadata üretiliyor ===")
    df = build(firemaker_regex=args.firemaker_regex)
    if df.empty:
        print("Hiç örnek bulunamadı; CSV yazılmadı. data/raw/ altına veri koyduktan sonra tekrar çalıştır.")
        return
    out = ROOT / "data" / "metadata" / "all_samples.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\n{len(df)} örnek yazıldı: {out}")
    if len(df):
        print(df.groupby("dataset").agg(ornek=("path", "size"), yazar=("writer_id", "nunique"),
                                        sayfa=("page_id", "nunique")))


if __name__ == "__main__":
    main()
