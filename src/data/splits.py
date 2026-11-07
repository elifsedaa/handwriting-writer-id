"""Gün 14: Bölme stratejisi. all_samples.csv'ye iki sütun ekler.

split_closed : closed-set için SAYFA bazında. Her yazarın sayfaları train/val/test'e
               dağıtılır; aynı sayfa (ve o sayfadan çıkan tüm satır/patch'ler) tek bölmede kalır.
               Sayfası MIN_PAGES'den az olan yazarlar "excluded" olur.
split_open   : open-set için YAZAR bazında. Bir yazarın tüm örnekleri tek bölmededir;
               test yazarları eğitimde hiç görülmez. Veri setleri ayrı ayrı bölünür.

Kural: ÖNCE sayfayı böl, SONRA patch çıkar (sızıntı olmasın). Seed sabittir; çıktı CSV repoda tutulur.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from src.utils import ROOT

MIN_PAGES = 3
FRACS = (0.70, 0.15, 0.15)   # train, val, test


def _counts(n: int, fracs=FRACS) -> tuple[int, int, int]:
    """n öğeyi (train, val, test) sayılarına böler; val ve test en az 1 olur."""
    n_val = max(1, round(n * fracs[1]))
    n_test = max(1, round(n * fracs[2]))
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def split_closed(df: pd.DataFrame, seed: int = 42, min_pages: int = MIN_PAGES) -> pd.Series:
    rng = np.random.default_rng(seed)
    out = pd.Series("excluded", index=df.index, dtype=object)
    page_split = {}
    for writer, g in df.groupby("writer_id", sort=True):
        pages = sorted(g["page_id"].unique())
        if len(pages) < min_pages:
            continue
        pages = list(rng.permutation(pages))
        n_tr, n_va, _ = _counts(len(pages))
        for i, pg in enumerate(pages):
            page_split[(writer, pg)] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    keys = list(zip(df["writer_id"], df["page_id"]))
    return pd.Series([page_split.get(k, "excluded") for k in keys], index=df.index, dtype=object)


def split_open(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed + 1)
    writer_split = {}
    for _, g in df.groupby("dataset", sort=True):
        writers = sorted(g["writer_id"].unique())
        writers = list(rng.permutation(writers))
        n_tr, n_va, _ = _counts(len(writers))
        for i, w in enumerate(writers):
            writer_split[w] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return df["writer_id"].map(writer_split)


def check_leakage(df: pd.DataFrame) -> list[str]:
    """Sızıntı denetimi. Sorun listesi döndürür (boşsa temiz)."""
    problems = []
    used = df[df["split_closed"] != "excluded"]
    n_splits = used.groupby(["writer_id", "page_id"])["split_closed"].nunique()
    if (n_splits > 1).any():
        problems.append("Bir sayfa birden fazla closed-set bölmesinde.")
    w_splits = df.groupby("writer_id")["split_open"].nunique()
    if (w_splits > 1).any():
        problems.append("Bir yazar birden fazla open-set bölmesinde.")
    return problems


def add_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    if "split" in df.columns:
        df = df.drop(columns="split")
    df["split_closed"] = split_closed(df, seed)
    df["split_open"] = split_open(df, seed)
    return df


def summary(df: pd.DataFrame) -> str:
    lines = ["closed-set (sayfa bazında) örnek sayıları:",
             df.groupby(["dataset", "split_closed"]).size().unstack(fill_value=0).to_string(), "",
             "open-set (yazar bazında) yazar sayıları:",
             df.drop_duplicates("writer_id").groupby(["dataset", "split_open"]).size()
               .unstack(fill_value=0).to_string()]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    csv = ROOT / "data" / "metadata" / "all_samples.csv"
    if not csv.exists():
        print("all_samples.csv yok. Önce: python scripts/day13_build_metadata.py")
        return
    df = pd.read_csv(csv)
    if df.empty:
        print("all_samples.csv boş; bölme yapılmadı.")
        return
    df = add_splits(df, args.seed)
    problems = check_leakage(df)
    df.to_csv(csv, index=False, encoding="utf-8")
    print(summary(df))
    print("\nSızıntı denetimi:", "TEMİZ" if not problems else problems)
    print("Yazıldı:", csv)


if __name__ == "__main__":
    main()
