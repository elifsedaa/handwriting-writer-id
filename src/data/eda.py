"""Gün 12: Keşifsel veri analizi (EDA). Girdi: data/metadata/all_samples.csv.

Üretilen grafikler (experiments/figures/): yazar sayısı, yazar başına örnek, görüntü boyutu histogramı.
Görüntü boyutları yalnızca dosya başlığından okunur (piksel yüklenmez), bu yüzden hızlıdır.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.utils import ROOT, fig_path


def image_sizes(df: pd.DataFrame, root: Path = ROOT, max_n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Her veri setinden en fazla max_n görüntünün (width, height) değerini okur."""
    parts = []
    for _, g in df.groupby("dataset"):
        parts.append(g.sample(min(len(g), max_n), random_state=seed))
    rows = []
    for _, r in pd.concat(parts).iterrows():
        p = root / r["path"]
        try:
            with Image.open(p) as im:
                rows.append((r["dataset"], im.size[0], im.size[1]))
        except Exception:
            continue
    return pd.DataFrame(rows, columns=["dataset", "width", "height"])


def per_dataset_table(df: pd.DataFrame) -> pd.DataFrame:
    per_writer = df.groupby(["dataset", "writer_id"]).size().rename("n").reset_index()
    t = per_writer.groupby("dataset")["n"].agg(yazar="size", ornek="sum", min="min",
                                               medyan="median", max="max")
    pages = df.groupby("dataset")["page_id"].nunique().rename("sayfa")
    return t.join(pages)


def plot_all(df: pd.DataFrame, sizes: pd.DataFrame, prefix: str = "day12") -> list[Path]:
    out = []
    t = per_dataset_table(df)

    fig, ax = plt.subplots(figsize=(6, 4))
    t["yazar"].plot.bar(ax=ax, color="#4c72b0")
    ax.set_title("Veri setlerine göre yazar sayısı"); ax.set_ylabel("yazar"); ax.set_xlabel("")
    for i, v in enumerate(t["yazar"]):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout(); p = fig_path(f"{prefix}_yazar_sayisi.png"); fig.savefig(p, dpi=150); plt.close(fig)
    out.append(p)

    per_writer = df.groupby(["dataset", "writer_id"]).size().rename("n").reset_index()
    names = sorted(per_writer["dataset"].unique())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.5), squeeze=False)
    for ax, name in zip(axes[0], names):
        ax.hist(per_writer.loc[per_writer["dataset"] == name, "n"], bins=20, color="#55a868")
        ax.set_title(name); ax.set_xlabel("yazar başına örnek"); ax.set_ylabel("yazar")
    fig.tight_layout(); p = fig_path(f"{prefix}_yazar_basina_ornek.png"); fig.savefig(p, dpi=150); plt.close(fig)
    out.append(p)

    if len(sizes):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        for name, g in sizes.groupby("dataset"):
            axes[0].hist(g["width"], bins=30, alpha=0.6, label=name)
            axes[1].hist(g["height"], bins=30, alpha=0.6, label=name)
        axes[0].set_title("Görüntü genişliği (px)"); axes[1].set_title("Görüntü yüksekliği (px)")
        axes[0].legend()
        fig.tight_layout(); p = fig_path(f"{prefix}_goruntu_boyutu.png"); fig.savefig(p, dpi=150); plt.close(fig)
        out.append(p)
    return out


def main() -> None:
    csv = ROOT / "data" / "metadata" / "all_samples.csv"
    if not csv.exists():
        print("all_samples.csv yok. Önce: python scripts/day13_build_metadata.py")
        return
    df = pd.read_csv(csv)
    if df.empty:
        print("all_samples.csv boş; EDA yapılmadı.")
        return
    print(per_dataset_table(df).to_string())
    sizes = image_sizes(df)
    print("\nGrafikler:")
    for p in plot_all(df, sizes):
        print(" ", p)


if __name__ == "__main__":
    main()
