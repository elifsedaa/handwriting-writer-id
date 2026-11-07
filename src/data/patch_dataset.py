"""Gün 17: Sayfalardan patch veri seti üretme ve görselleştirme.

Girdi : data/metadata/all_samples.csv (split_closed sütunu dolu olmalı, bkz. splits.py)
Çıktı : data/processed/patches/<split>/<writer_id>/<page_id>_<NNN>.png  +  patches_index.csv
Sızıntı kuralı: bölme sayfa düzeyinde yapıldığı için bir sayfanın tüm patch'leri aynı klasöre gider.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.data.crop import crop_to_ink, extract_patches_np, filter_blank_np, pad_min_size, trim_border, stretch_contrast
from src.data.preprocess import load_gray
from src.utils import ROOT, fig_path

PATCH_DIR = ROOT / "data" / "processed" / "patches"
INDEX_COLS = ["patch_path", "dataset", "writer_id", "page_id", "split", "patch_idx", "ink_ratio"]


def page_to_patches(path, patch: int = 128, stride: int | None = None, scale: float = 0.5,
                    min_ink: float = 0.02, trim: float = 0.0, margin: int = 16):
    """Tek sayfa/satır görüntüsü -> (patch'ler [N,P,P], ink_ratio [N]).

    Sıra: gri oku -> kenarlardan kırp (trim) -> yazı bölgesine kırp -> ölçekle -> en az PxP'ye doldur
    -> patch çıkar -> boş patch'leri ele. `scale` 300 dpi taramayı küçültür (0.5 ≈ 150 dpi).
    """
    arr = load_gray(path)
    arr = trim_border(arr, trim)
    arr = stretch_contrast(arr)
    arr = crop_to_ink(arr, margin=margin)
    if scale != 1.0:
        h, w = arr.shape
        im = Image.fromarray((arr * 255).astype(np.uint8))
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = pad_min_size(arr, patch)
    patches = extract_patches_np(arr, patch, stride)
    return filter_blank_np(patches, min_ink)


def build_patch_dataset(meta: pd.DataFrame, root: Path = ROOT, out_dir: Path | None = None,
                        split_col: str = "split_closed", patch: int = 128, stride: int | None = None,
                        scale: float = 0.5, min_ink: float = 0.02, trim: float = 0.0,
                        max_per_page: int | None = 200, seed: int = 42, verbose: bool = True) -> pd.DataFrame:
    """meta satırlarını (excluded/boş hariç) patch'lere çevirip diske yazar, indeks DataFrame'i döndürür.

    max_per_page: bir sayfadan en fazla bu kadar patch (rastgele, seed'li) -> sayfa uzunluğu baskın olmasın.
    """
    out_dir = out_dir or (root / "data" / "processed" / "patches")
    rng = np.random.default_rng(seed)
    rows, n_fail = [], 0
    todo = meta[meta[split_col].notna() & (meta[split_col] != "excluded")]
    for i, r in enumerate(todo.itertuples(index=False)):
        try:
            patches, ratios = page_to_patches(root / r.path, patch, stride, scale, min_ink, trim)
        except Exception as e:                      # bozuk dosya: atla, say
            n_fail += 1
            if verbose:
                print(f"  atlandı ({type(e).__name__}): {r.path}")
            continue
        if max_per_page and len(patches) > max_per_page:
            keep = np.sort(rng.choice(len(patches), max_per_page, replace=False))
            patches, ratios = patches[keep], ratios[keep]
        split = getattr(r, split_col)
        folder = out_dir / split / r.writer_id
        folder.mkdir(parents=True, exist_ok=True)
        for k, (p, ink) in enumerate(zip(patches, ratios)):
            f = folder / f"{r.page_id}_{k:03d}.png"
            Image.fromarray((p * 255).astype(np.uint8)).save(f)
            rows.append((f.relative_to(root).as_posix(), r.dataset, r.writer_id, r.page_id, split, k, float(ink)))
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(todo)} sayfa işlendi")
    idx = pd.DataFrame(rows, columns=INDEX_COLS)
    if verbose and n_fail:
        print(f"UYARI: {n_fail} dosya okunamadı.")
    return idx


def plot_patch_grid(index: pd.DataFrame, root: Path = ROOT, n: int = 16, seed: int = 42,
                    name: str = "day17_patch_grid.png") -> Path | None:
    """İndeksten rastgele n patch seçip ızgara olarak çizer (başlık: yazar kodu)."""
    if index.empty:
        return None
    sample = index.sample(min(n, len(index)), random_state=seed)
    cols = 4
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), sample.itertuples(index=False)):
        ax.imshow(Image.open(root / r.patch_path), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{r.writer_id}\n{r.split}", fontsize=8)
    fig.tight_layout()
    p = fig_path(name)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_patch_summary(index: pd.DataFrame, name: str = "day17_patch_ozeti.png") -> Path | None:
    """Bölmeye göre patch sayısı ve yazar başına patch dağılımı."""
    if index.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    index.groupby("split").size().reindex(["train", "val", "test"]).fillna(0).plot.bar(ax=axes[0], color="#4c72b0")
    axes[0].set_title("Bölmeye göre patch sayısı"); axes[0].set_xlabel("")
    axes[1].hist(index.groupby("writer_id").size(), bins=20, color="#55a868")
    axes[1].set_title("Yazar başına patch"); axes[1].set_xlabel("patch sayısı")
    fig.tight_layout()
    p = fig_path(name)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> None:
    csv = ROOT / "data" / "metadata" / "all_samples.csv"
    if not csv.exists():
        print("all_samples.csv yok. Sırayla: day13_build_metadata.py, day14_make_splits.py")
        return
    meta = pd.read_csv(csv)
    if meta.empty or "split_closed" not in meta.columns:
        print("CSV boş ya da split_closed sütunu yok (day14_make_splits.py çalıştırıldı mı?).")
        return
    from src.config import Config
    cfg = Config()
    print(f"Patch üretimi: patch={cfg.patch_size}, scale=0.5, min_ink=0.02")
    idx = build_patch_dataset(meta, patch=cfg.patch_size)
    if idx.empty:
        print("Hiç patch üretilemedi.")
        return
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    idx.to_csv(PATCH_DIR / "patches_index.csv", index=False, encoding="utf-8")
    print(f"{len(idx)} patch yazıldı -> {PATCH_DIR}")
    print(idx.groupby(["dataset", "split"]).size().unstack(fill_value=0))
    for p in (plot_patch_grid(idx), plot_patch_summary(idx)):
        print("Grafik:", p)


if __name__ == "__main__":
    main()
