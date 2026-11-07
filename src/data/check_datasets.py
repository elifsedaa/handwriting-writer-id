"""Gün 6: data/raw altındaki veri setlerinin durumunu raporlar (docs/dataset_status.md)."""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import ROOT

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DATASETS = {
    "iam": "data/raw/iam",
    "cvl": "data/raw/cvl",
    "firemaker": "data/raw/firemaker",
}


def scan(folder: Path):
    """(görüntü sayısı, toplam MB) döndürür."""
    if not folder.exists():
        return 0, 0.0
    n, size = 0, 0
    for p in folder.rglob("*"):
        if p.is_file():
            size += p.stat().st_size
            if p.suffix.lower() in IMG_EXTS:
                n += 1
    return n, size / 1024 ** 2


def iam_writer_count():
    """IAM ascii/forms.txt dosyasından farklı yazar sayısını okumaya çalışır."""
    f = ROOT / "data/raw/iam/ascii/forms.txt"
    if not f.exists():
        return None
    writers = set()
    for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            writers.add(parts[1])
    return len(writers)


def main() -> None:
    rows = []
    for name, rel in DATASETS.items():
        n, mb = scan(ROOT / rel)
        rows.append((name, rel, n, mb, "VAR" if n > 0 else "YOK"))

    lines = [f"# Veri seti durumu ({datetime.now():%Y-%m-%d %H:%M})", "",
             "| Veri seti | Klasör | Görüntü sayısı | Boyut (MB) | Durum |",
             "|---|---|---|---|---|"]
    for name, rel, n, mb, st in rows:
        lines.append(f"| {name} | `{rel}` | {n} | {mb:.1f} | {st} |")

    n_writers = iam_writer_count()
    lines.append("")
    if n_writers is None:
        lines.append("IAM `ascii/forms.txt` bulunamadı (henüz indirilmedi ya da yol farklı).")
    else:
        lines.append(f"IAM forms.txt içinden okunan farklı yazar sayısı: **{n_writers}** "
                     "(IAM sayfasındaki sayıyla karşılaştır; farklıysa dosyayı elle incele).")

    report = "\n".join(lines) + "\n"
    print(report)
    out = ROOT / "docs" / "dataset_status.md"
    out.write_text(report, encoding="utf-8")
    print("Rapor yazıldı:", out)


if __name__ == "__main__":
    main()
