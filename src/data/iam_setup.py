"""Gün 11: IAM arşivlerini klasörleme ve doğrulama.

Beklenen girdi: indirilen arşivler (ascii.tgz, lines.tgz, xml.tgz, formsA-D.tgz ...)
`data/raw/iam/_downloads/` içinde durur. Betik bunları `data/raw/iam/<ad>/` altına açar.
Arşiv iç yapısı IAM sürümüne göre değişebileceği için açtıktan sonra doğrulama yapılır
ve gördüğü yapıyı raporlar; varsayım yapmaz.
"""
from __future__ import annotations
import sys
import tarfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import ROOT

IAM_DIR = ROOT / "data" / "raw" / "iam"
DOWNLOADS = IAM_DIR / "_downloads"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _safe_members(tar: tarfile.TarFile, dest: Path):
    """Hedef klasörün dışına yazmaya çalışan (../) üyeleri atlar."""
    dest = dest.resolve()
    for m in tar.getmembers():
        if (dest / m.name).resolve().is_relative_to(dest):
            yield m


def extract_archives(downloads: Path = DOWNLOADS, iam_dir: Path = IAM_DIR) -> list[str]:
    """downloads altındaki .tgz/.tar.gz/.zip dosyalarını iam_dir/<arşiv adı>/ içine açar.

    Zaten açılmış (klasörü dolu) arşivler atlanır. Açılan arşiv adlarını döndürür.
    """
    done = []
    if not downloads.exists():
        return done
    for arc in sorted(downloads.iterdir()):
        name = arc.name
        for suf in (".tar.gz", ".tgz", ".zip"):
            if name.lower().endswith(suf):
                stem = name[: -len(suf)]
                break
        else:
            continue
        dest = iam_dir / stem
        if dest.exists() and any(dest.iterdir()):
            print(f"  atlandı (zaten açık): {stem}")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        if suf == ".zip":
            with zipfile.ZipFile(arc) as z:
                z.extractall(dest)
        else:
            with tarfile.open(arc) as t:
                t.extractall(dest, members=_safe_members(t, dest))
        print(f"  açıldı: {name} -> {dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest}")
        done.append(stem)
    return done


def verify(iam_dir: Path = IAM_DIR) -> dict:
    """Klasör yapısını sayarak özet döndürür (hiçbir şeyi değiştirmez)."""
    res = {"forms_txt": None, "n_images": {}, "folders": []}
    if not iam_dir.exists():
        return res
    res["folders"] = sorted(p.name for p in iam_dir.iterdir() if p.is_dir())
    hits = list(iam_dir.rglob("forms.txt"))
    res["forms_txt"] = str(hits[0].relative_to(iam_dir)) if hits else None
    for sub in res["folders"]:
        if sub.startswith("_"):
            continue
        n = sum(1 for p in (iam_dir / sub).rglob("*") if p.suffix.lower() in IMG_EXTS)
        res["n_images"][sub] = n
    return res


def read_forms_txt(path: Path) -> dict[str, str]:
    """forms.txt -> {form_id: yazar_no}.

    VARSAYIM: satır biçimi `<form_id> <yazar_no> ...`, '#' ile başlayan satırlar yorum.
    IAM'in gerçek dosyasını görünce `tests/test_week3.py` ve bu fonksiyonu doğrula.
    """
    mapping = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            mapping[parts[0]] = parts[1]
    return mapping


def main() -> None:
    print("=== IAM klasörleme ve doğrulama ===")
    if not IAM_DIR.exists():
        print(f"{IAM_DIR} yok. Önce klasörü oluşturup arşivleri _downloads/ içine koy.")
        return
    print("Arşivler açılıyor:")
    extract_archives()
    r = verify()
    print("\nKlasörler:", r["folders"])
    print("Görüntü sayıları:", r["n_images"])
    if r["forms_txt"]:
        m = read_forms_txt(IAM_DIR / r["forms_txt"])
        print(f"forms.txt: {r['forms_txt']}  ->  {len(m)} form, {len(set(m.values()))} farklı yazar")
        print("(IAM sayfasında yazılı yazar sayısıyla karşılaştır.)")
    else:
        print("forms.txt BULUNAMADI (ascii arşivi indirilip açıldı mı?).")


if __name__ == "__main__":
    main()
