from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import re

archive = Path(r"C:\Users\elifs\Downloads\archive.zip")

with ZipFile(archive) as z:
    names = z.namelist()

    # IAM görüntü dosyalarının yollarını bul
    images = [
        name for name in names
        if name.lower().endswith(".png")
        and "/words/" in f"/{name}"
    ]

    # Form kimliklerini çıkar: ör. a01-000u
    forms = set()
    for name in images:
        parts = name.split("/")
        if len(parts) >= 4:
            forms.add(parts[-2])

    # Form kimliklerindeki sayısal bölümü yalnızca inceleme amacıyla çıkar
    candidate_ids = Counter()
    for form in forms:
        match = re.search(r"-(\d{3})[a-zA-Z]?$", form)
        if match:
            candidate_ids[match.group(1)] += 1

    print("=== IAM ARŞİV İNCELEMESİ ===")
    print("Görüntü sayısı:", len(images))
    print("Benzersiz form sayısı:", len(forms))
    print("Örnek görüntü yolları:")
    for name in images[:10]:
        print(" ", name)

    print("\nÖrnek form kimlikleri:")
    for form in sorted(forms)[:20]:
        print(" ", form)

    print("\nForm adlarından çıkarılan olası kimlik sayısı:",
          len(candidate_ids))
    print("İlk 20 olası kimlik ve form sayısı:")
    for writer_id, count in candidate_ids.most_common(20):
        print(f"  {writer_id}: {count} form")