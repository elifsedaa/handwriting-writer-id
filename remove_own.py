"""Kendi veri toplama kodunu/dokümanını projeden çıkarır. Proje kökünden çalıştır: python remove_own.py
Tekrar çalıştırılabilir (idempotent). Git gerektirmez."""
import re, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 1) Silinecekler
for rel in ["docs/data_collection_protocol.md", "docs/consent_form_template.md", "docs/scan_standard.md",
            "scripts/day07_make_form.py", "scripts/day09_check_own_scans.py", "src/data/scan_checks.py",
            "data/metadata/own_samples.csv"]:
    p = ROOT / rel
    if p.exists():
        p.unlink(); print("silindi:", rel)
for rel in ["docs/forms", "data/raw/own"]:
    p = ROOT / rel
    if p.exists():
        shutil.rmtree(p); print("silindi:", rel)


def edit(rel, pairs):
    p = ROOT / rel
    t = p.read_text(encoding="utf-8")
    for old, new in pairs:
        if old in t:
            t = t.replace(old, new)
        else:
            print(f"  UYARI: {rel} içinde bulunamadı (zaten değişmiş olabilir): {old[:50]!r}")
    p.write_text(t, encoding="utf-8"); print("düzenlendi:", rel)


# 2) Kaynak kod
edit("src/data/check_datasets.py", [('    "own": "data/raw/own",\n', "")])
edit("src/data/report.py", [
    ('["iam", "cvl", "firemaker", "own"]', '["iam", "cvl", "firemaker"]'),
    ("  |  Kendi veri toplama: hedef yazar [ ], toplanan [ ]", ""),
])
p = ROOT / "src/data/build_metadata.py"
t = p.read_text(encoding="utf-8")
t = t.replace("iam_000, cvl_0052, own_W001 ...", "iam_000, cvl_0052, firemaker_017 ...")
t = re.sub(r'OWN_RE = .*\n', "", t)
t = re.sub(r'def collect_own\(root.*?\n\n\n', "", t, flags=re.S)
t = t.replace(", collect_own(root)]", "]")
p.write_text(t, encoding="utf-8"); print("düzenlendi: src/data/build_metadata.py")

# 3) Testler: sentetik "own" yazarlar yerine sentetik "firemaker" yazarlar kullanılır
FM = r"^W(\d{3})_p(\d{2})$"   # yalnızca test için sahte dosya adı düzeni
edit("tests/test_week3.py", [
    ("6 yazar x 4 metin ; own: 8 yazar x 4 sayfa", "6 yazar x 4 metin ; firemaker (sahte düzen): 8 yazar x 4 sayfa"),
    ('data/raw/own/W{w:03d}', 'data/raw/firemaker/W{w:03d}'),
    ('data/raw/own/kotu_ad.png', 'data/raw/firemaker/kotu_ad.png'),
    ("df = bm.build(root)", "df = bm.build(root, firemaker_regex=FM_RE)"),
    ('"own": 8}', '"firemaker": 8}'),
    ('(df["dataset"] == "own").sum() == 32', '(df["dataset"] == "firemaker").sum() == 32'),
    ('own = s[s.dataset == "own"]', 'own = s[s.dataset == "firemaker"]'),
    ('(s.dataset == "own") & (s.split_closed', '(s.dataset == "firemaker") & (s.split_closed'),
    ('data/raw/own/W001/W001_p01.png', 'data/raw/firemaker/W001/W001_p01.png'),
    ("def save_img(", 'FM_RE = r"^W(\\d{3})_p(\\d{2})$"   # sahte Firemaker dosya adı düzeni (yalnızca test)\n\n\ndef save_img('),
])
edit("tests/test_week4.py", [
    ('data/raw/own/W{w:03d}', 'data/raw/firemaker/W{w:03d}'),
    ('("own", f"own_W{w:03d}", f"own_W{w:03d}_p{p:02d}"', '("firemaker", f"firemaker_W{w:03d}", f"firemaker_W{w:03d}_p{p:02d}"'),
    ('data/raw/own/yok.png', 'data/raw/firemaker/yok.png'),
    ('and "own" in txt2', 'and "firemaker" in txt2'),
    ('"| own | 20 | 5 | 20 | 20 |"', '"| firemaker | 20 | 5 | 20 | 20 |"'),
    ('l.startswith("| own")', 'l.startswith("| firemaker")'),
])

# 4) Dokümanlar
edit("README.md", [
    ("| 7 | `python scripts/day07_make_form.py` | Yazı toplama formu PDF |\n", ""),
    ("| 9 | `python scripts/day09_check_own_scans.py` | Kendi taramalarını denetle, `own_samples.csv` |\n", ""),
])
edit("docs/dataset_access_log.md", [
    ("| Kendi verim | - | [ ] | Planlandı | Onamlı formlu, kendi protokol | - | `docs/data_collection_protocol.md` |\n", ""),
    ("| Kendi verim | - | [ ] | Planlandı | Onam formlu, kendi protokol | - | `docs/data_collection_protocol.md` |\n", ""),
])
p = ROOT / "docs/datasets.md"
lines = [l for l in p.read_text(encoding="utf-8").splitlines(keepends=True)
         if "Kendi toplayacağım veri" not in l and "Kendi veri toplamamda" not in l]
txt = "".join(lines)
if txt.rstrip().endswith("## Not"):
    txt = txt.rstrip() + "\nEl yazısı örnekleri kişisel veri sayılabilir; yalnızca hazır veri setleri kullanılır ve lisans koşullarına uyulur.\n"
p.write_text(txt, encoding="utf-8"); print("düzenlendi: docs/datasets.md")
edit("docs/plan.md", [("Veri toplama / hazırlama, EDA", "Veri hazırlama, EDA")])
edit("scripts/day05_gantt.py", [("Veri seti araştırması / veri toplama", "Veri seti araştırması / hazırlama")])
print("\nBitti. Şimdi: python tests\\test_week3.py ; python tests\\test_week4.py")
