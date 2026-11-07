"""Gün 5: Proje planını (Gantt şeması) PNG olarak üretir."""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from src.utils import fig_path

# (görev, başlangıç, bitiş)
TASKS = [
    ("Kurulum ve literatür taraması", date(2026, 9, 28), date(2026, 10, 9)),
    ("Veri seti araştırması / hazırlama", date(2026, 10, 5), date(2026, 11, 6)),
    ("EDA ve ön işleme", date(2026, 10, 12), date(2026, 10, 30)),
    ("Baseline (MLP / CNN)", date(2026, 10, 19), date(2026, 11, 13)),
    ("Modüler kod yapısı", date(2026, 11, 9), date(2026, 11, 13)),
    ("Transfer öğrenme (ResNet-50)", date(2026, 11, 16), date(2026, 11, 27)),
    ("DINOv2 embedding", date(2026, 11, 26), date(2026, 12, 4)),
    ("Siamese / metric learning", date(2026, 12, 2), date(2026, 12, 11)),
    ("Değerlendirme (EER, unseen-writer)", date(2026, 12, 11), date(2026, 12, 18)),
    ("Demo (Gradio / HF Spaces)", date(2026, 12, 14), date(2026, 12, 18)),
    ("Hızlandırma ve toparlama", date(2026, 12, 21), date(2026, 12, 25)),
    ("Rapor ve sunum", date(2026, 12, 28), date(2026, 12, 31)),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (name, start, end) in enumerate(TASKS):
        ax.barh(i, (end - start).days + 1, left=mdates.date2num(start), height=0.6)
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels([t[0] for t in TASKS])
    ax.invert_yaxis()
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("El Yazısından Yazar Tanıma: 14 haftalık proje planı")
    fig.autofmt_xdate()
    out = fig_path("day05_gantt.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Gantt şeması kaydedildi:", out)


if __name__ == "__main__":
    main()
