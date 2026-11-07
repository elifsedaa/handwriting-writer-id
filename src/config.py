"""Deney ayarları. Tüm sabitler tek yerde toplanır (Gün 8)."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    seed: int = 42
    patch_size: int = 128
    batch_size: int = 32
    num_workers: int = 0        # Windows'ta başlangıçta 0 tut
    lr: float = 1e-3
    epochs: int = 10
    data_dir: str = "data"
    run_name: str = "debug"

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Config":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
