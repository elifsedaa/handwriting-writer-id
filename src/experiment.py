"""Her deney için tarih damgalı klasör açar; config, ortam ve git sürümünü kaydeder."""
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.config import Config
from src.utils import ROOT, get_env_info


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "yok"
    except Exception:
        return "yok"


def make_run_dir(cfg: Config) -> Path:
    """experiments/runs/<tarih_saat>_<run_name>/ klasörünü oluşturur."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "experiments" / "runs" / f"{stamp}_{cfg.run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg.save(run_dir / "config.json")
    env = get_env_info()
    env["git_commit"] = git_commit()
    (run_dir / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    return run_dir
