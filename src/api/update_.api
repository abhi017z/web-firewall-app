from __future__ import annotations

import glob
import os
import shutil
import threading
import time
from typing import Dict

from fastapi import BackgroundTasks, FastAPI, HTTPException

from ..utils.config import load_config, CONFIG_PATH
from ..utils.logger import logger, setup_logging


app = FastAPI(title="WAF Update API")
CONFIG = load_config(CONFIG_PATH)
STATE = {"running": False, "last_status": "idle", "version": 0}


def _models_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models", "checkpoints"))


def _list_versions() -> list[str]:
    base = _models_dir()
    return sorted(glob.glob(os.path.join(base, "model_v*.pt")))


def _current_version_path() -> str:
    base = _models_dir()
    return os.path.join(base, "best.pt")


def _rotate_versions(new_path: str) -> None:
    """Keep last 5 versioned checkpoints."""
    versions = _list_versions()
    while len(versions) > 4:
        old = versions.pop(0)
        try:
            os.remove(old)
        except FileNotFoundError:
            pass
    # Link/copy new as best
    best = _current_version_path()
    shutil.copyfile(new_path, best)


def _train_incremental(checkpoint_in: str, checkpoint_out: str, delta_path: str | None = None) -> None:
    """Fine-tune on delta benign data only if provided; else do a short full fine-tune."""
    import subprocess
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if delta_path and os.path.exists(os.path.join(root, delta_path)):
        # Temporarily swap train.jsonl with delta for a short run
        # The quick trainer reads data/train/train.jsonl, so point it to delta by copying
        tmp_backup = os.path.join(root, "data", "train", "_backup_train.jsonl")
        train_file = os.path.join(root, "data", "train", "train.jsonl")
        if os.path.exists(train_file):
            shutil.copyfile(train_file, tmp_backup)
        shutil.copyfile(os.path.join(root, delta_path), train_file)
        try:
            subprocess.check_call(["python", "scripts/train_quick.py", "--epochs", "2", "--batch", "64"], cwd=root)
        finally:
            if os.path.exists(tmp_backup):
                shutil.copyfile(tmp_backup, train_file)
    else:
        subprocess.check_call(["python", "scripts/train_quick.py", "--epochs", "2", "--batch", "64"], cwd=root)
    src = os.path.join(root, "models", "checkpoints", "best.pt")
    shutil.copyfile(src, checkpoint_out)


def incremental_retrain(delta: str | None = None) -> None:
    if STATE["running"]:
        return
    STATE["running"] = True
    STATE["last_status"] = "starting"
    try:
        models_dir = _models_dir()
        os.makedirs(models_dir, exist_ok=True)
        current = _current_version_path()
        if not os.path.exists(current):
            STATE["last_status"] = "no_base_model"
            return
        STATE["last_status"] = "training"
        next_version = len(_list_versions()) + 1
        out_path = os.path.join(models_dir, f"model_v{next_version}.pt")
        _train_incremental(current, out_path, delta_path=delta)
        STATE["last_status"] = "validating"
        # TODO: add real validation
        _rotate_versions(out_path)
        STATE["version"] = next_version
        STATE["last_status"] = "completed"
    except Exception as exc:
        logger.exception("Retrain failed: %s", exc)
        STATE["last_status"] = "failed"
    finally:
        STATE["running"] = False


@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks, delta_path: str | None = None):
    if STATE["running"]:
        return {"status": "already_running"}
    background_tasks.add_task(incremental_retrain, delta=delta_path)
    return {"status": "started"}


@app.get("/retrain/status")
async def retrain_status() -> Dict[str, object]:
    return dict(STATE)


@app.post("/add_benign_data")
async def add_benign_data() -> Dict[str, str]:
    # Placeholder endpoint for uploading/recording new benign samples
    return {"status": "accepted"}


