from __future__ import annotations

import os
from typing import Any, Dict
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


CONFIG_PATH = os.environ.get("WAF_CONFIG_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "config.yaml"))


__all__ = ["load_config", "CONFIG_PATH"]


