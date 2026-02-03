from __future__ import annotations
from typing import Any, Dict
import yaml
import copy

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(d)
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out

def parse_overrides(overrides: list[str]) -> Dict[str, Any]:
    """
    CLI overrides: key1.key2=value
    """
    out: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override: {item} (expected key=value)")
        key, raw = item.split("=", 1)
        if raw.lower() in {"true", "false"}:
            val: Any = raw.lower() == "true"
        else:
            try:
                val = float(raw) if "." in raw else int(raw)
            except ValueError:
                val = raw
        cur = out
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    return out
