from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_selection_policy_config(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base or {})
    for k, v in dict(override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_profile_config(
    cfg: dict[str, Any],
    *,
    symbol: str,
    side: str,
    profile: str | None = None,
) -> dict[str, Any]:
    profile_name = str(profile or cfg.get("default_profile", "research"))
    base = dict((cfg.get("profiles", {}) or {}).get(profile_name, {}) or {})

    asset_over = dict((cfg.get("asset_overrides", {}) or {}).get(symbol, {}) or {})
    asset_side_over = dict((cfg.get("asset_side_overrides", {}) or {}).get(f"{symbol}::{side}", {}) or {})

    merged = _deep_merge(base, asset_over)
    merged = _deep_merge(merged, asset_side_over)
    return merged
