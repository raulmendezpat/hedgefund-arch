from __future__ import annotations

"""
Canonical P_win trainer entrypoint.

Current production/research canonical P_win artifact:
- Registry: artifacts/p_win/asset_side/pwin_asset_side_model_registry.json
- Model:    artifacts/p_win/asset_side/models/p_win_global_model.pkl

The active canonical runtime uses a global P_win model only.
Asset/side local models were tested and rejected because the global model won
across the robust 1M / 3M / 6M / 12M comparison.
"""

from pathlib import Path
import json


def main() -> int:
    registry = Path("artifacts/p_win/asset_side/pwin_asset_side_model_registry.json")
    if not registry.exists():
        raise SystemExit(f"Missing canonical P_win registry: {registry}")

    data = json.loads(registry.read_text())
    print("canonical_name:", data.get("canonical_name"))
    print("version:", data.get("version"))
    print("feature_policy:", data.get("feature_policy"))
    print("groups_count:", len(data.get("groups") or {}))
    print("global_fallback_enabled:", (data.get("global_fallback") or {}).get("enabled"))
    print("global_model_path:", (data.get("global_fallback") or {}).get("model_path"))
    print("feature_cols:")
    for c in (data.get("global_fallback") or {}).get("feature_cols", []):
        print(" -", c)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
