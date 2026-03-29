from __future__ import annotations

import json
from pathlib import Path


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


class PWinCalibrationTable:
    def __init__(self, artifact_path: str | None = None, key_mode: str = "strategy_side"):
        self.artifact_path = str(artifact_path or "").strip()
        self.key_mode = str(key_mode or "strategy_side").strip().lower()
        self.enabled = False
        self.default_bins = []
        self.table = {}

        if self.artifact_path:
            p = Path(self.artifact_path)
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                self.default_bins = list(data.get("default_bins", []) or [])
                self.table = dict(data.get("table", {}) or {})
                self.enabled = True

    def _build_key(self, strategy_id: str, side: str, symbol: str = "") -> str:
        strategy_id = str(strategy_id or "")
        side = str(side or "").lower()
        symbol = str(symbol or "")
        if self.key_mode == "strategy_side":
            return f"{strategy_id}|{side}"
        if self.key_mode == "symbol_side":
            return f"{symbol}|{side}"
        return f"{strategy_id}|{side}"

    def _lookup_bins(self, key: str):
        bins = self.table.get(key)
        if bins:
            return list(bins)
        return list(self.default_bins or [])

    def calibrate(self, *, p_win: float, strategy_id: str, side: str, symbol: str = "") -> tuple[float, dict]:
        raw = _clip(float(p_win), 0.0, 1.0)

        if not self.enabled:
            return raw, {
                "pwin_calibration_enabled": False,
                "pwin_calibration_key": "",
                "pwin_calibration_hit": False,
            }

        key = self._build_key(strategy_id=strategy_id, side=side, symbol=symbol)
        bins = self._lookup_bins(key)

        if not bins:
            return raw, {
                "pwin_calibration_enabled": True,
                "pwin_calibration_key": key,
                "pwin_calibration_hit": False,
            }

        chosen = None
        for b in bins:
            lo = float(b.get("p_min", 0.0))
            hi = float(b.get("p_max", 1.0))
            if raw >= lo and raw <= hi:
                chosen = b
                break

        if chosen is None:
            chosen = bins[-1]

        calibrated = _clip(float(chosen.get("empirical_win_rate", raw)), 0.0, 1.0)

        return calibrated, {
            "pwin_calibration_enabled": True,
            "pwin_calibration_key": key,
            "pwin_calibration_hit": True,
            "pwin_calibration_raw": float(raw),
            "pwin_calibration_calibrated": float(calibrated),
            "pwin_calibration_bin_min": float(chosen.get("p_min", 0.0)),
            "pwin_calibration_bin_max": float(chosen.get("p_max", 1.0)),
            "pwin_calibration_bin_count": int(chosen.get("count", 0)),
        }
