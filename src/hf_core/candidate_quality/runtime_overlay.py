from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .config import CandidateQualityConfig
from .model import CandidateQualityModel


@lru_cache(maxsize=8)
def _get_model(model_path: str, manifest_path: str) -> CandidateQualityModel:
    return CandidateQualityModel(model_path=model_path, manifest_path=manifest_path)


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        out = float(value)
        if not np.isfinite(out):
            return float(default)
        return float(out)
    except Exception:
        return float(default)


def _score_candidate_quality(candidate, cfg: CandidateQualityConfig) -> tuple[float, str]:
    if not cfg.model_path or not cfg.manifest_path:
        return float("nan"), "missing_model_or_manifest_path"

    try:
        model = _get_model(cfg.model_path, cfg.manifest_path)
        return float(model.score_candidate(candidate)), ""
    except Exception as exc:
        return float("nan"), f"{type(exc).__name__}:{exc}"


def apply_candidate_quality_shadow_to_candidate(candidate, args):
    """
    Research-only candidate quality overlay.

    Contract:
    - Does not mutate p_win, p_win_prod, p_win_ml_raw, p_win_effective_runtime.
    - In shadow mode, only exports diagnostics into candidate.signal_meta.
    - In gate mode, this function only computes/exports score; the actual filter
      is applied by apply_candidate_quality_gate_to_selection.
    """
    cfg = CandidateQualityConfig.from_args(args)

    if not cfg.enabled:
        return candidate

    sm = dict(getattr(candidate, "signal_meta", {}) or {})
    sm["candidate_quality_mode"] = cfg.mode
    sm["candidate_quality_model_path"] = cfg.model_path
    sm["candidate_quality_manifest_path"] = cfg.manifest_path
    sm["candidate_quality_score_field"] = cfg.score_field
    sm["candidate_quality_gate_threshold"] = float(cfg.gate_threshold)

    if not (cfg.shadow_enabled or cfg.gate_enabled):
        sm["candidate_quality_error"] = f"unsupported_mode:{cfg.mode}"
        sm["candidate_quality_shadow_applied"] = False
        candidate.signal_meta = sm
        return candidate

    score, error = _score_candidate_quality(candidate, cfg)
    sm[cfg.score_field] = float(score)
    sm["candidate_quality_score_v0_47"] = float(score)
    sm["candidate_quality_shadow_applied"] = bool(error == "")
    sm["candidate_quality_error"] = str(error or "")

    candidate.signal_meta = sm
    return candidate


def _normalize_gate_symbol(value) -> str:
    s = str(value or "").upper().strip()
    s = s.replace("/USDT:USDT", "")
    s = s.replace("_USDT_USDT", "")
    s = s.replace("USDT", "")
    s = s.replace("/", "")
    s = s.replace(":", "")
    s = s.replace("_", "")
    return s


def _normalize_gate_side(value) -> str:
    s = str(value or "").lower().strip()
    if s in {"buy", "long"}:
        return "long"
    if s in {"sell", "short"}:
        return "short"
    return s


@lru_cache(maxsize=16)
def _load_gate_config(config_path: str) -> dict:
    path = str(config_path or "").strip()
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {
            "_load_error": f"missing_gate_config:{path}",
            "default_action": "pass",
            "groups": [],
        }
    try:
        data = json.loads(p.read_text())
        if not isinstance(data, dict):
            return {
                "_load_error": f"invalid_gate_config_not_object:{path}",
                "default_action": "pass",
                "groups": [],
            }
        return data
    except Exception as exc:
        return {
            "_load_error": f"{type(exc).__name__}:{exc}",
            "default_action": "pass",
            "groups": [],
        }


@lru_cache(maxsize=16)
def _build_gate_rule_map(config_path: str) -> tuple[dict, str, str]:
    cfg = _load_gate_config(config_path)
    default_action = str(cfg.get("default_action", "pass") or "pass").lower().strip()
    load_error = str(cfg.get("_load_error", "") or "")

    rules = {}
    for raw in cfg.get("groups", []) or []:
        if not isinstance(raw, dict):
            continue

        symbol = _normalize_gate_symbol(raw.get("symbol", ""))
        side = _normalize_gate_side(raw.get("side", ""))
        strategy_id = str(raw.get("strategy_id", "") or "").strip()

        if not symbol or not side or not strategy_id:
            continue

        try:
            threshold = float(raw.get("threshold", 0.30))
        except Exception:
            threshold = 0.30
        threshold = max(0.0, min(1.0, float(threshold)))

        key = (symbol, side, strategy_id)
        rules[key] = {
            "threshold": threshold,
            "reason": str(raw.get("reason", "") or ""),
        }

    return rules, default_action, load_error


def _resolve_candidate_gate_rule(candidate, cfg: CandidateQualityConfig) -> tuple[bool, float, str]:
    """
    Returns:
    - should_evaluate_gate: True means score and compare against threshold.
    - threshold: group-specific or global threshold.
    - reason: diagnostic reason/scope.
    """
    config_path = str(getattr(cfg, "gate_config_json", "") or "").strip()
    if not config_path:
        return True, float(cfg.gate_threshold), "global_threshold"

    rules, default_action, load_error = _build_gate_rule_map(config_path)
    if load_error:
        # Safety-first: if config cannot be loaded, do not unexpectedly block trades.
        return False, float(cfg.gate_threshold), f"gate_config_error_pass_through:{load_error}"

    symbol = _normalize_gate_symbol(getattr(candidate, "symbol", ""))
    side = _normalize_gate_side(getattr(candidate, "side", ""))
    strategy_id = str(getattr(candidate, "strategy_id", "") or "").strip()
    key = (symbol, side, strategy_id)

    rule = rules.get(key)
    if rule is not None:
        return True, float(rule["threshold"]), f"group_rule:{symbol}|{side}|{strategy_id}"

    if default_action == "block":
        return True, float(cfg.gate_threshold), f"default_block:{symbol}|{side}|{strategy_id}"

    return False, float(cfg.gate_threshold), f"default_pass:{symbol}|{side}|{strategy_id}"


def apply_candidate_quality_gate_to_selection(selected_candidates, selected_decisions, args):
    """
    Research-only hard gate after selection/prod-selection and before allocation.

    Contract:
    - mode=off/shadow: exact pass-through.
    - mode=gate without config: global threshold filter.
    - mode=gate with config default_action=pass: only configured groups are filtered.
    - Does not mutate p_win, p_win_prod, p_win_ml_raw, p_win_effective_runtime.
    """
    cfg = CandidateQualityConfig.from_args(args)

    if not cfg.gate_enabled:
        return selected_candidates, selected_decisions, {
            "candidate_quality_gate_enabled": False,
            "candidate_quality_gate_applied": False,
            "candidate_quality_gate_threshold": float(cfg.gate_threshold),
            "candidate_quality_gate_in": int(len(selected_candidates or [])),
            "candidate_quality_gate_out": int(len(selected_candidates or [])),
            "candidate_quality_gate_blocked": 0,
        }

    kept_candidates = []
    kept_decisions = []
    blocked = 0
    rows = []

    for candidate, decision in zip(list(selected_candidates or []), list(selected_decisions or [])):
        should_eval, threshold, scope_reason = _resolve_candidate_gate_rule(candidate, cfg)

        sm = dict(getattr(candidate, "signal_meta", {}) or {})
        sm["candidate_quality_gate_config_json"] = str(getattr(cfg, "gate_config_json", "") or "")
        sm["candidate_quality_gate_scope_reason"] = str(scope_reason)
        sm["candidate_quality_gate_enabled"] = True
        sm["candidate_quality_gate_applied"] = bool(should_eval)
        sm["candidate_quality_gate_threshold"] = float(threshold)

        score = float("nan")
        error = ""

        if should_eval:
            candidate = apply_candidate_quality_shadow_to_candidate(candidate, args)
            sm = dict(getattr(candidate, "signal_meta", {}) or {})
            sm["candidate_quality_gate_config_json"] = str(getattr(cfg, "gate_config_json", "") or "")
            sm["candidate_quality_gate_scope_reason"] = str(scope_reason)
            sm["candidate_quality_gate_enabled"] = True
            sm["candidate_quality_gate_applied"] = True
            sm["candidate_quality_gate_threshold"] = float(threshold)

            score = _safe_float(sm.get(cfg.score_field, sm.get("candidate_quality_score_v0_47", float("nan"))))
            error = str(sm.get("candidate_quality_error", "") or "")
            gate_pass = bool(np.isfinite(score) and score >= float(threshold) and not error)
            reason = "pass" if gate_pass else ("score_below_threshold" if not error else f"score_error:{error}")
        else:
            gate_pass = True
            reason = str(scope_reason)

        sm["candidate_quality_gate_pass"] = bool(gate_pass)
        sm["candidate_quality_selected_after_gate"] = bool(gate_pass)
        sm["candidate_quality_gate_reason"] = reason

        candidate.signal_meta = sm

        rows.append({
            "symbol": str(getattr(candidate, "symbol", "") or ""),
            "strategy_id": str(getattr(candidate, "strategy_id", "") or ""),
            "side": str(getattr(candidate, "side", "") or ""),
            "score": float(score) if np.isfinite(score) else None,
            "threshold": float(threshold),
            "pass": bool(gate_pass),
            "reason": reason,
            "gate_applied": bool(should_eval),
            "scope_reason": str(scope_reason),
            "gate_config_json": str(getattr(cfg, "gate_config_json", "") or ""),
        })

        if gate_pass:
            kept_candidates.append(candidate)
            kept_decisions.append(decision)
        else:
            blocked += 1

    meta = {
        "candidate_quality_gate_enabled": True,
        "candidate_quality_gate_applied": True,
        "candidate_quality_gate_threshold": float(cfg.gate_threshold),
        "candidate_quality_gate_in": int(len(selected_candidates or [])),
        "candidate_quality_gate_out": int(len(kept_candidates)),
        "candidate_quality_gate_blocked": int(blocked),
        "candidate_quality_gate_rows": rows,
    }

    return kept_candidates, kept_decisions, meta
