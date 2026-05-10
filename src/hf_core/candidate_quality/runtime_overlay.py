from __future__ import annotations

from functools import lru_cache

from .config import CandidateQualityConfig
from .model import CandidateQualityModel


@lru_cache(maxsize=8)
def _get_model(model_path: str, manifest_path: str) -> CandidateQualityModel:
    return CandidateQualityModel(model_path=model_path, manifest_path=manifest_path)


def apply_candidate_quality_shadow_to_candidate(candidate, args):
    """
    Research-only candidate quality overlay.

    Contract:
    - Does not mutate p_win, p_win_prod, p_win_ml_raw, p_win_effective_runtime.
    - In shadow mode, only exports diagnostics into candidate.signal_meta.
    - Gate and sizing effects are intentionally not implemented here yet.
    """
    cfg = CandidateQualityConfig.from_args(args)

    if not cfg.enabled:
        return candidate

    sm = dict(getattr(candidate, "signal_meta", {}) or {})
    sm["candidate_quality_mode"] = cfg.mode
    sm["candidate_quality_model_path"] = cfg.model_path
    sm["candidate_quality_manifest_path"] = cfg.manifest_path
    sm["candidate_quality_score_field"] = cfg.score_field

    if not cfg.shadow_enabled:
        sm["candidate_quality_error"] = f"unsupported_mode_for_initial_patch:{cfg.mode}"
        sm["candidate_quality_shadow_applied"] = False
        candidate.signal_meta = sm
        return candidate

    if not cfg.model_path or not cfg.manifest_path:
        sm["candidate_quality_error"] = "missing_model_or_manifest_path"
        sm["candidate_quality_shadow_applied"] = False
        candidate.signal_meta = sm
        return candidate

    try:
        model = _get_model(cfg.model_path, cfg.manifest_path)
        score = model.score_candidate(candidate)
        sm[cfg.score_field] = float(score)
        sm["candidate_quality_score_v0_47"] = float(score)
        sm["candidate_quality_shadow_applied"] = True
        sm["candidate_quality_error"] = ""
    except Exception as exc:
        sm[cfg.score_field] = float("nan")
        sm["candidate_quality_score_v0_47"] = float("nan")
        sm["candidate_quality_shadow_applied"] = False
        sm["candidate_quality_error"] = f"{type(exc).__name__}:{exc}"

    candidate.signal_meta = sm
    return candidate
