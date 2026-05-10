from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CandidateQualityConfig:
    mode: str = "off"
    model_path: str = ""
    manifest_path: str = ""
    score_field: str = "candidate_quality_score_v0_47"

    @classmethod
    def from_args(cls, args) -> "CandidateQualityConfig":
        mode = str(getattr(args, "candidate_quality_mode", "off") or "off").strip().lower()
        model_path = str(getattr(args, "candidate_quality_model_path", "") or "").strip()
        manifest_path = str(getattr(args, "candidate_quality_manifest_path", "") or "").strip()
        score_field = str(
            getattr(args, "candidate_quality_score_field", "candidate_quality_score_v0_47")
            or "candidate_quality_score_v0_47"
        ).strip()

        if not manifest_path and model_path:
            candidate_manifest = Path(model_path).with_name("pwin_v0_47_raw_only_manifest.json")
            if candidate_manifest.exists():
                manifest_path = str(candidate_manifest)

        return cls(
            mode=mode,
            model_path=model_path,
            manifest_path=manifest_path,
            score_field=score_field,
        )

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def shadow_enabled(self) -> bool:
        return self.mode == "shadow"
