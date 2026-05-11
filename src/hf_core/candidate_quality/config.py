from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class CandidateQualityConfig:
    mode: str = "off"
    model_path: str = ""
    manifest_path: str = ""
    score_field: str = "candidate_quality_score_v0_47"
    gate_threshold: float = 0.30
    gate_config_json: str = ""

    @classmethod
    def from_args(cls, args) -> "CandidateQualityConfig":
        mode = str(getattr(args, "candidate_quality_mode", "off") or "off").strip().lower()
        model_path = str(getattr(args, "candidate_quality_model_path", "") or "").strip()
        manifest_path = str(getattr(args, "candidate_quality_manifest_path", "") or "").strip()
        score_field = str(
            getattr(args, "candidate_quality_score_field", "candidate_quality_score_v0_47")
            or "candidate_quality_score_v0_47"
        ).strip()

        raw_gate_threshold = getattr(args, "candidate_quality_gate_threshold", 0.30)
        if raw_gate_threshold is None:
            raw_gate_threshold = 0.30
        try:
            gate_threshold = float(raw_gate_threshold)
        except Exception:
            gate_threshold = 0.30
        gate_threshold = max(0.0, min(1.0, float(gate_threshold)))
        gate_config_json = str(getattr(args, "candidate_quality_gate_config_json", "") or "").strip()

        if not manifest_path and model_path:
            candidate_manifest = Path(model_path).with_name("pwin_v0_47_raw_only_manifest.json")
            if candidate_manifest.exists():
                manifest_path = str(candidate_manifest)

        # Allow callers to pass only the manifest path. The manifest is the canonical
        # deployment artifact and can point to the model. This keeps runtime CLI usage
        # stable while avoiding silent missing_model_or_manifest_path failures.
        if manifest_path and not model_path:
            try:
                manifest_p = Path(manifest_path)
                manifest_obj = json.loads(manifest_p.read_text())

                def _walk_model_refs(obj):
                    refs = []
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            kl = str(k).lower()
                            if isinstance(v, str) and (
                                "model" in kl
                                or "joblib" in kl
                                or v.endswith(".joblib")
                                or v.endswith(".pkl")
                            ):
                                refs.append(v)
                            refs.extend(_walk_model_refs(v))
                    elif isinstance(obj, list):
                        for item in obj:
                            refs.extend(_walk_model_refs(item))
                    return refs

                candidate_model_refs = []
                candidate_model_refs.extend([
                    manifest_obj.get("model_path"),
                    manifest_obj.get("model"),
                    manifest_obj.get("model_file"),
                    manifest_obj.get("reference_model"),
                    (manifest_obj.get("outputs") or {}).get("model"),
                    (manifest_obj.get("artifacts") or {}).get("model"),
                ])
                candidate_model_refs.extend(_walk_model_refs(manifest_obj))

                seen = set()
                for ref in candidate_model_refs:
                    if not ref:
                        continue
                    ref = str(ref).strip()
                    if not ref or ref in seen:
                        continue
                    seen.add(ref)

                    ref_path = Path(ref)
                    candidates = [
                        ref_path,
                        manifest_p.parent / ref,
                        manifest_p.parent / ref_path.name,
                    ]

                    for candidate_path in candidates:
                        if candidate_path.is_file():
                            model_path = str(candidate_path)
                            break

                    if model_path:
                        break

                # Last-resort safe fallback: use the single model artifact colocated
                # with the manifest. This is valid for v0.47 raw-only candidate quality.
                if not model_path:
                    colocated_models = sorted(
                        list(manifest_p.parent.glob("*.joblib"))
                        + list(manifest_p.parent.glob("*.pkl"))
                    )
                    if len(colocated_models) == 1:
                        model_path = str(colocated_models[0])

            except Exception:
                # Validation/fail-fast is handled by research_runtime and scorer.
                pass

        return cls(
            mode=mode,
            model_path=model_path,
            manifest_path=manifest_path,
            score_field=score_field,
            gate_threshold=gate_threshold,
            gate_config_json=gate_config_json,
        )

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def shadow_enabled(self) -> bool:
        return self.mode == "shadow"

    @property
    def gate_enabled(self) -> bool:
        return self.mode == "gate"
