from __future__ import annotations


def resolve_post_ml_score(meta: dict) -> float:
    try:
        return float(
            meta.get(
                "post_ml_competitive_score",
                meta.get(
                    "meta_post_ml_competitive_score",
                    meta.get(
                        "post_ml_score",
                        meta.get("meta_post_ml_score", 0.0),
                    ),
                ),
            ) or 0.0
        )
    except Exception:
        return 0.0


def parse_strategy_side_post_ml_weight_rules(raw: str | None) -> dict[tuple[str, str], tuple[float, float, float]]:
    out: dict[tuple[str, str], tuple[float, float, float]] = {}
    if not raw:
        return out

    for chunk in str(raw).split(","):
        item = str(chunk or "").strip()
        if not item:
            continue

        parts = [p.strip() for p in item.split("|")]
        if len(parts) != 5:
            continue

        sid, side, ref, min_mult, max_mult = parts
        try:
            out[(str(sid).lower(), str(side).lower())] = (
                float(ref),
                float(min_mult),
                float(max_mult),
            )
        except Exception:
            continue

    return out
