from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RET_COLS = ["ret_1h", "ret_4h", "ret_12h", "ret_24h", "ret_48h"]


def summarize(df: pd.DataFrame, group_cols: list[str], ret_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row["rows"] = len(g)

        for rc in ret_cols:
            s = pd.to_numeric(g.get(rc), errors="coerce")
            row[f"{rc}_mean"] = float(s.mean()) if len(s) else 0.0
            row[f"{rc}_median"] = float(s.median()) if len(s) else 0.0
            row[f"{rc}_hit"] = float((s > 0).mean()) if len(s) else 0.0

        # mejor horizonte por media
        best_col = None
        best_val = None
        for rc in ret_cols:
            v = row.get(f"{rc}_mean")
            if best_val is None or (v is not None and v > best_val):
                best_val = v
                best_col = rc
        row["best_horizon"] = best_col
        row["best_horizon_mean"] = best_val
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("best_horizon_mean", ascending=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    p = Path(f"results/performance_attribution_{args.name}.csv")
    if not p.exists():
        raise SystemExit(f"Missing file: {p}")

    df = pd.read_csv(p)

    ret_cols = [c for c in RET_COLS if c in df.columns]
    if not ret_cols:
        raise SystemExit("No return horizon columns found")

    accepted = df[df["accept"] == True].copy()

    outputs: list[tuple[str, pd.DataFrame]] = []

    outputs.append(("BY_SIDE", summarize(accepted, ["side"], ret_cols)))
    outputs.append(("BY_STRATEGY", summarize(accepted, ["strategy_id"], ret_cols)))
    outputs.append(("BY_STRATEGY_SIDE", summarize(accepted, ["strategy_id", "side"], ret_cols)))

    if "score_bucket" in accepted.columns:
        outputs.append(("BY_SCORE_BUCKET", summarize(accepted, ["score_bucket"], ret_cols)))
    if "pwin_bucket" in accepted.columns:
        outputs.append(("BY_PWIN_BUCKET", summarize(accepted, ["pwin_bucket"], ret_cols)))
    if "band" in accepted.columns:
        outputs.append(("BY_BAND", summarize(accepted, ["band"], ret_cols)))

    txt_path = Path(f"results/holding_horizon_{args.name}.txt")
    csv_base = Path(f"results/holding_horizon_{args.name}")

    with txt_path.open("w", encoding="utf-8") as f:
        for title, out in outputs:
            f.write(f"\n=== {title} ===\n")
            if out.empty:
                f.write("no rows\n")
            else:
                f.write(out.to_string(index=False))
                f.write("\n")

    for title, out in outputs:
        out.to_csv(csv_base.with_name(f"{csv_base.name}_{title.lower()}.csv"), index=False)

    print(f"saved: {txt_path}")
    for title, _ in outputs:
        print(f"saved: {csv_base.with_name(f'{csv_base.name}_{title.lower()}.csv')}")

    print("\n=== PREVIEW ===")
    for title, out in outputs:
        print(f"\n=== {title} ===")
        if out.empty:
            print("no rows")
        else:
            print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
