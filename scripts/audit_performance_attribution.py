from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hf.data.ohlcv import fetch_ohlcv_ccxt, dt_to_ms_utc


def _safe_qcut(series: pd.Series, q: int, prefix: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.nunique() < 2:
        return pd.Series([f"{prefix}_single"] * len(series), index=series.index)
    try:
        return pd.qcut(
            s.rank(method="first"),
            q=q,
            labels=[f"{prefix}_{i+1}" for i in range(q)],
        )
    except Exception:
        return pd.Series([f"{prefix}_fallback"] * len(series), index=series.index)


def _side_sign(side: str) -> float:
    s = str(side or "").lower()
    if s == "long":
        return 1.0
    if s == "short":
        return -1.0
    return 0.0


def load_registry_symbols(path: str) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    seen = set()
    for r in list(data or []):
        sym = str((r or {}).get("symbol", "") or "")
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def load_symbol_close_map(symbol: str, start: str, exchange: str, cache_dir: str) -> pd.Series:
    df = fetch_ohlcv_ccxt(
        symbol=symbol,
        timeframe="1h",
        start_ms=dt_to_ms_utc(start),
        end_ms=None,
        exchange_id=exchange,
        cache_dir=cache_dir,
        use_cache=True,
        refresh_if_no_end=True,
    ).copy()

    df["ts"] = pd.to_datetime(
        pd.to_numeric(df["timestamp"], errors="coerce"),
        unit="ms",
        utc=True,
        errors="coerce",
    )
    df = df.dropna(subset=["ts"]).sort_values("ts").copy()
    df = df[df["ts"] >= pd.Timestamp(start, tz="UTC")]
    s = pd.to_numeric(df["close"], errors="coerce")
    s.index = df["ts"]
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--strategy-registry", default="artifacts/strategy_registry.json")
    ap.add_argument("--start", default="2025-09-07 00:00:00")
    ap.add_argument("--exchange", default="binanceusdm")
    ap.add_argument("--cache-dir", default="data/cache")
    args = ap.parse_args()

    base = Path("results")
    cand_path = base / f"research_runtime_candidates_{args.name}.csv"
    metrics_path = base / f"research_runtime_metrics_{args.name}.json"

    if not cand_path.exists():
        raise SystemExit(f"Missing candidate file: {cand_path}")

    cands = pd.read_csv(cand_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    cands["ts"] = pd.to_datetime(cands["ts"], utc=True, errors="coerce")
    cands = cands.dropna(subset=["ts"]).sort_values(["ts"]).copy()

    symbols = load_registry_symbols(args.strategy_registry)
    close_by_symbol = {}
    print("=== LOADING SYMBOL CLOSE SERIES ===")
    for sym in symbols:
        try:
            s = load_symbol_close_map(sym, args.start, args.exchange, args.cache_dir)
            close_by_symbol[sym] = s
            print(sym, "rows:", len(s), "min:", s.index.min(), "max:", s.index.max())
        except Exception as e:
            print("FAILED", sym, repr(e))

    rows = []
    unmatched = set()

    for _, row in cands.iterrows():
        symbol = str(row.get("symbol", "") or "")
        if symbol not in close_by_symbol:
            unmatched.add(symbol)
            continue

        close_s = close_by_symbol[symbol]
        ts = row["ts"]
        if ts not in close_s.index:
            continue

        loc = close_s.index.get_loc(ts)
        if hasattr(loc, "__len__") and not isinstance(loc, int):
            loc = loc[-1]
        i = int(loc)

        if i + 1 >= len(close_s):
            continue

        px0 = float(close_s.iloc[i])
        if not np.isfinite(px0) or px0 == 0.0:
            continue

        side_sign = _side_sign(row.get("side", ""))

        horizon_returns = {}
        available_any = False
        for h in [1, 4, 12, 24, 48]:
            col = f"ret_{h}h"
            if i + h >= len(close_s):
                horizon_returns[col] = np.nan
                continue
            px1 = float(close_s.iloc[i + h])
            if not np.isfinite(px1):
                horizon_returns[col] = np.nan
                continue
            raw_ret = (px1 / px0) - 1.0
            horizon_returns[col] = float(side_sign * raw_ret)
            available_any = True

        if not available_any:
            continue

        best_ret = np.nan
        worst_ret = np.nan
        if i + 24 < len(close_s):
            future = pd.to_numeric(close_s.iloc[i + 1:i + 25], errors="coerce").dropna()
            if len(future) > 0:
                rets = side_sign * ((future / px0) - 1.0)
                best_ret = float(rets.max())
                worst_ret = float(rets.min())

        rows.append({
            "ts": ts,
            "symbol": symbol,
            "strategy_id": row.get("strategy_id"),
            "side": row.get("side"),
            "signal_strength": pd.to_numeric(row.get("signal_strength"), errors="coerce"),
            "base_weight": pd.to_numeric(row.get("base_weight"), errors="coerce"),
            "p_win": pd.to_numeric(row.get("p_win"), errors="coerce"),
            "expected_return": pd.to_numeric(row.get("expected_return"), errors="coerce"),
            "score": pd.to_numeric(row.get("score"), errors="coerce"),
            "size_mult": pd.to_numeric(row.get("size_mult"), errors="coerce"),
            "accept": bool(row.get("accept", False)),
            "band": row.get("band"),
            "reason": row.get("reason"),
            "ctx_backdrop": row.get("ctx_backdrop"),
            "ctx_side_backdrop_alignment": pd.to_numeric(row.get("ctx_side_backdrop_alignment"), errors="coerce"),
            "ctx_expected_holding_bars": pd.to_numeric(row.get("ctx_expected_holding_bars"), errors="coerce"),
            "ctx_exit_profile": row.get("ctx_exit_profile"),
            "adx_below_min": row.get("adx_below_min"),
            "ema_gap_below_min": row.get("ema_gap_below_min"),
            "atrp_low": row.get("atrp_low"),
            "adx_low": row.get("adx_low"),
            "range_expansion_low": row.get("range_expansion_low"),
            "mfe_24h": best_ret,
            "mae_24h": worst_ret,
            **horizon_returns,
        })

    if unmatched:
        print("\n=== UNMATCHED SYMBOLS ===")
        for s in sorted(unmatched):
            print(s)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No attribution rows generated after OHLCV alignment")

    df["pwin_bucket"] = _safe_qcut(df["p_win"], 5, "pwin")
    df["score_bucket"] = _safe_qcut(df["score"], 5, "score")
    df["size_bucket"] = _safe_qcut(df["size_mult"], 5, "size")
    df["alignment_bucket"] = _safe_qcut(df["ctx_side_backdrop_alignment"], 5, "align")

    ret_main = "ret_24h" if df["ret_24h"].notna().any() else (
        "ret_12h" if df["ret_12h"].notna().any() else (
            "ret_4h" if df["ret_4h"].notna().any() else "ret_1h"
        )
    )

    out_csv = base / f"performance_attribution_{args.name}.csv"
    df.to_csv(out_csv, index=False)

    summary_lines = []
    summary_lines.append("=== GLOBAL ===")
    summary_lines.append(f"rows: {len(df)}")
    summary_lines.append(f"accept_rate: {float(df['accept'].mean())}")
    summary_lines.append(f"main_return_column: {ret_main}")
    for c in ["ret_1h", "ret_4h", "ret_12h", "ret_24h", "ret_48h", "mfe_24h", "mae_24h"]:
        if c in df.columns and df[c].notna().any():
            summary_lines.append(f"{c}_mean: {float(pd.to_numeric(df[c], errors='coerce').mean())}")

    def add_group(title: str, group_cols: list[str], ret_col: str):
        g = (
            df.groupby(group_cols, dropna=False)
            .agg(
                rows=("strategy_id", "size"),
                accept_rate=("accept", "mean"),
                avg_pwin=("p_win", "mean"),
                avg_score=("score", "mean"),
                avg_size=("size_mult", "mean"),
                avg_ret=(ret_col, "mean"),
                med_ret=(ret_col, "median"),
                avg_mfe_24h=("mfe_24h", "mean"),
                avg_mae_24h=("mae_24h", "mean"),
                hit_rate=(ret_col, lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            )
            .sort_values(["avg_ret", "hit_rate"], ascending=False)
        )
        summary_lines.append("")
        summary_lines.append(f"=== {title} ({ret_col}) ===")
        summary_lines.append(g.to_string())

    add_group("BY STRATEGY", ["strategy_id"], ret_main)
    add_group("BY SIDE", ["side"], ret_main)
    add_group("BY STRATEGY+SIDE", ["strategy_id", "side"], ret_main)
    add_group("BY BAND", ["band"], ret_main)
    add_group("BY EXIT PROFILE", ["ctx_exit_profile"], ret_main)
    add_group("BY BACKDROP", ["ctx_backdrop"], ret_main)
    add_group("BY PWIN BUCKET", ["pwin_bucket"], ret_main)
    add_group("BY SCORE BUCKET", ["score_bucket"], ret_main)
    add_group("BY SIZE BUCKET", ["size_bucket"], ret_main)
    add_group("BY ALIGNMENT BUCKET", ["alignment_bucket"], ret_main)

    acc = df[df["accept"] == True].copy()
    if not acc.empty:
        def add_acc_group(title: str, cols: list[str], ret_col: str):
            g = (
                acc.groupby(cols, dropna=False)
                .agg(
                    rows=("strategy_id", "size"),
                    avg_pwin=("p_win", "mean"),
                    avg_score=("score", "mean"),
                    avg_size=("size_mult", "mean"),
                    avg_ret=(ret_col, "mean"),
                    med_ret=(ret_col, "median"),
                    avg_mfe_24h=("mfe_24h", "mean"),
                    avg_mae_24h=("mae_24h", "mean"),
                    hit_rate=(ret_col, lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
                )
                .sort_values(["avg_ret", "hit_rate"], ascending=False)
            )
            summary_lines.append("")
            summary_lines.append(f"=== ACCEPTED ONLY - {title} ({ret_col}) ===")
            summary_lines.append(g.to_string())

        add_acc_group("BY STRATEGY", ["strategy_id"], ret_main)
        add_acc_group("BY SIDE", ["side"], ret_main)
        add_acc_group("BY STRATEGY+SIDE", ["strategy_id", "side"], ret_main)
        add_acc_group("BY EXIT PROFILE", ["ctx_exit_profile"], ret_main)
        add_acc_group("BY BACKDROP", ["ctx_backdrop"], ret_main)
        add_acc_group("BY PWIN BUCKET", ["pwin_bucket"], ret_main)
        add_acc_group("BY SCORE BUCKET", ["score_bucket"], ret_main)
        add_acc_group("BY ALIGNMENT BUCKET", ["alignment_bucket"], ret_main)

    weak = acc[acc["score_bucket"].astype(str).isin(["score_1", "score_2"])]
    strong = acc[acc["score_bucket"].astype(str).isin(["score_4", "score_5"])]
    summary_lines.append("")
    summary_lines.append("=== WEAK VS STRONG ===")
    if not weak.empty:
        summary_lines.append(
            f"weak_rows={len(weak)} weak_{ret_main}_mean={float(pd.to_numeric(weak[ret_main], errors='coerce').mean())} weak_hit_rate={float((pd.to_numeric(weak[ret_main], errors='coerce') > 0).mean())}"
        )
    if not strong.empty:
        summary_lines.append(
            f"strong_rows={len(strong)} strong_{ret_main}_mean={float(pd.to_numeric(strong[ret_main], errors='coerce').mean())} strong_hit_rate={float((pd.to_numeric(strong[ret_main], errors='coerce') > 0).mean())}"
        )

    out_txt = base / f"performance_attribution_{args.name}.txt"
    out_txt.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\nsaved: {out_csv}")
    print(f"saved: {out_txt}")
    print("\n".join(summary_lines[:60]))

    if metrics:
        print("\n=== RUNTIME METRICS ===")
        for k in [
            "total_return_pct",
            "sharpe_annual",
            "max_drawdown_pct",
            "win_rate_pct",
            "avg_n_opps",
            "avg_n_accepts",
            "avg_active_symbols",
            "avg_gross_weight",
            "total_seconds",
        ]:
            print(f"{k}: {metrics.get(k)}")


if __name__ == "__main__":
    main()
