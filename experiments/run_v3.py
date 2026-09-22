"""
생성기×가중치 상호작용 (H4) + 채널 조합별 Detection Delay.

    python -m experiments.run_v3 --part all

사전등록 v3에 고정된 설계:
  - H4: floor 생성기에서 가중치 탐색(200~207) → 평가(100~107)
        H4-a 세 채널 모두 유지 / H4-b 고정 가중치 대비 우위
  - 병행: 채널 조합 7종별 Detection Delay (기술 통계, 합격기준 없음)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from detect import channels, fusion
from detect.cluster import cluster
from experiments import metrics as M
from experiments.run import (
    DEFAULT_CONFIG,
    RESULTS_DIR,
    _build,
    _delay_cluster_fn,
    load_config,
)
from experiments.run_v2 import (
    EVAL_SEEDS,
    MIN_CLUSTER_SIZE,
    SEARCH_SEEDS,
    _paired_diff_ci,
    _weight_grid,
)

FLOOR_MODE = {"time_window_mode": "floor", "time_window_floor_days": None}


def _cfg_floor(cfg: dict) -> dict:
    c = dict(cfg)
    c["data"] = {**cfg["data"], **FLOOR_MODE}
    return c


def _ari_list(cfg, seeds, weights, embedder) -> list[float]:
    out = []
    for seed in seeds:
        res, ch = _build(cfg, 1.0, seed, embedder)
        labels = cluster(fusion.fuse(ch, weights), MIN_CLUSTER_SIZE)
        out.append(M.ari(res["truth"]["cluster_id"], labels))
    return out


# ===========================================================================
# H4 — floor 생성기 + 가중치 탐색
# ===========================================================================

def run_h4(cfg: dict, embedder) -> dict:
    print("[H4] floor 생성기 + 가중치 탐색")
    cfg_f = _cfg_floor(cfg)
    고정 = cfg["detect"]["default_weights"]

    # 탐색 (탐색 시드에서만)
    rows = []
    for w in _weight_grid():
        aris = _ari_list(cfg_f, SEARCH_SEEDS, w, embedder)
        rows.append({**w, "ARI_mean": float(np.mean(aris))})
    search = pd.DataFrame(rows).sort_values("ARI_mean", ascending=False)
    best = {k: float(search.iloc[0][k]) for k in ("sem", "lcn", "time")}

    print(f"  선택 가중치 = {best} (탐색 ARI {search.iloc[0]['ARI_mean']:.3f})")
    print("  탐색 상위 5:")
    print(search.head(5).round(3).to_string(index=False))

    # H4-a: 세 채널 모두 유지되는가
    h4a = bool(all(v > 0 for v in best.values()))
    print(f"  H4-a 세 채널 유지: {'충족' if h4a else '불충족'} "
          f"(v2 Arm A(legacy)에서는 (0, 0, 1)로 수렴했었음)")

    # H4-b: 평가 시드에서 고정 가중치 대비
    탐색성능 = _ari_list(cfg_f, EVAL_SEEDS, best, embedder)
    고정성능 = _ari_list(cfg_f, EVAL_SEEDS, 고정, embedder)
    ci = _paired_diff_ci(탐색성능, 고정성능)
    h4b = bool(np.mean(탐색성능) >= np.mean(고정성능) and ci["excludes_zero"])

    print(f"  탐색 {np.mean(탐색성능):.3f} vs 고정 {np.mean(고정성능):.3f} "
          f"| Δ CI [{ci['lo']:.3f}, {ci['hi']:.3f}] "
          f"→ H4-b {'충족' if h4b else '차이 없음' if not ci['excludes_zero'] else '불충족'}")

    return {
        "search": search, "best_weights": best,
        "searched": 탐색성능, "fixed": 고정성능, "diff_ci": ci,
        "H4a_pass": h4a, "H4b_pass": h4b,
    }


# ===========================================================================
# 채널 조합별 Detection Delay (기술 통계)
# ===========================================================================

def run_delay_by_channel(cfg: dict, embedder, mode: str = "legacy") -> pd.DataFrame:
    print(f"[병행] 채널 조합별 Detection Delay ({mode} 생성기)")
    c = _cfg_floor(cfg) if mode == "floor" else cfg
    조합 = cfg["ablation"]["channel_sets"]

    rows = []
    for name, w in 조합.items():
        for seed in EVAL_SEEDS:
            res, ch = _build(c, 1.0, seed, embedder)
            labels = cluster(fusion.fuse(ch, w), MIN_CLUSTER_SIZE)
            dd = M.detection_delay(
                res["obs"], res["truth"],
                _delay_cluster_fn(res, w, MIN_CLUSTER_SIZE, embedder),
                purity_target=0.8, n_steps=8,
            )
            도달 = [v for v in dd["delays"].values() if v is not None]
            rows.append({
                "channels": name, "seed": seed,
                "ARI": M.ari(res["truth"]["cluster_id"], labels),
                "reach_rate": dd["reach_rate"],
                "delay_mean": float(np.mean(도달)) if 도달 else None,
                "delay_min": int(min(도달)) if 도달 else None,
            })
        최근 = [r for r in rows if r["channels"] == name]
        print(f"  {name:>14}: ARI {np.mean([r['ARI'] for r in 최근]):.3f} | "
              f"도달률 {np.mean([r['reach_rate'] for r in 최근]):.3f} | "
              f"지연 {np.nanmean([r['delay_mean'] or np.nan for r in 최근]):.0f}건")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--part", default="all", choices=["h4", "delay", "all"])
    ap.add_argument("--delay-mode", default="legacy", choices=["legacy", "floor"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    embedder = channels.KoSBERTEmbedder(cfg["detect"]["model_name"])
    out = Path(RESULTS_DIR) / "v3"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    결과: dict = {}

    if args.part in ("h4", "all"):
        h = run_h4(cfg, embedder)
        h["search"].to_csv(out / "search.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"seed": EVAL_SEEDS, "searched": h["searched"],
                      "fixed": h["fixed"]}).to_csv(
            out / "eval.csv", index=False, encoding="utf-8-sig")
        결과["h4"] = {
            "best_weights": h["best_weights"],
            "searched_mean": float(np.mean(h["searched"])),
            "fixed_mean": float(np.mean(h["fixed"])),
            "diff_ci": h["diff_ci"],
            "H4a_pass": h["H4a_pass"], "H4b_pass": h["H4b_pass"],
        }

    if args.part in ("delay", "all"):
        d = run_delay_by_channel(cfg, embedder, args.delay_mode)
        d.to_csv(out / f"delay_by_channel_{args.delay_mode}.csv",
                 index=False, encoding="utf-8-sig")
        s = d.groupby("channels", as_index=False).agg(
            ARI_mean=("ARI", "mean"), reach_mean=("reach_rate", "mean"),
            delay_mean=("delay_mean", "mean"), delay_min=("delay_min", "min"),
        ).sort_values("ARI_mean", ascending=False)
        s.to_csv(out / f"delay_by_channel_{args.delay_mode}_summary.csv",
                 index=False, encoding="utf-8-sig")
        결과["delay_mode"] = args.delay_mode

    결과["elapsed_sec"] = round(time.time() - t0, 1)
    (out / "verdicts.json").write_text(
        json.dumps(결과, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"[완료] {결과['elapsed_sec']}초 · {out}")


if __name__ == "__main__":
    main()
