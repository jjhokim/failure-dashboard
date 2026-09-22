"""
C4·C6 후속 재실험 실행기 (사전등록 v2).

    python -m experiments.run_v2 --arm all

사전등록 v2에 고정된 설계:
  - Arm A: 생성기 유지 + 가중치 격자 탐색(탐색 시드 200~207) → 평가(100~107)
  - Arm B: 시각 밀집폭 floor 모드 + 고정 가중치 → 평가(100~107)
  - C6   : 치명도 3변형(S0/S1/S2) 페어링 비교

판정은 모두 **동일 시드 페어링 차이의 부트스트랩 95% CI**로 한다.
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
from experiments.run import RESULTS_DIR, _build, load_config, DEFAULT_CONFIG
from prioritize import scoring

EVAL_SEEDS = list(range(100, 108))    # 평가 전용
SEARCH_SEEDS = list(range(200, 208))  # 가중치 탐색 전용
MIN_CLUSTER_SIZE = 12                 # 1차 선택 규칙 결과를 그대로 사용
SINGLE_CHANNELS = {
    "sem": {"sem": 1.0, "lcn": 0.0, "time": 0.0},
    "lcn": {"sem": 0.0, "lcn": 1.0, "time": 0.0},
    "time": {"sem": 0.0, "lcn": 0.0, "time": 1.0},
}


def _weight_grid(step: float = 0.25) -> list[dict]:
    """합이 1인 (sem, lcn, time) 가중치 격자."""
    vals = np.arange(0, 1 + 1e-9, step)
    out = []
    for s in vals:
        for l in vals:
            t = 1.0 - s - l
            if -1e-9 <= t <= 1 + 1e-9:
                out.append({"sem": round(float(s), 3),
                            "lcn": round(float(l), 3),
                            "time": round(max(0.0, t), 3)})
    return out


def _paired_diff_ci(a: list[float], b: list[float], seed: int = 0) -> dict:
    """동일 시드 페어링 차이 (a-b)의 부트스트랩 95% CI."""
    pairs = [(x, y) for x, y in zip(a, b)
             if x is not None and y is not None
             and not pd.isna(x) and not pd.isna(y)]
    diffs = [x - y for x, y in pairs]
    ci = M.bootstrap_ci(diffs, seed=seed)
    ci["excludes_zero"] = bool(
        ci["lo"] is not None and (ci["lo"] > 0 or ci["hi"] < 0)
    )
    return ci


def _ari_for(cfg, delta, seeds, weights, embedder, mode="legacy") -> list[float]:
    """조건별 시드 ARI 목록."""
    out = []
    for seed in seeds:
        c = dict(cfg)
        c["data"] = {**cfg["data"], "time_window_mode": mode}
        res, ch = _build(c, delta, seed, embedder)
        labels = cluster(fusion.fuse(ch, weights), MIN_CLUSTER_SIZE)
        out.append(M.ari(res["truth"]["cluster_id"], labels))
    return out


# ===========================================================================
# Arm A — 가중치 탐색
# ===========================================================================

def arm_a(cfg, embedder) -> dict:
    print("[Arm A] 가중치 탐색 (탐색 시드 200~207)")
    grid = _weight_grid()

    # 탐색 시드에서만 최적 가중치 선택
    search_rows = []
    for w in grid:
        aris = _ari_for(cfg, 1.0, SEARCH_SEEDS, w, embedder)
        search_rows.append({**w, "ARI_mean": float(np.mean(aris))})
    search = pd.DataFrame(search_rows).sort_values("ARI_mean", ascending=False)
    best = {k: float(search.iloc[0][k]) for k in ("sem", "lcn", "time")}
    print(f"  선택 가중치 = {best} (탐색 ARI {search.iloc[0]['ARI_mean']:.3f})")

    # 평가 시드에서 융합 vs 단일채널 최선
    fused = _ari_for(cfg, 1.0, EVAL_SEEDS, best, embedder)
    singles = {n: _ari_for(cfg, 1.0, EVAL_SEEDS, w, embedder)
               for n, w in SINGLE_CHANNELS.items()}
    best_single_name = max(singles, key=lambda n: float(np.mean(singles[n])))
    best_single = singles[best_single_name]

    ci = _paired_diff_ci(fused, best_single)
    verdict = bool(np.mean(fused) >= np.mean(best_single) and ci["excludes_zero"])

    print(f"  융합 {np.mean(fused):.3f} vs 단일최선({best_single_name}) "
          f"{np.mean(best_single):.3f} | 차이 CI [{ci['lo']:.3f}, {ci['hi']:.3f}] "
          f"→ A1 {'충족' if verdict else '불충족'}")

    return {
        "search_table": search, "best_weights": best,
        "fused": fused, "singles": singles,
        "best_single_name": best_single_name, "diff_ci": ci, "A1_pass": verdict,
    }


# ===========================================================================
# Arm B — 생성기 시각 밀집폭 수정
# ===========================================================================

def arm_b(cfg, embedder) -> dict:
    print("[Arm B] 시각 밀집폭 floor 모드")
    w_fixed = cfg["detect"]["default_weights"]

    fused = _ari_for(cfg, 1.0, EVAL_SEEDS, w_fixed, embedder, mode="floor")
    singles = {n: _ari_for(cfg, 1.0, EVAL_SEEDS, w, embedder, mode="floor")
               for n, w in SINGLE_CHANNELS.items()}
    best_single_name = max(singles, key=lambda n: float(np.mean(singles[n])))
    best_single = singles[best_single_name]

    ci = _paired_diff_ci(fused, best_single)
    verdict = bool(np.mean(fused) >= np.mean(best_single) and ci["excludes_zero"])

    # B2: δ 스윕 건전성 재보고 (C1·C3)
    sweep = []
    for delta in cfg["ablation"]["deltas"]:
        aris = _ari_for(cfg, delta, EVAL_SEEDS, w_fixed, embedder, mode="floor")
        sweep.append({"delta": delta, "ARI_mean": float(np.mean(aris)),
                      "ARI_std": float(np.std(aris))})

    print(f"  융합 {np.mean(fused):.3f} vs 단일최선({best_single_name}) "
          f"{np.mean(best_single):.3f} | 차이 CI [{ci['lo']:.3f}, {ci['hi']:.3f}] "
          f"→ B1 {'충족' if verdict else '불충족'}")
    print(f"  B2: 시간단독 ARI = {np.mean(singles['time']):.3f} (1차 legacy 0.853)")

    return {"fused": fused, "singles": singles, "best_single_name": best_single_name,
            "diff_ci": ci, "B1_pass": verdict, "sweep": pd.DataFrame(sweep)}


# ===========================================================================
# C6 — 치명도 3변형
# ===========================================================================

def c6_variants(cfg, embedder, weights) -> dict:
    print("[C6] 치명도 3변형 비교 (S0/S1/S2)")
    rows = []
    for seed in EVAL_SEEDS:
        res, ch = _build(cfg, 1.0, seed, embedder)
        obs, truth = res["obs"], res["truth"]
        y = truth["cluster_id"].to_numpy()
        labels = cluster(fusion.fuse(ch, weights), MIN_CLUSTER_SIZE)

        base = scoring.score_clusters(
            obs, labels, total_op_hours=res["counts"].get("total_op_hours")
        )
        정답 = scoring.truth_ranking(truth)
        for variant in ("S0", "S1", "S2"):
            t = scoring.rescore_with_variant(base, variant)
            r = M.ranking_scores(t, 정답, labels, y, k=5)
            corr = (float(t["건수"].corr(t["치명도_obs"]))
                    if len(t) >= 3 else None)
            rows.append({"seed": seed, "variant": variant,
                         "spearman": r["spearman"],
                         "precision_at_k": r["precision_at_k"],
                         "corr_size_crit": corr})

    raw = pd.DataFrame(rows)
    piv = raw.pivot(index="seed", columns="variant", values="spearman")

    비교 = {}
    for v in ("S1", "S2"):
        ci = _paired_diff_ci(list(piv[v]), list(piv["S0"]))
        개선 = bool(np.nanmean(piv[v]) > np.nanmean(piv["S0"]))
        비교[v] = {"diff_ci": ci, "improved": 개선,
                   "pass": bool(개선 and ci["excludes_zero"])}
        print(f"  {v}-S0: Δρ mean={ci['mean']:.3f} CI[{ci['lo']:.3f},{ci['hi']:.3f}] "
              f"→ {'채택후보' if 비교[v]['pass'] else '기준 미충족'}")

    summary = raw.groupby("variant", as_index=False).agg(
        spearman_mean=("spearman", "mean"), spearman_std=("spearman", "std"),
        pak_mean=("precision_at_k", "mean"),
        corr_size_crit_mean=("corr_size_crit", "mean"),
    )
    print(summary.round(3).to_string(index=False))
    return {"raw": raw, "summary": summary, "comparison": 비교}


# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--arm", default="all", choices=["A", "B", "c6", "all"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    embedder = channels.KoSBERTEmbedder(cfg["detect"]["model_name"])
    out = Path(RESULTS_DIR) / "v2"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    결과: dict = {}

    if args.arm in ("A", "all"):
        a = arm_a(cfg, embedder)
        a["search_table"].to_csv(out / "armA_search.csv", index=False,
                                 encoding="utf-8-sig")
        pd.DataFrame({"seed": EVAL_SEEDS, "fused": a["fused"],
                      **{f"single_{k}": v for k, v in a["singles"].items()}}
                     ).to_csv(out / "armA_eval.csv", index=False,
                              encoding="utf-8-sig")
        결과["armA"] = {"best_weights": a["best_weights"],
                       "best_single": a["best_single_name"],
                       "fused_mean": float(np.mean(a["fused"])),
                       "best_single_mean": float(np.mean(a["singles"][a["best_single_name"]])),
                       "diff_ci": a["diff_ci"], "A1_pass": a["A1_pass"]}

    if args.arm in ("B", "all"):
        b = arm_b(cfg, embedder)
        b["sweep"].to_csv(out / "armB_sweep.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"seed": EVAL_SEEDS, "fused": b["fused"],
                      **{f"single_{k}": v for k, v in b["singles"].items()}}
                     ).to_csv(out / "armB_eval.csv", index=False,
                              encoding="utf-8-sig")
        결과["armB"] = {"best_single": b["best_single_name"],
                       "fused_mean": float(np.mean(b["fused"])),
                       "best_single_mean": float(np.mean(b["singles"][b["best_single_name"]])),
                       "time_only_mean": float(np.mean(b["singles"]["time"])),
                       "diff_ci": b["diff_ci"], "B1_pass": b["B1_pass"]}

    if args.arm in ("c6", "all"):
        w = 결과.get("armA", {}).get("best_weights") or cfg["detect"]["default_weights"]
        c = c6_variants(cfg, embedder, w)
        c["raw"].to_csv(out / "c6_raw.csv", index=False, encoding="utf-8-sig")
        c["summary"].to_csv(out / "c6_summary.csv", index=False, encoding="utf-8-sig")
        결과["c6"] = {k: v for k, v in c["comparison"].items()}

    결과["elapsed_sec"] = round(time.time() - t0, 1)
    (out / "verdicts.json").write_text(
        json.dumps(결과, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"[완료] {결과['elapsed_sec']}초 · {out}")


if __name__ == "__main__":
    main()
