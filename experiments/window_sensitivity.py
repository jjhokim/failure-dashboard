"""
시간 밀집폭 민감도 분석 (후보작업 2).

    python -m experiments.window_sensitivity

## 왜 민감도 분석인가 — '실측 근거'가 불가능한 이유

1. 본 과제에는 **실제 무기체계 정비 데이터가 없다**(전부 가상/합성).
2. 문헌상으로도 공통원인 고장(CCF)의 시간 창은 **표준 고정값이 아니다**.
   NRC의 CCF 정의는 "PRA 임무 성공이 불확실해지는 **선택된(selected) 기간** 내에
   고장이 발생할 것"으로, 임무·점검주기에 따라 분석자가 정하는 값이다
   (NUREG/CR-6268, NUREG/CR-5485).

따라서 "올바른 밀집폭"을 확보하는 것은 원리적으로 불가능하며, 올바른 처리는
**결론이 이 가정에 얼마나 의존하는지를 구간 전체에서 특성화**하는 것이다.

## 성격 — 탐색적(exploratory)

이 분석은 **사전등록된 가설검정이 아니다.** 합격/불합격을 판정하지 않으며,
"융합이 단일채널을 앞서기 시작하는 밀집폭 임계"를 기술적으로 보고할 뿐이다.
논문에서는 주 결과(legacy)의 한계를 정량화하는 **민감도 분석**으로 인용한다.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from detect import channels, fusion
from detect.cluster import cluster
from experiments import metrics as M
from experiments.run import DEFAULT_CONFIG, RESULTS_DIR, _build, load_config
from experiments.run_v2 import MIN_CLUSTER_SIZE, SINGLE_CHANNELS, _paired_diff_ci

# 스윕할 밀집폭(일). 1일 = 1차 legacy와 동등, 30일 ≈ 월 단위 점검주기
FLOOR_DAYS = [1.0, 2.0, 3.65, 7.0, 14.0, 30.0]
EVAL_SEEDS = list(range(100, 108))


def _ari_at(cfg, floor_days, seeds, weights, embedder) -> list[float]:
    out = []
    for seed in seeds:
        c = dict(cfg)
        c["data"] = {**cfg["data"], "time_window_mode": "floor",
                     "time_window_floor_days": floor_days}
        res, ch = _build(c, 1.0, seed, embedder)
        labels = cluster(fusion.fuse(ch, weights), MIN_CLUSTER_SIZE)
        out.append(M.ari(res["truth"]["cluster_id"], labels))
    return out


def run() -> pd.DataFrame:
    cfg = load_config(DEFAULT_CONFIG)
    embedder = channels.KoSBERTEmbedder(cfg["detect"]["model_name"])
    w_fused = cfg["detect"]["default_weights"]

    t0 = time.time()
    rows = []
    for floor in FLOOR_DAYS:
        fused = _ari_at(cfg, floor, EVAL_SEEDS, w_fused, embedder)
        singles = {n: _ari_at(cfg, floor, EVAL_SEEDS, w, embedder)
                   for n, w in SINGLE_CHANNELS.items()}
        best_name = max(singles, key=lambda n: float(np.mean(singles[n])))
        ci = _paired_diff_ci(fused, singles[best_name])

        rows.append({
            "floor_days": floor,
            "fused_mean": float(np.mean(fused)),
            "fused_std": float(np.std(fused)),
            "time_mean": float(np.mean(singles["time"])),
            "sem_mean": float(np.mean(singles["sem"])),
            "lcn_mean": float(np.mean(singles["lcn"])),
            "best_single": best_name,
            "best_single_mean": float(np.mean(singles[best_name])),
            "diff_mean": ci["mean"], "diff_lo": ci["lo"], "diff_hi": ci["hi"],
            "fusion_wins": bool(ci["mean"] > 0 and ci["excludes_zero"]),
        })
        print(f"  floor={floor:>5.2f}일 | 융합 {rows[-1]['fused_mean']:.3f} "
              f"vs {best_name} {rows[-1]['best_single_mean']:.3f} "
              f"| Δ CI [{ci['lo']:.3f}, {ci['hi']:.3f}] "
              f"→ {'융합 우위' if rows[-1]['fusion_wins'] else '우위 아님'}")

    df = pd.DataFrame(rows)
    out = Path(RESULTS_DIR) / "window_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sweep.csv", index=False, encoding="utf-8-sig")

    이긴다 = df[df["fusion_wins"]]
    임계 = float(이긴다["floor_days"].min()) if not 이긴다.empty else None
    (out / "summary.json").write_text(json.dumps({
        "floor_days": FLOOR_DAYS, "eval_seeds": EVAL_SEEDS,
        "fusion_win_threshold_days": 임계,
        "elapsed_sec": round(time.time() - t0, 1),
        "note": "탐색적 민감도 분석. 사전등록된 가설검정이 아님.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[요약] 융합이 단일채널 최선을 앞서기 시작하는 밀집폭 = "
          f"{임계 if 임계 is not None else '구간 내 없음'}일")
    print(f"[완료] {time.time() - t0:.1f}초 · {out}")
    return df


if __name__ == "__main__":
    run()
