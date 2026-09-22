"""
실험 실행기 (Phase 4).

    python -m experiments.run --config experiments/configs/ablation.yaml --seeds 0-7
    python -m experiments.run --timing          # 1조합×1시드 소요시간만 측정

사전등록(experiments/preregistration.md)에 고정된 조건·기준을 그대로 실행한다.
모든 수치는 시드 집합에 대한 mean±std로 보고한다(R4).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from detect import channels, fusion
from detect.cluster import cluster
from experiments import metrics as M
from prioritize import scoring
from synth import generate
from synth.config import SynthConfig

_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _ROOT / "results"
DEFAULT_CONFIG = _ROOT / "experiments" / "configs" / "ablation.yaml"


# ===========================================================================
# 설정
# ===========================================================================

def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_seeds(spec: str) -> list[int]:
    """'0-7' 또는 '0,1,2' 형식을 시드 목록으로 변환."""
    spec = str(spec).strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def _synth_config(cfg: dict, delta: float, seed: int) -> SynthConfig:
    d = dict(cfg["data"])
    d.update(delta=float(delta), seed=int(seed))
    return SynthConfig(**d)


# ===========================================================================
# 단일 실행 단위
# ===========================================================================

def _build(cfg: dict, delta: float, seed: int, embedder):
    """합성 생성 + 3채널 거리행렬 (조건별로 1회만 계산해 재사용)."""
    res = generate(_synth_config(cfg, delta, seed), save=False)
    ch = channels.build_channels(
        res["obs"],
        embedder=embedder,
        tau_days=float(cfg["detect"]["tau_days"]),
        use_cache=bool(cfg["detect"].get("use_cache", True)),
    )
    return res, ch


def _delay_cluster_fn(res: dict, weights: dict, min_cluster_size: int, embedder):
    """
    Detection Delay용 부분 데이터 군집화 함수를 만든다.

    슬라이딩(누적) 윈도우마다 3채널 거리행렬을 다시 만들고 군집화한다.
    truth·라벨을 참조하지 않는다.
    """
    def _fn(sub_obs):
        ch = channels.build_channels(sub_obs, embedder=embedder, use_cache=False)
        return cluster(fusion.fuse(ch, weights), min_cluster_size)
    return _fn


def evaluate_once(
    res: dict,
    ch: dict,
    weights: dict,
    min_cluster_size: int,
    k: int = 5,
    permute_labels: bool = False,
    permute_seed: int = 0,
    embedder=None,
    with_delay: bool = False,
    delay_steps: int = 8,
) -> dict:
    """한 조건(가중치·m)에 대한 지표 일괄 산출."""
    obs, truth = res["obs"], res["truth"]
    y = truth["cluster_id"].to_numpy()

    if permute_labels:  # 음성 대조군 2: 정답 라벨 무작위 순열
        y = np.random.RandomState(permute_seed).permutation(y)

    labels = cluster(fusion.fuse(ch, weights), min_cluster_size)

    out: dict = {
        "ARI": M.ari(y, labels),
        "noise_ratio": M.noise_ratio(labels),
        "n_clusters_found": int(len({int(v) for v in labels if v != -1})),
    }
    out.update({f"pairwise_{k2}": v
                for k2, v in M.pairwise_precision_recall(y, labels).items()})

    # 순위 지표 (치명도는 관측치 대체값만 사용)
    표 = scoring.score_clusters(
        obs, labels, total_op_hours=res["counts"].get("total_op_hours")
    )
    정답 = scoring.truth_ranking(
        truth.assign(cluster_id=y) if permute_labels else truth
    )
    out.update(M.ranking_scores(표, 정답, labels, y, k=k))

    # 축 상관 진단 (C6)
    if len(표) >= 3:
        out["corr_size_crit"] = float(표["건수"].corr(표["치명도_obs"]))
        out["corr_size_trend"] = float(표["건수"].corr(표["추세"]))
        out["corr_trend_crit"] = float(표["추세"].corr(표["치명도_obs"]))

    # Detection Delay (사전등록 §6) — 순도 0.8 최초 도달까지의 누적 고장 건수
    if with_delay and embedder is not None and not permute_labels:
        dd = M.detection_delay(
            obs, truth,
            _delay_cluster_fn(res, weights, min_cluster_size, embedder),
            purity_target=0.8, n_steps=delay_steps,
        )
        도달 = [v for v in dd["delays"].values() if v is not None]
        out["delay_reach_rate"] = dd["reach_rate"]
        # 미도달은 NA로 두고 도달 건에 대해서만 평균(사전등록 §6)
        out["delay_mean_records"] = float(np.mean(도달)) if 도달 else None
        out["delay_min_records"] = int(min(도달)) if 도달 else None
    return out


# ===========================================================================
# min_cluster_size 사전 선택 규칙 (사전등록 §3-1)
# ===========================================================================

def select_min_cluster_size(cfg: dict, seeds: list[int], embedder) -> dict:
    """
    δ=0(무주입) 대조군에서 노이즈 비율 ≥ 임계를 만족하는 후보 중 최솟값을 선택한다.
    선택은 귀무조건만으로 이뤄지며 ARI를 보지 않는다(순환성 방지).
    """
    후보 = list(cfg["detect"]["min_cluster_sizes"])
    임계 = float(cfg["criteria"]["null_noise_ratio_min"])

    기록 = []
    for seed in seeds:
        res, ch = _build(cfg, 0.0, seed, embedder)
        D = fusion.fuse(ch, cfg["detect"]["default_weights"])
        for m in 후보:
            labels = cluster(D, m)
            기록.append({
                "min_cluster_size": m,
                "seed": seed,
                "noise_ratio": M.noise_ratio(labels),
                "ARI": M.ari(res["truth"]["cluster_id"], labels),
            })

    df = pd.DataFrame(기록)
    요약 = df.groupby("min_cluster_size", as_index=False)[["noise_ratio", "ARI"]].mean()
    충족 = 요약[요약["noise_ratio"] >= 임계]["min_cluster_size"]
    선택 = int(충족.min()) if len(충족) else int(요약["min_cluster_size"].max())

    return {"selected": 선택, "table": 요약, "raw": df,
            "criterion_met": bool(len(충족))}


# ===========================================================================
# 전체 실행
# ===========================================================================

def run(
    config_path: str | Path = DEFAULT_CONFIG,
    seeds: str = "0-7",
    run_id: str | None = None,
    out_dir: str | Path = RESULTS_DIR,
) -> dict:
    cfg = load_config(config_path)
    seed_list = parse_seeds(seeds)
    embedder = channels.KoSBERTEmbedder(cfg["detect"]["model_name"])
    k = int(cfg["criteria"].get("precision_at_k", 5))

    t0 = time.time()

    # 1) m 사전 선택 (δ=0 조건만 사용)
    sel = select_min_cluster_size(cfg, seed_list, embedder)
    m = sel["selected"]
    print(f"[선택] min_cluster_size = {m} (δ=0 노이즈기준 충족={sel['criterion_met']})")
    print(sel["table"].to_string(index=False))

    rows: list[dict] = []

    # 2) δ 스윕 (기본 가중치) — 단조성 C3, 무주입 C1
    #    Detection Delay는 누적 재군집화라 비용이 크므로 이 조건에서만 산출한다.
    with_delay = bool(cfg.get("ablation", {}).get("detection_delay", True))
    for delta in cfg["ablation"]["deltas"]:
        for seed in seed_list:
            res, ch = _build(cfg, delta, seed, embedder)
            r = evaluate_once(
                res, ch, cfg["detect"]["default_weights"], m, k=k,
                embedder=embedder,
                with_delay=with_delay and delta > 0,  # δ=0은 정답 군집이 없음
            )
            rows.append({"조건": "delta_sweep", "delta": delta, "seed": seed,
                         "channels": "sem+lcn+time", **r})

    # 3) 채널 ablation (δ=1.0) — C4
    for name, w in cfg["ablation"]["channel_sets"].items():
        for seed in seed_list:
            res, ch = _build(cfg, 1.0, seed, embedder)
            r = evaluate_once(res, ch, w, m, k=k)
            rows.append({"조건": "channel_ablation", "delta": 1.0, "seed": seed,
                         "channels": name, **r})

    # 4) 음성 대조군: 라벨 무작위 순열 (δ=1.0) — C2
    for seed in seed_list:
        res, ch = _build(cfg, 1.0, seed, embedder)
        r = evaluate_once(res, ch, cfg["detect"]["default_weights"], m, k=k,
                          permute_labels=True, permute_seed=seed)
        rows.append({"조건": "label_permutation", "delta": 1.0, "seed": seed,
                     "channels": "sem+lcn+time", **r})

    # 5) 베이스라인 (δ=1.0)
    from detect import baselines
    for seed in seed_list:
        res, _ = _build(cfg, 1.0, seed, embedder)
        obs, y = res["obs"], res["truth"]["cluster_id"]
        for bname, labels in (
            ("baseline_tfidf", baselines.baseline_tfidf(list(obs["고장증상"]), m)),
            ("baseline_rules", baselines.baseline_rules(list(obs["고장증상"]), m)),
        ):
            rows.append({
                "조건": "baseline", "delta": 1.0, "seed": seed, "channels": bname,
                "ARI": M.ari(y, labels), "noise_ratio": M.noise_ratio(labels),
                "n_clusters_found": int(len({int(v) for v in labels if v != -1})),
            })

    raw = pd.DataFrame(rows)
    value_cols = [c for c in raw.columns
                  if c not in ("조건", "delta", "seed", "channels")]
    summary = M.summarize(raw, ["조건", "channels", "delta"], value_cols)

    소요 = time.time() - t0

    # 저장
    run_id = run_id or f"run_{len(seed_list)}seeds_m{m}"
    out = Path(out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "raw.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    sel["table"].to_csv(out / "min_cluster_size_selection.csv",
                        index=False, encoding="utf-8-sig")
    (out / "config.json").write_text(
        json.dumps({
            "config": cfg, "seeds": seed_list,
            "selected_min_cluster_size": m,
            "elapsed_sec": round(소요, 1),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[완료] {소요:.1f}초 · 결과 {out}")
    return {"raw": raw, "summary": summary, "selection": sel,
            "elapsed_sec":소요, "out_dir": str(out), "min_cluster_size": m}


def measure_timing(config_path: str | Path = DEFAULT_CONFIG, seed: int = 0) -> float:
    """사전등록 §7: 전체 실행 전 1조합×1시드 소요시간 측정."""
    cfg = load_config(config_path)
    embedder = channels.KoSBERTEmbedder(cfg["detect"]["model_name"])

    t0 = time.time()
    res, ch = _build(cfg, 1.0, seed, embedder)
    evaluate_once(res, ch, cfg["detect"]["default_weights"],
                  cfg["detect"]["min_cluster_sizes"][0])
    소요 = time.time() - t0
    print(f"[측정] 1조합×1시드 = {소요:.1f}초")
    return 소요


def main() -> None:
    ap = argparse.ArgumentParser(description="공통원인 탐지 실험 실행기")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--timing", action="store_true",
                    help="1조합×1시드 소요시간만 측정")
    args = ap.parse_args()

    if args.timing:
        measure_timing(args.config)
    else:
        run(args.config, args.seeds, args.run_id)


if __name__ == "__main__":
    main()
