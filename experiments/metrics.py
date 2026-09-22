"""
실험 평가지표 (Phase 4).

정의는 experiments/preregistration.md §6에 고정되어 있으며, 이 모듈은 그 정의를
그대로 구현한다. 실행 후 정의를 바꾸지 않는다(R5).

  - ARI                 : 노이즈(-1)는 '하나의 노이즈 군집' 라벨로 유지한 채 계산
  - Pairwise P/R        : 동일 군집 쌍 기준, 노이즈 쌍 제외
  - Detection Delay     : 순도 0.8 최초 도달까지의 누적 고장 건수 (미도달 시 None)
  - Spearman ρ / P@k    : 주입 Cm 정답순위 vs 산출순위, 다수결 매칭(C5-1)
  - bootstrap_ci        : 시드 평균에 대한 부트스트랩 95% CI
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

NOISE_LABEL = -1


# ===========================================================================
# 군집 품질
# ===========================================================================

def ari(truth_labels, pred_labels) -> float:
    """
    Adjusted Rand Index.

    사전등록 고정: 노이즈(-1)를 개별 군집으로 분해하지 않고 **하나의 노이즈 군집
    라벨로 유지**한 채 계산한다(라벨을 그대로 전달).
    """
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(list(truth_labels), list(pred_labels)))


def pairwise_precision_recall(truth_labels, pred_labels) -> dict:
    """
    동일 군집 쌍 기준 Precision / Recall.

    노이즈 쌍 제외: 정답·예측 어느 쪽이든 -1인 레코드가 포함된 쌍은 계산에서 뺀다.
    """
    y = np.asarray(list(truth_labels), dtype=int)
    p = np.asarray(list(pred_labels), dtype=int)
    유효 = (y != NOISE_LABEL) & (p != NOISE_LABEL)
    y, p = y[유효], p[유효]

    if len(y) < 2:
        return {"precision": None, "recall": None, "f1": None}

    tp = fp = fn = 0
    for i, j in itertools.combinations(range(len(y)), 2):
        같음_정답 = y[i] == y[j]
        같음_예측 = p[i] == p[j]
        if 같음_예측 and 같음_정답:
            tp += 1
        elif 같음_예측 and not 같음_정답:
            fp += 1
        elif (not 같음_예측) and 같음_정답:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall else None
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def noise_ratio(pred_labels) -> float:
    """예측 라벨 중 노이즈(-1) 비율."""
    p = np.asarray(list(pred_labels), dtype=int)
    return float((p == NOISE_LABEL).mean()) if len(p) else 0.0


# ===========================================================================
# 탐지 지연
# ===========================================================================

def detection_delay(
    obs: pd.DataFrame,
    truth: pd.DataFrame,
    cluster_fn,
    purity_target: float = 0.8,
    n_steps: int = 10,
    time_col: str = "발생일시",
) -> dict:
    """
    Detection Delay — 주입 시점 t*_c 이후 슬라이딩 윈도우 재군집화에서
    해당 군집의 순도가 purity_target에 처음 도달할 때까지의 **누적 고장 건수**.

    구현: 데이터를 발생일시 순으로 정렬하고 n_steps 지점에서 누적 재군집화한다.
    각 정답 군집 c에 대해, 산출 군집 중 c를 가장 많이 포함한 군집의 순도가
    purity_target 이상이 되는 최초 시점의 누적 레코드 수를 기록한다.
    미도달 시 None (사전등록: NA로 기록하고 도달률 병기).

    Parameters
    ----------
    cluster_fn : callable(obs_subset) -> labels
        부분 데이터에 대해 라벨을 반환하는 함수(탐지 파이프라인 주입).

    Returns
    -------
    {"delays": {cluster_id: 누적건수 or None}, "reach_rate": float}
    """
    순서 = pd.to_datetime(obs[time_col], errors="coerce").sort_values().index
    obs_s = obs.loc[순서].reset_index(drop=True)
    truth_s = truth.loc[순서].reset_index(drop=True)

    정답군집 = sorted({int(c) for c in truth_s["cluster_id"] if c != NOISE_LABEL})
    delays: dict[int, int | None] = {c: None for c in 정답군집}

    n = len(obs_s)
    체크지점 = [int(n * (k + 1) / n_steps) for k in range(n_steps)]

    for cut in 체크지점:
        if cut < 5:
            continue
        sub_obs = obs_s.iloc[:cut]
        sub_truth = truth_s.iloc[:cut]
        try:
            labels = np.asarray(cluster_fn(sub_obs), dtype=int)
        except Exception:
            continue
        if len(labels) != cut:
            continue

        for c in 정답군집:
            if delays[c] is not None:
                continue
            c_mask = sub_truth["cluster_id"].to_numpy() == c
            if c_mask.sum() == 0:
                continue
            # c를 가장 많이 포함한 산출 군집의 순도
            최고순도 = 0.0
            for pl in set(labels[labels != NOISE_LABEL]):
                멤버 = labels == pl
                if 멤버.sum() == 0:
                    continue
                순도 = float((c_mask & 멤버).sum() / 멤버.sum())
                최고순도 = max(최고순도, 순도)
            if 최고순도 >= purity_target:
                delays[c] = int(cut)

    도달 = [v for v in delays.values() if v is not None]
    return {
        "delays": delays,
        "reach_rate": len(도달) / len(정답군집) if 정답군집 else 0.0,
    }


# ===========================================================================
# 순위 지표
# ===========================================================================

def match_clusters(pred_labels, truth_labels) -> dict[int, int]:
    """
    산출 군집 → 정답 군집 대응 (사전등록 C5-1: 다수결).
    각 산출 군집 내 최빈 정답 라벨로 대응하며, 노이즈는 제외한다.
    """
    p = np.asarray(list(pred_labels), dtype=int)
    y = np.asarray(list(truth_labels), dtype=int)

    매핑: dict[int, int] = {}
    for c in sorted(set(p[p != NOISE_LABEL])):
        후보 = y[(p == c) & (y != NOISE_LABEL)]
        if len(후보):
            매핑[int(c)] = int(pd.Series(후보).value_counts().idxmax())
    return 매핑


def ranking_scores(
    score_table: pd.DataFrame,
    truth_rank: pd.DataFrame,
    pred_labels,
    truth_labels,
    k: int = 5,
) -> dict:
    """
    Spearman ρ 와 Precision@k.

    score_table : prioritize.score_clusters() 결과 (군집·순위)
    truth_rank  : prioritize.truth_ranking() 결과 (군집·순위)
    """
    from scipy.stats import spearmanr

    if score_table.empty or truth_rank.empty:
        return {"spearman": None, "precision_at_k": None, "n_pairs": 0}

    매핑 = match_clusters(pred_labels, truth_labels)
    산출순위 = score_table.set_index("군집")["순위"]
    정답순위 = truth_rank.set_index("군집")["순위"]

    쌍 = [
        (float(산출순위[c]), float(정답순위[t]))
        for c, t in 매핑.items()
        if c in 산출순위.index and t in 정답순위.index
    ]
    if len(쌍) < 3:
        return {"spearman": None, "precision_at_k": None, "n_pairs": len(쌍)}

    rho = spearmanr([a for a, _ in 쌍], [b for _, b in 쌍]).statistic
    rho = None if rho is None or np.isnan(rho) else float(rho)

    # Precision@k: 산출 상위 k 군집이 대응하는 정답군집이 정답 상위 k에 드는 비율
    상위_산출 = list(score_table.head(k)["군집"])
    정답_상위 = set(truth_rank.head(k)["군집"])
    적중 = sum(
        1 for c in 상위_산출 if 매핑.get(int(c)) in 정답_상위
    )
    pak = 적중 / min(k, len(상위_산출)) if 상위_산출 else None

    return {"spearman": rho, "precision_at_k": pak, "n_pairs": len(쌍)}


# ===========================================================================
# 집계 / 부트스트랩
# ===========================================================================

def bootstrap_ci(
    values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict:
    """
    시드 평균에 대한 부트스트랩 신뢰구간 (사전등록 C5).
    난수는 seed 인자로 통제한다(원칙 #5).
    """
    v = np.asarray([x for x in values if x is not None and not pd.isna(x)], dtype=float)
    if len(v) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    if len(v) == 1:
        return {"mean": float(v[0]), "lo": float(v[0]), "hi": float(v[0]), "n": 1}

    rng = np.random.RandomState(seed)
    means = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)]
    return {
        "mean": float(v.mean()),
        "lo": float(np.percentile(means, 100 * alpha / 2)),
        "hi": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "n": int(len(v)),
    }


def summarize(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    """조건별 mean±std 요약 (R4: 단일 시드 수치는 보고하지 않는다)."""
    agg = {c: ["mean", "std", "count"] for c in value_cols if c in df.columns}
    if not agg:
        return pd.DataFrame()
    out = df.groupby(group_cols, as_index=False).agg(agg)
    out.columns = [
        c[0] if not c[1] else f"{c[0]}_{c[1]}" for c in out.columns.to_flat_index()
    ]
    return out
