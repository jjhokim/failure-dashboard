"""
공통원인 후보 군집 우선순위화 (Phase 3).

군집 단위 점수:
    S_c = w_size·size_c + w_trend·slope_c + w_crit·crit_c      (각 항 표준화)

| 축     | 산출 |
|--------|------|
| 규모   | 군집 내 고장 건수 |
| 추세   | 최근 N주 단위기간 발생률의 선형 기울기 |
| 치명도 | Cm = β·α·λ·t 의 군집 내 합산 |

중요 제약: **탐지기는 주입된 β·α·λ를 알 수 없다.** 따라서 치명도는 관측치 기반
대체값으로만 산출한다.
    λ_obs = 군집 건수 / Σ운용시간
    β·α   ≈ effect_class(EFF) 비율
정답 순위(주입 Cm 기반)는 truth 파일에서 truth_ranking()으로 **별도** 산출하며,
산출 점수(score_clusters)에는 truth가 흘러들어가지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NOISE_LABEL = -1

# 가중치 초기값 (규모, 추세, 치명도)
DEFAULT_WEIGHTS: dict[str, float] = {"size": 0.4, "trend": 0.3, "crit": 0.3}


# ===========================================================================
# 표준화
# ===========================================================================

def standardize(values) -> np.ndarray:
    """
    z-점수 표준화. 표준편차가 0이면(모두 동일) 0 벡터를 반환한다.
    """
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return v
    sd = v.std()
    if sd <= 1e-12:
        return np.zeros_like(v)
    return (v - v.mean()) / sd


# ===========================================================================
# 축별 산출
# ===========================================================================

def cluster_trend(
    times, labels, period_days: float = 7.0, n_periods: int = 8
) -> dict[int, float]:
    """
    군집별 추세 = 해당 군집 **자체 활동기간** 내 단위기간 발생 건수의 선형 기울기.

    창 정의(사용자 확정): 전역 '최근 N주'가 아니라 각 군집의 첫 발생~마지막 발생
    구간을 창으로 삼는다. 전역 최근창을 쓰면 과거에 발생한 군집의 기울기가
    일괄 0이 되어 추세 축이 무력화되기 때문이다. 과거 군집도 '당시 가속되었는가'로
    평가된다.

    단위기간 길이는 period_days와 (활동기간 / n_periods) 중 큰 값을 쓴다.
    → 활동기간이 길면 n_periods개 구간으로 나누고, 짧으면 period_days를 유지한다.

    기울기 > 0 이면 활동기간 동안 발생률이 증가했음을 뜻한다.
    유효 구간이 2개 미만이면 0.0.
    """
    ts = pd.to_datetime(pd.Series(list(times)), errors="coerce").reset_index(drop=True)
    labels = np.asarray(list(labels), dtype=int)
    if ts.notna().sum() == 0:
        return {}

    out: dict[int, float] = {}
    for c in sorted(set(labels[labels != NOISE_LABEL])):
        c_ts = ts[labels == c].dropna()
        if len(c_ts) < 2:
            out[c] = 0.0
            continue

        활동일수 = (c_ts.max() - c_ts.min()).total_seconds() / 86400.0
        if 활동일수 <= 0:
            out[c] = 0.0
            continue

        폭 = max(float(period_days), 활동일수 / max(1, n_periods))
        days = (c_ts - c_ts.min()).dt.total_seconds() / 86400.0
        bins = np.floor(days / 폭).astype(int)

        counts = pd.Series(bins).value_counts().sort_index()
        full = pd.Series(0.0, index=range(int(bins.max()) + 1))
        full.loc[counts.index] = counts.values.astype(float)
        if len(full) < 2:
            out[c] = 0.0
            continue

        x = np.arange(len(full), dtype=float)
        out[c] = float(np.polyfit(x, full.to_numpy(dtype=float), 1)[0])
    return out


def cluster_criticality_observed(
    df: pd.DataFrame,
    labels,
    total_op_hours: float | None = None,
    effect_col: str = "effect_class",
    op_hours_col: str = "op_hours_at_failure",
    time_col: str = "발생일시",
) -> dict[int, float]:
    """
    관측치 기반 치명도 대체값 Cm_obs를 군집별로 합산한다.

    Cm_obs(record) = (β·α)_obs · λ_obs · t
      - λ_obs   = 군집 건수 / Σ운용시간   (운용시간 미등록 시 군집 건수 / 전체 건수)
      - (β·α)_obs = 군집 내 effect_class == 'EFF' 비율 (컬럼 부재 시 1.0)
      - t        = op_hours_at_failure, 없으면 데이터 시작으로부터의 경과일

    ※ 주입된 β·α·λ(truth)는 사용하지 않는다.
    """
    labels = np.asarray(list(labels), dtype=int)
    n = len(df)

    # 노출시간 t
    if op_hours_col in df.columns and df[op_hours_col].notna().any():
        t_all = pd.to_numeric(df[op_hours_col], errors="coerce").fillna(0.0).to_numpy()
    else:
        ts = pd.to_datetime(df[time_col], errors="coerce")
        t_all = ((ts - ts.min()).dt.total_seconds() / 86400.0).fillna(0.0).to_numpy()

    분모 = float(total_op_hours) if total_op_hours else float(max(n, 1))

    out: dict[int, float] = {}
    for c in sorted(set(labels[labels != NOISE_LABEL])):
        mask = labels == c
        size = int(mask.sum())
        lam_obs = size / 분모 if 분모 > 0 else 0.0

        if effect_col in df.columns and df.loc[mask, effect_col].notna().any():
            ba_obs = float((df.loc[mask, effect_col] == "EFF").mean())
        else:
            ba_obs = 1.0

        out[c] = float(np.sum(ba_obs * lam_obs * t_all[mask]))
    return out


# ===========================================================================
# 대표 정보
# ===========================================================================

def _대표_LCN(s: pd.Series) -> str | None:
    s = s.dropna()
    return None if s.empty else str(s.value_counts().idxmax())


def _대표_문장(s: pd.Series, k: int = 3) -> list[str]:
    """군집을 대표하는 문장 k개 (빈도 우선, 동률은 사전순으로 결정론적 선택)."""
    s = s.dropna().astype(str)
    if s.empty:
        return []
    vc = s.value_counts()
    순서 = sorted(vc.index, key=lambda t: (-vc[t], t))
    return list(순서[:k])


# ===========================================================================
# 종합 점수
# ===========================================================================

def score_clusters(
    df: pd.DataFrame,
    labels,
    weights: dict[str, float] | None = None,
    total_op_hours: float | None = None,
    period_days: float = 7.0,
    n_periods: int = 8,
    text_col: str = "고장증상",
    lcn_col: str = "lcn",
    time_col: str = "발생일시",
) -> pd.DataFrame:
    """
    군집별 우선순위 점수와 순위 테이블을 산출한다.

    Returns
    -------
    pd.DataFrame
        columns: 군집, 점수, 대표LCN, 대표문장1~3, 건수, 추세,
                 치명도_obs, z_size, z_trend, z_crit
        점수 내림차순 정렬, 순위 컬럼 포함.
    """
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    labels = np.asarray(list(labels), dtype=int)

    군집들 = sorted(set(labels[labels != NOISE_LABEL]))
    cols = ["순위", "군집", "점수", "건수", "추세", "치명도_obs",
            "대표LCN", "대표문장1", "대표문장2", "대표문장3",
            "z_size", "z_trend", "z_crit"]
    if not 군집들:
        return pd.DataFrame(columns=cols)

    sizes = {c: int((labels == c).sum()) for c in 군집들}
    trends = cluster_trend(df[time_col], labels, period_days, n_periods)
    crits = cluster_criticality_observed(
        df, labels, total_op_hours, time_col=time_col
    )

    z_size = standardize([sizes[c] for c in 군집들])
    z_trend = standardize([trends.get(c, 0.0) for c in 군집들])
    z_crit = standardize([crits.get(c, 0.0) for c in 군집들])

    점수 = (
        w.get("size", 0.0) * z_size
        + w.get("trend", 0.0) * z_trend
        + w.get("crit", 0.0) * z_crit
    )

    rows = []
    for i, c in enumerate(군집들):
        mask = labels == c
        sub = df[mask]
        문장 = _대표_문장(sub[text_col]) if text_col in df.columns else []
        문장 += [None] * (3 - len(문장))
        rows.append({
            "군집": int(c),
            "점수": float(점수[i]),
            "건수": sizes[c],
            "추세": float(trends.get(c, 0.0)),
            "치명도_obs": float(crits.get(c, 0.0)),
            "대표LCN": _대표_LCN(sub[lcn_col]) if lcn_col in df.columns else None,
            "대표문장1": 문장[0], "대표문장2": 문장[1], "대표문장3": 문장[2],
            "z_size": float(z_size[i]),
            "z_trend": float(z_trend[i]),
            "z_crit": float(z_crit[i]),
        })

    out = pd.DataFrame(rows).sort_values("점수", ascending=False).reset_index(drop=True)
    out.insert(0, "순위", np.arange(1, len(out) + 1))
    return out[cols]


# ===========================================================================
# 민감도 분석
# ===========================================================================

def sensitivity_analysis(
    df: pd.DataFrame,
    labels,
    weight_grid: list[dict[str, float]] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    가중치 조합을 바꿔가며 순위가 얼마나 흔들리는지 본다.

    Returns
    -------
    pd.DataFrame
        columns: w_size, w_trend, w_crit, 1위군집, 상위3군집, 기본대비_스피어만
        (기본 가중치 결과와의 순위 상관을 함께 보고)
    """
    from scipy.stats import spearmanr

    if weight_grid is None:
        weight_grid = [
            {"size": 0.4, "trend": 0.3, "crit": 0.3},   # 기본
            {"size": 1.0, "trend": 0.0, "crit": 0.0},
            {"size": 0.0, "trend": 1.0, "crit": 0.0},
            {"size": 0.0, "trend": 0.0, "crit": 1.0},
            {"size": 0.34, "trend": 0.33, "crit": 0.33},
            {"size": 0.6, "trend": 0.2, "crit": 0.2},
            {"size": 0.2, "trend": 0.6, "crit": 0.2},
            {"size": 0.2, "trend": 0.2, "crit": 0.6},
        ]

    기본 = score_clusters(df, labels, DEFAULT_WEIGHTS, **kwargs)
    기본순위 = 기본.set_index("군집")["순위"]

    rows = []
    for w in weight_grid:
        r = score_clusters(df, labels, w, **kwargs)
        if r.empty:
            continue
        순위 = r.set_index("군집")["순위"].reindex(기본순위.index)
        rho = (
            float(spearmanr(기본순위.to_numpy(), 순위.to_numpy()).statistic)
            if len(기본순위) > 1 else 1.0
        )
        rows.append({
            "w_size": w.get("size", 0.0),
            "w_trend": w.get("trend", 0.0),
            "w_crit": w.get("crit", 0.0),
            "1위군집": int(r.iloc[0]["군집"]),
            "상위3군집": list(r.head(3)["군집"].astype(int)),
            "기본대비_스피어만": rho,
        })
    return pd.DataFrame(rows)


# ===========================================================================
# 정답 순위 (truth 기반 — 평가 전용, 산출 점수와 분리)
# ===========================================================================

def truth_ranking(truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    정답 순위: truth 파일의 주입 Cm을 군집별로 합산해 내림차순 정렬한다.

    ※ 평가(Phase 4 Spearman ρ·Precision@k) 전용이다.
      score_clusters()에는 이 정보가 전달되지 않는다.
    """
    t = truth_df[truth_df["cluster_id"] != NOISE_LABEL]
    if t.empty:
        return pd.DataFrame(columns=["순위", "군집", "Cm_주입합"])

    agg = (
        t.groupby("cluster_id", as_index=False)["Cm"].sum()
        .rename(columns={"cluster_id": "군집", "Cm": "Cm_주입합"})
        .sort_values("Cm_주입합", ascending=False)
        .reset_index(drop=True)
    )
    agg.insert(0, "순위", np.arange(1, len(agg) + 1))
    return agg
