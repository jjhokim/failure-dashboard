"""
Phase 3 우선순위화 검증.

  - standardize / cluster_trend / cluster_criticality_observed 축별 산출
  - score_clusters 순위 테이블 (가중치 반영, 대표 LCN·문장)
  - sensitivity_analysis 가중치 민감도
  - truth_ranking 은 평가 전용이며 점수 산출과 분리(β·α·λ 미사용)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prioritize import scoring  # noqa: E402


def _df(n_a: int = 10, n_b: int = 5) -> tuple[pd.DataFrame, np.ndarray]:
    """군집 0(대형·최근 증가) / 군집 1(소형) 고정 입력."""
    rows, labels = [], []
    base = pd.Timestamp("2025-01-01")
    # 군집 0: 뒤로 갈수록 촘촘 (증가 추세)
    for i in range(n_a):
        rows.append({
            "발생일시": base + pd.Timedelta(days=int(40 - i * 40 / n_a)),
            "고장증상": "구동모터 과열 발생", "lcn": "A.01.01.01",
            "effect_class": "EFF",
        })
        labels.append(0)
    for i in range(n_b):
        rows.append({
            "발생일시": base + pd.Timedelta(days=i * 6),
            "고장증상": "통신모듈 신호 두절", "lcn": "B.02.02.02",
            "effect_class": "NEFF",
        })
        labels.append(1)
    return pd.DataFrame(rows), np.array(labels)


# ---------------------------------------------------------------------------
# 표준화
# ---------------------------------------------------------------------------

def test_standardize_평균0_표준편차1():
    z = scoring.standardize([1, 2, 3, 4])
    assert z.mean() == pytest.approx(0.0, abs=1e-9)
    assert z.std() == pytest.approx(1.0, abs=1e-9)


def test_standardize_상수는_0벡터():
    assert np.allclose(scoring.standardize([5, 5, 5]), 0.0)


# ---------------------------------------------------------------------------
# 축별 산출
# ---------------------------------------------------------------------------

def test_cluster_trend_증가군집이_더_큼():
    df, labels = _df()
    tr = scoring.cluster_trend(df["발생일시"], labels)
    assert set(tr) == {0, 1}
    assert tr[0] > tr[1]     # 군집 0이 최근 증가 추세


def test_cluster_trend_과거군집도_기울기산출():
    """
    회귀 테스트: 창을 '전역 최근 N주'로 잡으면 데이터 span 초반에 발생한 군집의
    기울기가 일괄 0이 되어 추세 축이 죽는다. 군집 자체 활동기간 기준이므로
    긴 span의 초반 군집도 0이 아닌 기울기를 가져야 한다.
    """
    base = pd.Timestamp("2025-01-01")
    times, labels = [], []
    # 군집 0: span 맨 앞 30일 구간에서 가속 (뒤로 갈수록 촘촘)
    for i in range(12):
        times.append(base + pd.Timedelta(days=30 - int(30 * (1 - i / 12) ** 2)))
        labels.append(0)
    # 군집 1: span 맨 뒤에 존재 (전역 '최근'에 해당)
    for i in range(8):
        times.append(base + pd.Timedelta(days=330 + i))
        labels.append(1)

    tr = cluster_trend_호출(times, labels)
    assert tr[0] != 0.0, "초반 군집의 추세가 0이면 추세 축이 무력화된 것"


def cluster_trend_호출(times, labels):
    return scoring.cluster_trend(pd.Series(times), np.array(labels))


def test_cluster_trend_노이즈_제외():
    df, labels = _df()
    labels = labels.copy()
    labels[:3] = scoring.NOISE_LABEL
    tr = scoring.cluster_trend(df["발생일시"], labels)
    assert scoring.NOISE_LABEL not in tr


def test_criticality_EFF비율_반영():
    df, labels = _df()
    crit = scoring.cluster_criticality_observed(df, labels)
    # 군집 0은 EFF 100%, 군집 1은 NEFF 100% → 군집 1 치명도 0
    assert crit[0] > 0
    assert crit[1] == pytest.approx(0.0)


def test_criticality_truth컬럼_미사용():
    """β·α·λ(truth)가 df에 있어도 치명도 산출에 쓰이지 않아야 한다."""
    df, labels = _df()
    base = scoring.cluster_criticality_observed(df, labels)

    오염 = df.copy()
    오염["beta"] = 999.0
    오염["alpha"] = 999.0
    오염["lambda_"] = 999.0
    오염["Cm"] = 999.0
    after = scoring.cluster_criticality_observed(오염, labels)
    assert base == after


# ---------------------------------------------------------------------------
# 종합 점수
# ---------------------------------------------------------------------------

def test_score_clusters_순위테이블():
    df, labels = _df()
    r = scoring.score_clusters(df, labels)

    assert list(r["순위"]) == [1, 2]
    assert r.iloc[0]["군집"] == 0            # 규모·추세·치명도 모두 우위
    assert r.iloc[0]["건수"] == 10
    assert r.iloc[0]["대표LCN"] == "A.01.01.01"
    assert r.iloc[0]["대표문장1"] == "구동모터 과열 발생"
    # 점수 내림차순
    assert r.iloc[0]["점수"] >= r.iloc[1]["점수"]


def test_score_clusters_가중치가_순위를_뒤집음():
    """
    규모는 크지만 비임무(NEFF)인 군집 0 vs 규모는 작지만 임무영향(EFF)인 군집 1.
    → size 가중이면 0이 1위, crit 가중이면 1이 1위로 뒤집혀야 한다.
    """
    base = pd.Timestamp("2025-01-01")
    rows, labels = [], []
    for i in range(12):   # 군집 0: 대형·NEFF
        rows.append({"발생일시": base + pd.Timedelta(days=i),
                     "고장증상": "소음", "lcn": "A.01.01.01",
                     "effect_class": "NEFF"})
        labels.append(0)
    for i in range(5):    # 군집 1: 소형·EFF
        rows.append({"발생일시": base + pd.Timedelta(days=i),
                     "고장증상": "통신두절", "lcn": "B.02.02.02",
                     "effect_class": "EFF"})
        labels.append(1)
    df, labels = pd.DataFrame(rows), np.array(labels)

    r_size = scoring.score_clusters(df, labels, {"size": 1, "trend": 0, "crit": 0})
    r_crit = scoring.score_clusters(df, labels, {"size": 0, "trend": 0, "crit": 1})

    assert r_size.iloc[0]["군집"] == 0      # 규모 우선 → 대형 군집
    assert r_crit.iloc[0]["군집"] == 1      # 치명도 우선 → EFF 군집으로 역전


def test_score_clusters_노이즈만이면_빈테이블():
    df, labels = _df()
    labels = np.full(len(df), scoring.NOISE_LABEL)
    r = scoring.score_clusters(df, labels)
    assert r.empty
    assert "점수" in r.columns


# ---------------------------------------------------------------------------
# 민감도 분석
# ---------------------------------------------------------------------------

def test_sensitivity_analysis():
    df, labels = _df()
    s = scoring.sensitivity_analysis(df, labels)
    assert len(s) >= 4
    assert {"w_size", "w_trend", "w_crit", "1위군집", "기본대비_스피어만"} <= set(s.columns)


# ---------------------------------------------------------------------------
# 정답 순위 (평가 전용)
# ---------------------------------------------------------------------------

def test_truth_ranking_Cm합산_내림차순():
    truth = pd.DataFrame({
        "cluster_id": [0, 0, 1, 1, -1],
        "Cm": [1.0, 1.0, 5.0, 5.0, 99.0],   # 노이즈(-1)는 제외되어야 함
    })
    r = scoring.truth_ranking(truth)
    assert list(r["군집"]) == [1, 0]         # Cm 합 10 > 2
    assert r.iloc[0]["Cm_주입합"] == 10.0
    assert -1 not in set(r["군집"])


# ---------------------------------------------------------------------------
# 통합 (스모크): 합성→탐지→우선순위 배선이 정상 동작하는지만 확인.
#
# ※ 순위 정확도(Spearman ρ·Precision@k)는 여기서 단정하지 않는다.
#    단일 시드 ρ는 시드에 따라 0.85 ~ -0.33까지 흔들리므로(R4),
#    성능 판정은 Phase 4에서 8시드 mean±std로만 수행한다.
# ---------------------------------------------------------------------------

def test_통합_스모크_탐지결과로_순위산출():
    from detect import channels, fusion
    from detect.cluster import cluster
    from synth import generate
    from synth.config import SynthConfig

    res = generate(
        SynthConfig(n_records=220, n_clusters=4, noise_ratio=0.2,
                    delta=1.0, seed=5),
        save=False,
    )
    obs, truth = res["obs"], res["truth"]

    ch = channels.build_channels(
        obs, embedder=channels.HashingEmbedder(), use_cache=False
    )
    labels = cluster(fusion.fuse(ch), min_cluster_size=5)

    산출 = scoring.score_clusters(obs, labels)
    정답 = scoring.truth_ranking(truth)

    # 배선 검증: 순위 테이블이 well-formed 하고, 정답 순위가 분리 산출된다
    assert not 산출.empty and not 정답.empty
    assert list(산출["순위"]) == list(range(1, len(산출) + 1))
    assert 산출["점수"].is_monotonic_decreasing
    assert 산출["건수"].sum() <= len(obs)
    assert 산출["대표문장1"].notna().all()
    assert scoring.NOISE_LABEL not in set(산출["군집"])
    # 정답 순위는 truth의 주입 Cm에서만 나온다
    assert "Cm_주입합" in 정답.columns
