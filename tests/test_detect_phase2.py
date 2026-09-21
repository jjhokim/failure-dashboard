"""
Phase 2 탐지 파이프라인 검증.

임베딩이 필요한 테스트는 결정론적 HashingEmbedder(오프라인 더미)를 쓴다.
실제 실험은 KoSBERT로 수행하며, 여기서는 배선(거리행렬·융합·군집화)의
정확성과 ablation 동작을 검증한다.

R2 검증: detect가 synth를 import하지 않음 / 규칙사전이 생성기 사전과 분리됨.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import detect.cluster as clu  # noqa: E402
from detect import baselines, channels, fusion  # noqa: E402


# ---------------------------------------------------------------------------
# 구조(LCN) 채널
# ---------------------------------------------------------------------------

def test_lcn_distance_동일_LCN은_0():
    D = channels.lcn_distance(["A.01.02", "A.01.02"])
    assert D[0, 1] == 0.0
    assert np.allclose(np.diag(D), 0.0)


def test_lcn_distance_형제가_먼친척보다_가까움():
    # A.01.01 vs A.01.02 (형제, 거리 2) < A.01.01 vs B.05.09 (다른 루트)
    D = channels.lcn_distance(["A.01.01", "A.01.02", "B.05.09"])
    assert D[0, 1] < D[0, 2]
    assert D.max() <= 1.0 and D.min() >= 0.0
    assert np.allclose(D, D.T)


# ---------------------------------------------------------------------------
# 시간 채널
# ---------------------------------------------------------------------------

def test_time_distance_단조증가_및_tau():
    times = ["2025-01-01", "2025-01-02", "2025-06-01"]
    D = channels.time_distance(times, tau_days=30.0)
    assert D[0, 1] < D[0, 2]             # 가까운 시점이 더 작은 거리
    assert np.allclose(np.diag(D), 0.0)
    # Δt = τ 이면 1 - e^-1 ≈ 0.632
    D2 = channels.time_distance(["2025-01-01", "2025-01-31"], tau_days=30.0)
    assert D2[0, 1] == pytest.approx(1 - np.exp(-1), abs=1e-3)


# ---------------------------------------------------------------------------
# 의미 채널 (오프라인 더미 임베더)
# ---------------------------------------------------------------------------

def test_semantic_distance_동일문장은_0():
    texts = ["구동모터 과열 발생", "구동모터 과열 발생", "통신모듈 신호 끊김"]
    D = channels.semantic_distance(
        texts, embedder=channels.HashingEmbedder(), use_cache=False
    )
    assert D[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert D[0, 2] > D[0, 1]
    assert np.allclose(D, D.T)


# ---------------------------------------------------------------------------
# 융합 (ablation 포함)
# ---------------------------------------------------------------------------

def test_minmax_normalize_범위():
    D = np.array([[0.0, 2.0, 4.0], [2.0, 0.0, 6.0], [4.0, 6.0, 0.0]])
    N = fusion.minmax_normalize(D)
    off = ~np.eye(3, dtype=bool)
    assert N[off].min() == pytest.approx(0.0)
    assert N[off].max() == pytest.approx(1.0)


def test_fuse_가중치0은_채널_off():
    n = 4
    rng = np.random.RandomState(0)
    A = np.abs(rng.rand(n, n)); A = (A + A.T) / 2; np.fill_diagonal(A, 0)
    B = np.abs(rng.rand(n, n)); B = (B + B.T) / 2; np.fill_diagonal(B, 0)
    ch = {"sem": A, "lcn": B, "time": B}

    # time 가중치 0 → sem/lcn만 사용한 결과와 동일해야 함 (ablation 동작)
    f1 = fusion.fuse(ch, {"sem": 0.5, "lcn": 0.5, "time": 0.0})
    f2 = fusion.fuse({"sem": A, "lcn": B}, {"sem": 0.5, "lcn": 0.5})
    assert np.allclose(f1, f2)


def test_fuse_단일채널은_정규화결과와_동일():
    n = 5
    rng = np.random.RandomState(1)
    A = np.abs(rng.rand(n, n)); A = (A + A.T) / 2; np.fill_diagonal(A, 0)
    ch = {"sem": A, "lcn": A * 3, "time": A * 7}
    f = fusion.fuse(ch, {"sem": 1.0, "lcn": 0.0, "time": 0.0})
    assert np.allclose(f, fusion.minmax_normalize(A))


def test_fuse_가중치_전부0이면_에러():
    A = np.zeros((3, 3))
    with pytest.raises(ValueError):
        fusion.fuse({"sem": A}, {"sem": 0.0})


# ---------------------------------------------------------------------------
# 군집화
# ---------------------------------------------------------------------------

def _두덩어리_거리행렬(n_each: int = 6) -> np.ndarray:
    n = n_each * 2
    D = np.full((n, n), 0.9)
    for blk in (range(n_each), range(n_each, n)):
        for i in blk:
            for j in blk:
                D[i, j] = 0.05
    np.fill_diagonal(D, 0.0)
    return D


def test_cluster_명확한_두군집_분리():
    labels = clu.cluster(_두덩어리_거리행렬(6), min_cluster_size=5)
    stats = clu.cluster_stats(labels)
    assert stats["n_clusters"] == 2
    assert labels[0] == labels[1]          # 같은 덩어리는 같은 라벨
    assert labels[0] != labels[-1]         # 다른 덩어리는 다른 라벨


def test_cluster_sweep_사전고정범위():
    out = clu.cluster_sweep(_두덩어리_거리행렬(8))
    assert list(out) == clu.MIN_CLUSTER_SIZES == [5, 8, 12]


def test_cluster_stats_노이즈비율():
    labels = np.array([-1, -1, 0, 0, 0, 1, 1, 1])
    st = clu.cluster_stats(labels)
    assert st["n_clusters"] == 2
    assert st["noise_ratio"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 베이스라인
# ---------------------------------------------------------------------------

def test_baseline_tfidf_군집분리():
    texts = ["구동모터 과열 발생"] * 6 + ["통신모듈 신호 두절 확인"] * 6
    labels = baselines.baseline_tfidf(texts, min_cluster_size=5)
    assert len(set(labels[:6])) == 1
    assert labels[0] != labels[-1]


def test_baseline_rules_범주그룹핑():
    texts = ["구동모터 과열 발생"] * 5 + ["통신모듈 신호 두절"] * 5 + ["설명 없음"] * 2
    labels = baselines.baseline_rules(texts, min_group_size=5)
    assert labels[0] == labels[4]          # 열관련 그룹
    assert labels[5] == labels[9]          # 통신계통 그룹
    assert labels[0] != labels[5]
    assert labels[-1] == clu.NOISE_LABEL   # 미매칭 → 노이즈


# ---------------------------------------------------------------------------
# R2: 생성기·탐지기 분리
# ---------------------------------------------------------------------------

def test_R2_detect가_synth를_import하지_않음():
    detect_dir = Path(__file__).resolve().parent.parent / "detect"
    for py in detect_dir.glob("*.py"):
        src = py.read_text(encoding="utf-8")
        assert "import synth" not in src and "from synth" not in src, py.name


def test_R2_규칙사전이_생성기사전과_다름():
    from synth.narrator import 증상_동의어

    # 사전 객체를 공유하지 않아야 하고, 분류 체계(키)도 독립이어야 한다
    assert baselines.정비_키워드 is not 증상_동의어
    assert set(baselines.정비_키워드) != set(증상_동의어)


# ---------------------------------------------------------------------------
# 통합: 합성데이터에서 주입 신호 회수
# ---------------------------------------------------------------------------

def test_통합_delta1이_delta0보다_ARI_높음():
    from sklearn.metrics import adjusted_rand_score

    from synth import generate
    from synth.config import SynthConfig

    emb = channels.HashingEmbedder()

    def run(delta):
        res = generate(
            SynthConfig(n_records=180, n_clusters=3, noise_ratio=0.2,
                        delta=delta, seed=5), save=False
        )
        obs, truth = res["obs"], res["truth"]
        ch = channels.build_channels(obs, embedder=emb, use_cache=False)
        labels = clu.cluster(fusion.fuse(ch), min_cluster_size=5)
        return adjusted_rand_score(truth["cluster_id"], labels)

    ari0, ari1 = run(0.0), run(1.0)
    assert ari1 > ari0            # 주입 강도가 높을수록 회수 성능 상승
    assert ari0 < 0.25            # 무주입 대조군은 낮아야 함
