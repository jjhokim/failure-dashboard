"""
Phase 4 실험 지표 검증.

사전등록(§6)에 고정한 정의대로 동작하는지 손계산 가능한 입력으로 확인한다.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import metrics as M  # noqa: E402
from experiments.run import parse_seeds  # noqa: E402


# ---------------------------------------------------------------------------
# ARI / 노이즈
# ---------------------------------------------------------------------------

def test_ari_완전일치는_1():
    assert M.ari([0, 0, 1, 1], [5, 5, 9, 9]) == pytest.approx(1.0)


def test_ari_노이즈를_단일군집으로_유지():
    """사전등록: -1을 개별 군집으로 분해하지 않고 하나의 라벨로 유지."""
    y = [0, 0, 1, 1, -1, -1]
    p = [0, 0, 1, 1, -1, -1]
    assert M.ari(y, p) == pytest.approx(1.0)   # 노이즈끼리 같은 군집 취급


def test_noise_ratio():
    assert M.noise_ratio([-1, -1, 0, 0]) == pytest.approx(0.5)
    assert M.noise_ratio([]) == 0.0


# ---------------------------------------------------------------------------
# Pairwise
# ---------------------------------------------------------------------------

def test_pairwise_완전일치():
    r = M.pairwise_precision_recall([0, 0, 1, 1], [0, 0, 1, 1])
    assert r["precision"] == pytest.approx(1.0)
    assert r["recall"] == pytest.approx(1.0)


def test_pairwise_노이즈쌍_제외():
    # 노이즈 레코드를 제외하면 완전일치가 되어야 함
    r = M.pairwise_precision_recall([0, 0, 1, 1, -1], [0, 0, 1, 1, 7])
    assert r["precision"] == pytest.approx(1.0)
    assert r["recall"] == pytest.approx(1.0)


def test_pairwise_전부_한군집이면_recall1_precision낮음():
    r = M.pairwise_precision_recall([0, 0, 1, 1], [0, 0, 0, 0])
    assert r["recall"] == pytest.approx(1.0)
    assert r["precision"] < 1.0


# ---------------------------------------------------------------------------
# 매칭 / 순위
# ---------------------------------------------------------------------------

def test_match_clusters_다수결():
    pred = [0, 0, 0, 1, 1]
    truth = [7, 7, 9, 3, 3]
    assert M.match_clusters(pred, truth) == {0: 7, 1: 3}


def test_match_clusters_노이즈_제외():
    assert -1 not in M.match_clusters([-1, -1, 0, 0], [5, 5, 2, 2])


def test_ranking_scores_완전일치면_rho1():
    score = pd.DataFrame({"순위": [1, 2, 3], "군집": [0, 1, 2]})
    truth = pd.DataFrame({"순위": [1, 2, 3], "군집": [10, 11, 12]})
    pred_labels = [0] * 3 + [1] * 3 + [2] * 3
    truth_labels = [10] * 3 + [11] * 3 + [12] * 3

    r = M.ranking_scores(score, truth, pred_labels, truth_labels, k=3)
    assert r["spearman"] == pytest.approx(1.0)
    assert r["precision_at_k"] == pytest.approx(1.0)
    assert r["n_pairs"] == 3


def test_ranking_scores_역순이면_rho_음수():
    score = pd.DataFrame({"순위": [1, 2, 3], "군집": [0, 1, 2]})
    truth = pd.DataFrame({"순위": [1, 2, 3], "군집": [12, 11, 10]})
    pred_labels = [0] * 3 + [1] * 3 + [2] * 3
    truth_labels = [10] * 3 + [11] * 3 + [12] * 3
    assert M.ranking_scores(score, truth, pred_labels, truth_labels, k=3)["spearman"] < 0


def test_ranking_scores_쌍부족이면_None():
    score = pd.DataFrame({"순위": [1], "군집": [0]})
    truth = pd.DataFrame({"순위": [1], "군집": [10]})
    r = M.ranking_scores(score, truth, [0, 0], [10, 10], k=5)
    assert r["spearman"] is None


# ---------------------------------------------------------------------------
# 부트스트랩 / 요약
# ---------------------------------------------------------------------------

def test_bootstrap_ci_결정론성_및_구간():
    vals = [0.2, 0.4, 0.5, 0.6, 0.55, 0.45, 0.35, 0.5]
    a = M.bootstrap_ci(vals, seed=0)
    b = M.bootstrap_ci(vals, seed=0)
    assert a == b                                  # 시드 고정 → 재현
    assert a["lo"] <= a["mean"] <= a["hi"]
    assert a["n"] == 8


def test_bootstrap_ci_None_무시():
    assert M.bootstrap_ci([None, None])["n"] == 0
    assert M.bootstrap_ci([1.0, None])["n"] == 1


def test_summarize_mean_std():
    df = pd.DataFrame({
        "조건": ["a", "a", "b", "b"],
        "channels": ["x"] * 4, "delta": [1.0] * 4,
        "ARI": [0.2, 0.4, 0.8, 1.0],
    })
    out = M.summarize(df, ["조건", "channels", "delta"], ["ARI"])
    assert set(["ARI_mean", "ARI_std", "ARI_count"]) <= set(out.columns)
    assert out.loc[out["조건"] == "a", "ARI_mean"].iloc[0] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 탐지 지연
# ---------------------------------------------------------------------------

def test_detection_delay_도달과_미도달():
    n = 40
    obs = pd.DataFrame({
        "발생일시": pd.date_range("2025-01-01", periods=n, freq="D"),
    })
    truth = pd.DataFrame({"cluster_id": [0] * 20 + [1] * 20})

    # 완벽 분리 군집화 → 순도 1.0, 조기 도달
    def perfect(sub):
        return truth["cluster_id"].to_numpy()[: len(sub)]

    r = M.detection_delay(obs, truth, perfect, n_steps=4)
    assert r["reach_rate"] > 0
    assert any(v is not None for v in r["delays"].values())

    # 전부 노이즈 → 미도달
    r2 = M.detection_delay(obs, truth, lambda sub: np.full(len(sub), -1), n_steps=4)
    assert r2["reach_rate"] == 0.0
    assert all(v is None for v in r2["delays"].values())


# ---------------------------------------------------------------------------
# CLI 인자
# ---------------------------------------------------------------------------

def test_evaluate_once_detection_delay_배선():
    """
    후보작업 3: Detection Delay가 ablation 루프(evaluate_once)에 연결되었는지 확인.
    오프라인 더미 임베더로 배선만 검증한다.
    """
    import warnings

    from detect import channels
    from experiments.run import DEFAULT_CONFIG, _build, evaluate_once, load_config

    warnings.filterwarnings("ignore")
    cfg = load_config(DEFAULT_CONFIG)
    cfg["data"] = {**cfg["data"], "n_records": 100, "n_clusters": 3}
    emb = channels.HashingEmbedder()
    res, ch = _build(cfg, 1.0, 902, emb)

    # with_delay=False면 지연 컬럼이 없어야 한다
    없음 = evaluate_once(res, ch, cfg["detect"]["default_weights"], 5)
    assert "delay_reach_rate" not in 없음

    있음 = evaluate_once(res, ch, cfg["detect"]["default_weights"], 5,
                        embedder=emb, with_delay=True, delay_steps=4)
    assert "delay_reach_rate" in 있음
    assert 0.0 <= 있음["delay_reach_rate"] <= 1.0
    # 도달했다면 누적 건수는 양수이고 전체 건수를 넘지 않는다
    if 있음["delay_mean_records"] is not None:
        assert 0 < 있음["delay_mean_records"] <= len(res["obs"])
        assert 있음["delay_min_records"] <= 있음["delay_mean_records"]


def test_parse_seeds():
    assert parse_seeds("0-7") == list(range(8))
    assert parse_seeds("0,2,5") == [0, 2, 5]
    assert parse_seeds("3") == [3]
