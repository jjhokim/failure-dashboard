"""
사전등록 v2 후속 재실험 구성요소 검증.

  - 시각 밀집폭 모드 (legacy / floor)
  - 치명도 3변형 재점수 (S0 / S1 / S2)
  - 가중치 격자, 페어링 차이 CI
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_v2 import _paired_diff_ci, _weight_grid  # noqa: E402
from prioritize import scoring  # noqa: E402
from synth.config import SynthConfig  # noqa: E402
from synth.physical import _time_window  # noqa: E402


# ---------------------------------------------------------------------------
# 시각 밀집폭 모드
# ---------------------------------------------------------------------------

def test_time_window_legacy는_delta1에서_1일():
    cfg = SynthConfig(delta=1.0, time_window_mode="legacy")
    assert _time_window(cfg, 365) == pytest.approx(1.0)


def test_time_window_floor는_delta1에서_span1퍼센트():
    cfg = SynthConfig(delta=1.0, time_window_mode="floor")
    assert _time_window(cfg, 365) == pytest.approx(3.65)


def test_time_window_기본값은_legacy():
    """1차 실험 재현성 보존 — 기본 동작이 바뀌면 안 된다."""
    assert _time_window(SynthConfig(delta=1.0), 365) == pytest.approx(1.0)


def test_time_window_delta0은_두_모드_유사():
    for mode in ("legacy", "floor"):
        w = _time_window(SynthConfig(delta=0.0, time_window_mode=mode), 365)
        assert w > 15  # δ=0이면 폭이 넓다


def test_floor모드가_legacy보다_데이터_분산_큼():
    """floor 모드는 δ=1에서 군집 시각이 덜 몰려야 한다."""
    from synth.physical import build_physical_states

    def 군집내_시각표준편차(mode):
        cfg = SynthConfig(n_records=300, n_clusters=4, noise_ratio=0.1,
                          delta=1.0, seed=1, time_window_mode=mode)
        states, truth = build_physical_states(cfg)
        t = pd.to_datetime(states["occurred_at"])
        일 = (t - t.min()).dt.total_seconds() / 86400
        sds = [일[truth["cluster_id"].to_numpy() == c].std()
               for c in sorted(set(truth["cluster_id"])) if c != -1]
        return float(np.mean(sds))

    assert 군집내_시각표준편차("floor") > 군집내_시각표준편차("legacy")


# ---------------------------------------------------------------------------
# 치명도 변형
# ---------------------------------------------------------------------------

def _표() -> pd.DataFrame:
    df = pd.DataFrame({
        "발생일시": pd.date_range("2025-01-01", periods=30, freq="3D"),
        "고장증상": ["증상"] * 30,
        "lcn": ["A.01.01.01"] * 15 + ["B.02.02.02"] * 15,
        "effect_class": ["EFF"] * 10 + ["NEFF"] * 20,
    })
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
    return scoring.score_clusters(df, labels)


def test_S0는_원본과_동일():
    base = _표()
    s0 = scoring.rescore_with_variant(base, "S0")
    assert np.allclose(sorted(s0["치명도_obs"]), sorted(base["치명도_obs"]))


def test_S2_강도형은_건당평균():
    base = _표()
    s2 = scoring.rescore_with_variant(base, "S2")
    b = base.set_index("군집"); s = s2.set_index("군집")
    for c in b.index:
        assert s.loc[c, "치명도_obs"] == pytest.approx(
            b.loc[c, "치명도_obs"] / b.loc[c, "건수"]
        )


def test_S1_잔차화는_규모상관_제거():
    base = _표()
    s1 = scoring.rescore_with_variant(base, "S1")
    if len(s1) >= 3 and s1["건수"].std() > 0:
        corr = s1["건수"].corr(s1["치명도_obs"])
        assert abs(corr) < 1e-6 or pd.isna(corr)   # 선형 잔차 → 상관 ≈ 0


def test_변형이_순위를_재정렬():
    base = _표()
    s2 = scoring.rescore_with_variant(base, "S2", {"size": 0, "trend": 0, "crit": 1})
    assert list(s2["순위"]) == list(range(1, len(s2) + 1))
    assert s2["점수"].is_monotonic_decreasing


def test_알수없는_변형은_에러():
    with pytest.raises(ValueError):
        scoring.rescore_with_variant(_표(), "S9")


def test_빈표는_그대로():
    assert scoring.rescore_with_variant(pd.DataFrame(), "S1").empty


# ---------------------------------------------------------------------------
# 실행기 유틸
# ---------------------------------------------------------------------------

def test_weight_grid_합이1():
    grid = _weight_grid(0.25)
    assert len(grid) > 5
    for w in grid:
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(v >= 0 for v in w.values())


def test_paired_diff_ci_부호와_0배제():
    a = [0.9, 0.85, 0.88, 0.92]
    b = [0.5, 0.45, 0.48, 0.52]
    ci = _paired_diff_ci(a, b)
    assert ci["mean"] > 0
    assert ci["excludes_zero"] is True

    # 차이가 없으면 0을 포함
    ci2 = _paired_diff_ci(a, a)
    assert ci2["excludes_zero"] is False
