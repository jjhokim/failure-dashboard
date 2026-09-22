"""
Phase 5 검증 — BIT 조인 / LCN 파레토 / 모의 BIT 생성.

R3 검증: BIT 생성기가 군집 라벨·탐지 결과를 참조하지 않는다.
"""

import inspect
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import crud, database  # noqa: E402
from backend.models import BIT이벤트, 고장이력  # noqa: E402
from synth.bit import generate_bit_events  # noqa: E402


@pytest.fixture
def 임시DB(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "t.db")
    database.init_db()
    return tmp_path


# ---------------------------------------------------------------------------
# BIT 조인
# ---------------------------------------------------------------------------

def _고장(lcn, dt, 증상="증상"):
    return 고장이력(
        제대구분="부대정비", 발생일시=dt, 체계명="체계-A", LRU명="LRU-001",
        고장유형="전기적", 고장증상=증상, 처리상태="수리완료",
        등록일시=dt, lcn=lcn,
    )


def test_BIT조인_윈도우_내만_연결(임시DB):
    기준 = datetime(2025, 5, 1, 12, 0, 0)
    crud.고장이력_저장(_고장("A.01.01.01", 기준))

    crud.BIT이벤트_저장([
        # 같은 LCN, 2시간 차 → 연결
        BIT이벤트(lcn="A.01.01.01", bit_code="BIT-1",
                 occurred_at=datetime(2025, 5, 1, 14, 0, 0)),
        # 같은 LCN, 100시간 차 → 제외
        BIT이벤트(lcn="A.01.01.01", bit_code="BIT-2",
                 occurred_at=datetime(2025, 5, 5, 16, 0, 0)),
        # 다른 LCN, 시각 근접 → 제외
        BIT이벤트(lcn="B.02.02.02", bit_code="BIT-3",
                 occurred_at=datetime(2025, 5, 1, 12, 30, 0)),
    ])

    조인 = crud.BIT_정비기록_조인(윈도우_시간=24)
    assert len(조인) == 1
    assert 조인.iloc[0]["bit_code"] == "BIT-1"
    assert 조인.iloc[0]["시간차_h"] == pytest.approx(2.0)


def test_BIT조인_윈도우_확대시_증가(임시DB):
    기준 = datetime(2025, 5, 1, 12, 0, 0)
    crud.고장이력_저장(_고장("A.01.01.01", 기준))
    crud.BIT이벤트_저장([
        BIT이벤트(lcn="A.01.01.01", bit_code="BIT-far",
                 occurred_at=datetime(2025, 5, 2, 12, 0, 0)),  # 24h
    ])
    assert len(crud.BIT_정비기록_조인(윈도우_시간=12)) == 0
    assert len(crud.BIT_정비기록_조인(윈도우_시간=25)) == 1


def test_BIT조인_데이터없으면_빈결과(임시DB):
    조인 = crud.BIT_정비기록_조인()
    assert 조인.empty
    assert "시간차_h" in 조인.columns


# ---------------------------------------------------------------------------
# LCN 파레토
# ---------------------------------------------------------------------------

def test_LCN파레토_레벨별_집계():
    df = pd.DataFrame({"lcn": ["A.01.01", "A.01.02", "A.02.01", "B.01.01"]})
    p2 = crud.LCN_파레토(df, level=2)
    assert list(p2["LCN노드"])[0] == "A.01"      # 2건으로 최다
    assert p2.iloc[0]["건수"] == 2
    assert p2["누적비율_%"].iloc[-1] == pytest.approx(100.0, abs=0.5)

    p1 = crud.LCN_파레토(df, level=1)
    assert set(p1["LCN노드"]) == {"A", "B"}
    assert p1.iloc[0]["건수"] == 3               # A가 3건


def test_LCN파레토_빈입력():
    assert crud.LCN_파레토(pd.DataFrame()).empty
    assert crud.LCN_파레토(pd.DataFrame({"lcn": [None, None]})).empty


# ---------------------------------------------------------------------------
# 모의 BIT 생성 (R3)
# ---------------------------------------------------------------------------

def _obs(n=40):
    return pd.DataFrame({
        "lcn": [f"A.0{i % 3 + 1}.01.01" for i in range(n)],
        "발생일시": pd.date_range("2025-01-01", periods=n, freq="6h").strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "system_id": ["SYS-A-001"] * n,
    })


def test_BIT생성_결정론성_및_형식():
    a = generate_bit_events(_obs(), seed=1)
    b = generate_bit_events(_obs(), seed=1)
    assert a.equals(b)
    assert list(a.columns) == ["system_id", "lcn", "bit_code", "occurred_at"]
    assert len(a) > 0


def test_BIT생성_R3_라벨_미참조():
    """생성 함수 시그니처·결과 어디에도 군집 라벨이 없어야 한다."""
    params = set(inspect.signature(generate_bit_events).parameters)
    assert "cluster_id" not in params
    assert "truth" not in params
    assert "labels" not in params

    out = generate_bit_events(_obs(), seed=0)
    assert "cluster_id" not in out.columns


def test_BIT생성_bit_code는_LCN에서만_파생():
    from synth.bit import _bit_code

    # 동일 상위 LCN → 동일 코드, 다른 상위 LCN → 다른 코드
    assert _bit_code("A.01.05.09") == _bit_code("A.01.99.99")
    assert _bit_code("A.01.01.01") != _bit_code("B.02.01.01")


def test_BIT생성_coverage_반영():
    적게 = generate_bit_events(_obs(100), seed=2, coverage=0.2, false_alarm_ratio=0)
    많이 = generate_bit_events(_obs(100), seed=2, coverage=0.9, false_alarm_ratio=0)
    assert len(적게) < len(많이)


def test_BIT생성_빈입력():
    assert generate_bit_events(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# 통합: 생성 → 저장 → 조인
# ---------------------------------------------------------------------------

def test_통합_BIT생성후_조인(임시DB):
    obs = _obs(30)
    for _, r in obs.iterrows():
        crud.고장이력_저장(_고장(r["lcn"], pd.Timestamp(r["발생일시"]).to_pydatetime()))

    ev = generate_bit_events(obs, seed=0, coverage=1.0,
                             window_hours=2.0, false_alarm_ratio=0.0)
    crud.BIT이벤트_저장([
        BIT이벤트(system_id=r["system_id"], lcn=r["lcn"], bit_code=r["bit_code"],
                 occurred_at=pd.Timestamp(r["occurred_at"]).to_pydatetime())
        for _, r in ev.iterrows()
    ])

    조인 = crud.BIT_정비기록_조인(윈도우_시간=3)
    assert not 조인.empty
    assert (조인["시간차_h"] <= 3).all()
