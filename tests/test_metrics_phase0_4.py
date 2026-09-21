"""
Phase 0-4 지표 재구현 검증 (backend/metrics.py).

손계산 가능한 소규모 고정 입력으로 순수 함수 기대값을 검증한다.
  - mtbf_logistics / mtbf_mission : 운용시간 기반 MTBF
  - mttr_km                       : Kaplan-Meier 우측절단 보정 MTTR
  - availability_inherent / _operational : Ai / Ao
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import metrics  # noqa: E402


def _failures(rows: list[dict]) -> pd.DataFrame:
    cols = ["발생일시", "처리상태", "수리소요시간_h", "수리시작시간",
            "수리완료시간", "effect_class", "mdt_hours"]
    return pd.DataFrame(rows, columns=cols)


def _systems(op_hours: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "system_id": [f"SYS-{i}" for i in range(len(op_hours))],
        "cumulative_op_hours": op_hours,
    })


# ---------------------------------------------------------------------------
# 운용시간 기반 MTBF
# ---------------------------------------------------------------------------

def test_total_op_hours():
    assert metrics.total_op_hours(_systems([1000, 1000])) == 2000.0
    assert metrics.total_op_hours(pd.DataFrame()) == 0.0


def test_mtbf_logistics():
    failures = _failures([{"처리상태": "수리완료"}] * 4)
    # Σ운용시간 2000 / 고장 4건 = 500
    assert metrics.mtbf_logistics(failures, _systems([1000, 1000])) == 500.0


def test_mtbf_logistics_운용시간_없음():
    failures = _failures([{"처리상태": "수리완료"}] * 4)
    assert metrics.mtbf_logistics(failures, pd.DataFrame()) is None


def test_mtbf_mission_eff만_계수():
    failures = _failures([
        {"처리상태": "수리완료", "effect_class": "EFF"},
        {"처리상태": "수리완료", "effect_class": "EFF"},
        {"처리상태": "수리완료", "effect_class": "NEFF"},
        {"처리상태": "수리완료", "effect_class": "NEFF"},
    ])
    # Σ운용시간 2000 / EFF 2건 = 1000
    assert metrics.mtbf_mission(failures, _systems([1000, 1000])) == 1000.0


# ---------------------------------------------------------------------------
# MTTR — Kaplan-Meier 절단 보정
# ---------------------------------------------------------------------------

def test_mttr_km_절단없음_naive와_동일():
    # 완료건 [2, 4]만 → KM RMST = 단순평균 3.0
    df = _failures([
        {"처리상태": "수리완료", "수리소요시간_h": 2.0},
        {"처리상태": "수리완료", "수리소요시간_h": 4.0},
    ])
    assert metrics.mttr_km(df) == 3.0
    assert metrics.mttr_naive(df) == 3.0


def test_mttr_km_수리중만_절단반영():
    # 완료 [2,4] + 수리중 1건(수리시작 후 10h 경과, 절단)
    df = _failures([
        {"처리상태": "수리완료", "수리소요시간_h": 2.0},
        {"처리상태": "수리완료", "수리소요시간_h": 4.0},
        {"처리상태": "수리중", "수리소요시간_h": None,
         "수리시작시간": "2025-01-01 00:00:00"},
    ])
    km = metrics.mttr_km(df, as_of="2025-01-01 10:00:00")
    # 절단(진행 중 장기수리) 반영으로 MTTR 상향: 5.33 > naive 3.0
    assert km == 5.33
    assert km > metrics.mttr_naive(df)


def test_mttr_km_미해결은_제외():
    # (b) 결정: '미해결'은 절단 표본에서 제외 → 완료 [2,4]만 남아 3.0
    df = _failures([
        {"처리상태": "수리완료", "수리소요시간_h": 2.0},
        {"처리상태": "수리완료", "수리소요시간_h": 4.0},
        {"처리상태": "미해결", "수리소요시간_h": None,
         "수리시작시간": "2025-01-01 00:00:00"},
    ])
    assert metrics.mttr_km(df, as_of="2025-01-01 10:00:00") == 3.0


def test_mttr_km_완료건_없음():
    # 수리중만 있고 완료 사건이 없으면 KM 추정 불가 → None
    df = _failures([{"처리상태": "수리중", "수리소요시간_h": None,
                     "수리시작시간": "2025-01-01 00:00:00"}])
    assert metrics.mttr_km(df, as_of="2025-01-01 10:00:00") is None


def test_mttr_km_빈입력():
    assert metrics.mttr_km(_failures([])) is None


# ---------------------------------------------------------------------------
# 가동도 Ai / Ao
# ---------------------------------------------------------------------------

def test_availability_inherent():
    assert metrics.availability_inherent(90.0, 10.0) == 0.9
    assert metrics.availability_inherent(100.0, None) is None
    assert metrics.availability_inherent(0.0, 0.0) is None


def test_availability_operational_결측이면_none():
    # MDT 결측 → 억지 계산 없이 None
    assert metrics.availability_operational(100.0, None) is None
    assert metrics.availability_operational(None, 5.0) is None
    # 둘 다 있으면 정상 산출: 95/(95+5)=0.95
    assert metrics.availability_operational(95.0, 5.0) == 0.95


def test_mean_mdt():
    df = _failures([
        {"처리상태": "수리완료", "mdt_hours": 2.0},
        {"처리상태": "수리완료", "mdt_hours": 4.0},
        {"처리상태": "미해결", "mdt_hours": None},
    ])
    assert metrics.mean_mdt(df) == 3.0
    # 전부 결측 → None
    df2 = _failures([{"처리상태": "수리완료", "mdt_hours": None}])
    assert metrics.mean_mdt(df2) is None
