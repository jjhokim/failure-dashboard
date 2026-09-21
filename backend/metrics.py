"""
RAM 분석 지표 순수 함수 모듈 (Phase 0-4).

원칙(#3): 모든 지표 계산 로직은 이 모듈의 순수 함수로 집중한다.
UI(app.py)·CRUD에는 계산식을 두지 않는다. 각 함수는 입력(DataFrame/스칼라)만으로
결정되며 전역 난수·시계에 의존하지 않는다(절단 기준시각 as_of는 인자로 받거나
데이터의 최댓값으로 결정론적으로 도출).

근거: 방위사업청 「무기체계 RAM 업무지침」(2018).

정의식:
  MTBF(군수) = Σ운용시간 / 전체 고장건수                    (mtbf_logistics)
  MTBF(임무) = Σ운용시간 / 임무영향(EFF) 고장건수           (mtbf_mission)
  MTTR       = 평균 수리시간, 우측절단은 Kaplan-Meier 보정   (mttr_km)
  Ai         = MTBF / (MTBF + MTTR)   [고유가용도]           (availability_inherent)
  Ao         = MTBM / (MTBM + MDT)    [운용가용도]           (availability_operational)

용어:
  ALDT = 행정·군수지연시간, MDT = 정비중단시간, MTBM = 평균정비간격.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ===========================================================================
# 운용시간 / 건수 헬퍼
# ===========================================================================

def total_op_hours(systems: Optional[pd.DataFrame]) -> float:
    """systems.cumulative_op_hours 합계(Σ운용시간). 데이터 없으면 0.0."""
    if systems is None or systems.empty or "cumulative_op_hours" not in systems.columns:
        return 0.0
    return float(systems["cumulative_op_hours"].fillna(0).sum())


# ===========================================================================
# MTBF (평균고장간격)
# ===========================================================================

def mtbf_logistics(
    failures: pd.DataFrame, systems: Optional[pd.DataFrame]
) -> Optional[float]:
    """
    군수 MTBF = Σ운용시간 / 전체 고장건수  (운용시간 기준).

    기존 오류 #2(달력 경과시간 ÷ 건수, 모수 역민감) 정정판.
    운용시간 합이 0이거나(=systems 미등록) 고장건수가 0이면 None 반환.
    근거: 「무기체계 RAM 업무지침」(2018).
    """
    op = total_op_hours(systems)
    n = len(failures)
    if op <= 0 or n == 0:
        return None
    return round(op / n, 2)


def mtbf_mission(
    failures: pd.DataFrame, systems: Optional[pd.DataFrame]
) -> Optional[float]:
    """
    임무 MTBF = Σ운용시간 / 임무영향(effect_class=='EFF') 고장건수.

    임무 수행에 영향을 준 고장만 분모(건수)에 계수한다.
    effect_class 컬럼이 없거나 EFF 건수·운용시간이 0이면 None 반환.
    """
    op = total_op_hours(systems)
    if "effect_class" not in failures.columns:
        return None
    n_eff = int((failures["effect_class"] == "EFF").sum())
    if op <= 0 or n_eff == 0:
        return None
    return round(op / n_eff, 2)


def mtbf_calendar(
    failures: pd.DataFrame, 관측기간_h: Optional[float] = None
) -> float:
    """
    [임시·근사] 달력 경과시간 ÷ 고장건수.

    ⚠ 오류 #2: 운용시간 기준이 아니며 모수(fleet size)에 역민감하다.
    운용시간(systems.cumulative_op_hours) 데이터가 없을 때의 폴백 전용이며,
    데이터가 갖춰지면 mtbf_logistics로 대체한다.
    """
    건수 = len(failures)
    if 건수 == 0:
        return 0.0

    if 관측기간_h is None:
        dates = pd.to_datetime(failures["발생일시"], errors="coerce").dropna()
        if len(dates) < 2:
            return 0.0
        관측기간_h = (dates.max() - dates.min()).total_seconds() / 3600

    if 관측기간_h <= 0:
        return 0.0

    return round(관측기간_h / 건수, 2)


# ===========================================================================
# MTTR (평균수리시간)
# ===========================================================================

def _수리완료_durations(failures: pd.DataFrame) -> list[float]:
    """수리완료 & 수리소요시간_h 유효(>0) 건의 수리시간 목록(관측·비절단)."""
    완료 = failures[
        (failures["처리상태"] == "수리완료")
        & failures["수리소요시간_h"].notna()
        & (failures["수리소요시간_h"] > 0)
    ]
    return [float(x) for x in 완료["수리소요시간_h"]]


def mttr_naive(failures: pd.DataFrame) -> Optional[float]:
    """
    [비교·폴백용] 수리완료 건만의 단순 평균 수리시간.

    ⚠ 오류 #3: 미완료(진행 중) 수리건을 제외하므로 우측절단 편향으로
    MTTR을 과소평가한다. 정식 산출은 mttr_km를 사용한다.
    """
    durations = _수리완료_durations(failures)
    if not durations:
        return None
    return round(float(np.mean(durations)), 2)


def mttr_km(
    failures: pd.DataFrame, as_of: Optional[object] = None
) -> Optional[float]:
    """
    Kaplan-Meier로 우측절단을 처리한 평균 수리시간(MTTR).

    오류 #3 정정: 진행 중('수리중') 수리건을 우측절단(censored=True)으로 포함한다.
      - '수리완료' 건: duration=수리소요시간_h, event_observed=1 (관측)
      - '수리중'  건: duration=(as_of - 수리시작시간), event_observed=0 (절단)
        (수리시작시간이 없으면 절단 관측을 구성할 수 없어 제외)
      - '미해결'  건: 능동적 수리가 진행되는 상태가 아니므로 MTTR 표본에서 제외한다.
        (사용자 결정 (b): 수리중만 절단, 미해결 제외)

    MTTR 추정 = 제한평균생존시간(RMST) = ∫₀^{t_max} S(t) dt  (스텝 적분).
    as_of 미지정 시 데이터 내 최종 시각(수리완료/수리시작/발생일시의 최댓값)을
    기준시각으로 사용해 결정론성을 유지한다.
    관측된 완료건이 하나도 없으면(KM 추정 불가) None 반환.

    근거: 우측절단 하 신뢰성 추정의 표준(비모수 Kaplan-Meier).
    """
    if failures is None or failures.empty:
        return None

    durations = _수리완료_durations(failures)
    events = [1] * len(durations)

    # '수리중'(진행 중) 건만 우측절단으로 편입 ('미해결'은 제외; 수리시작시간 필요)
    수리중 = failures[failures["처리상태"] == "수리중"]
    if "수리시작시간" in failures.columns and not 수리중.empty:
        start = pd.to_datetime(수리중["수리시작시간"], errors="coerce")

        if as_of is None:
            후보 = []
            for col in ("수리완료시간", "수리시작시간", "발생일시"):
                if col in failures.columns:
                    후보.append(pd.to_datetime(failures[col], errors="coerce"))
            기준 = pd.concat(후보) if 후보 else pd.Series(dtype="datetime64[ns]")
            as_of_ts = 기준.max()
        else:
            as_of_ts = pd.to_datetime(as_of)

        if pd.notna(as_of_ts):
            elapsed = (as_of_ts - start).dt.total_seconds() / 3600
            for e in elapsed.dropna():
                if e > 0:
                    durations.append(float(e))
                    events.append(0)

    # 관측된 완료 사건이 없으면 KM 추정 불가
    if not durations or sum(events) == 0:
        return None

    from lifelines import KaplanMeierFitter
    from lifelines.utils import restricted_mean_survival_time

    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events)
    t_max = max(durations)
    mttr = float(restricted_mean_survival_time(kmf, t=t_max))
    return round(mttr, 2)


# ===========================================================================
# 가동도 (Ai / Ao)
# ===========================================================================

def availability_inherent(
    mtbf: Optional[float], mttr: Optional[float]
) -> Optional[float]:
    """
    고유가용도 Ai = MTBF / (MTBF + MTTR).  [설계단계 고유가용도]

    예방정비·군수지연(ALDT)을 포함하지 않는다(운용가용도 Ao 아님).
    입력이 None이거나 분모가 0이면 None 반환.
    """
    if mtbf is None or mttr is None:
        return None
    분모 = mtbf + mttr
    if 분모 == 0:
        return None
    return round(mtbf / 분모, 4)


def availability_operational(
    mtbm: Optional[float], mdt: Optional[float]
) -> Optional[float]:
    """
    운용가용도 Ao = MTBM / (MTBM + MDT).

    MTBM(평균정비간격)·MDT(정비중단시간, ALDT 포함) 중 하나라도 결측이면
    None 반환한다. 결측 데이터를 임의값으로 대체해 억지 계산하지 않는다.
    (현 단계 샘플데이터는 MDT/ALDT가 없어 실무상 None이 정상이다.)
    """
    if mtbm is None or mdt is None:
        return None
    분모 = mtbm + mdt
    if 분모 == 0:
        return None
    return round(mtbm / 분모, 4)


def mean_mdt(failures: pd.DataFrame) -> Optional[float]:
    """평균 정비중단시간(MDT) = mdt_hours 평균. 전부 결측/부재면 None."""
    if failures is None or failures.empty or "mdt_hours" not in failures.columns:
        return None
    vals = pd.to_numeric(failures["mdt_hours"], errors="coerce").dropna()
    if vals.empty:
        return None
    return round(float(vals.mean()), 2)
