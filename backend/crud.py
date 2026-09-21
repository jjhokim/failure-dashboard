"""
CRUD 및 분석 지표 계산 모듈

구성:
  1. 입력     - 고장이력 단건 저장, 기존 레코드 수정
  2. 조회     - 전체/필터 조회, 단건 조회
  3. 삭제     - 단건 삭제
  4. 분석     - MTBF, MTTR, Ai(고유가용도) 계산
  5. 집계     - 파레토, 월별 트렌드, 제대별 비교, KPI 요약

반환 타입:
  - 조회·집계 함수는 모두 pandas DataFrame 반환 (Streamlit에서 바로 사용)
  - KPI 요약은 dict 반환
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from backend import metrics
from backend.database import get_connection
from backend.models import BIT이벤트, LCN노드, 고장이력, 장비


# ===========================================================================
# 1. 입력
# ===========================================================================

def 고장이력_저장(record: 고장이력) -> int:
    """
    고장이력 단건을 DB에 저장하고, 생성된 id를 반환한다.

    Parameters
    ----------
    record : 고장이력
        models.py의 고장이력 dataclass 인스턴스.
        id 필드는 무시된다 (AUTOINCREMENT).

    Returns
    -------
    int
        DB에서 할당된 새 레코드의 id.
    """
    sql = """
        INSERT INTO 고장이력 (
            제대구분, 발생일시, 체계명, LRU명, 고장유형, 고장증상,
            수리시작시간, 수리완료시간, 정비조치내용, 처리상태, 등록일시,
            op_hours_at_failure, aldt_hours, mdt_hours, lcn,
            effect_class, narrative, system_id, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        record.제대구분,
        _fmt_dt(record.발생일시),
        record.체계명,
        record.LRU명,
        record.고장유형,
        record.고장증상,
        _fmt_dt(record.수리시작시간),
        _fmt_dt(record.수리완료시간),
        record.정비조치내용,
        record.처리상태,
        _fmt_dt(record.등록일시),
        # --- Phase 0-2 확장 필드 ---
        record.op_hours_at_failure,
        record.aldt_hours,
        record.mdt_hours,
        record.lcn,
        record.effect_class,
        record.narrative,
        record.system_id,
        record.source,
    )
    with get_connection() as conn:
        cursor = conn.execute(sql, params)
        return cursor.lastrowid


_고장이력_삽입컬럼: list[str] = [
    "제대구분", "발생일시", "체계명", "LRU명", "고장유형", "고장증상",
    "수리시작시간", "수리완료시간", "정비조치내용", "처리상태", "등록일시",
    "op_hours_at_failure", "aldt_hours", "mdt_hours", "lcn",
    "effect_class", "narrative", "system_id", "source",
]


def 고장이력_DataFrame저장(표준df: pd.DataFrame, source: str = "import") -> int:
    """
    표준화된 DataFrame(adapters.표준화 결과)을 고장이력에 일괄 저장한다.

    - 없는 컬럼은 NULL로 채운다. 처리상태 결측은 '미해결', 등록일시는 현재시각.
    - 모든 레코드의 source를 지정값(기본 'import')으로 기록한다.
    반환: 저장 건수.
    """
    if 표준df is None or 표준df.empty:
        return 0

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    values: list[tuple] = []
    for _, r in 표준df.iterrows():
        d = {c: (None if (c not in 표준df.columns or pd.isna(r[c])) else r[c])
             for c in _고장이력_삽입컬럼}
        d["처리상태"] = d.get("처리상태") or "미해결"
        d["등록일시"] = now
        d["source"]  = source
        values.append(tuple(d[c] for c in _고장이력_삽입컬럼))

    placeholders = ", ".join(["?"] * len(_고장이력_삽입컬럼))
    cols = ", ".join(_고장이력_삽입컬럼)
    with get_connection() as conn:
        conn.executemany(
            f"INSERT INTO 고장이력 ({cols}) VALUES ({placeholders})", values
        )
    return len(values)


def 고장이력_수정(id: int, 수정내용: dict) -> None:
    """
    기존 레코드를 부분 수정한다.

    Parameters
    ----------
    id : int
        수정할 레코드의 id.
    수정내용 : dict
        {컬럼명: 새값} 형태. 예) {"처리상태": "수리완료", "수리완료시간": "2026-01-01 12:00:00"}
    """
    if not 수정내용:
        return

    # datetime 객체가 들어오면 문자열로 변환
    normalized = {k: (_fmt_dt(v) if isinstance(v, datetime) else v)
                  for k, v in 수정내용.items()}

    set_clause = ", ".join(f"{col} = ?" for col in normalized)
    values     = list(normalized.values()) + [id]

    with get_connection() as conn:
        conn.execute(f"UPDATE 고장이력 SET {set_clause} WHERE id = ?", values)


# ===========================================================================
# 2. 조회
# ===========================================================================

def 고장이력_전체조회(
    제대구분:   Optional[str]  = None,
    체계명:     Optional[str]  = None,
    고장유형:   Optional[str]  = None,
    처리상태:   Optional[str]  = None,
    날짜_시작:  Optional[str]  = None,   # "YYYY-MM-DD"
    날짜_종료:  Optional[str]  = None,   # "YYYY-MM-DD"
) -> pd.DataFrame:
    """
    필터 조건에 맞는 고장이력을 DataFrame으로 반환한다.
    모든 필터는 선택적이며, 지정하지 않으면 전체 조회된다.

    Returns
    -------
    pd.DataFrame
        발생일시 내림차순 정렬. 수리소요시간_h 파생 컬럼 포함.
    """
    sql    = "SELECT * FROM 고장이력 WHERE 1=1"
    params: list = []

    if 제대구분:
        sql += " AND 제대구분 = ?"
        params.append(제대구분)
    if 체계명:
        sql += " AND 체계명 = ?"
        params.append(체계명)
    if 고장유형:
        sql += " AND 고장유형 = ?"
        params.append(고장유형)
    if 처리상태:
        sql += " AND 처리상태 = ?"
        params.append(처리상태)
    if 날짜_시작:
        sql += " AND 발생일시 >= ?"
        params.append(f"{날짜_시작} 00:00:00")
    if 날짜_종료:
        sql += " AND 발생일시 <= ?"
        params.append(f"{날짜_종료} 23:59:59")

    sql += " ORDER BY 발생일시 DESC"

    with get_connection() as conn:
        df = pd.read_sql(sql, conn, params=params)

    return _add_파생컬럼(df)


def 고장이력_단건조회(id: int) -> Optional[pd.Series]:
    """id로 단건을 조회해 Series로 반환한다. 없으면 None."""
    with get_connection() as conn:
        df = pd.read_sql(
            "SELECT * FROM 고장이력 WHERE id = ?", conn, params=[id]
        )
    if df.empty:
        return None
    return _add_파생컬럼(df).iloc[0]


def 미해결_고장_조회(limit: int = 10) -> pd.DataFrame:
    """
    처리상태가 '수리중' 또는 '미해결'인 건을 발생일시 오름차순으로 반환한다.
    대시보드 요약 패널의 '최근 미해결 목록'에 사용된다.
    """
    sql = """
        SELECT * FROM 고장이력
        WHERE 처리상태 IN ('수리중', '미해결')
        ORDER BY 발생일시 ASC
        LIMIT ?
    """
    with get_connection() as conn:
        df = pd.read_sql(sql, conn, params=[limit])
    return _add_파생컬럼(df)


# ===========================================================================
# 3. 삭제
# ===========================================================================

def 고장이력_삭제(id: int) -> None:
    """id에 해당하는 레코드를 삭제한다."""
    with get_connection() as conn:
        conn.execute("DELETE FROM 고장이력 WHERE id = ?", [id])


# ===========================================================================
# 4. 분석 지표 계산
# ===========================================================================

# 지표 계산 로직은 backend/metrics.py 순수 함수에 집중한다(원칙 #3).
# 아래는 기존 호출부·테스트 호환을 위한 얇은 위임 래퍼다.

def MTTR_계산(df: pd.DataFrame) -> Optional[float]:
    """
    [레거시·비교용] 수리완료 건 단순 평균 MTTR → metrics.mttr_naive 위임.

    ⚠ 우측절단 미처리(오류 #3). 정식 산출은 metrics.mttr_km(KPI_요약이 사용).
    """
    return metrics.mttr_naive(df)


def MTBF_계산(df: pd.DataFrame, 관측기간_h: Optional[float] = None) -> float:
    """
    [레거시·폴백용] 달력 경과시간 기반 MTBF → metrics.mtbf_calendar 위임.

    ⚠ 운용시간 기준 아님(오류 #2). 운용시간 데이터가 있으면
    metrics.mtbf_logistics를 사용한다(KPI_요약이 자동 선택).
    """
    return metrics.mtbf_calendar(df, 관측기간_h)


def 고유가용도_Ai_계산(mtbf: float, mttr: Optional[float]) -> Optional[float]:
    """고유가용도 Ai = MTBF/(MTBF+MTTR) → metrics.availability_inherent 위임."""
    return metrics.availability_inherent(mtbf, mttr)


# ===========================================================================
# 5. 집계
# ===========================================================================

def KPI_요약(
    df: pd.DataFrame,
    systems: Optional[pd.DataFrame] = None,
    관측기간_h: Optional[float] = None,
) -> dict:
    """
    대시보드 KPI 카드용 요약 지표를 반환한다 (계산은 metrics.py 위임).

    - MTBF: 운용시간 기반(mtbf_logistics) 우선, systems 미등록 시 달력근사 폴백.
    - MTTR: Kaplan-Meier 절단보정(mttr_km). lifelines 미설치 시 naive 폴백.
    - Ai  : availability_inherent. Ao: availability_operational(MDT 결측 시 None).

    Parameters
    ----------
    systems : pd.DataFrame, optional
        장비 모집단. 미지정 시 장비_전체조회()로 자동 조회.

    Returns
    -------
    dict
        총_고장건수 / 미완료_건수 / MTBF_h / MTBF_기준 /
        MTTR_h / MTTR_완료건수 / 가용도_Ai / 가용도_Ao
    """
    if systems is None:
        try:
            systems = 장비_전체조회()
        except Exception:
            systems = pd.DataFrame()

    # MTBF: 운용시간 기반 우선, 없으면 달력근사
    mtbf_log = metrics.mtbf_logistics(df, systems)
    if mtbf_log is not None:
        mtbf, mtbf_기준 = mtbf_log, "운용시간"
    else:
        mtbf, mtbf_기준 = metrics.mtbf_calendar(df, 관측기간_h), "달력근사"

    # MTTR: KM 절단보정 (lifelines 미설치 시 naive 폴백)
    try:
        mttr = metrics.mttr_km(df)
    except ImportError:
        mttr = metrics.mttr_naive(df)

    ai = metrics.availability_inherent(mtbf, mttr)

    # Ao: MTBM·MDT 필요. 현재 MTBM 미산출·MDT 결측이므로 실무상 None.
    ao = metrics.availability_operational(None, metrics.mean_mdt(df))

    미완료 = df[df["처리상태"] != "수리완료"]
    if df.empty:
        mttr_완료건수 = 0
    else:
        mttr_완료건수 = int(
            ((df["처리상태"] == "수리완료") & df["수리소요시간_h"].notna()).sum()
        )

    return {
        "총_고장건수":   len(df),
        "미완료_건수":   len(미완료),
        "MTBF_h":        mtbf,
        "MTBF_기준":     mtbf_기준,
        "MTTR_h":        mttr,
        "MTTR_완료건수": mttr_완료건수,
        "가용도_Ai":     ai,
        "가용도_Ao":     ao,
    }


def 월별_고장건수(df: pd.DataFrame) -> pd.DataFrame:
    """
    월별 고장 트렌드 집계.

    Returns
    -------
    pd.DataFrame
        columns: ["연월", "고장건수", "제대구분"]
        연월 형식: "YYYY-MM"
    """
    if df.empty:
        return pd.DataFrame(columns=["연월", "고장건수", "제대구분"])

    tmp = df.copy()
    tmp["연월"] = pd.to_datetime(tmp["발생일시"], errors="coerce").dt.strftime("%Y-%m")
    result = (
        tmp.groupby(["연월", "제대구분"], as_index=False)
           .size()
           .rename(columns={"size": "고장건수"})
           .sort_values("연월")
    )
    return result


def 파레토_분석(df: pd.DataFrame, 기준컬럼: str = "고장유형") -> pd.DataFrame:
    """
    파레토 분석 — 고장유형(또는 LRU명)별 빈도 및 누적 비율 계산.

    Parameters
    ----------
    df : pd.DataFrame
        고장이력_전체조회() 결과.
    기준컬럼 : str
        "고장유형" 또는 "LRU명".

    Returns
    -------
    pd.DataFrame
        columns: [기준컬럼, "건수", "비율_%", "누적비율_%"]
        건수 내림차순 정렬.
    """
    if df.empty or 기준컬럼 not in df.columns:
        return pd.DataFrame(columns=[기준컬럼, "건수", "비율_%", "누적비율_%"])

    freq = (
        df[기준컬럼]
        .value_counts()
        .reset_index()
        .rename(columns={"index": 기준컬럼, "count": "건수",
                         기준컬럼: 기준컬럼, "proportion": "비율_%"})
    )
    # pandas 버전 호환: value_counts() 결과 컬럼명 정규화
    if "count" not in freq.columns and freq.shape[1] == 2:
        freq.columns = [기준컬럼, "건수"]

    freq = freq.sort_values("건수", ascending=False).reset_index(drop=True)
    전체 = freq["건수"].sum()
    freq["비율_%"]    = (freq["건수"] / 전체 * 100).round(1)
    freq["누적비율_%"] = freq["비율_%"].cumsum().round(1)
    return freq


def 제대별_MTTR(df: pd.DataFrame) -> pd.DataFrame:
    """
    제대구분별 평균 수리시간(MTTR)과 완료/미완료 건수를 병기한다.

    - MTTR_h    : 해당 제대의 '수리완료' 건 중 수리소요시간_h 평균.
    - 완료건수  : 처리상태 == '수리완료' 이고 수리소요시간_h 가 유효한 건수.
    - 미완료건수: 처리상태 != '수리완료' 건수.

    주의(알려진 오류 #3): 현재 MTTR은 완료건만 평균에 반영하므로
    우측절단(미완료 수리건 누락) 편향으로 MTTR 과소·Ai 과대 가능성이 있다.
    Phase 0-4에서 lifelines Kaplan-Meier로 절단 처리 예정.
    완료/미완료 건수를 병기하는 것은 이 편향을 화면에서 드러내기 위함이다.

    Returns
    -------
    pd.DataFrame
        columns: ["제대구분", "MTTR_h", "완료건수", "미완료건수"]
    """
    cols = ["제대구분", "MTTR_h", "완료건수", "미완료건수"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for 제대, g in df.groupby("제대구분"):
        완료 = g[(g["처리상태"] == "수리완료") & g["수리소요시간_h"].notna()]
        미완료 = g[g["처리상태"] != "수리완료"]
        mttr = round(완료["수리소요시간_h"].mean(), 2) if not 완료.empty else None
        rows.append({
            "제대구분":   제대,
            "MTTR_h":     mttr,
            "완료건수":   int(len(완료)),
            "미완료건수": int(len(미완료)),
        })

    return (
        pd.DataFrame(rows, columns=cols)
        .sort_values("제대구분")
        .reset_index(drop=True)
    )


def 체계별_MTTR(df: pd.DataFrame) -> pd.DataFrame:
    """
    체계명 × 제대구분별 평균 수리소요시간 집계.

    Returns
    -------
    pd.DataFrame
        columns: ["체계명", "제대구분", "평균MTTR_h", "고장건수"]
    """
    완료건 = df[df["처리상태"] == "수리완료"].copy()
    if 완료건.empty:
        return pd.DataFrame(columns=["체계명", "제대구분", "평균MTTR_h", "고장건수"])

    result = (
        완료건.groupby(["체계명", "제대구분"], as_index=False)
              .agg(평균MTTR_h=("수리소요시간_h", "mean"), 고장건수=("id", "count"))
    )
    result["평균MTTR_h"] = result["평균MTTR_h"].round(2)
    return result.sort_values(["체계명", "제대구분"])


def 제대별_고장현황(df: pd.DataFrame) -> pd.DataFrame:
    """
    제대구분 × 처리상태별 고장건수 집계.
    대시보드의 제대별 고장현황 비교 차트에 사용된다.

    Returns
    -------
    pd.DataFrame
        columns: ["제대구분", "처리상태", "건수"]
    """
    if df.empty:
        return pd.DataFrame(columns=["제대구분", "처리상태", "건수"])

    result = (
        df.groupby(["제대구분", "처리상태"], as_index=False)
          .size()
          .rename(columns={"size": "건수"})
    )
    return result.sort_values(["제대구분", "처리상태"])


def 불가동_추세(df: pd.DataFrame) -> pd.DataFrame:
    """
    월별 불가동 건수 추세 집계 (처리상태 = '수리중' 또는 '미해결').

    Returns
    -------
    pd.DataFrame
        columns: ["연월", "체계명", "불가동건수"]
    """
    불가동 = df[df["처리상태"] != "수리완료"].copy()
    if 불가동.empty:
        return pd.DataFrame(columns=["연월", "체계명", "불가동건수"])

    불가동["연월"] = pd.to_datetime(불가동["발생일시"], errors="coerce").dt.strftime("%Y-%m")
    result = (
        불가동.groupby(["연월", "체계명"], as_index=False)
              .size()
              .rename(columns={"size": "불가동건수"})
              .sort_values("연월")
    )
    return result


# ===========================================================================
# 6. 확장 테이블 CRUD (Phase 0-2: systems / lcn_tree / bit_events)
# ===========================================================================
#
# 스키마를 실제로 사용·검증 가능하게 하는 최소 입출력 함수.
# 본격적인 활용(운용시간 기반 MTBF, LCN 구조거리, BIT 조인)은
# Phase 0-4 / Phase 5에서 이 함수들 위에 구현한다.

def 장비_저장(장비목록: "장비 | list[장비]") -> int:
    """
    systems(장비 모집단) 레코드를 일괄 저장한다 (system_id 기준 UPSERT).
    단건 또는 리스트를 받아 저장 건수를 반환한다.
    """
    목록 = list(장비목록) if isinstance(장비목록, (list, tuple)) else [장비목록]
    if not 목록:
        return 0
    sql = """
        INSERT OR REPLACE INTO systems
            (system_id, system_type, echelon, commissioned_at, cumulative_op_hours)
        VALUES (?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.executemany(sql, [
            (s.system_id, s.system_type, s.echelon,
             _fmt_dt(s.commissioned_at), s.cumulative_op_hours)
            for s in 목록
        ])
    return len(목록)


def 장비_전체조회() -> pd.DataFrame:
    """systems 테이블 전체를 DataFrame으로 반환한다."""
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM systems ORDER BY system_id", conn)


def LCN트리_저장(노드목록: "LCN노드 | list[LCN노드]") -> int:
    """lcn_tree 레코드를 일괄 저장한다 (lcn 기준 UPSERT). 저장 건수 반환."""
    목록 = list(노드목록) if isinstance(노드목록, (list, tuple)) else [노드목록]
    if not 목록:
        return 0
    sql = """
        INSERT OR REPLACE INTO lcn_tree (lcn, parent_lcn, level, description)
        VALUES (?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.executemany(
            sql, [(n.lcn, n.parent_lcn, n.level, n.description) for n in 목록]
        )
    return len(목록)


def LCN트리_전체조회() -> pd.DataFrame:
    """lcn_tree 테이블 전체를 DataFrame으로 반환한다."""
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM lcn_tree ORDER BY lcn", conn)


def BIT이벤트_저장(이벤트목록: "BIT이벤트 | list[BIT이벤트]") -> int:
    """
    bit_events 레코드를 일괄 저장한다 (event_id 자동증가). 저장 건수 반환.
    ※ R3: BIT는 탐지 채널 입력이 아니라 Phase 5 조인(LCN·시각)용이다.
    """
    목록 = list(이벤트목록) if isinstance(이벤트목록, (list, tuple)) else [이벤트목록]
    if not 목록:
        return 0
    sql = """
        INSERT INTO bit_events (system_id, lcn, bit_code, occurred_at)
        VALUES (?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.executemany(
            sql,
            [(e.system_id, e.lcn, e.bit_code, _fmt_dt(e.occurred_at)) for e in 목록],
        )
    return len(목록)


def BIT이벤트_전체조회() -> pd.DataFrame:
    """bit_events 테이블 전체를 DataFrame으로 반환한다."""
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM bit_events ORDER BY occurred_at", conn)


# ===========================================================================
# 내부 헬퍼
# ===========================================================================

def _fmt_dt(value) -> Optional[str]:
    """datetime → 'YYYY-MM-DD HH:MM:SS' 문자열 변환. None이면 None 반환."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value  # 이미 문자열인 경우


def _add_파생컬럼(df: pd.DataFrame) -> pd.DataFrame:
    """
    조회 결과 DataFrame에 분석용 파생 컬럼을 추가한다.

    추가 컬럼:
      - 수리소요시간_h : 수리완료시간 - 수리시작시간 (시간 단위, float)
      - 발생_연월      : 발생일시의 "YYYY-MM" 문자열 (월별 집계용)
    """
    if df.empty:
        df["수리소요시간_h"] = pd.Series(dtype=float)
        df["발생_연월"]     = pd.Series(dtype=str)
        return df

    시작 = pd.to_datetime(df["수리시작시간"], errors="coerce")
    완료 = pd.to_datetime(df["수리완료시간"], errors="coerce")
    df["수리소요시간_h"] = (완료 - 시작).dt.total_seconds() / 3600

    df["발생_연월"] = (
        pd.to_datetime(df["발생일시"], errors="coerce").dt.strftime("%Y-%m")
    )
    return df
